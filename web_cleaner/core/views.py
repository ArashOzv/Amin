import base64
import io
import json
import logging
import os
from typing import Optional

import requests
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

WORKER_URL = os.getenv("FLUX_WORKER_URL", "https://dry-butterfly-fc08.sasimankan12man.workers.dev/")


class WorkerResponseError(RuntimeError):
    pass


def _b64_to_image(data_url: str) -> Image.Image:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid data URL payload")
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def _image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _parse_worker_error(response: requests.Response) -> str:
    content_type = (response.headers.get("content-type") or "").lower()
    req_id = response.headers.get("x-upstream-request-id") or response.headers.get("cf-ray") or ""

    if "application/json" in content_type:
        try:
            body = response.json()
            details = body.get("details") or body.get("error") or json.dumps(body)
        except Exception:
            details = response.text[:2000]
    elif "text" in content_type:
        details = response.text[:2000]
    else:
        details = f"non-text response content_type={content_type or 'unknown'}"

    req_part = f" request_id={req_id}" if req_id else ""
    return f"Worker failed ({response.status_code}).{req_part} {details}".strip()


def _parse_worker_image(response: requests.Response) -> Image.Image:
    content_type = (response.headers.get("content-type") or "").lower()
    req_id = response.headers.get("x-upstream-request-id") or response.headers.get("cf-ray") or ""

    if content_type.startswith("image/"):
        try:
            return Image.open(io.BytesIO(response.content)).convert("RGBA")
        except UnidentifiedImageError as exc:
            raise WorkerResponseError("Worker returned invalid image bytes") from exc

    if "application/json" in content_type:
        try:
            body = response.json()
        except Exception as exc:
            raise WorkerResponseError("Worker returned malformed JSON") from exc

        message = body.get("error") or body.get("note") or "Worker returned JSON instead of image"
        details = body.get("details")
        req_hint = body.get("request_id") or req_id
        text = message if not details else f"{message}: {details}"
        if req_hint:
            text = f"{text} (request_id={req_hint})"
        raise WorkerResponseError(text)

    preview = response.text[:300] if response.text else ""
    raise WorkerResponseError(
        f"Worker returned unsupported content type '{content_type or 'unknown'}' with preview: {preview}"
    )


def _call_flux(image: Image.Image, mask: Image.Image, guidance: float = 2.0, seed: Optional[int] = None) -> Image.Image:
    prompt = (
        "Inpaint only the white area from image 2 mask. "
        "Remove text, speech bubble text, SFX, watermark marks, and restore clean background. "
        "Outside mask, keep image exactly unchanged."
    )

    src_512 = image.convert("RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    mask_512 = mask.convert("L").resize((512, 512), Image.Resampling.NEAREST).convert("RGBA")

    payload = {
        "prompt": prompt,
        "image": _image_to_data_url(src_512),
        "image1": _image_to_data_url(src_512),
        "image2": _image_to_data_url(mask_512),
        "width": 512,
        "height": 512,
        "guidance": float(guidance),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    try:
        response = requests.post(WORKER_URL, json=payload, timeout=300)
    except requests.RequestException as exc:
        raise WorkerResponseError(f"Could not reach FLUX worker: {exc}") from exc

    if not response.ok:
        raise WorkerResponseError(_parse_worker_error(response))

    try:
        return _parse_worker_image(response)
    except WorkerResponseError:
        raise
    except Exception as exc:
        raise WorkerResponseError(f"Unexpected worker response parsing error: {exc}") from exc


def _composite_only_mask(original: Image.Image, cleaned: Image.Image, mask: Image.Image) -> Image.Image:
    w, h = original.size
    cleaned = cleaned.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    alpha = mask.convert("L").resize((w, h), Image.Resampling.NEAREST)
    return Image.composite(cleaned, original.convert("RGBA"), alpha)


def _mask_has_content(mask: Image.Image) -> bool:
    """Return True only when user actually painted non-zero mask pixels.

    We check both luminance and alpha so this is stable across browsers,
    PNG encoders, and platform-specific PIL behaviors.
    """
    gray = mask.convert("L")
    alpha = mask.split()[-1] if mask.mode in ("RGBA", "LA") else None

    gray_has = gray.getbbox() is not None and (gray.getextrema() or (0, 0))[1] > 0
    alpha_has = False
    if alpha is not None:
        alpha_has = alpha.getbbox() is not None and (alpha.getextrema() or (0, 0))[1] > 0

    # If an alpha channel exists, users draw into visible alpha too.
    # Require some non-zero luminance, and allow alpha as extra signal.
    return gray_has or alpha_has


@require_GET
def index(request: HttpRequest):
    return render(request, "core/index.html")


@require_POST
def clean_selection(request: HttpRequest):
    try:
        image_data = request.POST.get("image")
        mask_data = request.POST.get("mask")

        if not image_data or not mask_data:
            return JsonResponse({"ok": False, "error": "image and mask are required"}, status=400)

        guidance = float(request.POST.get("guidance", "2.0"))
        if guidance < 1.0 or guidance > 6.0:
            return JsonResponse({"ok": False, "error": "guidance must be between 1.0 and 6.0"}, status=400)

        seed_raw = request.POST.get("seed", "").strip()
        seed = int(seed_raw) if seed_raw else None

        original = _b64_to_image(image_data)
        mask = _b64_to_image(mask_data)

        if not _mask_has_content(mask):
            return JsonResponse({"ok": False, "error": "Mask is empty. Draw over area to remove first."}, status=400)

        cleaned_512 = _call_flux(original, mask, guidance=guidance, seed=seed)
        output = _composite_only_mask(original, cleaned_512, mask)

        return JsonResponse({"ok": True, "result": _image_to_data_url(output)})
    except WorkerResponseError as exc:
        logger.warning("Worker processing error: %s", exc)
        return JsonResponse({"ok": False, "error": str(exc), "type": "worker_error"}, status=502)
    except ValueError as exc:
        return JsonResponse({"ok": False, "error": str(exc), "type": "validation_error"}, status=400)
    except Exception as exc:
        logger.exception("Unexpected cleaning error")
        return JsonResponse(
            {
                "ok": False,
                "error": "Unexpected server error while cleaning image",
                "details": str(exc),
                "type": "server_error",
            },
            status=500,
        )
