import base64
import io
import os
from typing import Optional

import requests
from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from PIL import Image

WORKER_URL = os.getenv("FLUX_WORKER_URL", "https://dry-butterfly-fc08.sasimankan12man.workers.dev/")


def _b64_to_image(data_url: str) -> Image.Image:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid data URL")
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def _image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


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

    response = requests.post(WORKER_URL, json=payload, timeout=300)
    if not response.ok:
        content_type = (response.headers.get("content-type") or "").lower()
        details = response.text[:3000] if ("json" in content_type or "text" in content_type) else ""
        raise RuntimeError(f"Worker failed ({response.status_code}): {details}")

    return Image.open(io.BytesIO(response.content)).convert("RGBA")


def _composite_only_mask(original: Image.Image, cleaned: Image.Image, mask: Image.Image) -> Image.Image:
    w, h = original.size
    cleaned = cleaned.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")
    alpha = mask.convert("L").resize((w, h), Image.Resampling.NEAREST)
    return Image.composite(cleaned, original.convert("RGBA"), alpha)


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
        seed_raw = request.POST.get("seed", "").strip()
        seed = int(seed_raw) if seed_raw else None

        original = _b64_to_image(image_data)
        mask = _b64_to_image(mask_data)

        cleaned_512 = _call_flux(original, mask, guidance=guidance, seed=seed)
        output = _composite_only_mask(original, cleaned_512, mask)

        return JsonResponse({"ok": True, "result": _image_to_data_url(output)})
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)
