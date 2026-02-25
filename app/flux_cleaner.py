import base64
import io
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter


WORKER_URL = "https://dry-butterfly-fc08.sasimankan12man.workers.dev/"


# -----------------------------
# Worker client
# -----------------------------
def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _decode_worker_image(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGBA")


def call_flux_multi_ref(
    image0: Image.Image,
    image1: Image.Image,
    mask_hint: Image.Image,
    *,
    worker_url: str = WORKER_URL,
    model: str = "@cf/black-forest-labs/flux-2-klein-4b",
    seed: Optional[int] = None,
    guidance: float = 2.0,
    retries: int = 4,
    timeout_s: int = 300,
) -> Image.Image:
    prompt = (
        "Manga cleanup task. "
        "Image 1 is the source patch and image 2 is the strict mask hint. "
        "White area only: remove text, speech characters, SFX, captions, and watermark traces. "
        "Reconstruct original background naturally with matching gradients and texture. "
        "Black area must remain pixel-faithful to image 1. "
        "Do not alter composition, perspective, characters, line thickness, color palette, lighting, or shadows. "
        "No new objects, no blur, no stylization."
    )

    i0 = image0.convert("RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    i1 = image1.convert("RGBA").resize((512, 512), Image.Resampling.LANCZOS)
    i2 = mask_hint.convert("L").resize((512, 512), Image.Resampling.NEAREST).convert("RGBA")

    payload = {
        "prompt": prompt,
        "model": model,
        "image": pil_to_data_url(i0, "PNG"),
        "image1": pil_to_data_url(i1, "PNG"),
        "image2": pil_to_data_url(i2, "PNG"),
        "width": 512,
        "height": 512,
        "guidance": float(guidance),
    }
    if seed is not None:
        payload["seed"] = int(seed)

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(worker_url, json=payload, timeout=timeout_s)
            ct = (resp.headers.get("content-type") or "").lower()
            if not resp.ok:
                detail = resp.text[:3000] if ("json" in ct or "text" in ct) else ""
                raise RuntimeError(f"Worker HTTP {resp.status_code}: {detail}")
            return _decode_worker_image(resp.content)
        except Exception as exc:
            last_error = exc
            time.sleep(0.75 * attempt)
    raise RuntimeError(f"Worker failed after {retries} attempts") from last_error


# -----------------------------
# Mask geometry helpers
# -----------------------------
def ensure_mask_size(mask_path: str, width: int, height: int) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.shape[:2] != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def connected_boxes(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int, int]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: List[Tuple[int, int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((x, y, x + w, y + h, area))
    return out


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def square_crop_coords(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    pad: int,
    min_side: int,
    max_side: int,
) -> Tuple[int, int, int, int]:
    x1p = clamp(x1 - pad, 0, width)
    y1p = clamp(y1 - pad, 0, height)
    x2p = clamp(x2 + pad, 0, width)
    y2p = clamp(y2 + pad, 0, height)

    side = max(min_side, x2p - x1p, y2p - y1p)
    side = min(side, max_side, min(width, height))

    cx = (x1p + x2p) // 2
    cy = (y1p + y2p) // 2
    sx1 = clamp(cx - side // 2, 0, width - side)
    sy1 = clamp(cy - side // 2, 0, height - side)
    return sx1, sy1, sx1 + side, sy1 + side


def preprocess_mask(mask_u8: np.ndarray, dilate: int, smooth_px: int) -> np.ndarray:
    m = (mask_u8 > 127).astype(np.uint8) * 255
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        m = cv2.dilate(m, k, iterations=1)
    if smooth_px > 0:
        inv = 255 - (m > 127).astype(np.uint8) * 255
        dt = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        soft = (dt <= float(smooth_px)).astype(np.uint8) * 255
        m = np.maximum(m, soft)
    return m


# -----------------------------
# Image fusion helpers
# -----------------------------
def make_edit_patch_prefill_inpaint(patch_rgba: Image.Image, mask_u8: np.ndarray, noise_sigma: float = 1.75) -> Image.Image:
    patch = np.array(patch_rgba.convert("RGBA"), dtype=np.uint8)
    rgb = patch[..., :3]
    alpha = patch[..., 3]

    mask = (mask_u8 > 127).astype(np.uint8) * 255
    if cv2.countNonZero(mask) < 10:
        return patch_rgba.convert("RGBA")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    inpaint_telea = cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    inpaint_ns = cv2.inpaint(bgr, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
    inpainted = cv2.addWeighted(inpaint_telea, 0.7, inpaint_ns, 0.3, 0)

    inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB).astype(np.int16)
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, size=inpainted_rgb.shape).astype(np.int16)
        mask3 = mask > 0
        inpainted_rgb[mask3] = np.clip(inpainted_rgb[mask3] + noise[mask3], 0, 255)

    out = np.dstack([inpainted_rgb.astype(np.uint8), alpha])
    return Image.fromarray(out, "RGBA")


def color_match(clean: Image.Image, orig: Image.Image, mask_u8: np.ndarray) -> Image.Image:
    clean_a = np.array(clean.convert("RGBA"), dtype=np.float32)
    orig_a = np.array(orig.convert("RGBA"), dtype=np.float32)

    keep = (mask_u8 <= 127) & (orig_a[..., 3] > 0)
    if int(keep.sum()) < 500:
        return clean

    out = clean_a.copy()
    eps = 1e-6
    for c in range(3):
        o = orig_a[..., c][keep]
        k = clean_a[..., c][keep]
        o_mean, o_std = float(o.mean()), float(o.std())
        k_mean, k_std = float(k.mean()), float(k.std())
        scale = 1.0 if k_std < 1.0 else (o_std / (k_std + eps))
        shift = o_mean - scale * k_mean
        out[..., c] = out[..., c] * scale + shift

    out[..., :3] = np.clip(out[..., :3], 0, 255)
    out[..., 3] = clean_a[..., 3]
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def blend_only_mask(orig: Image.Image, clean: Image.Image, mask_u8: np.ndarray, feather: int) -> Image.Image:
    alpha = Image.fromarray(mask_u8.astype(np.uint8), "L")
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    return Image.composite(clean.convert("RGBA"), orig.convert("RGBA"), alpha)


def poisson_blend_mask(orig_patch: Image.Image, clean_patch: Image.Image, mask_u8: np.ndarray, pad: int = 20) -> Image.Image:
    orig_rgba = orig_patch.convert("RGBA")
    clean_rgba = clean_patch.convert("RGBA")

    orig = np.array(orig_rgba.convert("RGB"), dtype=np.uint8)
    clean = np.array(clean_rgba.convert("RGB"), dtype=np.uint8)
    h, w = orig.shape[:2]

    if clean.shape[:2] != (h, w):
        clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC)

    m = (mask_u8 > 127).astype(np.uint8) * 255
    if cv2.countNonZero(m) < 50:
        return orig_rgba

    border_touch = np.any(m[0, :] > 0) or np.any(m[-1, :] > 0) or np.any(m[:, 0] > 0) or np.any(m[:, -1] > 0)
    if border_touch:
        return blend_only_mask(orig_rgba, clean_rgba, m, feather=10)

    ys, xs = np.where(m > 0)
    cy = int(np.mean(ys)) if ys.size else h // 2
    cx = int(np.mean(xs)) if xs.size else w // 2
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    orig_p = cv2.copyMakeBorder(orig, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    clean_p = cv2.copyMakeBorder(clean, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    m_p = cv2.copyMakeBorder(m, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0)

    try:
        blended_p = cv2.seamlessClone(clean_p, orig_p, m_p, (cx + pad, cy + pad), cv2.NORMAL_CLONE)
        blended = blended_p[pad : pad + h, pad : pad + w]
        out = Image.fromarray(blended, "RGB").convert("RGBA")
        out.putalpha(orig_rgba.split()[-1])
        return out
    except cv2.error:
        return blend_only_mask(orig_rgba, clean_rgba, m, feather=10)


def seam_error(clean: Image.Image, orig: Image.Image, mask_u8: np.ndarray, ring: int = 8) -> float:
    """Compute boundary mismatch score between candidate and original patch.

    `clean` may come from FLUX at 512x512, while `orig` is patch-sized.
    We align `clean` + `mask_u8` to original size before scoring.
    """
    orig_w, orig_h = orig.size

    clean_rgb = clean.convert("RGB")
    if clean_rgb.size != (orig_w, orig_h):
        clean_rgb = clean_rgb.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    clean_np = np.array(clean_rgb, dtype=np.float32)
    orig_np = np.array(orig.convert("RGB"), dtype=np.float32)

    m = (mask_u8 > 127).astype(np.uint8) * 255
    if m.shape[:2] != (orig_h, orig_w):
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)

    outer = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1)), iterations=1)
    ring_m = (outer > 0) & (m == 0)

    diff = np.abs(clean_np - orig_np).mean(axis=2)
    if int(ring_m.sum()) < 20:
        return float(diff.mean())
    return float(diff[ring_m].mean())


def best_of_seeds(
    edit_patch: Image.Image,
    patch_orig: Image.Image,
    mask_hint: Image.Image,
    mask_np: np.ndarray,
    *,
    worker_url: str,
    guidance: float,
    seeds: Sequence[int],
) -> Image.Image:
    candidates: List[Tuple[float, Image.Image]] = []
    for s in seeds:
        out = call_flux_multi_ref(
            edit_patch,
            patch_orig,
            mask_hint,
            worker_url=worker_url,
            seed=s,
            guidance=guidance,
        )
        score = seam_error(out, patch_orig, mask_np)
        candidates.append((score, out))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


@dataclass
class Params:
    min_area: int = 220
    max_patches: int = 260
    min_side: int = 768
    max_side: int = 1536
    pad: int = 96

    mask_dilate: int = 7
    mask_smooth_px: int = 4
    blend_feather: int = 10

    guidance: float = 2.0
    seed: int = 123
    seed_variations: int = 2

    worker_url: str = WORKER_URL


def _seed_schedule(base_seed: int, n: int) -> List[int]:
    return [base_seed + i * 47 for i in range(max(1, n))]


def clean_long_page(page_path: str, mask_path: str, out_path: str, params: Params = Params()) -> None:
    page = Image.open(page_path).convert("RGBA")
    width, height = page.size
    base_mask = ensure_mask_size(mask_path, width, height)

    boxes = connected_boxes(base_mask, params.min_area)
    boxes.sort(key=lambda b: b[4], reverse=True)
    boxes = boxes[: params.max_patches]
    print(f"regions={len(boxes)} image=({width}, {height})")

    for idx, (x1, y1, x2, y2, area) in enumerate(boxes, start=1):
        sx1, sy1, sx2, sy2 = square_crop_coords(
            x1, y1, x2, y2, width, height, params.pad, params.min_side, params.max_side
        )
        side = sx2 - sx1

        patch_orig = page.crop((sx1, sy1, sx2, sy2)).convert("RGBA")
        mask_crop = base_mask[sy1:sy2, sx1:sx2].copy()
        if mask_crop.shape[:2] != (side, side):
            mask_crop = cv2.resize(mask_crop, (side, side), interpolation=cv2.INTER_NEAREST)

        mask_crop = preprocess_mask(mask_crop, params.mask_dilate, params.mask_smooth_px)
        if cv2.countNonZero(mask_crop) < 40:
            continue

        edit_patch = make_edit_patch_prefill_inpaint(patch_orig, mask_crop)
        mask_hint = Image.fromarray(mask_crop, "L")

        seeds = _seed_schedule(params.seed + idx * 101, params.seed_variations)
        cleaned_512 = best_of_seeds(
            edit_patch,
            patch_orig,
            mask_hint,
            mask_crop,
            worker_url=params.worker_url,
            guidance=params.guidance,
            seeds=seeds,
        )
        cleaned_back = cleaned_512.resize((side, side), Image.Resampling.LANCZOS)
        cleaned_matched = color_match(cleaned_back, patch_orig, mask_crop)

        blended = poisson_blend_mask(patch_orig, cleaned_matched, mask_crop)
        if params.blend_feather > 0:
            blended = blend_only_mask(patch_orig, blended, mask_crop, params.blend_feather)

        page.alpha_composite(blended, dest=(sx1, sy1))
        if idx % 10 == 0:
            print(f"processed {idx}/{len(boxes)}")

    page.save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    clean_long_page(
        page_path="./inputs/page.png",
        mask_path="./inputs/mask_all.png",
        out_path="./outputs/cleaned_flux_long.png",
        params=Params(),
    )
