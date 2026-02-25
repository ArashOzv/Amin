from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal, Tuple

import cv2
import easyocr
import numpy as np
from PIL import Image

_READER_JP = easyocr.Reader(["ja", "en"], gpu=False)
_READER_KR = easyocr.Reader(["ko", "en"], gpu=False)
_READER_EN = easyocr.Reader(["en"], gpu=False)


def _reader(lang: str):
    if lang == "jp":
        return _READER_JP
    if lang == "kr":
        return _READER_KR
    return _READER_EN


def _vertical_tiles(h: int, tile_h: int, overlap: int) -> Iterator[Tuple[int, int]]:
    tile_h = max(512, int(tile_h))
    overlap = max(0, int(overlap))
    step = max(1, tile_h - overlap)
    y = 0
    while y < h:
        y0 = y
        y1 = min(h, y0 + tile_h)
        yield y0, y1
        if y1 >= h:
            break
        y += step


def _clahe_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _close(mask: np.ndarray, ks: int) -> np.ndarray:
    if ks <= 1:
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=1)


def _open(mask: np.ndarray, ks: int) -> np.ndarray:
    if ks <= 1:
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=1)


def _expand_by_distance(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    m = (mask > 127).astype(np.uint8)
    inv = 1 - m
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    add = (dist <= float(px)).astype(np.uint8) * 255
    return np.maximum(mask, add)


def _keep_components(mask: np.ndarray, min_area: int, max_frac: float = 0.35) -> np.ndarray:
    _, m = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = m.shape[:2]
    img_area = h * w
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    out = np.zeros_like(m)
    for i in range(1, num):
        _, _, _, _, area = stats[i]
        if area < min_area or area > int(img_area * max_frac):
            continue
        out[labels == i] = 255
    return out


def build_bubble_mask(
    bgr: np.ndarray,
    language: str,
    *,
    tile_h: int = 2200,
    overlap: int = 420,
    ocr_scales: Tuple[float, ...] = (1.0, 1.25, 1.5),
    conf: float = 0.20,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    r = _reader(language)
    full_text = np.zeros((h, w), dtype=np.uint8)

    for y0, y1 in _vertical_tiles(h, tile_h, overlap):
        tile = bgr[y0:y1]
        tile_h_now = y1 - y0

        for s in ocr_scales:
            if s == 1.0:
                tile_s = tile
                scale_back = 1.0
            else:
                tile_s = cv2.resize(tile, (int(w * s), int(tile_h_now * s)), interpolation=cv2.INTER_CUBIC)
                scale_back = 1.0 / s

            gray = _clahe_gray(tile_s)
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=32, sigmaSpace=32)

            for bbox, txt, score in r.readtext(gray):
                if score < conf or not str(txt).strip():
                    continue

                pts = np.array(bbox, dtype=np.float32) * scale_back
                pts = np.round(pts).astype(np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)

                pad_x = max(8, int(bw * 0.08))
                pad_y = max(8, int(bh * 0.15))
                x1 = max(0, x - pad_x)
                y1r = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2r = min(tile_h_now, y + bh + pad_y)
                cv2.rectangle(full_text, (x1, y0 + y1r), (x2, y0 + y2r), 255, -1)

    full_text = _close(full_text, 17)
    full_text = _open(full_text, 3)
    full_text = _keep_components(full_text, min_area=180)

    # Bubble interior expansion: only in bright local regions near text
    gray_full = _clahe_gray(bgr)
    blur = cv2.GaussianBlur(gray_full, (5, 5), 0)
    bright_thr = int(np.percentile(blur, 66))
    bright = (blur >= bright_thr).astype(np.uint8) * 255
    bright = _open(bright, 3)

    seeds = cv2.dilate(full_text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    grown = cv2.bitwise_and(bright, seeds)
    grown = _close(grown, 11)

    bubble = cv2.bitwise_or(full_text, grown)
    bubble = _expand_by_distance(bubble, 4)
    bubble = _keep_components(bubble, min_area=220, max_frac=0.30)
    return bubble


def build_sfx_mask(bgr: np.ndarray, *, tile_h: int = 2200, overlap: int = 420) -> np.ndarray:
    h, w = bgr.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)

    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(24000)

    for y0, y1 in _vertical_tiles(h, tile_h, overlap):
        tile = bgr[y0:y1]
        th = y1 - y0
        gray = _clahe_gray(tile)

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

        _, bh_thr = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, gr_thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        regions, _ = mser.detectRegions(gray)
        mser_mask = np.zeros((th, w), dtype=np.uint8)
        for pts in regions:
            x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
            area = bw * bh
            if area < 120:
                continue
            if bw > w * 0.85 and bh > th * 0.25:
                continue
            cv2.rectangle(mser_mask, (x, y), (x + bw, y + bh), 255, -1)

        comb = cv2.bitwise_or(bh_thr, gr_thr)
        comb = cv2.bitwise_or(comb, mser_mask)
        comb = _close(comb, 19)
        comb = _open(comb, 3)
        comb = _expand_by_distance(comb, 3)
        comb = _keep_components(comb, min_area=170, max_frac=0.22)

        full[y0:y1, :] = cv2.bitwise_or(full[y0:y1, :], comb)

    return full


def create_masks(
    image_path: str,
    out_bubbles: str = "mask_bubbles.png",
    out_sfx: str = "mask_sfx.png",
    out_all: str = "mask_all.png",
    language: Literal["jp", "kr", "en"] = "kr",
) -> None:
    bgr = cv2.imread(str(Path(image_path)))
    if bgr is None:
        raise FileNotFoundError(image_path)

    bubbles = build_bubble_mask(bgr, language)
    sfx = build_sfx_mask(bgr)

    sfx = cv2.bitwise_and(sfx, cv2.bitwise_not(bubbles))
    all_mask = cv2.bitwise_or(bubbles, sfx)

    Image.fromarray(bubbles).save(out_bubbles)
    Image.fromarray(sfx).save(out_sfx)
    Image.fromarray(all_mask).save(out_all)

    h, w = bubbles.shape[:2]

    def cov(m: np.ndarray) -> float:
        return 100.0 * float(np.count_nonzero(m)) / float(h * w)

    print("saved:", out_bubbles, out_sfx, out_all)
    print("coverage bubbles %:", cov(bubbles))
    print("coverage sfx %:", cov(sfx))
    print("coverage all %:", cov(all_mask))


if __name__ == "__main__":
    import sys

    image = sys.argv[1] if len(sys.argv) > 1 else "./inputs/page.png"
    lang = sys.argv[2] if len(sys.argv) > 2 else "kr"
    create_masks(image, language=lang)
