#!/usr/bin/env python3
"""
Core Document Orientation & Deskew Pipeline.
Provides fast coarse orientation (OSD + projection profile + concurrent OCR fallback)
and sub-degree fine deskewing.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import pytesseract
    _TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH
    _HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    _HAS_PYTESSERACT = False

log = logging.getLogger("orientation_core")

DEEP_OCR_MAX_DIM = 1200
OSD_MAX_DIM = 1200
FAST_ANALYSIS_MAX_DIM = 1200
MIN_OSD_CONFIDENCE = 8.0

_OCR_FALLBACK_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ocr_fallback")


@dataclass
class OrientationResult:
    path: str
    coarse_angle: int = 0
    coarse_method: str = "none"
    coarse_confidence: float = 0.0
    fine_angle: float = 0.0
    total_rotation: float = 0.0
    elapsed_time_ms: float = 0.0


def check_tesseract_installed() -> Tuple[bool, str]:
    """Check if Tesseract OCR is installed and accessible."""
    if not _HAS_PYTESSERACT:
        return False, "pytesseract Python library is not installed (run `pip install pytesseract`)."
    
    try:
        ver = pytesseract.get_tesseract_version()
        return True, f"Tesseract v{ver} available."
    except Exception as e:
        return False, (
            f"Tesseract executable not found or not in PATH ({e}).\n"
            "Please install Tesseract OCR:\n"
            "  - Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  - Linux: sudo apt install tesseract-ocr\n"
            "  - macOS: brew install tesseract"
        )


def normalize_360(angle: float) -> float:
    return float(angle % 360.0)


def normalize_signed_angle(angle: float) -> float:
    value = angle % 360.0
    if value >= 180.0:
        value -= 360.0
    return float(value)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Image is None.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if img.shape[2] == 3:
            return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def read_image(path: str) -> np.ndarray:
    """Unicode-safe image reader."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"Image file is empty: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot decode image: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> bool:
    """Unicode-safe image writer."""
    img = ensure_bgr(img)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        ext = ".png"
        path = path + ext
    success, encoded = cv2.imencode(ext, img)
    if not success:
        return False
    encoded.tofile(path)
    return True


def rotate_90(img: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k=-k))


def resize_max_dimension(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return img
    scale = max_dim / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def fast_analysis_binary(gray: np.ndarray) -> np.ndarray:
    small_gray = resize_max_dimension(gray, FAST_ANALYSIS_MAX_DIM)
    blurred = cv2.GaussianBlur(small_gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def fast_osd_orientation(gray: np.ndarray) -> Tuple[Optional[int], float]:
    if not _HAS_PYTESSERACT:
        return None, 0.0
    try:
        proc_gray = resize_max_dimension(gray, OSD_MAX_DIM)
        osd = pytesseract.image_to_osd(proc_gray, config="--psm 0")
        angle = None
        confidence = 0.0

        for line in osd.splitlines():
            line = line.strip()
            if line.lower().startswith("rotate:"):
                try:
                    angle = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.lower().startswith("orientation confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass

        if angle is None:
            return None, 0.0

        correction = (360 - angle) % 360
        correction = int(round(correction / 90.0) * 90) % 360
        return correction, confidence
    except Exception as exc:
        log.debug(f"Tesseract OSD failed: {exc}")
        return None, 0.0


def fast_heuristic_score(binary: np.ndarray) -> float:
    row_sums = np.sum(binary, axis=1, dtype=np.float32)
    if row_sums.size == 0 or np.std(row_sums) == 0:
        return 0.0
    maximum = float(np.max(row_sums))
    if maximum <= 0:
        return 0.0
    norm = row_sums / (maximum + 1e-6)
    var_score = float(np.std(norm))
    gap_fraction = float(np.mean(norm < 0.05))
    return var_score * 0.6 + gap_fraction * 0.4


def fast_ocr_confidence_score(gray: np.ndarray) -> Tuple[float, int]:
    if not _HAS_PYTESSERACT:
        return 0.0, 0
    try:
        proc = resize_max_dimension(gray, DEEP_OCR_MAX_DIM)
        data = pytesseract.image_to_data(proc, config="--psm 11", output_type=pytesseract.Output.DICT)
    except Exception as exc:
        log.debug(f"Deep OCR failed: {exc}")
        return 0.0, 0

    confident = []
    texts = data.get("text", [])
    confidences = data.get("conf", [])

    for text, confidence in zip(texts, confidences):
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            continue
        cleaned = text.strip()
        if confidence_value >= 55.0 and len(cleaned) >= 2 and any(ch.isalnum() for ch in cleaned):
            confident.append(confidence_value)

    if not confident:
        return 0.0, 0

    score = float(np.mean(confident)) * math.log1p(len(confident))
    return score, len(confident)


def fast_coarse_orientation(gray: np.ndarray, min_osd_confidence: float = MIN_OSD_CONFIDENCE) -> Tuple[int, str, float]:
    angle, confidence = fast_osd_orientation(gray)
    if angle is not None and confidence >= min_osd_confidence:
        return int(angle), "tesseract_osd", float(confidence)

    scores = {}
    for k in range(4):
        rotation = rotate_90(gray, k)
        binary = fast_analysis_binary(rotation)
        scores[k * 90] = fast_heuristic_score(binary)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_angle, top_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
    ratio = top_score / runner_score if runner_score > 1e-9 else float("inf")
    ambiguous = runner_score <= 0 or ratio < 1.15

    if not ambiguous:
        return int(top_angle), "projection_heuristic", float(top_score)

    if not _HAS_PYTESSERACT:
        return int(top_angle), "projection_heuristic_fallback", float(top_score)

    futures = {}
    for k in range(4):
        rotated = rotate_90(gray, k)
        futures[k * 90] = _OCR_FALLBACK_POOL.submit(fast_ocr_confidence_score, rotated)

    ocr_results = {}
    for angle_value, future in futures.items():
        try:
            ocr_results[angle_value] = future.result()
        except Exception:
            ocr_results[angle_value] = (0.0, 0)

    if not ocr_results:
        return int(top_angle), "projection_heuristic_fallback", float(top_score)

    best_angle = max(ocr_results, key=lambda a: ocr_results[a][0])
    best_score, best_count = ocr_results[best_angle]

    if best_count >= 3:
        return int(best_angle), "deep_ocr_confidence", float(best_score)

    return int(top_angle), "projection_heuristic_fallback", float(top_score)


def fast_fine_skew(binary: np.ndarray, angle_range: float = 12.0) -> float:
    h, w = binary.shape[:2]
    if h < 10 or w < 10:
        return 0.0

    scale = min(1.0, 600.0 / max(h, w))
    small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else binary

    foreground_ratio = np.count_nonzero(small) / float(small.size)
    if foreground_ratio < 0.002 or foreground_ratio > 0.80:
        return 0.0

    sh, sw = small.shape
    center = (sw / 2.0, sh / 2.0)
    best_angle = 0.0
    best_var = -1.0

    for angle in np.arange(-angle_range, angle_range + 0.001, 0.5):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(small, matrix, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        row_sums = np.sum(rotated, axis=1, dtype=np.float32)
        variance = float(np.var(row_sums))
        if variance > best_var:
            best_var = variance
            best_angle = float(angle)

    for angle in np.arange(best_angle - 0.45, best_angle + 0.451, 0.05):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(small, matrix, (sw, sh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        row_sums = np.sum(rotated, axis=1, dtype=np.float32)
        variance = float(np.var(row_sums))
        if variance > best_var:
            best_var = variance
            best_angle = float(angle)

    return float(normalize_signed_angle(best_angle))


def fast_rotate_full_res(img: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_deg = normalize_signed_angle(angle_deg)
    if abs(angle_deg) < 0.05:
        return img.copy()

    img = ensure_bgr(img)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos_val = abs(matrix[0, 0])
    sin_val = abs(matrix[0, 1])
    new_w = max(1, int(round(h * sin_val + w * cos_val)))
    new_h = max(1, int(round(h * cos_val + w * sin_val)))

    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    return cv2.warpAffine(
        img, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
    )


def fast_autocrop(img: np.ndarray, pad: int = 8, threshold: int = 250) -> np.ndarray:
    if img is None:
        return img
    img = ensure_bgr(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return img

    scale = min(1.0, 800.0 / max(h, w))
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    _, mask = cv2.threshold(small, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return img

    x, y, bw, bh = cv2.boundingRect(coords)
    inv_scale = 1.0 / scale
    x0 = max(0, int(x * inv_scale) - pad)
    y0 = max(0, int(y * inv_scale) - pad)
    x1 = min(w, int((x + bw) * inv_scale) + pad)
    y1 = min(h, int((y + bh) * inv_scale) + pad)

    if x1 <= x0 or y1 <= y0:
        return img
    cropped = img[y0:y1, x0:x1]
    if cropped.shape[0] < h * 0.10 or cropped.shape[1] < w * 0.10:
        return img
    return cropped


def process_image(image_path: str, autocrop: bool = True) -> Tuple[np.ndarray, np.ndarray, OrientationResult]:
    start_time = time.perf_counter()
    raw_img = read_image(image_path)
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    coarse_angle, method, confidence = fast_coarse_orientation(gray, min_osd_confidence=MIN_OSD_CONFIDENCE)
    coarse_angle = int(coarse_angle) % 360

    coarse_k = (coarse_angle // 90) % 4
    img_coarse = rotate_90(raw_img, coarse_k)
    gray_coarse = rotate_90(gray, coarse_k)

    binary = fast_analysis_binary(gray_coarse)
    fine_angle = fast_fine_skew(binary)

    final_img = fast_rotate_full_res(img_coarse, fine_angle)
    if autocrop:
        final_img = fast_autocrop(final_img)

    total_rotation = normalize_signed_angle(coarse_angle + fine_angle)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    result = OrientationResult(
        path=image_path,
        coarse_angle=coarse_angle,
        coarse_method=method,
        coarse_confidence=confidence,
        fine_angle=fine_angle,
        total_rotation=total_rotation,
        elapsed_time_ms=elapsed_ms,
    )
    return raw_img, final_img, result
