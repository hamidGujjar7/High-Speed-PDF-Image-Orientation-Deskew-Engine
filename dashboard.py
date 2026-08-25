#!/usr/bin/env python3
"""
Dashboard UI Renderer for Side-by-Side Orientation Comparison.
Renders raw vs corrected views, telemetry badges, runtime time casting, and key prompts.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .core import OrientationResult, ensure_bgr, normalize_signed_angle
except ImportError:
    from core import OrientationResult, ensure_bgr, normalize_signed_angle



def resize_for_display(img: np.ndarray, target_w: int = 580, target_h: int = 640) -> np.ndarray:
    img = ensure_bgr(img)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = max(0.001, min(target_w / float(w), target_h / float(h)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC

    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = (24, 24, 28)

    x_off = max(0, (target_w - nw) // 2)
    y_off = max(0, (target_h - nh) // 2)
    paste_w = min(nw, target_w - x_off)
    paste_h = min(nh, target_h - y_off)

    if paste_w > 0 and paste_h > 0:
        canvas[y_off:y_off + paste_h, x_off:x_off + paste_w] = resized[:paste_h, :paste_w]

    cv2.rectangle(
        canvas,
        (x_off, y_off),
        (min(target_w - 1, x_off + paste_w), min(target_h - 1, y_off + paste_h)),
        (65, 65, 75),
        1
    )
    return canvas


def paste_image_safe(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    if image is None:
        return
    image = ensure_bgr(image)
    canvas_h, canvas_w = canvas.shape[:2]
    image_h, image_w = image.shape[:2]

    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    dst_x1 = min(canvas_w, x + image_w)
    dst_y1 = min(canvas_h, y + image_h)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return

    src_x0 = max(0, -x)
    src_y0 = max(0, -y)
    src_x1 = min(src_x0 + (dst_x1 - dst_x0), image_w)
    src_y1 = min(src_y0 + (dst_y1 - dst_y0), image_h)

    actual_w = src_x1 - src_x0
    actual_h = src_y1 - src_y0
    if actual_w <= 0 or actual_h <= 0:
        return

    canvas[dst_y0:dst_y0 + actual_h, dst_x0:dst_x0 + actual_w] = image[src_y0:src_y1, src_x0:src_x1]


def create_dashboard(
    orig_img: np.ndarray,
    corr_img: np.ndarray,
    img_name: str,
    index: int,
    total: int,
    result: OrientationResult,
    manual_offset: int = 0
) -> np.ndarray:
    card_w, card_h = 580, 680
    header_h, footer_h = 105, 80
    gap = 30
    margin_left, margin_right = 15, 15

    total_w = margin_left + card_w + gap + card_w + margin_right
    total_h = header_h + card_h + footer_h + 20

    dashboard = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    dashboard[:] = (18, 18, 22)

    # Header
    cv2.rectangle(dashboard, (0, 0), (total_w - 1, header_h), (28, 28, 36), -1)
    cv2.line(dashboard, (0, header_h), (total_w, header_h), (50, 50, 65), 2)
    cv2.putText(
        dashboard,
        "PDF & IMAGE ORIENTATION DASHBOARD",
        (25, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.80,
        (0, 220, 255),
        2,
        cv2.LINE_AA
    )
    progress_text = f"[{index + 1} / {total}]  {img_name}"
    cv2.putText(
        dashboard,
        progress_text,
        (25, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (200, 200, 210),
        1,
        cv2.LINE_AA
    )

    # Telemetry Badge
    badge_w = 500
    badge_x0 = total_w - badge_w - 20
    badge_x1 = total_w - 20
    speed_color = (0, 255, 120) if result.elapsed_time_ms < 200 else (0, 215, 255)

    cv2.rectangle(dashboard, (badge_x0, 12), (badge_x1, 92), (35, 35, 45), -1)
    cv2.rectangle(dashboard, (badge_x0, 12), (badge_x1, 92), (55, 55, 70), 1)

    time_text = f"Time: {result.elapsed_time_ms:.1f} ms ({result.elapsed_time_ms / 1000.0:.2f}s)"
    cv2.putText(
        dashboard,
        time_text,
        (badge_x0 + 15, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        speed_color,
        1,
        cv2.LINE_AA
    )

    telemetry = (
        f"Coarse: {result.coarse_angle}deg | "
        f"Fine: {result.fine_angle:+.2f}deg | "
        f"Total: {normalize_signed_angle(result.total_rotation + manual_offset):+.2f}deg"
    )
    cv2.putText(
        dashboard,
        telemetry,
        (badge_x0 + 15, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (220, 220, 230),
        1,
        cv2.LINE_AA
    )

    method_text = f"Method: {result.coarse_method} | Conf: {result.coarse_confidence:.2f}"
    cv2.putText(
        dashboard,
        method_text,
        (badge_x0 + 15, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (170, 170, 180),
        1,
        cv2.LINE_AA
    )

    # Panels
    panel_y = header_h + 15
    image_h = card_h - 55
    disp_orig = resize_for_display(orig_img, card_w, image_h)
    disp_corr = resize_for_display(corr_img, card_w, image_h)

    x1 = margin_left
    x2 = x1 + card_w + gap

    cv2.putText(
        dashboard,
        "ORIGINAL (RAW)",
        (x1 + 10, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (160, 160, 170),
        2,
        cv2.LINE_AA
    )
    paste_image_safe(dashboard, disp_orig, x1, panel_y + 35)
    cv2.rectangle(
        dashboard,
        (x1, panel_y),
        (min(total_w - 1, x1 + card_w), min(total_h - 1, panel_y + card_h)),
        (50, 50, 65),
        1
    )

    cv2.putText(
        dashboard,
        "CORRECTED (UPRIGHT)",
        (x2 + 10, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (100, 255, 120),
        2,
        cv2.LINE_AA
    )
    paste_image_safe(dashboard, disp_corr, x2, panel_y + 35)
    cv2.rectangle(
        dashboard,
        (x2, panel_y),
        (min(total_w - 1, x2 + card_w), min(total_h - 1, panel_y + card_h)),
        (50, 50, 65),
        1
    )

    # Footer
    footer_y = total_h - footer_h
    cv2.rectangle(dashboard, (0, footer_y), (total_w - 1, total_h - 1), (25, 25, 32), -1)
    cv2.line(dashboard, (0, footer_y), (total_w, footer_y), (50, 50, 65), 2)
    controls = "[Left/A/P] Prev   [Right/D/N/Space] Next   [R] +90   [L] -90   [S] Save   [Q/Esc] Exit"
    cv2.putText(
        dashboard,
        controls,
        (25, total_h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 215, 255),
        1,
        cv2.LINE_AA
    )

    return dashboard


def is_next_key(key: int) -> bool:
    return key in {ord("d"), ord("D"), ord("n"), ord("N"), 32, 83, 2555904, 65363}


def is_previous_key(key: int) -> bool:
    return key in {ord("a"), ord("A"), ord("p"), ord("P"), 81, 2424832, 65361}
