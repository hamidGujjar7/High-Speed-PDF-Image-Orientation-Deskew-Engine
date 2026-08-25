#!/usr/bin/env python3
"""
PDF & Image Orientation / Deskew Interactive Application.

Usage:
  python app.py                                   # Runs with default test images in assist/images
  python app.py "path/to/document.pdf"            # Runs on specific PDF
  python app.py "path/to/image_folder"            # Runs on specific image directory
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List

import cv2
import numpy as np

# Handle local package imports whether run directly or as a module
try:
    from .core import (
        process_image,
        rotate_90,
        save_image,
        check_tesseract_installed,
        OrientationResult,
        _OCR_FALLBACK_POOL,
    )
    from .pdf_converter import pdf_to_images
    from .dashboard import (
        create_dashboard,
        is_next_key,
        is_previous_key,
    )
except ImportError:
    from core import (
        process_image,
        rotate_90,
        save_image,
        check_tesseract_installed,
        OrientationResult,
        _OCR_FALLBACK_POOL,
    )
    from pdf_converter import pdf_to_images
    from dashboard import (
        create_dashboard,
        is_next_key,
        is_previous_key,
    )

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]


def collect_images(source: str) -> List[str]:
    paths = []
    if os.path.isfile(source):
        lower = source.lower()
        if any(lower.endswith(ext.replace("*", "")) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]):
            paths.append(source)
    elif os.path.isdir(source):
        for ext in IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(source, ext)))
            paths.extend(glob.glob(os.path.join(source, ext.upper())))
    return sorted(set(paths))


def get_default_input() -> str:
    """Finds default input: test images in assist/images or local PDF."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    assist_images_dir = os.path.join(app_dir, "assist", "images")
    if os.path.exists(assist_images_dir) and len(collect_images(assist_images_dir)) > 0:
        return assist_images_dir

    # Check parent directory for sample PDFs
    parent_dir = os.path.dirname(app_dir)
    pdfs = glob.glob(os.path.join(parent_dir, "*.pdf"))
    if pdfs:
        return pdfs[0]

    return "test"


def main():
    parser = argparse.ArgumentParser(description="High-Speed PDF & Image Orientation / Deskew Tool")
    parser.add_argument("input", nargs="?", default=None, help="Path to PDF file, image file, or directory (optional).")
    parser.add_argument("-o", "--out-dir", default="output_corrected", help="Directory to save corrected outputs.")
    parser.add_argument("--workers", type=int, default=4, help="Number of background page-processing workers.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rendered PDF pages.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Check Tesseract OCR installation
    tess_ok, tess_msg = check_tesseract_installed()
    if not tess_ok:
        print("\n" + "=" * 70)
        print("⚠️  TESSERACT OCR WARNING / ERROR")
        print("=" * 70)
        print(tess_msg)
        print("-" * 70)
        print("Note: The application will continue using projection profile heuristics,")
        print("but deep OCR fallback accuracy may be limited without Tesseract installed.")
        print("=" * 70 + "\n")
    else:
        print(f"[✓] {tess_msg}")

    # 2. Resolve Input
    target_input = args.input
    if target_input is None:
        target_input = get_default_input()
        print(f"[*] No input provided. Using default sample images: {target_input}")

    if not os.path.exists(target_input):
        print(f"[!] Error: Input path does not exist: {target_input}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    is_pdf = target_input.lower().endswith(".pdf")
    raw_images_dir = "extracted_raw_pages" if is_pdf else target_input

    # 3. Extract PDF if applicable
    if is_pdf:
        print(f"\n[*] Converting PDF '{target_input}' to lossless raw images in '{raw_images_dir}'...")
        pdf_to_images(
            pdf_path=target_input,
            out_dir=raw_images_dir,
            dpi=args.dpi,
            prefer_embedded=True,
            force_png=True,
            hash_suffix=True,
        )

    # 4. Collect Images
    image_paths = collect_images(raw_images_dir)
    if not image_paths:
        print(f"[!] No valid images found in: {raw_images_dir}")
        sys.exit(1)

    print(f"\n[*] Launching background processing ({args.workers} workers) for {len(image_paths)} image(s)...")

    # 5. Prefetch in Background Threads
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, OrientationResult]] = {}
    futures = {}
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="page_worker")

    for path in image_paths:
        futures[path] = executor.submit(process_image, path)

    # 6. Initialize UI Window
    window_name = "PDF & Image Orientation Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1240, 890)

    current_idx = 0
    total_imgs = len(image_paths)
    manual_offsets: Dict[str, int] = {}

    try:
        while True:
            img_path = image_paths[current_idx]
            img_name = os.path.basename(img_path)

            if img_path not in cache:
                raw_img, corr_img, result = futures[img_path].result()
                cache[img_path] = (raw_img, corr_img, result)
                print(
                    f"[{current_idx + 1}/{total_imgs}] {img_name:<30} -> "
                    f"Rotation: {result.total_rotation:+.2f}deg | "
                    f"Time: {result.elapsed_time_ms:.1f} ms ({result.elapsed_time_ms/1000.0:.2f}s)"
                )
                save_image(os.path.join(args.out_dir, f"{os.path.splitext(img_name)[0]}_corrected.png"), corr_img)
            else:
                raw_img, corr_img, result = cache[img_path]

            manual_offset = manual_offsets.get(img_path, 0)
            displayed_corr = rotate_90(corr_img, (manual_offset // 90) % 4) if manual_offset != 0 else corr_img

            dash_frame = create_dashboard(
                orig_img=raw_img,
                corr_img=displayed_corr,
                img_name=img_name,
                index=current_idx,
                total=total_imgs,
                result=result,
                manual_offset=manual_offset,
            )
            cv2.imshow(window_name, dash_frame)

            key = cv2.waitKey(0) & 0xFF

            if key in [ord("q"), ord("Q"), 27]:
                print("\nExiting dashboard.")
                break
            elif is_next_key(key):
                current_idx = (current_idx + 1) % total_imgs
            elif is_previous_key(key):
                current_idx = (current_idx - 1 + total_imgs) % total_imgs
            elif key in [ord("r"), ord("R")]:
                manual_offsets[img_path] = (manual_offsets.get(img_path, 0) + 90) % 360
                print(f"Manual override for {img_name}: {manual_offsets[img_path]}deg")
            elif key in [ord("l"), ord("L")]:
                manual_offsets[img_path] = (manual_offsets.get(img_path, 0) - 90) % 360
                print(f"Manual override for {img_name}: {manual_offsets[img_path]}deg")
            elif key in [ord("s"), ord("S")]:
                save_path = os.path.join(args.out_dir, f"{os.path.splitext(img_name)[0]}_manual_saved.png")
                save_image(save_path, displayed_corr)
                print(f"[OK] Saved adjusted image: {save_path}")

    finally:
        executor.shutdown(wait=False)
        _OCR_FALLBACK_POOL.shutdown(wait=False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
