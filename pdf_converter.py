#!/usr/bin/env python3
"""
High-accuracy PDF-to-image conversion module.
Extracts native embedded full-page scans losslessly without resampling,
or renders born-digital pages via PyMuPDF at high DPI.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

from PIL import Image

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

log = logging.getLogger("pdf_converter")


@dataclass
class PageExportResult:
    page_number: int
    path: str = ""
    method: str = ""
    native_dpi: Optional[float] = None
    output_dpi: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def check_pymupdf_installed() -> bool:
    return pymupdf is not None


def _page_effective_dpi(page: "pymupdf.Page", pixel_width: int) -> float:
    width_pt = page.rect.width
    width_in = width_pt / 72.0
    return pixel_width / width_in if width_in > 0 else 0.0


def _full_page_embedded_image(page: "pymupdf.Page", coverage_tolerance: float = 0.02):
    images = page.get_images(full=True)
    if len(images) != 1:
        return None
    xref = images[0][0]
    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return None
    if len(rects) != 1:
        return None
    rect = rects[0]
    page_rect = page.rect

    covers_width = abs(rect.width - page_rect.width) <= page_rect.width * coverage_tolerance
    covers_height = abs(rect.height - page_rect.height) <= page_rect.height * coverage_tolerance
    starts_at_origin = (
        abs(rect.x0 - page_rect.x0) <= page_rect.width * coverage_tolerance
        and abs(rect.y0 - page_rect.y0) <= page_rect.height * coverage_tolerance
    )

    if not (covers_width and covers_height and starts_at_origin):
        return None

    try:
        text_len = len(page.get_text("text").strip())
    except Exception:
        text_len = 0

    if text_len > 200:
        return None
    return xref


def _extract_embedded(doc: "pymupdf.Document", xref: int) -> tuple[bytes, str, int, int]:
    info = doc.extract_image(xref)
    return info["image"], info["ext"], info["width"], info["height"]


def _render_page(page: "pymupdf.Page", dpi: int) -> bytes:
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
    return pix.tobytes("png")


def pdf_to_images(
    pdf_path: str,
    out_dir: str,
    dpi: int = 300,
    pages: Optional[List[int]] = None,
    prefer_embedded: bool = True,
    force_png: bool = True,
    hash_suffix: bool = True,
    max_upsample_ratio: float = 1.15,
) -> List[PageExportResult]:
    """Convert PDF to lossless images."""
    if pymupdf is None:
        raise RuntimeError(
            "PyMuPDF is not installed. Please install it using:\n"
            "  pip install pymupdf"
        )

    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    results: List[PageExportResult] = []

    try:
        total_pages = len(doc)
        page_indices = [p - 1 for p in pages] if pages else list(range(total_pages))

        for idx in page_indices:
            if idx < 0 or idx >= total_pages:
                log.warning(f"Page {idx + 1} out of range ({total_pages} total).")
                continue

            page = doc[idx]
            page_num = idx + 1
            result = PageExportResult(page_number=page_num)

            xref = _full_page_embedded_image(page) if prefer_embedded else None

            if xref is not None:
                raw_bytes, ext, w, h = _extract_embedded(doc, xref)
                native_dpi = _page_effective_dpi(page, w)
                result.method = "embedded_extract"
                result.native_dpi = native_dpi
                result.output_dpi = native_dpi

                if native_dpi < dpi / max_upsample_ratio:
                    result.warnings.append(
                        f"Native scan resolution (~{native_dpi:.0f} DPI) is below requested {dpi} DPI."
                    )

                if force_png and ext.lower() != "png":
                    img = Image.open(io.BytesIO(raw_bytes))
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    raw_bytes = buf.getvalue()
                    ext = "png"

                out_bytes = raw_bytes
                out_ext = ext
            else:
                out_bytes = _render_page(page, dpi)
                out_ext = "png"
                result.method = "rendered"
                result.output_dpi = dpi

            if hash_suffix:
                digest = hashlib.md5(out_bytes).hexdigest()[:8]
                filename = f"page_{page_num:04d}_{digest}.{out_ext}"
            else:
                filename = f"page_{page_num:04d}.{out_ext}"

            out_path = os.path.join(out_dir, filename)
            with open(out_path, "wb") as f:
                f.write(out_bytes)

            result.path = out_path
            log.info(f"Extracted page {page_num}/{total_pages} -> {filename} ({result.method})")
            results.append(result)
    finally:
        doc.close()

    return results
