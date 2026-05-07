"""
Aircraft Journey Extractor, using PaddleOCR

Usage:
  python extractor.py image1.png image2.jpg --out output.json
  python extractor.py data\01_clean_typed.png data\02_with_defect.png data\03_missing_fields.png data\04_all_handwritten.png --out result\overall_output.json

Dependencies:
  pip install paddlepaddle paddleocr opencv-python pillow python-dotenv

No API key required — runs fully locally.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from paddleocr import PaddleOCR

load_dotenv()

# Suppress noisy PaddleOCR / PaddlePaddle logs
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Region-of-interest definitions
#
# Two layout presets — values are (x1, y1, x2, y2) as fractions of image size.
#
# LAYOUT_GENERATED  → synthetic test images created by generate_test_images.py
#                     (800 × 900 px, VALUE_X=320, ROW_H=62, HEADER_H=80)
#
# LAYOUT_ORIGINAL   → real scanned / PDF-rendered forms (portrait A4-ish).
#                     Tune these fractions if you have real form scans.
# ---------------------------------------------------------------------------

LAYOUT_GENERATED: Dict[str, Tuple[float, float, float, float]] = {
    "aircraft_model":      (0.41, 0.089, 0.98, 0.158),
    "registration_number": (0.41, 0.158, 0.98, 0.227),
    "departure_airport":   (0.41, 0.227, 0.98, 0.296),
    "arrival_airport":     (0.41, 0.296, 0.98, 0.364),
    "crew":                (0.41, 0.364, 0.98, 0.433),
    "fuel":                (0.41, 0.433, 0.98, 0.502),
    "load":                (0.41, 0.502, 0.98, 0.571),
    # Extend to 0.98 so long handwritten defect messages are not clipped
    "defect_message":      (0.02, 0.571, 0.98, 0.98),
}

LAYOUT_ORIGINAL: Dict[str, Tuple[float, float, float, float]] = {
    "aircraft_model":      (0.50, 0.14,  0.96, 0.20),
    "registration_number": (0.50, 0.20,  0.96, 0.255),
    "departure_airport":   (0.50, 0.255, 0.96, 0.31),
    "arrival_airport":     (0.50, 0.31,  0.96, 0.365),
    "crew":                (0.50, 0.365, 0.96, 0.42),
    "fuel":                (0.50, 0.42,  0.96, 0.475),
    "load":                (0.50, 0.475, 0.96, 0.535),
    # Extend to 0.98 so long handwritten defect messages are not clipped
    "defect_message":      (0.02, 0.535, 0.96, 0.98),
}


# ---------------------------------------------------------------------------
# Layout auto-detection
# ---------------------------------------------------------------------------

def _detect_layout(img: np.ndarray) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Pick a ROI layout based on image aspect ratio.
    Generated forms are ~800×900 (ratio ~0.89).
    Real scanned A4 forms are typically taller (ratio < 0.80).
    """
    h, w = img.shape[:2]
    ratio = w / h
    return LAYOUT_GENERATED if ratio > 0.84 else LAYOUT_ORIGINAL


# ---------------------------------------------------------------------------
# PaddleOCR singleton
# ---------------------------------------------------------------------------

_OCR: Optional[PaddleOCR] = None


def _get_ocr() -> PaddleOCR:
    global _OCR
    if _OCR is None:
        # use_angle_cls=True handles upside-down / rotated text in crops
        _OCR = PaddleOCR(use_angle_cls=True, lang="en")
    return _OCR


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def _deskew(img: np.ndarray) -> np.ndarray:
    """Correct small rotation angles (±5°) using Hough line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                             minLineLength=img.shape[1] // 3, maxLineGap=20)
    if lines is None:
        return img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return img
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5 or abs(median_angle) > 5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Upscale + denoise + threshold a field crop for better OCR."""
    # Upscale 2× for small crops
    h, w = crop.shape[:2]
    if h < 60 or w < 200:
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Mild denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    # Adaptive threshold handles uneven lighting / shadow
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    # Convert back to BGR — PaddleOCR expects colour input
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _crop(img: np.ndarray,
          roi: Tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = roi
    return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]


# ---------------------------------------------------------------------------
# OCR per field
# ---------------------------------------------------------------------------

def _ocr_crop(crop: np.ndarray, is_multiline: bool = False) -> str:
    """Run PaddleOCR on a single crop and return cleaned text."""
    ocr = _get_ocr()
    preprocessed = _preprocess_crop(crop)

    # PaddleOCR 2.x API: ocr(img, cls=True)
    result = ocr.ocr(preprocessed, cls=True)

    if not result or result[0] is None:
        return ""

    # Sort detections top-to-bottom, then left-to-right by bounding box origin
    detections = result[0]
    detections.sort(key=lambda d: (d[0][0][1], d[0][0][0]))

    # Filter by confidence > 50%
    texts = [det[1][0] for det in detections if det[1][1] > 0.5]

    sep = "\n" if is_multiline else " "
    return sep.join(texts).strip()


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------

def _norm_airport(raw: str) -> Optional[str]:
    """Extract a 3-letter IATA-like code."""
    v = re.sub(r"[^A-Za-z]", "", raw).upper()
    m = re.search(r"[A-Z]{3}", v)
    return m.group(0) if m else (v if len(v) == 3 else None)


def _norm_int(raw: str) -> Optional[int]:
    raw = raw.replace("I", "1").replace("l", "1").replace("O", "0")
    m = re.search(r"\d+", raw)
    return int(m.group(0)) if m else None


def _norm_fuel(raw: str) -> Optional[str]:
    v = raw.lower().replace(" ", "")
    m = re.search(r"(\d+(?:\.\d+)?)(k?)", v)
    if not m:
        return None
    return f"{m.group(1)}k" if m.group(2) == "k" else m.group(1)


def _norm_registration(raw: str) -> Optional[str]:
    v = re.sub(r"[^A-Za-z0-9\-]", "", raw).upper()
    return v if v else None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract(path: Path) -> Dict[str, Any]:
    img = _load(path)
    img = _deskew(img)
    rois = _detect_layout(img)

    raw: Dict[str, str] = {}
    for field, roi in rois.items():
        crop = _crop(img, roi)
        raw[field] = _ocr_crop(crop, is_multiline=(field == "defect_message"))

    result: Dict[str, Any] = {
        "source_file": path.name,
        "aircraft_model":      raw["aircraft_model"].strip() or None,
        "registration_number": _norm_registration(raw["registration_number"]),
        "departure_airport":   _norm_airport(raw["departure_airport"]),
        "arrival_airport":     _norm_airport(raw["arrival_airport"]),
        "crew":                _norm_int(raw["crew"]),
        "fuel_on_board":       _norm_fuel(raw["fuel"]),
        "load":                _norm_int(raw["load"]),
        "defect_message":      raw["defect_message"] or None,
        "raw_ocr":             raw,
        "warnings":            [],
    }

    required = ["aircraft_model", "registration_number",
                "departure_airport", "arrival_airport"]
    for key in required:
        if not result[key]:
            result["warnings"].append(f"Missing or low-confidence field: {key}")
    if result["defect_message"]:
        result["warnings"].append(
            "Defect message present — manual review recommended"
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured data from aircraft journey summary images."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    outputs: List[Dict[str, Any]] = [extract(p) for p in args.images]
    text = json.dumps(outputs if len(outputs) > 1 else outputs[0], indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
