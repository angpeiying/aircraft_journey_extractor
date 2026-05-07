# Aircraft Journey Extractor

## Objective
Prototype the smallest working backend path from an aircraft journey summary image to structured JSON.

## Scope and Assumptions
- The input is an image (PNG, JPEG, GIF, or WebP) of an aircraft journey summary form.
- Common case: one form per image; batch input can contain 1–3 images.
- Fields may be typed, printed, or handwritten — quality varies.
- The defect message field is typically handwritten and may be partially illegible.

## Approach

This prototype uses **PaddleOCR** (`paddleocr==2.9.1`) with **OpenCV** preprocessing — fully local, no API key required:

1. Load the image and correct small rotations (±5°) using Hough line detection.
2. Auto-detect the form layout (generated vs. real scan) from the image aspect ratio.
3. Crop each field using pre-calibrated region-of-interest (ROI) fractions.
4. Preprocess each crop: upscale 2×, denoise, adaptive threshold.
5. Run PaddleOCR on each crop and filter detections by confidence > 50%.
6. Normalise raw OCR text per field (airport codes, integers, fuel strings, registration).
7. Emit structured JSON with `warnings` for missing required fields or defect messages.

### Why PaddleOCR over Tesseract?

| | Tesseract + OpenCV | PaddleOCR |
|---|---|---|
| Accuracy on clean print | Good | Excellent |
| Accuracy on degraded scans | Poor | Good |
| Handwriting | Very poor | Moderate |
| No API key / fully local | Yes | Yes |
| GPU acceleration | No | Optional |
| Confidence scores per detection | No | Yes |
| Orientation correction | Manual | Built-in (`use_angle_cls=True`) |

### Key Design Choices
- **ROI-based extraction** — each field is defined as a fraction of image dimensions, making the extractor layout-aware without training a custom model.
- **Dual layout presets** (`LAYOUT_GENERATED` / `LAYOUT_ORIGINAL`) — auto-selected by aspect ratio so the same script handles both synthetic test images and real scanned forms.
- **Adaptive preprocessing per crop** — upscaling small crops 2× and applying adaptive thresholding significantly improves OCR accuracy on small or low-contrast fields.
- **Confidence filtering** — detections below 50% confidence are discarded rather than silently producing noisy output.
- **`defect_message` treated as multiline** — detections are sorted top-to-bottom and joined with newlines to preserve the structure of longer handwritten notes.
- **Warnings field** — missing required fields and defect messages are surfaced explicitly so downstream consumers know when manual review is needed.

## Install

```bash
pip install paddlepaddle==2.6.2 paddleocr==2.9.1 opencv-python pillow python-dotenv
```

> **Note:** PaddleOCR 3.x is not supported on Windows due to oneDNN incompatibility. Use `paddlepaddle==2.6.2` + `paddleocr==2.9.1`.

## Run

Single form:
```bash
python extractor.py sample.png --out output.json
```

Multiple forms:
```bash
python extractor.py form1.png form2.png form3.png --out output.json
```

## JSON Output (sample)

```json
{
  "source_file": "01_clean_typed.png",
  "aircraft_model": "Airbus A320",
  "registration_number": "9M-XX1",
  "departure_airport": "PEN",
  "arrival_airport": "KUL",
  "crew": 3,
  "fuel_on_board": "15k",
  "load": 180,
  "defect_message": null,
  "raw_ocr": {
    "aircraft_model": "Airbus A320",
    "registration_number": "9M-XX1",
    "departure_airport": "PEN",
    "arrival_airport": "KuL",
    "crew": "3",
    "fuel": "15K",
    "load": "180",
    "defect_message": ""
  },
  "warnings": []
}
```

## API Sketch

### Request
`POST /extract-journey`

Content type: `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `files` | File (×1–3) | Aircraft journey summary image(s) |

```bash
curl -X POST http://localhost:8000/extract-journey \
  -F "files=@form1.png" \
  -F "files=@form2.png"
```

### Response
```json
{
  "forms": [
    {
      "source_file": "form1.png",
      "aircraft_model": "Airbus A320",
      "registration_number": "9M-XX1",
      "departure_airport": "KUL",
      "arrival_airport": "SIN",
      "crew": 6,
      "fuel_on_board": "12k",
      "load": 150,
      "defect_message": null,
      "raw_ocr": { "...": "..." },
      "warnings": []
    }
  ]
}
```

## Constraints
- No API key required — runs fully locally.
- Requires `paddlepaddle==2.6.2` + `paddleocr==2.9.1` (PaddleOCR 3.x not supported on Windows).
- First run downloads ~50 MB of PaddleOCR model weights to `~/.paddleocr/`.
- ROI fractions are calibrated for two form layouts; a significantly different form design requires re-tuning the fractions.
- Handwritten defect messages are partially legible — always flagged for manual review.

## Future Improvements
- Add a FastAPI wrapper for real backend deployment.
- Fine-tune ROI fractions against a larger corpus of real scanned forms.
- Add per-field confidence scores to the output JSON.
- Validate airport codes and registration numbers against reference databases.
- Explore a layout-agnostic approach (e.g., key-value detection model) to remove dependency on hard-coded ROI fractions.
- GPU inference for faster batch processing.
