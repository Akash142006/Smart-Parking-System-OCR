import os
import re

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR - fast and accurate for plates
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
PLATE_PATTERN = re.compile(r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}")
FULL_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$")

LETTER_SUBS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

DIGIT_SUBS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}


def normalize_text(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def preprocess_variants(image_path, fast=False):
    """Create multiple OCR-friendly image variants."""
    img = cv2.imread(image_path)
    if img is None:
        return [image_path]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    bilateral = cv2.bilateralFilter(eq, 9, 75, 75)
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(eq, -1, kernel)

    base, ext = os.path.splitext(image_path)
    variant_paths = [image_path]

    if fast:
        variant_data = [
            ("eq", eq),
            ("otsu", otsu),
        ]
    else:
        variant_data = [
            ("eq", eq),
            ("otsu", otsu),
            ("adaptive", adaptive),
            ("sharp", sharp),
        ]

    for name, frame in variant_data:
        resized = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        out_path = f"{base}_{name}{ext}"
        cv2.imwrite(out_path, resized)
        variant_paths.append(out_path)

    return variant_paths


def detect_plate_regions(image_path):
    """Detect likely plate regions and save cropped images for OCR."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []

    h, w = gray.shape[:2]
    min_area = (w * h) * 0.005

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue

        ratio = cw / float(ch + 1e-6)
        if ratio < 1.8 or ratio > 7.5:
            continue

        pad_x = int(cw * 0.08)
        pad_y = int(ch * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        regions.append(crop)

    regions.sort(key=lambda c: c.shape[0] * c.shape[1], reverse=True)
    regions = regions[:3]

    base, ext = os.path.splitext(image_path)
    saved_paths = []
    for idx, region in enumerate(regions):
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        region_bin = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        region_up = cv2.resize(region_bin, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
        out_path = f"{base}_region_{idx}{ext}"
        cv2.imwrite(out_path, region_up)
        saved_paths.append(out_path)

    return saved_paths


def _position_fix(candidate):
    """Fix common OCR confusions based on Indian plate format positions."""
    text = normalize_text(candidate)
    if len(text) < 9:
        return text

    # Try windows that can fit [AA][00][A..A][0000]
    for start in range(max(1, len(text) - 11) + 1):
        for size in (10, 9, 11):
            chunk = text[start : start + size]
            if len(chunk) != size:
                continue

            chars = list(chunk)
            # First two must be letters
            for i in (0, 1):
                chars[i] = LETTER_SUBS.get(chars[i], chars[i])
            # Next two must be digits
            for i in (2, 3):
                chars[i] = DIGIT_SUBS.get(chars[i], chars[i])
            # Last four must be digits
            for i in range(size - 4, size):
                chars[i] = DIGIT_SUBS.get(chars[i], chars[i])
            # Middle part must be letters
            for i in range(4, size - 4):
                chars[i] = LETTER_SUBS.get(chars[i], chars[i])

            fixed = "".join(chars)
            if FULL_PLATE_PATTERN.match(fixed):
                return fixed

    return text


def extract_plate_text(image_path, fast=False):
    """Read possible plate-like text from multiple image variants using PaddleOCR."""
    try:
        variants = preprocess_variants(image_path, fast=fast)
        if not fast:
            variants.extend(detect_plate_regions(image_path))
        candidates = []
        seen = {}

        for variant in variants:
            try:
                # PaddleOCR returns: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]], ...]
                results = ocr.ocr(variant)
                if not results:
                    continue

                # Preserve left-to-right order for chunk merge (sort by x-coordinate)
                ordered = []
                # Flatten potential nested output safely across PaddleOCR versions.
                lines = []
                if isinstance(results, list):
                    if len(results) > 0 and isinstance(results[0], list):
                        for item in results:
                            lines.extend(item)
                    else:
                        lines = results

                for bbox_info in lines:
                    try:
                        if not isinstance(bbox_info, (list, tuple)) or len(bbox_info) < 2:
                            continue

                        text_info = bbox_info[1]
                        if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                            continue

                        text, confidence = text_info[0], float(text_info[1])
                        if confidence < 0.10:
                            continue

                        cleaned = normalize_text(text)
                        if not cleaned:
                            continue

                        fixed = _position_fix(cleaned)
                        best_text = fixed if FULL_PLATE_PATTERN.match(fixed) else cleaned

                        ordered.append(best_text)

                        if best_text not in seen or confidence > seen[best_text]:
                            seen[best_text] = confidence
                    except Exception:
                        continue

                # Merge adjacent chunks because plates often split across OCR tokens.
                for i in range(len(ordered) - 1):
                    merged2 = _position_fix(ordered[i] + ordered[i + 1])
                    if merged2 and merged2 not in seen:
                        seen[merged2] = 0.09

                for i in range(len(ordered) - 2):
                    merged3 = _position_fix(ordered[i] + ordered[i + 1] + ordered[i + 2])
                    if merged3 and merged3 not in seen:
                        seen[merged3] = 0.08

            except Exception as e:
                print(f"Error processing variant {variant}: {e}")
                continue

        # Also try merged chunks globally
        texts = list(seen.keys())
        for i in range(len(texts) - 1):
            merged_text = _position_fix(texts[i] + texts[i + 1])
            if merged_text not in seen:
                seen[merged_text] = min(seen[texts[i]], seen[texts[i + 1]])

        for text, score in seen.items():
            candidates.append((text, score))

        candidates.sort(
            key=lambda item: (
                1 if FULL_PLATE_PATTERN.match(item[0]) else 0,
                item[1],
                len(item[0]),
            ),
            reverse=True,
        )

        return [text for text, _ in candidates]
    except Exception as exc:
        print(f"OCR Error: {exc}")
        return []


def find_plate_number(image_path, fast=False):
    """Return valid plate if found; else return best available OCR value."""
    all_text = extract_plate_text(image_path, fast=fast)

    for text in all_text:
        repaired = _position_fix(text)
        match = PLATE_PATTERN.search(repaired)
        if match:
            return {"success": True, "plate": match.group(0), "candidates": all_text}

    if all_text:
        best_guess = all_text[0]
        return {
            "success": False,
            "plate": best_guess,
            "candidates": all_text,
            "note": "Plate format not confirmed; returning best OCR value",
        }

    return {
        "success": False,
        "plate": "UNREADABLE",
        "candidates": [],
        "note": "No text detected in image",
    }