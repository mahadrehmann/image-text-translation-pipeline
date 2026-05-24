from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

'''
To run:
uvicorn main:app --reload

and goto http://127.0.0.1:8000/docs
'''

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment / .env file")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ── Initialize EasyOCR once at startup ────────────────────────────────────────
reader = easyocr.Reader(['en'], gpu=False)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Processing API",
    description="Extracts text via EasyOCR, translates with Gemini, and returns an inpainted image with translated text.",
    version="0.3.0",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def translate_regions(texts: list[str]) -> list[str]:
    """
    Sends all detected text regions to Gemini in a single call.
    Asks for a JSON array of translations in the same order.
    Returns a list of translated strings (falls back to originals on failure).
    """
    numbered = "\n".join(f'{i + 1}. "{t}"' for i, t in enumerate(texts))
    prompt = (
        "You are a professional translator.\n"
        "Translate each of the following texts to English.\n"
        "Return ONLY a valid JSON array of translated strings in the SAME ORDER.\n"
        "Example output: [\"Hello\", \"World\"]\n\n"
        f"Texts to translate:\n{numbered}"
    )
    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()
    print (numbered, response)
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    translations = json.loads(raw.strip())
    if not isinstance(translations, list) or len(translations) != len(texts):
        raise ValueError("Gemini returned unexpected translation format.")
    return [str(t) for t in translations]


def inpaint_regions(img_bgr: np.ndarray, bboxes: list) -> np.ndarray:
    """
    Creates a binary mask from EasyOCR bounding-box polygons and inpaints them.
    bboxes: list of 4-corner polygons [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for bbox in bboxes:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Dilate slightly so edges are fully covered
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Loads a system font at the given size.
    Prefers regular weight; falls back to bold, then PIL default.
    """
    regular_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    bold_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    candidates = bold_candidates + regular_candidates if bold else regular_candidates + bold_candidates
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def estimate_font_size(box_h: int, scale: float = 0.72) -> int:
    """
    Estimates the original font size from the bounding-box height.
    The scale factor (default 0.72) accounts for line-height / descender
    padding that EasyOCR includes in the bbox.
    Returns a size clamped to a sensible minimum of 8px.
    """
    return max(8, int(box_h * scale))


def cluster_font_sizes(sizes: list[int], tolerance: float = 0.25) -> list[int]:
    """
    Snaps a list of font sizes to cluster medians so that lines whose
    original text was the same visual size don't render at slightly
    different sizes due to EasyOCR bbox height jitter.

    Two sizes belong to the same cluster when they are within `tolerance`
    (default 25%) of the smaller of the two.

    Returns a new list of the same length with each size replaced by
    the integer median of its cluster.
    """
    if not sizes:
        return sizes

    # Build clusters greedily on sorted sizes
    indexed = sorted(enumerate(sizes), key=lambda x: x[1])
    clusters: list[list[tuple[int, int]]] = []  # list of [(original_idx, size), ...]

    for orig_idx, sz in indexed:
        placed = False
        for cluster in clusters:
            rep = cluster[0][1]  # representative = first (smallest) size in cluster
            if rep > 0 and abs(sz - rep) / rep <= tolerance:
                cluster.append((orig_idx, sz))
                placed = True
                break
        if not placed:
            clusters.append([(orig_idx, sz)])

    # Compute median of each cluster and map back to original positions
    result = [0] * len(sizes)
    for cluster in clusters:
        cluster_sizes = sorted(s for _, s in cluster)
        median = cluster_sizes[len(cluster_sizes) // 2]
        for orig_idx, _ in cluster:
            result[orig_idx] = median

    return result


def merge_detections_into_lines(
    detections: list[dict],
    overlap_threshold: float = 0.5,
) -> list[dict]:
    """
    Groups EasyOCR chunk-level detections into whole-line detections.

    Two chunks belong to the same line when their vertical ranges overlap
    by at least `overlap_threshold` of the shorter chunk's height.

    Within each line, chunks are sorted left-to-right and their texts are
    joined with a space.  The merged bounding box spans the full horizontal
    extent of the line.

    Returns a list of merged line dicts:
        {
            "text": str,                  # space-joined chunk texts
            "chunk_indices": list[int],   # original detection indices in the line
            "bounding_box": {x_min, y_min, x_max, y_max},
        }
    """
    n = len(detections)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: detections[i]["bounding_box"]["y_min"])
    used = [False] * n
    merged_lines: list[dict] = []

    for i in order:
        if used[i]:
            continue
        group = [i]
        used[i] = True
        bb_i = detections[i]["bounding_box"]
        y_min_i, y_max_i = bb_i["y_min"], bb_i["y_max"]

        for j in order:
            if used[j]:
                continue
            bb_j = detections[j]["bounding_box"]
            y_min_j, y_max_j = bb_j["y_min"], bb_j["y_max"]

            overlap   = max(0, min(y_max_i, y_max_j) - max(y_min_i, y_min_j))
            shorter_h = min(y_max_i - y_min_i, y_max_j - y_min_j)
            if shorter_h > 0 and overlap / shorter_h >= overlap_threshold:
                group.append(j)
                used[j] = True

        # Sort chunks left-to-right within the line
        group.sort(key=lambda i: detections[i]["bounding_box"]["x_min"])

        # Build merged bounding box covering all chunks in this line
        x_mins = [detections[i]["bounding_box"]["x_min"] for i in group]
        y_mins = [detections[i]["bounding_box"]["y_min"] for i in group]
        x_maxs = [detections[i]["bounding_box"]["x_max"] for i in group]
        y_maxs = [detections[i]["bounding_box"]["y_max"] for i in group]

        merged_lines.append({
            "text": " ".join(detections[i]["text"] for i in group),
            "chunk_indices": group,
            "bounding_box": {
                "x_min": min(x_mins),
                "y_min": min(y_mins),
                "x_max": max(x_maxs),
                "y_max": max(y_maxs),
            },
        })

    # Sort lines top-to-bottom
    merged_lines.sort(key=lambda ln: ln["bounding_box"]["y_min"])
    return merged_lines


def draw_translated_lines(
    pil_img: Image.Image,
    merged_lines: list[dict],
    translations: list[str],
) -> Image.Image:
    """
    Renders each translated line string into the line's merged bounding box.
    - Font sizes are estimated from bbox heights then clustered to eliminate
      EasyOCR jitter — lines of similar original size render identically.
    - Text is left-aligned with a small horizontal padding.
    - Color is black or white based on background brightness.
    """
    draw = ImageDraw.Draw(pil_img)

    # Pre-compute and cluster font sizes across all lines
    box_heights = [
        ln["bounding_box"]["y_max"] - ln["bounding_box"]["y_min"]
        for ln in merged_lines
    ]
    raw_sizes    = [estimate_font_size(h) for h in box_heights]
    stable_sizes = cluster_font_sizes(raw_sizes)

    for line, translated, font_size in zip(merged_lines, translations, stable_sizes):
        if not translated.strip():
            continue

        bb = line["bounding_box"]
        x_min, y_min = bb["x_min"], bb["y_min"]
        x_max, y_max = bb["x_max"], bb["y_max"]
        box_w = x_max - x_min
        box_h = y_max - y_min

        if box_w < 5 or box_h < 5:
            continue

        font = get_font(font_size)

        # Vertically center; left-align with small padding
        tbbox = draw.textbbox((0, 0), translated, font=font)
        tw, th = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]
        padding = max(2, int(box_h * 0.05))
        x = x_min + padding
        y = y_min + (box_h - th) // 2

        # Black or white text based on background brightness
        region = np.array(pil_img.crop((x_min, y_min, x_max, y_max)))
        text_color = (0, 0, 0) if region.mean() > 128 else (255, 255, 255)

        draw.text((x, y), translated, font=font, fill=text_color)

    return pil_img


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post(
    "/process-image",
    responses={200: {"content": {"image/png": {}}}},
    response_class=StreamingResponse,
)
async def process_image(image: UploadFile = File(...)):
    """
    Full pipeline:
    1. Validate image
    2. EasyOCR — detect text regions + bounding boxes
    3. Gemini 2.5 Flash — translate all regions in one call
    4. cv2 TELEA inpainting — erase original text
    5. Pillow — render translated text back in the same boxes
    6. Return the modified image as PNG
    """

    # ── 1. Validate ───────────────────────────────────────────────────────────
    ALLOWED_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")
    if image.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{image.content_type}'. Allowed: {', '.join(ALLOWED_TYPES)}"},
        )

    # ── 2. Decode image ───────────────────────────────────────────────────────
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_image)
        # OpenCV uses BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": f"Failed to decode image: {e}"})

    # ── 3. OCR ────────────────────────────────────────────────────────────────
    try:
        ocr_results = reader.readtext(img_array, detail=1)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"OCR failed: {e}"})

    if not ocr_results:
        # Nothing detected — return original image unchanged
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png",
                                 headers={"X-Warning": "No text detected in image"})

    # ── Parse raw OCR chunks ──────────────────────────────────────────────────
    raw_bboxes = []   # 4-corner polygons for every chunk (used for inpaint mask)
    detections = []   # per-chunk structured dicts

    for bbox, text, confidence in ocr_results:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        raw_bboxes.append(bbox)
        detections.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "bounding_box": {
                "x_min": int(min(xs)), "y_min": int(min(ys)),
                "x_max": int(max(xs)), "y_max": int(max(ys)),
            },
        })

    # ── Merge chunks into whole lines ─────────────────────────────────────────
    merged_lines = merge_detections_into_lines(detections)
    line_texts   = [ln["text"] for ln in merged_lines]

    # ── 4. Translate one string per merged line ───────────────────────────────
    try:
        translations = translate_regions(line_texts)
    except Exception as e:
        translations = line_texts
        print(f"[WARN] Translation failed, using originals: {e}")

    # ── 5. Inpaint ALL original chunk bboxes ─────────────────────────────────
    try:
        inpainted_bgr = inpaint_regions(img_bgr, raw_bboxes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Inpainting failed: {e}"})

    # ── 6. Draw one translated string per merged line ─────────────────────────
    try:
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(inpainted_rgb)
        result_pil = draw_translated_lines(result_pil, merged_lines, translations)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Text rendering failed: {e}"})

    # ── 7. Encode and return ──────────────────────────────────────────────────
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Extracted-Text": " | ".join(line_texts),
            "X-Translated-Text": " | ".join(translations),
            "X-Num-Lines": str(len(merged_lines)),
        },
    )
