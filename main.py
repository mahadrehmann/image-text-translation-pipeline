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


def get_text_color_kmeans(region_rgb: np.ndarray) -> tuple[int, int, int]:
    """
    Applies K-means (k=2) to the pixels inside a bounding-box crop to find
    the dominant text color.

    Strategy:
    - The two clusters represent background and text.
    - The SMALLER cluster (fewer pixels) is assumed to be the text color,
      since text covers less area than the background.
    - Falls back to black or white (brightness-based) if K-means fails.

    Parameters
    ----------
    region_rgb : H x W x 3 uint8 numpy array (RGB)

    Returns
    -------
    (R, G, B) tuple for the text color.
    """
    try:
        pixels = region_rgb.reshape(-1, 3).astype(np.float32)

        if len(pixels) < 2:
            raise ValueError("Region too small")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 2, None, criteria,
            attempts=3, flags=cv2.KMEANS_RANDOM_CENTERS
        )

        labels = labels.flatten()
        count0 = int(np.sum(labels == 0))
        count1 = int(np.sum(labels == 1))

        # Text cluster = the minority cluster
        text_cluster = 0 if count0 < count1 else 1
        color = centers[text_cluster].astype(int)
        return (int(color[0]), int(color[1]), int(color[2]))

    except Exception:
        # Fallback: black on light backgrounds, white on dark
        brightness = float(region_rgb.mean())
        return (0, 0, 0) if brightness > 128 else (255, 255, 255)


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


def is_paragraph_line(
    idx: int,
    merged_lines: list[dict],
    x_threshold: int = 30,
    vertical_scan_factor: float = 5.0,
) -> bool:
    """
    Returns True if the line at `idx` has a vertically nearby neighbor
    (above or below) whose x_min is within `x_threshold` pixels.

    Logic:
    - In a paragraph, lines start from the same left margin → x_min values
      are close → classify as paragraph → left-align.
    - In a centred poster, each line has a different width and is centred
      independently → x_min values differ per line → classify as isolated
      → center-align.

    Only neighbors within `vertical_scan_factor × line_height` are considered
    so we don't compare against lines from a completely different section.
    """
    bb = merged_lines[idx]["bounding_box"]
    x_min_curr = bb["x_min"]
    y_min, y_max = bb["y_min"], bb["y_max"]
    line_h = max(1, y_max - y_min)
    v_window = vertical_scan_factor * line_h

    for j, other in enumerate(merged_lines):
        if j == idx:
            continue
        obb = other["bounding_box"]

        # Only look at vertically nearby lines
        gap_below = obb["y_min"] - y_max
        gap_above = y_min - obb["y_max"]
        is_nearby = (0 <= gap_below <= v_window) or (0 <= gap_above <= v_window)
        if not is_nearby:
            continue

        # If x_min values are close → paragraph
        if abs(obb["x_min"] - x_min_curr) <= x_threshold:
            return True

    return False


def draw_translated_lines(
    pil_img: Image.Image,
    merged_lines: list[dict],
    translations: list[str],
    original_img_array: np.ndarray,
) -> Image.Image:
    """
    Renders each translated line string into the line's merged bounding box.
    - Font sizes are estimated from bbox heights then clustered to eliminate
      EasyOCR jitter — lines of similar original size render identically.
    - Paragraph lines (close x_min neighbors) are left-aligned.
    - Isolated lines (headlines, labels) are center-aligned.
    - Text color is detected via K-means on the ORIGINAL (pre-inpaint) crop
      so the real text color is used, not the inpainted background.
    """
    draw = ImageDraw.Draw(pil_img)

    # Pre-compute and cluster font sizes across all lines
    box_heights = [
        ln["bounding_box"]["y_max"] - ln["bounding_box"]["y_min"]
        for ln in merged_lines
    ]
    raw_sizes    = [estimate_font_size(h) for h in box_heights]
    stable_sizes = cluster_font_sizes(raw_sizes)

    for idx, (line, translated, font_size) in enumerate(zip(merged_lines, translations, stable_sizes)):
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

        tbbox = draw.textbbox((0, 0), translated, font=font)
        tw, th = tbbox[2] - tbbox[0], tbbox[3] - tbbox[1]

        # Paragraph lines → left-align; isolated lines → center-align
        if is_paragraph_line(idx, merged_lines):
            padding = max(4, int(box_h * 0.05))
            x = x_min + padding
        else:
            x = x_min + (box_w - tw) // 2

        y = y_min + (box_h - th) // 2  # always vertically centered

        # K-means color detection on the ORIGINAL image crop (before inpainting)
        region = original_img_array[y_min:y_max, x_min:x_max]
        text_color = get_text_color_kmeans(region)

        draw.text((x, y), translated, font=font, fill=text_color)

    return pil_img


# ── Shared pipeline ───────────────────────────────────────────────────────────

ALLOWED_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")
INPUT_DIR     = "images"
OUTPUT_DIR    = "reconstructed-images"


def run_pipeline(image_bytes: bytes) -> Image.Image:
    """
    Runs the full OCR → translate → inpaint → render pipeline on raw image
    bytes.  Returns the reconstructed PIL Image.

    Raises exceptions on failure so callers can handle them appropriately.
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(pil_image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # OCR
    ocr_results = reader.readtext(img_array, detail=1)
    if not ocr_results:
        return pil_image  # nothing to do — return original

    # Parse chunks
    raw_bboxes, detections = [], []
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

    # Merge chunks into lines
    merged_lines = merge_detections_into_lines(detections)
    line_texts   = [ln["text"] for ln in merged_lines]

    # Translate
    try:
        translations = translate_regions(line_texts)
    except Exception as e:
        print(f"[WARN] Translation failed, using originals: {e}")
        translations = line_texts

    # Inpaint
    inpainted_bgr = inpaint_regions(img_bgr, raw_bboxes)

    # Render translated text
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    result_pil    = Image.fromarray(inpainted_rgb)
    result_pil    = draw_translated_lines(result_pil, merged_lines, translations, img_array)

    return result_pil


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/process-image",
    responses={200: {"content": {"image/png": {}}}},
    response_class=StreamingResponse,
)
async def process_image(image: UploadFile = File(...)):
    """
    Single-image pipeline: upload one image, receive the reconstructed PNG.

    Steps:
    1. Validate image type
    2. EasyOCR — detect text regions + bounding boxes
    3. Gemini 2.5 Flash — translate all lines in one call
    4. cv2 TELEA inpainting — erase original text
    5. K-means color detection + Pillow — render translated text
    6. Return the modified image as PNG
    """
    if image.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type '{image.content_type}'. Allowed: {', '.join(ALLOWED_TYPES)}"},
        )

    try:
        image_bytes = await image.read()
        result_pil  = run_pipeline(image_bytes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/batch-process")
async def batch_process():
    """
    Batch pipeline: reads every image from the `images/` folder, runs the
    full pipeline on each one, and saves the results to `reconstructed-images/`.

    Returns a JSON summary with per-file status (success / error).

    No file upload needed — images must already be present in the `images/`
    directory relative to where the server is running.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        return JSONResponse(
            status_code=404,
            content={"error": f"Input directory '{INPUT_DIR}' not found."},
        )

    # Collect supported files
    candidates = [
        f for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".webp")
    ]

    if not candidates:
        return JSONResponse(
            status_code=404,
            content={"error": f"No supported images found in '{INPUT_DIR}'."},
        )

    results = []
    for filename in sorted(candidates):
        input_path  = os.path.join(INPUT_DIR, filename)
        stem        = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{stem}.png")

        try:
            with open(input_path, "rb") as f:
                image_bytes = f.read()

            result_pil = run_pipeline(image_bytes)
            result_pil.save(output_path, format="PNG")

            results.append({
                "file": filename,
                "status": "success",
                "output": output_path,
            })
            print(f"[BATCH] ✓ {filename} → {output_path}")

        except Exception as e:
            results.append({
                "file": filename,
                "status": "error",
                "error": str(e),
            })
            print(f"[BATCH] ✗ {filename}: {e}")

    success_count = sum(1 for r in results if r["status"] == "success")
    return {
        "total": len(results),
        "succeeded": success_count,
        "failed": len(results) - success_count,
        "output_dir": OUTPUT_DIR,
        "results": results,
    }

