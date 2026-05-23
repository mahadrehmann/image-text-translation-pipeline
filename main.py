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


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Tries common system fonts; falls back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def fit_text_in_box(draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
    """
    Binary-searches for the largest font size where `text` fits inside
    the given box dimensions. Returns (font, font_size).
    """
    lo, hi = 8, max(box_h, 10)
    best_size = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        font = get_font(mid)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= box_w and th <= box_h:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return get_font(best_size), best_size


def draw_translated_text(
    pil_img: Image.Image,
    detections: list[dict],
    translations: list[str],
) -> Image.Image:
    """
    Draws each translated string centered inside its original bounding box.
    Auto-sizes the font to fill the box, picks black or white text for contrast.
    """
    draw = ImageDraw.Draw(pil_img)

    for det, translated in zip(detections, translations):
        bb = det["bounding_box"]
        x_min, y_min = bb["x_min"], bb["y_min"]
        x_max, y_max = bb["x_max"], bb["y_max"]
        box_w = x_max - x_min
        box_h = y_max - y_min

        if box_w < 5 or box_h < 5 or not translated.strip():
            continue

        font, _ = fit_text_in_box(draw, translated, box_w, box_h)

        # Center text in the box
        bbox = draw.textbbox((0, 0), translated, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x_min + (box_w - tw) // 2
        y = y_min + (box_h - th) // 2

        # Pick text color (black/white) based on background brightness
        region = np.array(pil_img.crop((x_min, y_min, x_max, y_max)))
        avg_brightness = region.mean()
        text_color = (0, 0, 0) if avg_brightness > 128 else (255, 255, 255)

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

    # Parse OCR output
    raw_bboxes = []     # 4-corner polygons (for inpainting mask)
    detections = []     # structured dicts (for drawing)
    texts = []          # plain strings (for translation)

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
        texts.append(text)

    # ── 4. Translate ──────────────────────────────────────────────────────────
    try:
        translations = translate_regions(texts)
    except Exception as e:
        # Non-fatal: fall back to original texts
        translations = texts
        print(f"[WARN] Translation failed, using originals: {e}")

    # ── 5. Inpaint original text ──────────────────────────────────────────────
    try:
        inpainted_bgr = inpaint_regions(img_bgr, raw_bboxes)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Inpainting failed: {e}"})

    # ── 6. Draw translated text ───────────────────────────────────────────────
    try:
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(inpainted_rgb)
        result_pil = draw_translated_text(result_pil, detections, translations)
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
            "X-Extracted-Text": "; ".join(texts),
            "X-Translated-Text": "; ".join(translations),
            "X-Num-Detections": str(len(detections)),
        },
    )
