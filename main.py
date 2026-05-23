from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import numpy as np
import io
import os
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
# Add more language codes here as needed, e.g. ['en', 'ar', 'es']
reader = easyocr.Reader(['en'], gpu=False)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Image Processing API",
    description="Extracts text from an image with EasyOCR and translates it using Gemini.",
    version="0.2.0",
)


async def translate_text(text: str) -> str:
    """
    Translates the given text to English using Gemini 2.5 Flash.
    Returns the translated string, or raises an exception on failure.
    """
    prompt = (
        "You are a professional translator. "
        "Translate the following text to English. "
        "Return ONLY the translated text with no extra commentary or explanation.\n\n"
        f"Text to translate:\n{text}"
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    """
    Accepts an image file, extracts text via EasyOCR,
    then translates the extracted text to English using Gemini 2.5 Flash.
    """
    # ── 1. Validate file type ─────────────────────────────────────────────────
    ALLOWED_TYPES = ("image/png", "image/jpeg", "image/jpg", "image/webp")
    if image.content_type not in ALLOWED_TYPES:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Unsupported file type: '{image.content_type}'. "
                    f"Allowed: {', '.join(ALLOWED_TYPES)}"
                )
            },
        )

    # ── 2. Read image bytes ───────────────────────────────────────────────────
    try:
        image_bytes = await image.read()
        file_size = len(image_bytes)

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_image)
    except Exception as e:
        return JSONResponse(
            status_code=422,
            content={"error": f"Failed to decode image: {str(e)}"},
        )

    # ── 3. Run OCR ────────────────────────────────────────────────────────────
    try:
        results = reader.readtext(img_array, detail=1)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"OCR processing failed: {str(e)}"},
        )

    # Build structured detections
    detections = []
    full_text_parts = []

    for bbox, text, confidence in results:
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        detections.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "bounding_box": {
                "x_min": int(min(xs)),
                "y_min": int(min(ys)),
                "x_max": int(max(xs)),
                "y_max": int(max(ys)),
            },
        })
        full_text_parts.append(text)

    extracted_text = " ".join(full_text_parts)

    # ── 4. Translate with Gemini ──────────────────────────────────────────────
    translated_text = None
    translation_error = None

    if extracted_text.strip():
        try:
            translated_text = await translate_text(extracted_text)
        except Exception as e:
            # Translation failure is non-fatal — still return OCR results
            translation_error = f"Translation failed: {str(e)}"
    else:
        translation_error = "No text detected in image — skipping translation."

    # ── 5. Build response ─────────────────────────────────────────────────────
    response = {
        "status": "success",
        "filename": image.filename,
        "content_type": image.content_type,
        "file_size_bytes": file_size,
        "image_width": pil_image.width,
        "image_height": pil_image.height,
        "extracted_text": extracted_text,
        "translated_text": translated_text,
        "num_detections": len(detections),
        "detections": detections,
    }

    if translation_error:
        response["translation_warning"] = translation_error

    return response
