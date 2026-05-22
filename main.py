from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

'''
To run:
uvicorn main:app --reload

and goto http://127.0.0.1:8000/docs
'''


app = FastAPI(
    title="Image Processing API",
    description="API that accepts an image and returns extracted text/translation data.",
    version="0.1.0",
)


@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    """
    Accepts an image file and returns mock JSON output.

    In the future this will run OCR + translation on the image.
    For now it returns a static mock response.
    """
    # Basic validation
    if image.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {image.content_type}. Please upload a PNG, JPEG, or WebP image."},
        )

    # Read file metadata (we don't process it yet)
    file_size = len(await image.read())

    # Mock response
    return {
        "status": "success",
        "filename": image.filename,
        "content_type": image.content_type,
        "file_size_bytes": file_size,
        "extracted_text": "هذا نص تجريبي مستخرج من الصورة",
        "translated_text": "This is a sample text extracted from the image",
        "confidence": 0.95,
        "language_detected": "ar",
        "language_translated": "en",
        "bounding_boxes": [
            {"x": 10, "y": 20, "width": 200, "height": 40, "text": "هذا نص تجريبي"},
            {"x": 10, "y": 70, "width": 250, "height": 40, "text": "مستخرج من الصورة"},
        ],
    }
