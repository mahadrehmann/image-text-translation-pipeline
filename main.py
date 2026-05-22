from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
from PIL import Image
import numpy as np
import io

'''
To run:
uvicorn main:app --reload

and goto http://127.0.0.1:8000/docs
'''

# Initialize EasyOCR reader once at startup (avoids reloading model per request)
# Add more language codes here as needed, e.g. ['en', 'ar', 'es']
reader = easyocr.Reader(['en'], gpu=False)

app = FastAPI(
    title="Image Processing API",
    description="API that accepts an image and returns extracted text/translation data.",
    version="0.1.0",
)


@app.post("/process-image")
async def process_image(image: UploadFile = File(...)):
    """
    Accepts an image file and extracts text using EasyOCR.

    Returns detected text regions with bounding boxes and confidence scores.
    """
    # Basic validation
    if image.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {image.content_type}. Please upload a PNG, JPEG, or WebP image."},
        )

    # Read the uploaded image bytes
    image_bytes = await image.read()
    file_size = len(image_bytes)

    # Decode to numpy array for EasyOCR (it doesn't accept BytesIO)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(pil_image)

    # Run EasyOCR — detail=1 returns (bbox, text, confidence) per region
    results = reader.readtext(img_array, detail=1)

    # Build structured detections
    detections = []
    full_text_parts = []

    for bbox, text, confidence in results:
        # bbox is a list of 4 corner points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
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

    return {
        "status": "success",
        "filename": image.filename,
        "content_type": image.content_type,
        "file_size_bytes": file_size,
        "image_width": pil_image.width,
        "image_height": pil_image.height,
        "extracted_text": " ".join(full_text_parts),
        "num_detections": len(detections),
        "detections": detections,
    }
