# ── Base image ────────────────────────────────────────────────────────────────
# slim-bookworm keeps the image small while providing the system libs
# needed by OpenCV and EasyOCR (libGL, libglib2, etc.)
FROM python:3.12-slim-bookworm

# ── System dependencies ────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0  → OpenCV headless (cv2)
# libgomp1               → PyTorch / EasyOCR CPU parallelism
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY main.py .

# Copy input images if present (optional — batch mode needs this)
COPY images/ ./images/

# ── Environment variables ──────────────────────────────────────────────────────
# The .env file is read at runtime by python-dotenv (load_dotenv() in main.py).
# Mount or COPY it here; it is NOT baked into the image layer for security.
# Option A (recommended): pass the key at run time via --env-file
#   docker run --env-file .env ...
# Option B: mount at run time
#   docker run -v $(pwd)/.env:/app/.env ...
# Option C (dev only): COPY it here
# COPY .env .

# ── Output directory ───────────────────────────────────────────────────────────
# Pre-create so the batch endpoint can write without permission errors
RUN mkdir -p /app/reconstructed-images

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
