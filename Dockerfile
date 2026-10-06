FROM python:3.11-slim

WORKDIR /app

# Keep memory/CPU usage low for constrained free-tier hosting
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static ./static

# Download the small u2netp model directly (same file rembg would normally
# fetch), avoiding rembg's much heavier dependency chain (numba, scipy,
# scikit-image, pymatting) which doesn't fit in 512MB free-tier RAM.
RUN mkdir -p model && \
    curl -L -o model/u2netp.onnx \
    https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx

EXPOSE 7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}