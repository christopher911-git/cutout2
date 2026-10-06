import io
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageFilter, ImageOps

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cutout")

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "u2netp.onnx"
MAX_BYTES = 15 * 1024 * 1024
MAX_SIDE = 1600  # keep processing fast, especially on free hosting
INPUT_SIZE = 320  # u2netp's expected input resolution

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="Cutout")

# Loading the model can itself fail. Keep the app alive either way and
# report a clear error per-request instead of crashing the whole server.
try:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=so, providers=["CPUExecutionProvider"]
    )
    INPUT_NAME = session.get_inputs()[0].name
    MODEL_ERROR = None
except Exception as e:  # pragma: no cover
    session = None
    INPUT_NAME = None
    MODEL_ERROR = str(e)
    log.exception("Failed to load background-removal model")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Server error: {exc}"})


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    if MODEL_ERROR:
        return JSONResponse(status_code=503, content={"status": "model_failed", "detail": MODEL_ERROR})
    return {"status": "ok"}


def predict_mask(img: Image.Image) -> Image.Image:
    """Run the model and return a single-channel mask at the image's original size."""
    small = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # NCHW

    outputs = session.run(None, {INPUT_NAME: arr})
    mask = outputs[0][0, 0]  # (320, 320)

    # Contrast-stretch the raw output, same as standard u2net postprocessing
    mn, mx = float(mask.min()), float(mask.max())
    if mx > mn:
        mask = (mask - mn) / (mx - mn)
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)

    mask_img = Image.fromarray(mask, mode="L").resize(img.size, Image.LANCZOS)
    return mask_img


@app.post("/remove")
async def remove_background(file: UploadFile = File(...), trim: bool = True, refine: bool = False):
    if session is None:
        raise HTTPException(503, f"Background-removal model is unavailable: {MODEL_ERROR}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file was empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB.")

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")

    img.thumbnail((MAX_SIDE, MAX_SIDE))

    try:
        mask = predict_mask(img)
    except Exception as e:
        log.exception("Background removal failed")
        raise HTTPException(500, f"Background removal failed: {e}")

    if refine:
        # Soften the mask edges a touch to reduce staircase/jagged artifacts
        # from upscaling the small model output back to full resolution.
        mask = mask.filter(ImageFilter.GaussianBlur(radius=1.5))

    result = img.copy()
    result.putalpha(mask)

    if trim and result.getbbox():
        result = result.crop(result.getbbox())

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")