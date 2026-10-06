import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps
from rembg import new_session, remove

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cutout")

BASE = Path(__file__).parent
MAX_BYTES = 15 * 1024 * 1024
MAX_SIDE = 1600  # keep processing fast, especially on free hosting

app = FastAPI(title="Cutout")

# Loading the model can itself fail (e.g. no network on first run). Keep the
# app alive either way and report a clear error per-request instead of
# crashing the whole server at startup.
try:
    session = new_session("isnet-general-use")
    MODEL_ERROR = None
except Exception as e:  # pragma: no cover
    session = None
    MODEL_ERROR = str(e)
    log.exception("Failed to load background-removal model")


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    # Without this, an unhandled error returns an empty/HTML 500 response,
    # which breaks the frontend's attempt to read a JSON error message.
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

    result = None
    if refine:
        # Alpha matting gives cleaner edges but is slower and occasionally
        # fails on tricky images — fall back to a plain removal if so.
        try:
            result = remove(img, session=session, alpha_matting=True)
        except Exception:
            log.warning("alpha matting failed, falling back", exc_info=True)
            result = None

    if result is None:
        try:
            result = remove(img, session=session, alpha_matting=False)
        except Exception as e:
            log.exception("Background removal failed")
            raise HTTPException(500, f"Background removal failed: {e}")

    if trim and result.getbbox():
        result = result.crop(result.getbbox())

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")
