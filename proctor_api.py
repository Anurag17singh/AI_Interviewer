import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import base64
import uuid

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT, "proctor_data")
os.makedirs(SAVE_DIR, exist_ok=True)

app = FastAPI(title="Proctor API")

# allow requests from Streamlit (running on localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/snapshot")
async def snapshot(candidate: str = Form(...), timestamp: str = Form(...), image_b64: str = Form(...)):
    """
    Receives base64-encoded image (data only, without "data:image/..." prefix)
    """
    try:
        data = base64.b64decode(image_b64)
        fname = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{candidate}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(SAVE_DIR, fname)
        with open(path, "wb") as f:
            f.write(data)
        return JSONResponse({"status": "ok", "path": path})
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)}, status_code=500)

@app.post("/violation")
async def violation(candidate: str = Form(...), timestamp: str = Form(...), event: str = Form(...), details: str = Form("")):
    """
    Receives string events such as 'tab_hidden', 'multiple_tabs', 'paste_attempt', etc.
    Logs to a file.
    """
    try:
        logpath = os.path.join(SAVE_DIR, f"violations_{candidate}.log")
        with open(logpath, "a", encoding="utf-8") as f:
            f.write(f"{datetime.utcnow().isoformat()} | EVENT={event} | DETAILS={details}\n")
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "details": str(e)}, status_code=500)
