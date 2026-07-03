"""
main.py — CervixAI FastAPI backend.
Run locally with:  uvicorn main:app --reload --port 8000
"""

import json
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import database
import model as model_lib

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CervixAI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten this to your deployed frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve uploaded images so the frontend can preview/report them
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def startup():
    database.init_db()
    try:
        model_lib.load_model()
        print("[CervixAI] Model loaded successfully.")
    except Exception as e:
        print(f"[CervixAI] WARNING — model failed to load at startup: {e}")


def _save_upload(file: UploadFile, scan_id: str) -> Path:
    ext = Path(file.filename).suffix or ".png"
    dest = UPLOAD_DIR / f"{scan_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


def _run_single_analysis(file: UploadFile, patient_name: Optional[str] = None) -> dict:
    scan_id = f"CX-{uuid.uuid4().hex[:8].upper()}"

    raw_bytes = file.file.read()
    file.file.seek(0)

    try:
        predicted_class, confidence, all_probs = model_lib.predict(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    risk = model_lib.get_risk(predicted_class)
    note = model_lib.get_clinical_note(predicted_class)
    edu = model_lib.get_educational_info(predicted_class)

    saved_path = _save_upload(file, scan_id)

    # Grad-CAM heatmap — best-effort, analysis still succeeds if this fails
    heatmap_url = None
    heatmap_rel_path = None
    try:
        class_idx = model_lib.CLASS_NAMES.index(predicted_class)
        overlay_img = model_lib.generate_gradcam(raw_bytes, class_idx)
        heatmap_path = UPLOAD_DIR / f"{scan_id}_heatmap.png"
        overlay_img.save(heatmap_path)
        heatmap_rel_path = str(heatmap_path.relative_to(BASE_DIR))
        heatmap_url = f"/uploads/{heatmap_path.name}"
    except Exception as e:
        print(f"[CervixAI] Grad-CAM generation failed for {scan_id}: {e}")

    database.insert_analysis(
        scan_id=scan_id,
        filename=file.filename,
        image_path=str(saved_path.relative_to(BASE_DIR)),
        heatmap_path=heatmap_rel_path,
        cell_type=predicted_class,
        confidence=confidence,
        risk_level=risk,
        probabilities=json.dumps(all_probs),
        patient_name=patient_name,
    )

    return {
        "scan_id": scan_id,
        "filename": file.filename,
        "cell_type": predicted_class,
        "confidence": confidence,
        "risk_level": risk,
        "probabilities": all_probs,
        "clinical_note": note,
        "educational_info": edu,
        "image_url": f"/uploads/{saved_path.name}",
        "heatmap_url": heatmap_url,
        "second_opinion_requested": False,
    }



@app.post("/api/analyze")
def analyze_single(file: UploadFile = File(...), patient_name: Optional[str] = Form(None)):
    """Single image analysis."""
    return _run_single_analysis(file, patient_name)


@app.post("/api/analyze/batch")
def analyze_batch(files: List[UploadFile] = File(...)):
    """Multiple image upload — analyzes each and returns a list of results."""
    results = []
    errors = []
    for f in files:
        try:
            results.append(_run_single_analysis(f))
        except HTTPException as e:
            errors.append({"filename": f.filename, "error": e.detail})
    return {"results": results, "errors": errors, "total": len(files)}


@app.get("/api/history")
def history(limit: int = 200):
    rows = database.get_all_analyses(limit=limit)
    for r in rows:
        r["probabilities"] = json.loads(r["probabilities"])
    return rows


@app.get("/api/report/{scan_id}")
def get_report(scan_id: str):
    row = database.get_analysis_by_scan_id(scan_id)
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
    row["probabilities"] = json.loads(row["probabilities"])
    row["clinical_note"] = model_lib.get_clinical_note(row["cell_type"])
    row["educational_info"] = model_lib.get_educational_info(row["cell_type"])
    if row.get("heatmap_path"):
        row["heatmap_url"] = f"/uploads/{Path(row['heatmap_path']).name}"
    row["image_url"] = f"/uploads/{Path(row['image_path']).name}"
    return row


class SecondOpinionRequest(BaseModel):
    note: Optional[str] = None


@app.post("/api/second-opinion/{scan_id}")
def request_second_opinion(scan_id: str, payload: SecondOpinionRequest = SecondOpinionRequest()):
    updated = database.set_second_opinion(scan_id, payload.note)
    if not updated:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"scan_id": scan_id, "second_opinion_requested": True, "note": payload.note}


@app.delete("/api/history/{scan_id}")
def delete_record(scan_id: str):
    deleted = database.delete_analysis(scan_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scan not found")
    return {"deleted": True, "scan_id": scan_id}


@app.get("/api/dashboard/stats")
def dashboard_stats():
    return database.get_dashboard_stats()


@app.get("/api/classes")
def get_classes():
    return {
        "classes": model_lib.CLASS_NAMES,
        "risk_map": model_lib.RISK_MAP,
    }


# ---------------------------------------------------------------------------
# Serve the frontend LAST so it doesn't shadow the /api/* routes above.
# Only active when a bundled ./static/index.html exists (e.g. Docker deploy).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Serve the frontend LAST so it doesn't shadow the /api/* routes above.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DEBUG: Check whether the frontend exists inside the Docker container
# ---------------------------------------------------------------------------

STATIC_DIR = BASE_DIR / "static"

print("BASE_DIR:", BASE_DIR)
print("STATIC_DIR:", STATIC_DIR)
print("STATIC EXISTS:", STATIC_DIR.exists())

if STATIC_DIR.exists():
    print("Serving frontend")

    @app.get("/", include_in_schema=False)
    def frontend():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

else:
    print("Serving API only")

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "status": "ok",
            "service": "CervixAI API",
            "base_dir": str(BASE_DIR),
            "static_dir": str(STATIC_DIR),
        }