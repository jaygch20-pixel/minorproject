from __future__ import annotations

import base64
import io
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.services.dataset import inspect_archive
from app.services.skin_analysis import (
    analyze_skin_image,
    evaluation_metrics,
    model_status as trained_model_status,
)

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent
ARCHIVE_PATH = ROOT_DIR / "archive.zip"
HISTORY_PATH = BASE_DIR / "data" / "analysis_history.db"
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 64
MIN_IMAGE_INTENSITY_STD = 2.0
MIN_IMAGE_SHARPNESS = 5.0
ANALYSIS_LOCK = threading.Lock()

app = FastAPI(title="DermaVision AI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_db() -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(HISTORY_PATH)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                image_data TEXT,
                prediction TEXT,
                confidence REAL,
                status TEXT,
                notes TEXT,
                result_json TEXT,
                source TEXT
            )
            """
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(analyses)").fetchall()
        }
        if "source" not in columns:
            conn.execute("ALTER TABLE analyses ADD COLUMN source TEXT")


init_db()


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(HISTORY_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        yield conn


@lru_cache(maxsize=2)
def dataset_summary_from_archive() -> dict:
    if not ARCHIVE_PATH.exists():
        return {
            "available": False,
            "message": "No archive.zip file was found in the project root.",
            "archive_name": None,
        }
    summary = inspect_archive(ARCHIVE_PATH)
    counts = summary["target_counts"]
    return {
        "available": summary["target_column"] == "target",
        "archive_name": summary["archive_name"],
        "metadata_file": summary["train_metadata_member"],
        "total_records": summary["train_records"],
        "test_records": summary["test_records"],
        "target_column": summary["target_column"],
        "image_column": summary["train_image_column"],
        "label_encoding": {"0": "non_melanoma", "1": "melanoma"},
        "class_distribution": {
            "non_melanoma": counts.get("0", 0),
            "melanoma": counts.get("1", 0),
        },
        "train_images_matched": summary["train_ids_matched"],
        "test_images_matched": summary["test_ids_matched"],
        "verified_masks_available": summary["verified_masks_available"],
        "test_labels_available": summary["test_labels_available"],
        "unique_patients": summary["unique_train_patients"],
        "note": "Metadata and image names are read from archive.zip without extracting images.",
    }


def save_history_record(filename: str, image_data: str | None, result: dict, source: str) -> dict:
    record_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    payload = json.dumps(result)
    with closing(sqlite3.connect(HISTORY_PATH)) as conn, conn:
        conn.execute(
            """
            INSERT INTO analyses (
                id, filename, created_at, image_data, prediction,
                confidence, status, notes, result_json, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                filename,
                datetime.now(timezone.utc).isoformat(),
                image_data,
                result.get("prediction"),
                result.get("confidence"),
                result.get("status"),
                result.get("warning"),
                payload,
                source,
            ),
        )
    return {"id": record_id, **result, "filename": filename, "created_at": datetime.now(timezone.utc).isoformat()}


def list_history_records() -> list[dict]:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY created_at DESC").fetchall()

    records = []
    for row in rows:
        payload = json.loads(row["result_json"]) if row["result_json"] else {}
        if payload.get("preprocessing"):
            payload["preprocessing"].pop("image_data", None)
        if payload.get("segmentation"):
            for field in ("mask_image_data", "segmented_image_data", "boundary_overlay_image_data"):
                payload["segmentation"].pop(field, None)
        records.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "created_at": row["created_at"],
                "prediction": row["prediction"],
                "confidence": row["confidence"],
                "status": row["status"],
                "notes": row["notes"],
                "image_data": row["image_data"],
                "source": row["source"] if "source" in row.keys() else "upload",
                "result": payload,
            }
        )
    return records


@app.get("/")
def root() -> dict:
    return {"message": "DermaVision AI backend is running."}


@app.get("/api/health")
def health() -> dict:
    summary = trained_model_status()
    return {
        "status": "ok",
        "backend": "DermaVision AI API",
        "model_available": summary["available"],
        "dataset_available": dataset_summary_from_archive().get("available", False),
        "mode": summary["status"],
        "model_name": summary.get("model_name"),
    }


@app.get("/api/dataset/summary")
def dataset_summary() -> dict:
    summary = dataset_summary_from_archive()
    if not summary.get("available"):
        return {"status": "unavailable", **summary}
    return {"status": "ok", **summary}


@app.get("/api/evaluation/metrics")
def get_evaluation_metrics() -> dict:
    metrics = evaluation_metrics()
    if metrics is None:
        return {
            "status": "unavailable",
            "message": "CNN evaluation metrics are unavailable or invalid.",
        }
    return {"status": "ok", **metrics}


def validate_upload_image(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
            image_format = uploaded_image.format
            if image_format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=400,
                    detail="Please upload a valid JPG or PNG image.",
                )
            width, height = uploaded_image.size
            if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image must be at least {MIN_IMAGE_SIDE} × {MIN_IMAGE_SIDE} pixels.",
                )
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions are too large to process safely.",
                )

            thumbnail = uploaded_image.convert("RGB")
            thumbnail.thumbnail((256, 256), Image.Resampling.BILINEAR)
            gray = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2GRAY)
            if float(gray.std()) < MIN_IMAGE_INTENSITY_STD:
                raise HTTPException(
                    status_code=400,
                    detail="Image has too little visible detail. Upload a well-lit lesion image.",
                )
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            if sharpness < MIN_IMAGE_SHARPNESS:
                raise HTTPException(
                    status_code=400,
                    detail="Image appears severely out of focus. Upload a sharper image.",
                )
    except HTTPException:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a JPG or PNG image.",
        ) from error
    return image_format


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    source: str = Form("upload"),
    realtime: str = Form("false"),
) -> dict:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=400, detail="Please upload a JPG or PNG image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image exceeds 10MB. Please choose a smaller file.")

    image_format = validate_upload_image(image_bytes)

    image_mime_type = "image/png" if image_format == "PNG" else "image/jpeg"
    image_data = f"data:{image_mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    normalized_source = "camera" if source.strip().lower() == "camera" else "upload"
    with ANALYSIS_LOCK:
        result = analyze_skin_image(image_bytes)
    result["source"] = normalized_source

    if realtime.strip().lower() == "true":
        return {
            **result,
            "filename": file.filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_data": None,
        }

    record = save_history_record(file.filename, image_data, result, normalized_source)

    return {
        **result,
        "id": record["id"],
        "filename": file.filename,
        "created_at": record["created_at"],
        "image_data": image_data,
    }


@app.get("/api/history")
def history() -> dict:
    return {"status": "ok", "items": list_history_records()}


@app.get("/api/history/{record_id}")
def get_history_record(record_id: str) -> dict:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (record_id,)).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="History record not found.")

    payload = json.loads(row["result_json"]) if row["result_json"] else {}
    return {
        "id": row["id"],
        "filename": row["filename"],
        "created_at": row["created_at"],
        "image_data": row["image_data"],
        "prediction": row["prediction"],
        "confidence": row["confidence"],
        "status": row["status"],
        "notes": row["notes"],
        "source": row["source"] if "source" in row.keys() else "upload",
        "result": payload,
    }


@app.delete("/api/history/{record_id}")
def delete_history_record(record_id: str) -> dict:
    with get_db_connection() as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (record_id,))
        conn.commit()

    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="History record not found.")

    return {"status": "deleted", "id": record_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
