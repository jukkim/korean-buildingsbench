"""
Korean_BB Forecast API — 168h→24h 시계열 예측 서비스 (:8040).

ems_transformer 게이트웨이의 backend 중 하나. 다른 클라이언트(building-energy-3d,
분석 스크립트 등)도 직접 호출 가능.

Run:
  cd Korean_BB
  uvicorn api.app:app --host 127.0.0.1 --port 8040
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "reverse"))

from modules.module_f_forecast.korean_bb_wrapper import KoreanBBForecaster

DEFAULT_CHECKPOINT = os.environ.get(
    "KOREAN_BB_CHECKPOINT",
    str(ROOT / "checkpoints" / "TransformerWithGaussian-M-v3-3k_bb700_s18000_revin_on_best.pt"),
)
DEFAULT_DEVICE = os.environ.get("KOREAN_BB_DEVICE", "cpu")

app = FastAPI(
    title="Korean_BB Forecast API",
    version="1.0.0",
    description="168h→24h building load forecast — TransformerWithGaussian-M (CVRMSE 12.93%)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_forecaster: KoreanBBForecaster | None = None


def _get_forecaster() -> KoreanBBForecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = KoreanBBForecaster(device=DEFAULT_DEVICE)
    return _forecaster


class PredictRequest(BaseModel):
    load_168h: list[float] = Field(..., description="과거 168시간 hourly 전력(kWh/h)")
    start_time: datetime | None = Field(default=None, description="첫 컨텍스트 시점 ISO 8601")
    device: Literal["auto", "cpu", "cuda"] = "cpu"


class BatchPredictRequest(BaseModel):
    series: list[list[float]] = Field(..., description="여러 건물의 168h 시계열 배열")
    start_times: list[datetime] | None = None
    device: Literal["auto", "cpu", "cuda"] = "cpu"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": "TransformerWithGaussian-M",
        "checkpoint": Path(DEFAULT_CHECKPOINT).name,
        "loaded": _forecaster is not None,
        "device": DEFAULT_DEVICE,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if len(req.load_168h) < 168:
        raise HTTPException(status_code=400, detail=f"load_168h must have >= 168 values (got {len(req.load_168h)})")
    arr = np.asarray(req.load_168h[-168:], dtype=np.float32).reshape(-1)
    fc = _get_forecaster()
    result = fc.forecast(arr, start_time=req.start_time)
    return {
        "model": {
            "id": "korean_bb_twgauss_m",
            "family": "transformer_with_gaussian",
            "training_data": "Korean_BB_700_sims_revin_on",
            "performance": {"cvrmse_pct": 12.93},
        },
        "result": result.to_dict(),
    }


@app.post("/predict_batch")
def predict_batch(req: BatchPredictRequest) -> dict:
    fc = _get_forecaster()
    out = []
    starts = req.start_times or [None] * len(req.series)
    for i, (s, st) in enumerate(zip(req.series, starts)):
        if len(s) < 168:
            out.append({"index": i, "error": f"need >= 168 values, got {len(s)}"})
            continue
        arr = np.asarray(s[-168:], dtype=np.float32).reshape(-1)
        try:
            r = fc.forecast(arr, start_time=st)
            out.append({"index": i, "result": r.to_dict()})
        except Exception as exc:
            out.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "model": {"id": "korean_bb_twgauss_m"},
        "count": len(out),
        "results": out,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8040"))
    uvicorn.run(app, host=host, port=port)
