# app/main.py
"""
FastAPI inference service for the Customer Churn & LTV Prediction Engine.

Endpoints:
    GET  /health              — liveness probe
    POST /predict/single      — single customer churn + LTV score
    POST /predict/batch       — batch scoring (CSV upload or JSON list)
    GET  /model/info          — model metadata

Run from project root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import io
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn & LTV Prediction API",
    description=(
        "XGBoost-powered inference service for predicting customer churn "
        "probability and lifetime value (LTV) from Telco customer profiles."
    ),
    version="2.0.0",
)

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.json")
CHURN_META_PATH  = os.path.join(MODELS_DIR, "churn_model_meta.joblib")
LTV_MODEL_PATH   = os.path.join(MODELS_DIR, "ltv_model.json")

# ---------------------------------------------------------------------------
# Model state (loaded once at startup)
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "churn_model":    None,
    "churn_meta":     None,
    "ltv_model":      None,
    "models_loaded":  False,
}


@app.on_event("startup")
def _load_models() -> None:
    """Load XGBoost models + preprocessing metadata on app startup."""
    try:
        if os.path.exists(CHURN_MODEL_PATH):
            _state["churn_model"] = xgb.XGBClassifier()
            _state["churn_model"].load_model(CHURN_MODEL_PATH)
            logger.info("Churn model loaded from %s", CHURN_MODEL_PATH)
        else:
            logger.warning("Churn model not found at %s", CHURN_MODEL_PATH)

        if os.path.exists(CHURN_META_PATH):
            _state["churn_meta"] = joblib.load(CHURN_META_PATH)
            logger.info("Churn preprocessing metadata loaded.")

        if os.path.exists(LTV_MODEL_PATH):
            _state["ltv_model"] = xgb.XGBRegressor()
            _state["ltv_model"].load_model(LTV_MODEL_PATH)
            logger.info("LTV model loaded from %s", LTV_MODEL_PATH)
        else:
            logger.warning("LTV model not found at %s", LTV_MODEL_PATH)

        _state["models_loaded"] = (
            _state["churn_model"] is not None
            and _state["ltv_model"] is not None
        )

    except Exception as exc:
        logger.error("Error loading models: %s", exc)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class CustomerRecord(BaseModel):
    """Single customer feature payload."""
    customerID:       str   = Field(...,  example="7590-VHVEG")
    gender:           str   = Field(...,  example="Female")
    SeniorCitizen:    int   = Field(...,  ge=0, le=1, example=0)
    Partner:          str   = Field(...,  example="Yes")
    Dependents:       str   = Field(...,  example="No")
    tenure:           float = Field(...,  ge=0, example=12)
    PhoneService:     str   = Field(...,  example="Yes")
    MultipleLines:    str   = Field(...,  example="No")
    InternetService:  str   = Field(...,  example="DSL")
    OnlineSecurity:   str   = Field(...,  example="Yes")
    OnlineBackup:     str   = Field(...,  example="No")
    DeviceProtection: str   = Field(...,  example="No")
    TechSupport:      str   = Field(...,  example="Yes")
    StreamingTV:      str   = Field(...,  example="No")
    StreamingMovies:  str   = Field(...,  example="No")
    Contract:         str   = Field(...,  example="Month-to-month")
    PaperlessBilling: str   = Field(...,  example="Yes")
    PaymentMethod:    str   = Field(...,  example="Electronic check")
    MonthlyCharges:   float = Field(...,  ge=0, example=55.85)
    TotalCharges:     float = Field(...,  ge=0, example=670.20)


class PredictionResult(BaseModel):
    customerID:        str
    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    churn_prediction:  int   = Field(..., description="Binary churn label (0=stay, 1=churn)")
    churn_risk_tier:   str   = Field(..., description="LOW / MEDIUM / HIGH")
    ltv_prediction:    float = Field(..., description="Predicted lifetime value (TotalCharges)")


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerRecord]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUM_COLS = ["tenure", "MonthlyCharges"]


def _preprocess(df: pd.DataFrame, meta: dict | None = None) -> pd.DataFrame:
    """
    Apply the same encoding used during training:
      1. pd.get_dummies for categoricals
      2. Standardise numericals using stored fit_stats
      3. Reindex to training feature columns
    """
    # Clean TotalCharges (shouldn't be needed for API input but be safe)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    cat_present = [c for c in CAT_COLS if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_present, drop_first=False)

    if meta:
        fit_stats = meta.get("fit_stats", {})
        feature_cols = meta.get("feature_columns", [])

        for col, stats in fit_stats.items():
            if col in df_enc.columns:
                std = stats["std"] if stats["std"] > 0 else 1.0
                df_enc[col] = (df_enc[col] - stats["mean"]) / std

        df_enc = df_enc.reindex(columns=feature_cols, fill_value=0)

    return df_enc


def _risk_tier(prob: float) -> str:
    if prob >= 0.65:
        return "HIGH"
    if prob >= 0.35:
        return "MEDIUM"
    return "LOW"


def _score_dataframe(df_raw: pd.DataFrame) -> list[dict]:
    """Run churn + LTV inference on a raw feature DataFrame."""
    churn_model = _state["churn_model"]
    ltv_model   = _state["ltv_model"]
    meta        = _state["churn_meta"]

    if churn_model is None or ltv_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Run notebooks 03 and 04 first to generate "
                "churn_model.json and ltv_model.json in the models/ directory."
            ),
        )

    X = df_raw.drop(columns=["customerID", "Churn"], errors="ignore").copy()
    X_enc = _preprocess(X, meta)

    churn_probs = churn_model.predict_proba(X_enc)[:, 1].tolist()
    churn_preds = churn_model.predict(X_enc).tolist()
    ltv_preds   = ltv_model.predict(X_enc).tolist()

    results = []
    for i, (_, row) in enumerate(df_raw.iterrows()):
        results.append({
            "customerID":        row.get("customerID", f"CUST_{i}"),
            "churn_probability": round(churn_probs[i], 4),
            "churn_prediction":  int(churn_preds[i]),
            "churn_risk_tier":   _risk_tier(churn_probs[i]),
            "ltv_prediction":    round(float(ltv_preds[i]), 2),
        })
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    """Liveness probe — returns model load status."""
    return {
        "status":        "ok",
        "models_loaded": _state["models_loaded"],
        "churn_model":   _state["churn_model"] is not None,
        "ltv_model":     _state["ltv_model"]   is not None,
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Returns metadata about the loaded models."""
    meta = _state["churn_meta"] or {}
    n_features = len(meta.get("feature_columns", []))
    return {
        "churn_model_path":  CHURN_MODEL_PATH,
        "ltv_model_path":    LTV_MODEL_PATH,
        "n_features":        n_features,
        "feature_columns":   meta.get("feature_columns", []),
        "models_loaded":     _state["models_loaded"],
    }


@app.post(
    "/predict/single",
    response_model=PredictionResult,
    tags=["Inference"],
    summary="Single customer churn + LTV prediction",
)
def predict_single(customer: CustomerRecord):
    """
    Score a single customer record.

    Returns churn probability, binary prediction, risk tier (LOW/MEDIUM/HIGH),
    and predicted LTV (TotalCharges).
    """
    df = pd.DataFrame([customer.model_dump()])
    results = _score_dataframe(df)
    return results[0]


@app.post(
    "/predict/batch",
    response_model=list[PredictionResult],
    tags=["Inference"],
    summary="Batch prediction from JSON list",
)
def predict_batch(request: BatchPredictionRequest):
    """
    Score a batch of customer records provided as a JSON list.

    Returns one prediction result per customer in the same order.
    """
    if not request.customers:
        raise HTTPException(status_code=400, detail="No customers provided.")
    if len(request.customers) > 5000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 5000 limit.")

    df = pd.DataFrame([c.model_dump() for c in request.customers])
    return _score_dataframe(df)


@app.post(
    "/predict/batch/csv",
    response_model=list[PredictionResult],
    tags=["Inference"],
    summary="Batch prediction from uploaded CSV file",
)
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file (same schema as the Telco dataset) and receive
    churn + LTV predictions for every row.

    The CSV must contain all feature columns. customerID is optional.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    if len(df) > 10_000:
        raise HTTPException(status_code=400, detail="CSV exceeds 10,000 row limit.")

    return _score_dataframe(df)
