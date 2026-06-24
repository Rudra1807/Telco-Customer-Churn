"""
generate_predictions.py
------------------------
Generates a single, dashboard-ready predictions.csv combining:
    - Original customer features (from X_test.csv)
    - True churn label (from y_test.csv, for reference/validation)
    - Predicted churn probability + risk tier (from churn_model_test.json)
    - Predicted LTV (from ltv_model.json)

Run this from the project root:
    cd "customer-churn-ltv-engine"
    python generate_predictions.py

Output:
    data/processed/predictions.csv
"""

import os
import sys
import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# Make sure we can import from src/ when run from project root
sys.path.insert(0, os.getcwd())

from src.preprocessing import preprocess_dataframe  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(DATA_DIR, "y_test.csv")

CHURN_MODEL_PATH = os.path.join(MODEL_DIR, "churn_model_test.json")
CHURN_META_PATH = os.path.join(MODEL_DIR, "churn_model_test_meta.joblib")
LTV_MODEL_PATH = os.path.join(MODEL_DIR, "ltv_model.json")

OUTPUT_PATH = os.path.join(DATA_DIR, "predictions.csv")


def risk_tier(prob: float) -> str:
    """Buckets a churn probability into a business-friendly risk tier."""
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def main():
    print("Loading test data...")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    # y_test might come as a DataFrame with one column — squeeze to Series
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    # Keep a copy of customerID (if present) for the final output, then
    # drop identifier columns before feeding into the model.
    customer_ids = None
    if "customerID" in X_test.columns:
        customer_ids = X_test["customerID"].copy()

    print(f"Loaded {len(X_test):,} rows.")

    # ── 1. Load churn model + its preprocessing metadata ───────────────
    print("Loading churn model artifacts...")
    churn_model = XGBClassifier()
    churn_model.load_model(CHURN_MODEL_PATH)

    churn_meta = joblib.load(CHURN_META_PATH)
    churn_fit_stats = churn_meta.get("fit_stats", {})
    churn_feature_columns = churn_meta.get("feature_columns", [])

    # Drop non-feature columns before preprocessing, same as training did
    X_for_churn = X_test.drop(columns=["customerID"], errors="ignore")

    X_churn_enc, _, _ = preprocess_dataframe(
        X_for_churn,
        reference_columns=churn_feature_columns,
        fit_stats=churn_fit_stats,
    )

    print("Scoring churn probability...")
    churn_proba = churn_model.predict_proba(X_churn_enc)[:, 1]
    churn_pred = churn_model.predict(X_churn_enc)

    # ── 2. Load LTV model and score ───────────────────────────────────
    # The LTV model was trained on the same X_train structure minus
    # TotalCharges (which is the LTV regression target). We reuse the
    # churn model's fit_stats/columns for consistency since both were
    # derived from the same preprocessing step, just excluding
    # TotalCharges from the feature set.
    print("Loading LTV model...")
    ltv_model = XGBRegressor()
    ltv_model.load_model(LTV_MODEL_PATH)

    # Extract the exact feature list the LTV model was trained on from
    # its booster — this is the safest way to guarantee alignment since
    # the LTV model may use a different subset of features than churn.
    ltv_feature_columns = ltv_model.get_booster().feature_names
    X_ltv_enc = X_churn_enc.reindex(columns=ltv_feature_columns, fill_value=0)

    print("Scoring predicted LTV...")
    predicted_ltv = ltv_model.predict(X_ltv_enc)

    # ── 3. Assemble final dashboard-ready table ─────────────────────────
    print("Assembling final output...")
    output = X_test.copy()

    if customer_ids is not None:
        output["customerID"] = customer_ids
    else:
        output["customerID"] = [f"CUST_{i}" for i in range(len(output))]

    output["actual_churn"] = y_test.values
    output["churn_probability"] = churn_proba.round(4)
    output["churn_prediction"] = churn_pred
    output["risk_tier"] = output["churn_probability"].apply(risk_tier)
    output["predicted_ltv"] = predicted_ltv.round(2)

    output.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. Saved {len(output):,} rows to:\n{OUTPUT_PATH}")
    print("\nPreview:")
    print(output[["customerID", "actual_churn", "churn_probability",
                   "risk_tier", "predicted_ltv"]].head())


if __name__ == "__main__":
    main()
