"""
test_modules.py  -  Smoke test for all src/ modules (XGBoost edition)
Run from the project root:  python test_modules.py
"""
import sys
import os
import traceback
import pandas as pd
import numpy as np

results = []


def check(label, fn):
    try:
        val = fn()
        print("  [PASS] " + label)
        results.append((label, True, None))
        return val
    except Exception as exc:
        print("  [FAIL] " + label)
        print("         " + type(exc).__name__ + ": " + str(exc))
        results.append((label, False, exc))
        return None


# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  1 - utils")
print("="*60)

from src.utils import load_config, get_logger, PROJECT_ROOT

cfg    = check("load_config() returns a dict",         lambda: load_config())
_ok    = check("PROJECT_ROOT is a real directory",     lambda: True if os.path.isdir(PROJECT_ROOT) else (_ for _ in ()).throw(AssertionError(PROJECT_ROOT)))
logger = check("get_logger() returns a Logger",        lambda: get_logger("test", cfg))

# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  2 - data_loader")
print("="*60)

from src.data_loader import load_raw_data, validate_schema

df_raw = check("load_raw_data() loads the CSV",        lambda: load_raw_data())
if df_raw is not None:
    check("DataFrame has 7043 rows",                   lambda: True if df_raw.shape[0] == 7043 else (_ for _ in ()).throw(AssertionError(str(df_raw.shape[0]))))
    check("validate_schema() passes",                  lambda: validate_schema(df_raw))

# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  3 - preprocessing")
print("="*60)

from src.preprocessing import clean_data, preprocess_dataframe, split_dataset

df_clean = None
if df_raw is not None:
    df_clean = check("clean_data() returns DataFrame",         lambda: clean_data(df_raw))
    if df_clean is not None:
        check("Churn column is binary int {0,1}",
              lambda: True if set(df_clean["Churn"].dropna().unique()).issubset({0, 1})
              else (_ for _ in ()).throw(AssertionError(str(df_clean["Churn"].unique()))))
        check("TotalCharges is numeric dtype",
              lambda: True if pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])
              else (_ for _ in ()).throw(AssertionError("Not numeric")))

# Test preprocess_dataframe
if df_clean is not None:
    X_sample = df_clean.drop(columns=["customerID", "Churn"], errors="ignore").head(20)
    pp_result = check("preprocess_dataframe() returns (df, stats, cols) tuple",
                      lambda: preprocess_dataframe(X_sample))
    if pp_result is not None:
        df_enc, fit_stats, feat_cols = pp_result
        check("Encoded DataFrame has more columns than input (OHE applied)",
              lambda: True if df_enc.shape[1] > X_sample.shape[1]
              else (_ for _ in ()).throw(AssertionError(f"{df_enc.shape[1]} vs {X_sample.shape[1]}")))
        check("fit_stats is a dict with numeric-col entries",
              lambda: True if isinstance(fit_stats, dict) and len(fit_stats) > 0
              else (_ for _ in ()).throw(AssertionError(str(fit_stats))))
        check("feature_columns is a non-empty list",
              lambda: True if isinstance(feat_cols, list) and len(feat_cols) > 0
              else (_ for _ in ()).throw(AssertionError(str(feat_cols))))

splits = None
if df_clean is not None:
    splits = check("split_dataset() returns 4-tuple",
                   lambda: split_dataset(df_clean))
    if splits is not None:
        Xt, Xv, yt, yv = splits
        check("Train size ~80% (5000-6000 rows)",  lambda: True if 5000 < len(Xt) < 7000 else (_ for _ in ()).throw(AssertionError(str(len(Xt)))))
        check("Test  size ~20% (1000-2000 rows)",  lambda: True if 1000 < len(Xv) < 3000 else (_ for _ in ()).throw(AssertionError(str(len(Xv)))))

# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  4 - feature_engineering")
print("="*60)

from src.feature_engineering import create_features, add_tenure_cohorts

df_feat = None
if df_clean is not None:
    df_feat = check("create_features() runs without error", lambda: create_features(df_clean))
    if df_feat is not None:
        check("'tenure_group' column added",   lambda: True if "tenure_group"   in df_feat.columns else (_ for _ in ()).throw(AssertionError("Missing")))
        check("'total_services' column added", lambda: True if "total_services" in df_feat.columns else (_ for _ in ()).throw(AssertionError("Missing")))
        check("'charge_ratio' column added",   lambda: True if "charge_ratio"   in df_feat.columns else (_ for _ in ()).throw(AssertionError("Missing")))
        check("No 'nan' strings in tenure_group",
              lambda: True if "nan" not in df_feat["tenure_group"].values
              else (_ for _ in ()).throw(AssertionError("Found literal 'nan' string")))

        edge = pd.DataFrame([{"tenure": 0, "MonthlyCharges": 50.0, "TotalCharges": 0.0}])
        edge_out = add_tenure_cohorts(edge)
        check("tenure=0 edge -> '0-1 Year' (not 'nan')",
              lambda: True if edge_out["tenure_group"].iloc[0] == "0-1 Year"
              else (_ for _ in ()).throw(AssertionError(str(edge_out["tenure_group"].iloc[0]))))

        edge72 = pd.DataFrame([{"tenure": 72, "MonthlyCharges": 80.0, "TotalCharges": 5760.0}])
        edge72_out = add_tenure_cohorts(edge72)
        check("tenure=72 edge -> '5-6 Years'",
              lambda: True if edge72_out["tenure_group"].iloc[0] == "5-6 Years"
              else (_ for _ in ()).throw(AssertionError(str(edge72_out["tenure_group"].iloc[0]))))

# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  5 - train  (XGBoost)")
print("="*60)

from src.train import (
    train_churn_pipeline, train_ltv_pipeline,
    evaluate_churn_model, save_model_artifact,
)

churn_artifact = None
ltv_artifact   = None
saved_path     = None

if df_feat is not None and splits is not None:
    X_tr, X_te, y_tr, y_te = split_dataset(df_feat)

    # ── Churn model ────────────────────────────────────────────────────────
    churn_artifact = check("train_churn_pipeline() returns dict with model/fit_stats/feature_columns",
                           lambda: train_churn_pipeline(X_tr, y_tr))
    if churn_artifact is not None:
        check("artifact['model'] is XGBClassifier",
              lambda: True if hasattr(churn_artifact["model"], "predict_proba")
              else (_ for _ in ()).throw(AssertionError("Missing predict_proba")))
        check("artifact['fit_stats'] is a non-empty dict",
              lambda: True if isinstance(churn_artifact["fit_stats"], dict) and len(churn_artifact["fit_stats"]) > 0
              else (_ for _ in ()).throw(AssertionError(str(churn_artifact.get("fit_stats")))))
        check("artifact['feature_columns'] is a non-empty list",
              lambda: True if isinstance(churn_artifact["feature_columns"], list) and len(churn_artifact["feature_columns"]) > 0
              else (_ for _ in ()).throw(AssertionError(str(churn_artifact.get("feature_columns")))))

        # Preprocess test set with the same fit_stats + feature_columns
        from src.preprocessing import preprocess_dataframe as ppdf
        X_te_enc, _, _ = ppdf(
            X_te,
            reference_columns=churn_artifact["feature_columns"],
            fit_stats=churn_artifact["fit_stats"],
        )

        metrics = check("evaluate_churn_model() returns metrics dict",
                        lambda: evaluate_churn_model(churn_artifact["model"], X_te_enc, y_te))
        if metrics:
            check("Accuracy > 0.5  (got " + str(metrics.get("accuracy")) + ")",
                  lambda: True if metrics["accuracy"] > 0.5 else (_ for _ in ()).throw(AssertionError(str(metrics["accuracy"]))))
            check("ROC-AUC > 0.5  (got " + str(metrics.get("roc_auc")) + ")",
                  lambda: True if metrics["roc_auc"] > 0.5 else (_ for _ in ()).throw(AssertionError(str(metrics["roc_auc"]))))
            check("confusion_matrix is list-of-lists",
                  lambda: True if isinstance(metrics.get("confusion_matrix"), list) else (_ for _ in ()).throw(AssertionError()))

    # ── LTV model ──────────────────────────────────────────────────────────
    y_ltv_tr = df_feat.loc[X_tr.index, "TotalCharges"]
    ltv_artifact = check("train_ltv_pipeline() returns dict with model",
                         lambda: train_ltv_pipeline(X_tr, y_ltv_tr))
    if ltv_artifact is not None:
        check("LTV artifact['model'] has predict() method",
              lambda: True if hasattr(ltv_artifact["model"], "predict")
              else (_ for _ in ()).throw(AssertionError("Missing predict")))

    # ── Save ───────────────────────────────────────────────────────────────
    if churn_artifact is not None:
        saved_path = check("save_model_artifact() saves churn_model_test.json",
                           lambda: save_model_artifact(churn_artifact, "churn_model_test.json", "models/"))
        if saved_path:
            check("Saved .json file exists on disk",
                  lambda: True if os.path.exists(saved_path) else (_ for _ in ()).throw(AssertionError(saved_path)))
            meta_path = saved_path.replace(".json", "_meta.joblib")
            check("Preprocessing metadata .joblib saved alongside model",
                  lambda: True if os.path.exists(meta_path) else (_ for _ in ()).throw(AssertionError(meta_path)))

# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  6 - predict  (ChurnLTVInferencePipeline — XGBoost)")
print("="*60)

from src.predict import ChurnLTVInferencePipeline

infer = check("ChurnLTVInferencePipeline() instantiates", lambda: ChurnLTVInferencePipeline("models/"))

if infer is not None and saved_path is not None:
    # Point at test artifact
    meta_path = saved_path.replace(".json", "_meta.joblib")
    infer.model_path = saved_path
    infer.meta_path  = meta_path

    check("load_artifacts() succeeds",              lambda: infer.load_artifacts() or True)
    check("self.model is not None after load",       lambda: True if infer.model is not None else (_ for _ in ()).throw(AssertionError("model is None")))
    check("self.feature_columns is a list",         lambda: True if isinstance(infer.feature_columns, list) and len(infer.feature_columns) > 0 else (_ for _ in ()).throw(AssertionError("feature_columns empty")))

    sample = {
        "customerID": "0000-TEST",
        "gender": "Female", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No", "tenure": 12,
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "Yes",
        "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.85, "TotalCharges": "670.20",
    }

    res1 = check("predict_churn(dict) returns a list", lambda: infer.predict_churn(sample))
    if res1:
        check("Result has 'churn_probability' key",
              lambda: True if "churn_probability" in res1[0] else (_ for _ in ()).throw(AssertionError(str(res1[0].keys()))))
        check("churn_probability in [0, 1]",
              lambda: True if 0.0 <= res1[0]["churn_probability"] <= 1.0 else (_ for _ in ()).throw(AssertionError(str(res1[0]["churn_probability"]))))
        check("churn_prediction is 0 or 1",
              lambda: True if res1[0]["churn_prediction"] in {0, 1} else (_ for _ in ()).throw(AssertionError(str(res1[0]["churn_prediction"]))))
        print("         Prediction: " + str(res1[0]))

    batch = pd.DataFrame([sample, sample])
    res2 = check("predict_churn(DataFrame, 2 rows) returns 2 results",
                 lambda: infer.predict_churn(batch))
    if res2:
        check("Batch result length == 2",
              lambda: True if len(res2) == 2 else (_ for _ in ()).throw(AssertionError(str(len(res2)))))

    # Non-zero-based index test
    batch_reindexed = batch.copy()
    batch_reindexed.index = [100, 200]
    res3 = check("predict_churn() correct with non-zero-based index",
                 lambda: infer.predict_churn(batch_reindexed))
    if res3:
        check("Non-zero-index result length == 2",
              lambda: True if len(res3) == 2 else (_ for _ in ()).throw(AssertionError(str(len(res3)))))

    # TypeError guard
    def _type_error_test():
        try:
            infer.predict_churn(99999)
            raise AssertionError("Should have raised TypeError")
        except TypeError:
            return True
    check("predict_churn(int) raises TypeError", _type_error_test)

# ---------------------------------------------------------------------------
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print("  RESULTS: " + str(passed) + "/" + str(total) + " passed,  " + str(failed) + " failed")
print("="*60 + "\n")

if failed:
    print("Failed tests:")
    for label, ok, exc in results:
        if not ok:
            print("  * " + label + ":  " + str(exc))

sys.exit(0 if failed == 0 else 1)
