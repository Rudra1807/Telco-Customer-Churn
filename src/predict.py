# src/predict.py
import os
import joblib
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from src.utils import get_logger, load_config, PROJECT_ROOT
from src.preprocessing import clean_data, preprocess_dataframe
from src.feature_engineering import create_features

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)


class ChurnLTVInferencePipeline:
    """
    Inference pipeline for churn probability scoring and LTV prediction.

    Loads the XGBoost model artifact (JSON) and preprocessing metadata
    (fit_stats + feature_columns) produced by train.save_model_artifact().
    Exposes a clean predict_churn() interface that accepts either a single
    customer dict or a batch pd.DataFrame.

    Usage:
        pipeline = ChurnLTVInferencePipeline()
        pipeline.load_artifacts()
        results = pipeline.predict_churn({"customerID": "X", "tenure": 12, ...})
    """

    def __init__(self, model_dir: str = "models/"):
        """
        Parameters:
            model_dir (str): Directory containing 'churn_model.json' and
                'churn_model_meta.joblib'. Relative paths are resolved against
                the project root.
        """
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(PROJECT_ROOT, model_dir)

        self.model_dir  = model_dir
        self.model_path = os.path.join(model_dir, "churn_model.json")
        self.meta_path  = os.path.join(model_dir, "churn_model_meta.joblib")

        self.model: XGBClassifier | None = None
        self.fit_stats: dict | None      = None
        self.feature_columns: list | None = None

    def load_artifacts(self) -> None:
        """
        Loads the XGBoost churn model and its preprocessing metadata from disk.

        Falls back to mock-scoring mode if the files are not found.
        """
        if os.path.exists(self.model_path):
            logger.info(f"Loading XGBoost churn model from: {self.model_path}")
            self.model = XGBClassifier()
            self.model.load_model(self.model_path)
            logger.info("XGBoost churn model loaded successfully.")
        else:
            logger.warning(
                f"Model not found at '{self.model_path}'. "
                "Inference will return mock scores until the model is trained."
            )

        if os.path.exists(self.meta_path):
            logger.info(f"Loading preprocessing metadata from: {self.meta_path}")
            meta = joblib.load(self.meta_path)
            self.fit_stats       = meta.get("fit_stats", {})
            self.feature_columns = meta.get("feature_columns", [])
        else:
            logger.warning(
                f"Preprocessing metadata not found at '{self.meta_path}'. "
                "Predictions may be unreliable."
            )

    def predict_churn(self, raw_data) -> list[dict]:
        """
        Scores one or more customer records for churn probability.

        Applies the same clean_data -> create_features -> preprocess_dataframe
        -> predict chain used during training, guaranteeing train/serve
        consistency.

        Parameters:
            raw_data (dict | pd.DataFrame): A single customer profile dict or
                a DataFrame of customer records.

        Returns:
            list[dict]: One result dict per input record, each containing:
                - customerID (str)
                - churn_probability (float)
                - churn_prediction (int, 0 or 1)
        """
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        elif isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        else:
            raise TypeError(
                f"raw_data must be a dict or pd.DataFrame, got {type(raw_data).__name__}."
            )

        logger.info(f"Received inference request — {len(df)} record(s).")

        # ── Pre-processing ─────────────────────────────────────────────────────
        df_cleaned  = clean_data(df)
        df_features = create_features(df_cleaned)

        # ── Scoring ───────────────────────────────────────────────────────────
        if self.model is None:
            logger.info("Mock scoring mode — no trained model artifact loaded.")
            probabilities = [0.15] * len(df)
            predictions   = [0]    * len(df)
        else:
            X = df_features.drop(columns=["customerID", "Churn"], errors="ignore")
            X_enc, _, _ = preprocess_dataframe(
                X,
                reference_columns=self.feature_columns,
                fit_stats=self.fit_stats,
            )
            probabilities = self.model.predict_proba(X_enc)[:, 1].tolist()
            predictions   = self.model.predict(X_enc).tolist()

        # ── Build result list ─────────────────────────────────────────────────
        results = []
        for list_pos, (df_idx, row) in enumerate(df.iterrows()):
            cust_id = row.get("customerID", f"TEMP_{df_idx}")
            results.append({
                "customerID":        cust_id,
                "churn_probability": round(probabilities[list_pos], 4),
                "churn_prediction":  int(predictions[list_pos]),
            })

        logger.info(f"Inference complete — {len(results)} result(s) returned.")
        return results


if __name__ == "__main__":
    pipeline = ChurnLTVInferencePipeline()
    pipeline.load_artifacts()

    sample_payload = {
        "customerID":     "0000-TEST",
        "gender":         "Female",
        "SeniorCitizen":  0,
        "Partner":        "Yes",
        "Dependents":     "No",
        "tenure":         12,
        "PhoneService":   "Yes",
        "MultipleLines":  "No",
        "InternetService":"DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup":   "No",
        "DeviceProtection":"No",
        "TechSupport":    "Yes",
        "StreamingTV":    "No",
        "StreamingMovies":"No",
        "Contract":       "Month-to-month",
        "PaperlessBilling":"Yes",
        "PaymentMethod":  "Electronic check",
        "MonthlyCharges": 55.85,
        "TotalCharges":   "670.20",
    }

    results = pipeline.predict_churn(sample_payload)
    print("Sample Inference Result:")
    for r in results:
        print(r)
