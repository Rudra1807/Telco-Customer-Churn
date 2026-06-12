# src/predict.py
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from src.utils import get_logger, load_config, PROJECT_ROOT
from src.preprocessing import clean_data
from src.feature_engineering import create_features

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)


class ChurnLTVInferencePipeline:
    """
    Inference pipeline for churn probability scoring and LTV prediction.

    Loads the unified sklearn Pipeline artifact produced by train.train_churn_pipeline()
    (a single .joblib file containing both the ColumnTransformer preprocessor and the
    classifier).  Exposes a clean predict_churn() interface that accepts either a
    single customer dict or a batch pd.DataFrame.

    Usage:
        pipeline = ChurnLTVInferencePipeline()
        pipeline.load_artifacts()
        results = pipeline.predict_churn({"customerID": "X", "tenure": 12, ...})
    """

    def __init__(self, model_dir: str = "models/"):
        """
        Parameters:
            model_dir (str): Directory containing 'churn_model.joblib'.
                Relative paths are resolved against the project root.
        """
        # Always resolve to an absolute path so the class works from any CWD
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(PROJECT_ROOT, model_dir)

        self.model_dir = model_dir
        # Single unified Pipeline artifact (preprocessor + classifier combined)
        self.model_path = os.path.join(model_dir, "churn_model.joblib")

        self.pipeline: Pipeline | None = None

    def load_artifacts(self) -> None:
        """
        Loads the unified churn Pipeline artifact from disk.

        The artifact is a single sklearn Pipeline (produced by
        train.train_churn_pipeline) containing both the ColumnTransformer
        preprocessor and the classifier.  If the file is not found the
        pipeline continues in mock-scoring mode.
        """
        if os.path.exists(self.model_path):
            logger.info(f"Loading unified churn pipeline from: {self.model_path}")
            self.pipeline = joblib.load(self.model_path)
            logger.info("Churn pipeline artifact loaded successfully.")
        else:
            logger.warning(
                f"Artifact not found at '{self.model_path}'. "
                "Inference will return mock scores until the model is trained."
            )

    def predict_churn(self, raw_data) -> list[dict]:
        """
        Scores one or more customer records for churn probability.

        Applies the same clean_data → create_features → transform → predict
        chain used during training, guaranteeing train/serve consistency.

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
        df_cleaned = clean_data(df)
        df_features = create_features(df_cleaned)

        # ── Scoring ───────────────────────────────────────────────────────────
        if self.pipeline is None:
            logger.info("Mock scoring mode — no trained pipeline artifact loaded.")
            probabilities = [0.15] * len(df)
            predictions = [0] * len(df)
        else:
            # The unified Pipeline handles both transform and predict internally.
            # No separate .transform() call is needed or correct here.
            X = df_features.drop(columns=["customerID", "Churn"], errors="ignore")
            probabilities = self.pipeline.predict_proba(X)[:, 1].tolist()
            predictions = self.pipeline.predict(X).tolist()

        # ── Build result list ─────────────────────────────────────────────────
        # FIX: use enumerate() for the list index — df.iterrows() yields the
        # DataFrame's .index values (could be non-zero-based after slicing),
        # which must NOT be used to index into the plain Python lists above.
        results = []
        for list_pos, (df_idx, row) in enumerate(df.iterrows()):
            cust_id = row.get("customerID", f"TEMP_{df_idx}")
            results.append({
                "customerID": cust_id,
                "churn_probability": round(probabilities[list_pos], 4),
                "churn_prediction": int(predictions[list_pos]),
            })

        logger.info(f"Inference complete — {len(results)} result(s) returned.")
        return results


if __name__ == "__main__":
    pipeline = ChurnLTVInferencePipeline()
    pipeline.load_artifacts()

    sample_payload = {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.85,
        "TotalCharges": "670.20",
    }

    results = pipeline.predict_churn(sample_payload)
    print("Sample Inference Result:")
    for r in results:
        print(r)
