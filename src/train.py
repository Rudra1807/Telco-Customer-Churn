# src/train.py
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.utils import get_logger, load_config, PROJECT_ROOT
from src.preprocessing import get_preprocessor_pipeline

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)


# ── Model factory ──────────────────────────────────────────────────────────────

def get_churn_model(model_type: str, model_params: dict):
    """
    Instantiates a churn classification model based on the config selection.

    class_weight='balanced' is applied to both classifiers to compensate for
    the ~26 % churn-class imbalance in the Telco dataset.

    Parameters:
        model_type (str): One of 'random_forest' or 'logistic_regression'.
        model_params (dict): Hyperparameter dict sourced from config.yaml.

    Returns:
        sklearn estimator: Unfitted classifier.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 6),
            random_state=model_params.get("random_state", 42),
            class_weight="balanced",    # handles class imbalance
            n_jobs=-1,                  # use all CPU cores
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=model_params.get("random_state", 42),
        )
    else:
        logger.warning(
            f"Unknown model type '{model_type}'. Defaulting to LogisticRegression."
        )
        return LogisticRegression(max_iter=1000, class_weight="balanced")


def get_ltv_model(model_params: dict):
    """
    Instantiates a Ridge regression model for LTV (TotalCharges) prediction.

    Parameters:
        model_params (dict): Hyperparameter dict from config['model']['ltv'].

    Returns:
        Ridge: Unfitted regressor.
    """
    return Ridge(
        alpha=model_params.get("alpha", 1.0),
        random_state=model_params.get("random_state", 42),
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_churn_pipeline(
    X_train,
    y_train,
    categorical_cols: list = None,
    numerical_cols: list = None,
    model_type: str = None,
) -> Pipeline:
    """
    Builds, fits, and returns a full sklearn Pipeline for churn classification.

    The pipeline contains:
        1. A ColumnTransformer preprocessor (imputation + scaling + OHE).
        2. A churn classifier (RandomForest or LogisticRegression).

    Parameters:
        X_train: Training features (pd.DataFrame).
        y_train: Binary churn labels (pd.Series, dtype int).
        categorical_cols (list, optional): Overrides config categorical features.
        numerical_cols (list, optional): Overrides config numerical features.
        model_type (str, optional): Overrides config model type.

    Returns:
        Pipeline: Fitted sklearn pipeline ready for prediction.
    """
    logger.info("Initialising churn training pipeline...")

    churn_cfg = config["model"]["churn"]
    if model_type is None:
        model_type = churn_cfg.get("model_type", "random_forest")

    model_params = {
        "n_estimators": churn_cfg.get("n_estimators", 100),
        "max_depth": churn_cfg.get("max_depth", 6),
        "random_state": churn_cfg.get("random_state", 42),
    }

    preprocessor = get_preprocessor_pipeline(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
    classifier = get_churn_model(model_type, model_params)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    logger.info(
        f"Fitting churn pipeline — model: {model_type}, "
        f"samples: {len(X_train):,}."
    )
    pipeline.fit(X_train, y_train)
    logger.info("Churn model training completed successfully.")
    return pipeline


def train_ltv_pipeline(
    X_train,
    y_train,
    categorical_cols: list = None,
    numerical_cols: list = None,
) -> Pipeline:
    """
    Builds, fits, and returns a full sklearn Pipeline for LTV regression.

    The LTV target is TotalCharges, which must be excluded from X_train
    before calling this function.

    Parameters:
        X_train: Training features (pd.DataFrame, TotalCharges excluded).
        y_train: Continuous LTV target (pd.Series).
        categorical_cols (list, optional): Overrides config categorical features.
        numerical_cols (list, optional): Overrides config numerical features.

    Returns:
        Pipeline: Fitted sklearn pipeline ready for prediction.
    """
    logger.info("Initialising LTV training pipeline...")

    ltv_cfg = config["model"]["ltv"]
    model_params = {
        "alpha": ltv_cfg.get("alpha", 1.0),
        "random_state": ltv_cfg.get("random_state", 42),
    }

    preprocessor = get_preprocessor_pipeline(
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )
    regressor = get_ltv_model(model_params)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    logger.info(f"Fitting LTV pipeline — samples: {len(X_train):,}.")
    pipeline.fit(X_train, y_train)
    logger.info("LTV model training completed successfully.")
    return pipeline


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_churn_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """
    Evaluates a fitted churn pipeline and returns a metrics dictionary.

    Metrics computed:
        - accuracy
        - roc_auc
        - classification_report (string)
        - confusion_matrix (list of lists)

    Parameters:
        pipeline (Pipeline): Fitted churn sklearn pipeline.
        X_test: Test features.
        y_test: True binary labels.

    Returns:
        dict: Evaluation metrics.
    """
    logger.info("Evaluating churn model performance...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(
        f"Churn model — Accuracy: {metrics['accuracy']}, ROC-AUC: {metrics['roc_auc']}"
    )
    logger.info(f"\n{metrics['classification_report']}")
    return metrics


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model_artifact(
    pipeline: Pipeline,
    artifact_name: str = "churn_model.joblib",
    output_dir: str = "models/",
) -> str:
    """
    Serialises a fitted pipeline to disk using joblib.

    Parameters:
        pipeline (Pipeline): Fitted sklearn pipeline to save.
        artifact_name (str): Output filename (e.g. 'churn_model.joblib').
        output_dir (str): Directory path (relative to project root or absolute).

    Returns:
        str: Absolute path of the saved artifact.
    """
    # Resolve output directory relative to project root
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)

    os.makedirs(output_dir, exist_ok=True)   # exist_ok removes the need for a prior exists() check

    artifact_path = os.path.join(output_dir, artifact_name)
    logger.info(f"Saving pipeline artifact to: {artifact_path}")
    joblib.dump(pipeline, artifact_path)
    logger.info("Artifact saved successfully.")
    return artifact_path


if __name__ == "__main__":
    logger.info("train.py loaded successfully.")
