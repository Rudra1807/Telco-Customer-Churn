# src/train.py
import os
import joblib
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from src.utils import get_logger, load_config, PROJECT_ROOT
from src.preprocessing import preprocess_dataframe

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)


# ── Model factory ──────────────────────────────────────────────────────────────

def get_churn_model(model_params: dict) -> XGBClassifier:
    """
    Instantiates an XGBoost churn classifier with the given parameters.

    Parameters:
        model_params (dict): Hyperparameter dict sourced from config.yaml.

    Returns:
        XGBClassifier: Unfitted XGBoost classifier.
    """
    return XGBClassifier(
        n_estimators     = model_params.get("n_estimators", 100),
        max_depth        = model_params.get("max_depth", 6),
        learning_rate    = model_params.get("learning_rate", 0.1),
        subsample        = model_params.get("subsample", 0.8),
        colsample_bytree = model_params.get("colsample_bytree", 0.8),
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = model_params.get("random_state", 42),
        n_jobs           = -1,
    )


def get_ltv_model(model_params: dict) -> XGBRegressor:
    """
    Instantiates an XGBoost regressor for LTV (TotalCharges) prediction.

    Parameters:
        model_params (dict): Hyperparameter dict from config['model']['ltv'].

    Returns:
        XGBRegressor: Unfitted XGBoost regressor.
    """
    return XGBRegressor(
        n_estimators     = model_params.get("n_estimators", 200),
        max_depth        = model_params.get("max_depth", 5),
        learning_rate    = model_params.get("learning_rate", 0.05),
        subsample        = model_params.get("subsample", 0.8),
        colsample_bytree = model_params.get("colsample_bytree", 0.8),
        objective        = "reg:squarederror",
        random_state     = model_params.get("random_state", 42),
        n_jobs           = -1,
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train_churn_pipeline(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    categorical_cols: list = None,
    numerical_cols: list = None,
) -> dict:
    """
    Preprocesses features and fits an XGBoost churn classifier.

    Returns a dict containing the fitted model, preprocessing stats, and
    the ordered feature column list — everything needed for consistent
    inference later.

    Parameters:
        X_train: Training features (pd.DataFrame, raw — before encoding).
        y_train: Binary churn labels (pd.Series, dtype int).
        X_val:   Optional validation features for early stopping.
        y_val:   Optional validation labels.
        categorical_cols (list, optional): Overrides config categorical features.
        numerical_cols   (list, optional): Overrides config numerical features.

    Returns:
        dict: {
            'model': XGBClassifier (fitted),
            'fit_stats': dict of scaling stats,
            'feature_columns': list of column names in training order,
        }
    """
    logger.info("Initialising churn training pipeline...")

    churn_cfg = config["model"]["churn"]
    model_params = {
        "n_estimators": churn_cfg.get("n_estimators", 100),
        "max_depth":    churn_cfg.get("max_depth", 6),
        "random_state": churn_cfg.get("random_state", 42),
    }

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_enc, fit_stats, feature_columns = preprocess_dataframe(
        X_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )

    eval_set = None
    if X_val is not None and y_val is not None:
        X_val_enc, _, _ = preprocess_dataframe(
            X_val,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            reference_columns=feature_columns,
            fit_stats=fit_stats,
        )
        eval_set = [(X_val_enc, y_val)]

    # ── Fit ───────────────────────────────────────────────────────────────────
    model = get_churn_model(model_params)
    logger.info(
        f"Fitting XGBoost churn classifier — "
        f"n_estimators: {model_params['n_estimators']}, "
        f"samples: {len(X_enc):,}."
    )
    model.fit(X_enc, y_train, eval_set=eval_set, verbose=False)
    logger.info("Churn model training completed successfully.")

    return {
        "model": model,
        "fit_stats": fit_stats,
        "feature_columns": feature_columns,
    }


def train_ltv_pipeline(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    categorical_cols: list = None,
    numerical_cols: list = None,
) -> dict:
    """
    Preprocesses features and fits an XGBoost LTV regressor.

    The LTV target is TotalCharges — it must be excluded from X_train
    before calling this function.

    Parameters:
        X_train: Training features (pd.DataFrame, TotalCharges excluded).
        y_train: Continuous LTV target (pd.Series).
        X_val:   Optional validation features.
        y_val:   Optional validation target.
        categorical_cols (list, optional): Overrides config categorical features.
        numerical_cols   (list, optional): Overrides config numerical features.

    Returns:
        dict: {
            'model': XGBRegressor (fitted),
            'fit_stats': dict of scaling stats,
            'feature_columns': list of column names in training order,
        }
    """
    logger.info("Initialising LTV training pipeline...")

    ltv_cfg = config["model"]["ltv"]
    model_params = {
        "random_state": ltv_cfg.get("random_state", 42),
    }

    X_enc, fit_stats, feature_columns = preprocess_dataframe(
        X_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
    )

    eval_set = None
    if X_val is not None and y_val is not None:
        X_val_enc, _, _ = preprocess_dataframe(
            X_val,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
            reference_columns=feature_columns,
            fit_stats=fit_stats,
        )
        eval_set = [(X_val_enc, y_val)]

    model = get_ltv_model(model_params)
    logger.info(f"Fitting XGBoost LTV regressor — samples: {len(X_enc):,}.")
    model.fit(X_enc, y_train, eval_set=eval_set, verbose=False)
    logger.info("LTV model training completed successfully.")

    return {
        "model": model,
        "fit_stats": fit_stats,
        "feature_columns": feature_columns,
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_churn_model(model, X_test, y_test) -> dict:
    """
    Evaluates a fitted XGBoost churn model and returns a metrics dictionary.

    Metrics computed entirely with numpy (no sklearn dependency):
        - accuracy
        - roc_auc  (trapezoidal approximation)
        - confusion_matrix (list of lists)

    Parameters:
        model:   Fitted XGBClassifier.
        X_test:  Pre-processed test features (pd.DataFrame).
        y_test:  True binary labels (pd.Series or np.ndarray).

    Returns:
        dict: Evaluation metrics.
    """
    import numpy as np

    logger.info("Evaluating churn model performance...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values if hasattr(y_test, "values") else y_test

    # Accuracy
    accuracy = float((y_pred == y_true).mean())

    # ROC-AUC (trapezoidal)
    thresholds = sorted(set(y_prob), reverse=True)
    pos = int(y_true.sum()); neg = len(y_true) - pos
    tpr_pts, fpr_pts = [0.0], [0.0]
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr_pts.append(tp / pos if pos > 0 else 0)
        fpr_pts.append(fp / neg if neg > 0 else 0)
    tpr_pts.append(1.0); fpr_pts.append(1.0)
    roc_auc = float(np.trapezoid(tpr_pts, fpr_pts))

    # Confusion matrix
    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t)][int(p)] += 1

    metrics = {
        "accuracy": round(accuracy, 4),
        "roc_auc":  round(roc_auc, 4),
        "confusion_matrix": cm,
    }

    logger.info(
        f"Churn model — Accuracy: {metrics['accuracy']}, ROC-AUC: {metrics['roc_auc']}"
    )
    return metrics


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model_artifact(
    artifact: dict,
    artifact_name: str = "churn_model.json",
    output_dir: str = "models/",
) -> str:
    """
    Saves the XGBoost model and its preprocessing metadata to disk.

    The XGBoost model is saved in native JSON format via .save_model().
    Preprocessing stats (fit_stats + feature_columns) are saved alongside
    as a joblib file so inference can replicate the exact same transformation.

    Parameters:
        artifact (dict): Output of train_churn_pipeline() or train_ltv_pipeline().
            Must contain keys: 'model', 'fit_stats', 'feature_columns'.
        artifact_name (str): Base filename e.g. 'churn_model.json'.
        output_dir (str): Directory (relative to project root or absolute).

    Returns:
        str: Absolute path of the saved XGBoost model JSON.
    """
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, artifact_name)
    meta_name  = artifact_name.replace(".json", "_meta.joblib")
    meta_path  = os.path.join(output_dir, meta_name)

    logger.info(f"Saving XGBoost model to: {model_path}")
    artifact["model"].save_model(model_path)

    meta = {
        "fit_stats":       artifact["fit_stats"],
        "feature_columns": artifact["feature_columns"],
    }
    joblib.dump(meta, meta_path)
    logger.info(f"Preprocessing metadata saved to: {meta_path}")

    return model_path


if __name__ == "__main__":
    logger.info("train.py loaded successfully.")
