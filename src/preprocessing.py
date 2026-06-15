# src/preprocessing.py
import pandas as pd
import numpy as np

from src.utils import get_logger, load_config

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)

# ---------------------------------------------------------------------------
# Feature column definitions (read from config)
# ---------------------------------------------------------------------------
_CAT_COLS = config["features"]["categorical"]
_NUM_COLS = config["features"]["numerical"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data: fixes the TotalCharges dtype, imputes missing
    values, and maps the Churn target column to a binary integer (1/0).

    Parameters:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame (original is not mutated).
    """
    df_clean = df.copy()

    # ── TotalCharges ─────────────────────────────────────────────────────────
    if "TotalCharges" in df_clean.columns:
        logger.info("Cleaning and casting 'TotalCharges' to numeric float...")

        df_clean["TotalCharges"] = (
            df_clean["TotalCharges"]
            .replace(r"^\s*$", np.nan, regex=True)
        )
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

        zero_tenure_mask = df_clean["tenure"] == 0
        if zero_tenure_mask.any():
            logger.info(
                f"Imputing {zero_tenure_mask.sum()} TotalCharges entries "
                "for tenure-0 customers to 0.0."
            )
            df_clean.loc[zero_tenure_mask, "TotalCharges"] = 0.0

        null_count = df_clean["TotalCharges"].isnull().sum()
        if null_count > 0:
            median_val = df_clean["TotalCharges"].median()
            logger.info(
                f"Imputing {null_count} remaining NaN TotalCharges with median: {median_val:.2f}"
            )
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_val)

    # ── Churn target ──────────────────────────────────────────────────────────
    target_churn = config["data"]["target_col_churn"]
    if target_churn in df_clean.columns:
        if not pd.api.types.is_integer_dtype(df_clean[target_churn]):
            logger.info(f"Mapping target column '{target_churn}' -> binary integer (1/0).")
            df_clean[target_churn] = df_clean[target_churn].map({"Yes": 1, "No": 0})

    logger.info("Data cleansing completed.")
    return df_clean


def preprocess_dataframe(
    df: pd.DataFrame,
    categorical_cols: list = None,
    numerical_cols: list = None,
    reference_columns: list = None,
    fit_stats: dict = None,
) -> tuple:
    """
    Applies full preprocessing to a DataFrame using pure pandas/numpy.
    Replaces the sklearn ColumnTransformer + Pipeline pattern.

    Steps:
        1. One-hot encode categorical columns (pd.get_dummies).
        2. Standardise numerical columns using mean/std from training data.
        3. Align columns to a reference (for inference-time consistency).

    Parameters:
        df (pd.DataFrame): Input feature DataFrame (no target column).
        categorical_cols (list, optional): Categorical column names.
        numerical_cols (list, optional): Numerical column names.
        reference_columns (list, optional): Exact column order to align to
            (used at inference to match training feature matrix).
        fit_stats (dict, optional): {'col': {'mean': ..., 'std': ...}}
            Pre-computed scaling stats from training. If None, stats are
            computed from `df` (training mode).

    Returns:
        tuple: (df_encoded, fit_stats, feature_columns)
            - df_encoded (pd.DataFrame): Fully preprocessed feature matrix.
            - fit_stats (dict): Scaling stats (save these for inference).
            - feature_columns (list): Column names of df_encoded.
    """
    if categorical_cols is None:
        categorical_cols = _CAT_COLS
    if numerical_cols is None:
        numerical_cols = _NUM_COLS

    cat_present = [c for c in categorical_cols if c in df.columns]
    num_present = [c for c in numerical_cols if c in df.columns]

    logger.info(
        f"Preprocessing {len(num_present)} numerical + {len(cat_present)} categorical features."
    )

    # ── One-hot encoding (explicit categorical list) ──────────────────────────
    df_enc = pd.get_dummies(df, columns=cat_present, drop_first=False)

    # ── Auto-encode any remaining non-numeric columns (e.g. tenure_group
    #    added by feature_engineering that wasn't in the explicit cat list) ──────
    extra_obj_cols = df_enc.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if extra_obj_cols:
        logger.info(f"Auto-encoding extra non-numeric columns: {extra_obj_cols}")
        df_enc = pd.get_dummies(df_enc, columns=extra_obj_cols, drop_first=False)

    logger.info(f"After OHE: {df_enc.shape[1]} columns.")

    # ── Standardise numericals ────────────────────────────────────────────────
    if fit_stats is None:
        fit_stats = {}
        for col in num_present:
            if col in df_enc.columns:
                mean = df_enc[col].mean()
                std  = df_enc[col].std()
                fit_stats[col] = {"mean": float(mean), "std": float(std)}

    for col, stats in fit_stats.items():
        if col in df_enc.columns:
            std = stats["std"] if stats["std"] > 0 else 1.0
            df_enc[col] = (df_enc[col] - stats["mean"]) / std

    # ── Align to reference columns (inference mode) ───────────────────────────
    if reference_columns is not None:
        df_enc = df_enc.reindex(columns=reference_columns, fill_value=0)

    feature_columns = list(df_enc.columns)
    logger.info(f"Preprocessing complete. Output shape: {df_enc.shape}")
    return df_enc, fit_stats, feature_columns


# ---------------------------------------------------------------------------
# Backward-compatible alias so old callers don't break
# ---------------------------------------------------------------------------
def get_preprocessor_pipeline(categorical_cols=None, numerical_cols=None):
    """
    Deprecated shim — returns a callable that wraps preprocess_dataframe().
    Retained for backward compatibility with any code that calls this function.
    Use preprocess_dataframe() directly in new code.
    """
    logger.warning(
        "get_preprocessor_pipeline() is deprecated. "
        "Use preprocess_dataframe() directly."
    )

    class _PandasPreprocessor:
        def __init__(self):
            self.fit_stats = None
            self.feature_columns = None

        def fit_transform(self, df):
            enc, self.fit_stats, self.feature_columns = preprocess_dataframe(
                df, categorical_cols, numerical_cols
            )
            return enc

        def transform(self, df):
            enc, _, _ = preprocess_dataframe(
                df, categorical_cols, numerical_cols,
                reference_columns=self.feature_columns,
                fit_stats=self.fit_stats,
            )
            return enc

    return _PandasPreprocessor()


def split_dataset(
    df: pd.DataFrame,
    target_col: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Splits the cleaned DataFrame into train / test subsets using pandas .sample().
    Replaces sklearn train_test_split.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.
        target_col (str, optional): Target column name.
        test_size (float): Fraction to reserve for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_col is None:
        target_col = config["data"]["target_col_churn"]

    X = df.drop(columns=["customerID", target_col], errors="ignore")
    y = df[target_col]

    logger.info(
        f"Splitting dataset — test_size={test_size}, random_state={random_state}."
    )

    df_temp = X.copy()
    df_temp["_y"] = y.values
    df_train = df_temp.sample(frac=1 - test_size, random_state=random_state)
    df_test  = df_temp.drop(df_train.index)

    X_train = df_train.drop(columns=["_y"])
    y_train = df_train["_y"].astype(int)
    X_test  = df_test.drop(columns=["_y"])
    y_test  = df_test["_y"].astype(int)

    logger.info(
        f"Split complete — train: {len(X_train):,}, test: {len(X_test):,}"
    )
    return X_train, X_test, y_train, y_test
