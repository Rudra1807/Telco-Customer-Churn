# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import get_logger, load_config

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)


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
    # The IBM Telco dataset ships TotalCharges as a string; new customers
    # (tenure == 0) have a blank string instead of 0.
    if "TotalCharges" in df_clean.columns:
        logger.info("Cleaning and casting 'TotalCharges' to numeric float...")

        # Replace whitespace-only strings with NaN, then cast
        df_clean["TotalCharges"] = (
            df_clean["TotalCharges"]
            .replace(r"^\s*$", np.nan, regex=True)
        )
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

        # New customers (tenure == 0) legitimately have 0 total charges
        zero_tenure_mask = df_clean["tenure"] == 0
        if zero_tenure_mask.any():
            logger.info(
                f"Imputing {zero_tenure_mask.sum()} TotalCharges entries "
                "for tenure-0 customers to 0.0."
            )
            df_clean.loc[zero_tenure_mask, "TotalCharges"] = 0.0

        # Any remaining NaNs are imputed with the median
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
        # Guard: only map if the column is NOT already integer-typed.
        # We check with is_integer_dtype because df_clean[target_churn].dtype == object
        # is False for PyArrow-backed string columns (ArrowDtype), causing the
        # mapping to be silently skipped on newer pandas/pyarrow installs.
        if not pd.api.types.is_integer_dtype(df_clean[target_churn]):
            logger.info(f"Mapping target column '{target_churn}' -> binary integer (1/0).")
            df_clean[target_churn] = df_clean[target_churn].map({"Yes": 1, "No": 0})


    logger.info("Data cleansing completed.")
    return df_clean


def get_preprocessor_pipeline(
    categorical_cols: list = None,
    numerical_cols: list = None,
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer preprocessing pipeline.

    Numerical features → median imputation → StandardScaler.
    Categorical features → mode imputation → OneHotEncoder (handle_unknown='ignore').

    Parameters:
        categorical_cols (list, optional): Categorical feature names.
            Defaults to config['features']['categorical'].
        numerical_cols (list, optional): Numerical feature names.
            Defaults to config['features']['numerical'].

    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline.
    """
    if categorical_cols is None:
        categorical_cols = config["features"]["categorical"]
    if numerical_cols is None:
        numerical_cols = config["features"]["numerical"]

    logger.info(
        f"Building preprocessing pipeline — "
        f"{len(numerical_cols)} numerical, {len(categorical_cols)} categorical features."
    )

    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def split_dataset(
    df: pd.DataFrame,
    target_col: str = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Splits the cleaned DataFrame into stratified train / test subsets.

    Uses stratification when the target column is integer-typed (i.e. after
    the Yes/No → 1/0 mapping in clean_data), ensuring class proportions are
    preserved in both splits.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame.
        target_col (str, optional): Target column name.
            Defaults to config['data']['target_col_churn'].
        test_size (float): Fraction of samples to reserve for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) as DataFrames / Series.
    """
    if target_col is None:
        target_col = config["data"]["target_col_churn"]

    X = df.drop(columns=["customerID", target_col], errors="ignore")
    y = df[target_col]

    # FIX: use pd.api.types.is_integer_dtype — y.dtype == int is always False
    # for numpy int64 (the result of .map({'Yes': 1, 'No': 0})).
    should_stratify = pd.api.types.is_integer_dtype(y)
    logger.info(
        f"Splitting dataset — test_size={test_size}, "
        f"random_state={random_state}, stratify={should_stratify}."
    )

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if should_stratify else None,
    )
