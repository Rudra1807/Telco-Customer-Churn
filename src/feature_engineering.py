# src/feature_engineering.py
import pandas as pd
import numpy as np
from src.utils import get_logger, load_config

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)

# ── Constants ──────────────────────────────────────────────────────────────────
_TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
_TENURE_LABELS = ["0-1 Year", "1-2 Years", "2-3 Years", "3-4 Years", "4-5 Years", "5-6 Years"]

_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins 'tenure' (months) into discrete labelled cohort groups.

    Values outside the defined range (> 72) are clamped to the last bin
    rather than becoming NaN, preventing unknown categories in the OHE step.

    New column added:
        tenure_group (str): e.g. '0-1 Year', '1-2 Years', ...

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'tenure'.

    Returns:
        pd.DataFrame: Copy with 'tenure_group' appended.
    """
    df_feat = df.copy()

    if "tenure" not in df_feat.columns:
        logger.warning("'tenure' column not found — skipping tenure cohort feature.")
        return df_feat

    logger.info("Adding feature: 'tenure_group' cohort bins.")

    # Clamp out-of-range tenures so pd.cut never produces NaN
    clamped = df_feat["tenure"].clip(lower=_TENURE_BINS[0], upper=_TENURE_BINS[-1])

    tenure_group = pd.cut(
        clamped,
        bins=_TENURE_BINS,
        labels=_TENURE_LABELS,
        include_lowest=True,
        right=True,
    )

    # Cast to str; any residual NaN (shouldn't occur after clamp) → fallback label
    df_feat["tenure_group"] = tenure_group.astype(str).replace("nan", _TENURE_LABELS[0])

    return df_feat


def calculate_service_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts the number of optional services a customer has subscribed to.

    Checks for each of: OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies.

    New column added:
        total_services (int): Count of services with value 'Yes' (0–6).

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Copy with 'total_services' appended.
    """
    df_feat = df.copy()

    available = [col for col in _SERVICE_COLS if col in df_feat.columns]

    if not available:
        logger.warning("No service columns found — skipping service density feature.")
        return df_feat

    logger.info(f"Adding feature: 'total_services' (checking {len(available)} service columns).")
    df_feat["total_services"] = (df_feat[available] == "Yes").sum(axis=1)

    return df_feat


def add_billing_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives billing ratio features that capture spending patterns.

    New columns added:
        charge_ratio (float):
            MonthlyCharges / TotalCharges.  Zero when TotalCharges == 0
            (new customers) to avoid division-by-zero.
        charges_difference (float):
            TotalCharges − (tenure × MonthlyCharges).  Positive values
            indicate discounts or promotional credits applied to the account.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Copy with billing ratio columns appended.
    """
    df_feat = df.copy()

    has_monthly = "MonthlyCharges" in df_feat.columns
    has_total = "TotalCharges" in df_feat.columns
    has_tenure = "tenure" in df_feat.columns

    if not (has_monthly and has_total):
        logger.warning(
            "Missing MonthlyCharges or TotalCharges — skipping billing ratio features."
        )
        return df_feat

    logger.info("Adding feature: 'charge_ratio' (MonthlyCharges / TotalCharges).")
    df_feat["charge_ratio"] = np.where(
        df_feat["TotalCharges"] > 0,
        df_feat["MonthlyCharges"] / df_feat["TotalCharges"],
        0.0,
    )

    if has_tenure:
        logger.info("Adding feature: 'charges_difference' (TotalCharges − expected).")
        expected_total = df_feat["tenure"] * df_feat["MonthlyCharges"]
        df_feat["charges_difference"] = df_feat["TotalCharges"] - expected_total

    return df_feat


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the full feature engineering pipeline.

    Applies, in order:
        1. Tenure cohort binning
        2. Service subscription density
        3. Billing ratio derivations

    Parameters:
        df (pd.DataFrame): Cleaned input DataFrame (output of clean_data()).

    Returns:
        pd.DataFrame: Feature-enriched DataFrame.
    """
    logger.info("Starting feature engineering pipeline...")

    df_feat = df.copy()
    df_feat = add_tenure_cohorts(df_feat)
    df_feat = calculate_service_density(df_feat)
    df_feat = add_billing_ratio_features(df_feat)

    logger.info(
        f"Feature engineering complete. "
        f"Shape: {df_feat.shape[0]} rows × {df_feat.shape[1]} columns."
    )
    return df_feat
