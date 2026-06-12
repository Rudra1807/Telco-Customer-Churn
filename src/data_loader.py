# src/data_loader.py
import os
import pandas as pd
from src.utils import get_logger, load_config, PROJECT_ROOT

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Loads raw CSV data from the specified path.

    Resolves relative paths against the project root so the module works
    correctly from any working directory (notebooks, src, project root, etc.).

    Parameters:
        file_path (str, optional): Custom path to the raw dataset.
            If None, the path is read from config['data']['raw_path'].

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        FileNotFoundError: If the dataset cannot be found.
    """
    if file_path is None:
        file_path = config["data"]["raw_path"]

    # Resolve to absolute path anchored at project root
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "Please download the IBM Telco Churn dataset and place it in data/raw/."
        )

    logger.info(f"Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validates that the input DataFrame matches the expected column schema
    defined in config.yaml.

    Parameters:
        df (pd.DataFrame): DataFrame to validate.

    Returns:
        bool: True if schema is valid.

    Raises:
        ValueError: If one or more expected columns are missing.
    """
    expected_categorical = config["features"]["categorical"]
    expected_numerical = config["features"]["numerical"]
    target_churn = config["data"]["target_col_churn"]

    required_cols = set(expected_categorical + expected_numerical + [target_churn])

    # customerID is in the raw dataset but excluded from the feature lists
    if "customerID" in df.columns:
        required_cols.add("customerID")

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        error_msg = f"Schema validation failed. Missing columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Schema validation passed: all expected columns are present.")
    return True


if __name__ == "__main__":
    try:
        data = load_raw_data()
        validate_schema(data)
    except Exception as exc:
        logger.error(f"Error during data loading test: {exc}")
