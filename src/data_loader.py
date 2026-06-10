# src/data_loader.py
import os
import pandas as pd
from src.utils import get_logger, load_config

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)

def load_raw_data(file_path=None):
    """
    Loads raw CSV data from the specified path.
    
    Parameters:
        file_path (str, optional): Custom path to the raw dataset. If None, loaded from config.
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if file_path is None:
        file_path = config['data']['raw_path']
        
    # Handle absolute/relative path resolving if needed
    if not os.path.exists(file_path):
        parent_path = os.path.join("..", file_path)
        if os.path.exists(parent_path):
            file_path = parent_path
        else:
            raise FileNotFoundError(
                f"Dataset not found at '{file_path}' or '{parent_path}'. "
                "Please download the IBM Telco Churn dataset and place it in the data/raw/ directory."
            )
            
    logger.info(f"Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Successfully loaded dataset with shape: {df.shape}")
    return df

def validate_schema(df):
    """
    Validates that the input DataFrame matches the expected column schema defined in config.
    
    Parameters:
        df (pd.DataFrame): Dataframe to validate.
        
    Returns:
        bool: True if schema is valid, raises ValueError otherwise.
    """
    expected_categorical = config['features']['categorical']
    expected_numerical = config['features']['numerical']
    target_churn = config['data']['target_col_churn']
    
    required_cols = set(expected_categorical + expected_numerical + [target_churn])
    actual_cols = set(df.columns)
    
    # CustomerID is typically part of the raw dataset but not in feature lists
    if 'customerID' in df.columns:
        required_cols.add('customerID')
        
    missing_cols = required_cols - actual_cols
    
    if missing_cols:
        error_msg = f"Schema validation failed! Missing expected columns: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info("Schema validation passed: All expected features and target columns are present.")
    return True

if __name__ == "__main__":
    # Test script execution
    try:
        data = load_raw_data()
        validate_schema(data)
    except Exception as e:
        logger.error(f"Error during data loading testing: {e}")
