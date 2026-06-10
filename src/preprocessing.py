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

def clean_data(df):
    """
    Cleans raw customer data, specifically casting numeric values and mapping churn.
    
    Parameters:
        df (pd.DataFrame): Raw input DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df_clean = df.copy()
    
    # 1. Cast TotalCharges to numerical. The IBM dataset contains white space characters " " for new customers (tenure = 0).
    if 'TotalCharges' in df_clean.columns:
        logger.info("Cleaning and casting 'TotalCharges' to numeric float...")
        # Replace empty/spaces with NaN
        df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Impute new customer TotalCharges (tenure == 0) with 0.0 or MonthlyCharges
        zero_tenure_mask = df_clean['tenure'] == 0
        if zero_tenure_mask.any():
            logger.info(f"Imputing {zero_tenure_mask.sum()} missing TotalCharges for customers with 0 tenure to 0.0.")
            df_clean.loc[zero_tenure_mask, 'TotalCharges'] = 0.0
            
        # For remaining NaNs (if any), impute with median
        null_count = df_clean['TotalCharges'].isnull().sum()
        if null_count > 0:
            median_val = df_clean['TotalCharges'].median()
            logger.info(f"Imputing {null_count} remaining missing TotalCharges with median: {median_val}")
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val)
            
    # 2. Standardize target column (Yes/No to 1/0)
    target_churn = config['data']['target_col_churn']
    if target_churn in df_clean.columns:
        logger.info(f"Mapping target column '{target_churn}' to binary (1/0)...")
        df_clean[target_churn] = df_clean[target_churn].map({'Yes': 1, 'No': 0})
        
    logger.info("Data cleansing completed.")
    return df_clean

def get_preprocessor_pipeline(categorical_cols=None, numerical_cols=None):
    """
    Creates an Sklearn preprocessor using ColumnTransformer.
    
    Parameters:
        categorical_cols (list, optional): List of categorical features.
        numerical_cols (list, optional): List of numerical features.
        
    Returns:
        ColumnTransformer: Pre-configured preprocessing pipeline.
    """
    if categorical_cols is None:
        categorical_cols = config['features']['categorical']
    if numerical_cols is None:
        numerical_cols = config['features']['numerical']
        
    logger.info("Creating preprocessing pipelines for numerical and categorical features...")
    
    # Pipeline for numerical features
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

def split_dataset(df, target_col=None, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and test splits.
    
    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame.
        target_col (str, optional): Target column name. If None, loaded from config.
        test_size (float): Portion of dataset to assign to test set.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_col is None:
        target_col = config['data']['target_col_churn']
        
    X = df.drop(columns=[target_col, 'customerID'], errors='ignore')
    y = df[target_col]
    
    logger.info(f"Splitting dataset with test size = {test_size} and random_state = {random_state}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if y.dtype == int else None)
