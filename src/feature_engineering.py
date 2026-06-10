# src/feature_engineering.py
import pandas as pd
import numpy as np
from src.utils import get_logger, load_config

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)

def add_tenure_cohorts(df):
    """
    Groups customer tenure into discrete buckets.
    """
    df_feat = df.copy()
    if 'tenure' in df_feat.columns:
        logger.info("Adding feature: 'tenure_group' cohorts...")
        bins = [0, 12, 24, 36, 48, 60, 72]
        labels = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5-6 Years']
        
        # Tenure 0 will fall into '0-1 Year' cohort
        df_feat['tenure_group'] = pd.cut(df_feat['tenure'], bins=bins, labels=labels, include_lowest=True)
        # Cast to object/string for OHE or categorical handling
        df_feat['tenure_group'] = df_feat['tenure_group'].astype(str)
    return df_feat

def calculate_service_density(df):
    """
    Calculates number of secondary services a user subscribed to.
    """
    df_feat = df.copy()
    services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Check if all listed services exist in the DataFrame
    available_services = [s for s in services if s in df_feat.columns]
    
    if available_services:
        logger.info(f"Adding feature: 'total_services' count out of {len(available_services)}...")
        # Map 'Yes' to 1 and other values to 0
        service_counts = df_feat[available_services].apply(lambda col: col == 'Yes').sum(axis=1)
        df_feat['total_services'] = service_counts
    return df_feat

def add_billing_ratio_features(df):
    """
    Calculates charges ratios (e.g. MonthlyCharges vs. TotalCharges).
    """
    df_feat = df.copy()
    if 'MonthlyCharges' in df_feat.columns and 'TotalCharges' in df_feat.columns:
        logger.info("Adding feature: 'charge_ratio' (MonthlyCharges / TotalCharges)...")
        # To avoid division by zero or large scale issues when tenure is small
        df_feat['charge_ratio'] = np.where(
            df_feat['TotalCharges'] > 0,
            df_feat['MonthlyCharges'] / df_feat['TotalCharges'],
            0.0
        )
        
        logger.info("Adding feature: 'expected_vs_actual_charges' ratio...")
        # tenure * MonthlyCharges is what they were expected to pay
        expected_total = df_feat['tenure'] * df_feat['MonthlyCharges']
        df_feat['charges_difference'] = df_feat['TotalCharges'] - expected_total
        
    return df_feat

def create_features(df):
    """
    Runs the entire feature engineering pipeline.
    
    Parameters:
        df (pd.DataFrame): Cleaned input DataFrame.
        
    Returns:
        pd.DataFrame: Feature enriched DataFrame.
    """
    logger.info("Starting feature engineering pipeline...")
    
    df_feat = df.copy()
    df_feat = add_tenure_cohorts(df_feat)
    df_feat = calculate_service_density(df_feat)
    df_feat = add_billing_ratio_features(df_feat)
    
    # Add new categorical columns to the config settings temporarily or warn logger if config wasn't updated
    logger.info(f"Feature engineering pipeline completed. Total features: {df_feat.shape[1]}")
    return df_feat
