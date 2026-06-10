# src/predict.py
import os
import joblib
import pandas as pd
from src.utils import get_logger, load_config
from src.preprocessing import clean_data
from src.feature_engineering import create_features

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)

class ChurnLTVInferencePipeline:
    def __init__(self, model_dir="models/"):
        """
        Loads pre-trained model and preprocessing pipeline artifacts.
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "churn_model.joblib")
        self.preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
        
        self.model = None
        self.preprocessor = None
        
        # Check if one level up (if running from subdirectories)
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join("..", self.model_path)
            self.preprocessor_path = os.path.join("..", self.preprocessor_path)

    def load_artifacts(self):
        """
        Loads joblib serialized preprocessor pipeline and model files.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            logger.info(f"Loading inference model artifact from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"Loading preprocessor artifact from: {self.preprocessor_path}")
            self.preprocessor = joblib.load(self.preprocessor_path)
        else:
            logger.warning(
                f"Artifacts not found at '{self.model_path}'. "
                "Inference will rely on dummy outputs until model training is executed."
            )

    def predict_churn(self, raw_data):
        """
        Transforms raw data and makes predictions.
        
        Parameters:
            raw_data (pd.DataFrame or dict): Input subscriber profile data.
            
        Returns:
            dict: Churn predictions containing probabilities and classification labels.
        """
        if isinstance(raw_data, dict):
            df = pd.DataFrame([raw_data])
        else:
            df = raw_data.copy()
            
        logger.info(f"Received inference request for {len(df)} records.")
        
        # 1. Clean and engineer features
        df_cleaned = clean_data(df)
        df_features = create_features(df_cleaned)
        
        # 2. Check if artifacts are loaded
        if self.model is None or self.preprocessor is None:
            logger.info("Serving mock inference scores (no trained artifacts loaded)...")
            # Mock scoring
            probabilities = [0.15] * len(df)
            predictions = [0] * len(df)
        else:
            # Drop unnecessary columns that are not features
            X = df_features.drop(columns=['customerID', 'Churn'], errors='ignore')
            # Transform
            X_trans = self.preprocessor.transform(X)
            # Predict
            probabilities = self.model.predict_proba(X_trans)[:, 1].tolist()
            predictions = self.model.predict(X_trans).tolist()
            
        results = []
        for idx, row in df.iterrows():
            cust_id = row.get('customerID', f"TEMP_{idx}")
            results.append({
                "customerID": cust_id,
                "churn_probability": probabilities[idx],
                "churn_prediction": int(predictions[idx])
            })
            
        logger.info("Inference calculations completed successfully.")
        return results

if __name__ == "__main__":
    # Test script execution
    pipeline = ChurnLTVInferencePipeline()
    pipeline.load_artifacts()
    
    # Sample customer payload
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
        "TotalCharges": "670.20"
    }
    
    predictions = pipeline.predict_churn(sample_payload)
    print("Sample Inference Prediction Result:")
    print(predictions)
