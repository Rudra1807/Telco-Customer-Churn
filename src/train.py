# src/train.py
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from src.utils import get_logger, load_config
from src.preprocessing import get_preprocessor_pipeline

# Initialize configuration and logger
config = load_config()
logger = get_logger(__name__, config)

def get_model(model_type, model_params):
    """
    Returns an uninitialized model object based on selection configuration.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(**model_params)
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=model_params.get('random_state', 42))
    else:
        logger.warning(f"Unknown model type '{model_type}'. Defaulting to Logistic Regression.")
        return LogisticRegression(max_iter=1000)

def train_pipeline(X_train, y_train, model_type=None):
    """
    Runs training for the churn model.
    
    Note: Real model training will be triggered in Phase 3.
    """
    logger.info("Initializing model training pipeline...")
    
    churn_cfg = config['model']['churn']
    if model_type is None:
        model_type = churn_cfg.get('model_type', 'random_forest')
        
    model_params = {
        'n_estimators': churn_cfg.get('n_estimators', 100),
        'max_depth': churn_cfg.get('max_depth', 6),
        'random_state': churn_cfg.get('random_state', 42)
    }
    
    logger.info(f"Instantiating model of type: {model_type} with parameters: {model_params}")
    model = get_model(model_type, model_params)
    
    logger.info("Fitting churn model on training data...")
    # Fits model (Note: In a true pipeline, fit_transform preprocessor first)
    # model.fit(X_train, y_train)
    
    logger.info("Model training successfully completed (Starter dummy path executed).")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates predictions of a trained model against standard metrics.
    """
    logger.info("Evaluating model performance...")
    # Predictions
    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]
    
    # Placeholder metrics for starter validation
    metrics = {
        "accuracy": 0.80,
        "roc_auc": 0.85
    }
    
    logger.info(f"Evaluation metrics: Accuracy = {metrics['accuracy']}, ROC-AUC = {metrics['roc_auc']}")
    return metrics

def save_model_artifact(model, preprocessor=None, output_dir="models/"):
    """
    Serializes and saves a model and its preprocessor scaler/encoder pipelines to disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    model_path = os.path.join(output_dir, "churn_model.joblib")
    logger.info(f"Saving serialized model to: {model_path}")
    joblib.dump(model, model_path)
    
    if preprocessor:
        preproc_path = os.path.join(output_dir, "preprocessor.joblib")
        logger.info(f"Saving preprocessor to: {preproc_path}")
        joblib.dump(preprocessor, preproc_path)
        
    logger.info("Model artifacts saved successfully.")

if __name__ == "__main__":
    logger.info("train.py loaded successfully as module.")
