# Customer Churn & Lifetime Value (LTV) Engine

This repository contains a production-ready, end-to-end Machine Learning system for predicting **Customer Churn** and forecasting **Customer Lifetime Value (LTV)** using the IBM Telco Customer Churn dataset.

The system is designed with a modular architecture, splitting the pipeline into clear data-loading, preprocessing, feature-engineering, training, and scoring components. It features configuration-driven parameters to ensure reproducibility, a Streamlit dashboard for interactive business insights, and a FastAPI service for real-time model serving.

---

## Project Structure

```text
customer-churn-ltv-engine/
│
├── data/
│   ├── raw/                 # Original, immutable datasets (e.g. IBM Telco CSV)
│   └── processed/           # Cleaned and processed datasets ready for training
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis and data profiling
│   ├── 02_Data_Preprocessing.ipynb  # Prototyping data cleansing and transformations
│   ├── 03_Model_Training.ipynb     # Model training, validation, and churn evaluation
│   └── 04_LTV_Prediction.ipynb     # Modeling customer lifetime value (LTV)
│
├── src/                     # Core Python package with production modules
│   ├── __init__.py
│   ├── data_loader.py       # Methods to fetch, load, and validate input schema
│   ├── preprocessing.py     # Clean data (e.g., handling missing charges, scaling)
│   ├── feature_engineering.py # Deriving advanced features (cohorts, usage metrics)
│   ├── train.py             # Model training pipelines, cross-validation, and metrics
│   ├── predict.py           # Single/batch inference scoring pipelines
│   └── utils.py             # System helpers (logging, path resolution, plotting)
│
├── models/                  # Serialized models, scalers, and encoder pipelines (.joblib)
├── reports/                 # Evaluation plots, confusion matrices, and tracking logs
├── dashboard/               # Interactive UI for business metrics (Streamlit dashboard)
├── app/                     # Serving layer (FastAPI application for REST APIs)
├── tests/                   # Unit test cases for critical source code modules
│
├── requirements.txt         # Package dependencies for execution
├── config.yaml              # Centralized hyperparameters, path definitions, and feature configurations
├── .gitignore               # Environment, data, and cache exclusion patterns
└── README.md                # System documentation
```

---

## File Explanations

- **`config.yaml`**: The single source of truth for all configurations. It isolates raw/processed paths, data column names, categorical/numerical lists, train-test splits, random seeds, and model hyperparameters so you never have to hardcode them.
- **`src/data_loader.py`**: Handles initial data collection and validation. It ensures the incoming data conforms to the expected columns and types, preventing "garbage in, garbage out" at the ingestion layer.
- **`src/preprocessing.py`**: Performs cleaning routines. In the Telco dataset, `TotalCharges` is often read as an object due to blank spaces; this file handles casting, missing value imputation, scaling, and categorical encoding.
- **`src/feature_engineering.py`**: Generates domain-specific features. It extracts service combination density, monthly charging ratios, and customer tenure cohorts to enrich model performance.
- **`src/train.py`**: Executes structured training, cross-validation, and performance logging for both the binary Churn classifier and the continuous LTV regressor. It serializes the best models to the `models/` folder.
- **`src/predict.py`**: Controls inference. It exposes functions to load saved pipelines and score new raw payloads, ensuring the training and serving logic remain perfectly identical.
- **`src/utils.py`**: Handles common utilities like configuring professional stdout/file loggers, resolving directory paths, and saving plots.

---

## Virtual Environment Setup

To run this engine locally, configure a clean Python virtual environment.

### 1. Clone the repository and navigate to the project directory:
```bash
cd customer-churn-ltv-engine
```

### 2. Create a virtual environment:
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **macOS/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Upgrade pip and install dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Getting Started: Phase 1 (EDA)

1. Download the raw IBM Telco Customer Churn dataset.
2. Place the CSV file into `data/raw/` and name it exactly:  
   `WA_Fn-UseC_-Telco-Customer-Churn.csv` (or configure its path in `config.yaml`).
3. Open a Jupyter server:
   ```bash
   jupyter notebook
   ```
4. Navigate to `notebooks/01_EDA.ipynb` to begin exploratory data analysis.
