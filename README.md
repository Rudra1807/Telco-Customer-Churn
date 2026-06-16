# Customer Churn & Lifetime Value (LTV) Prediction Engine

A production-ready, end-to-end Machine Learning system for predicting **Customer Churn** and forecasting **Customer Lifetime Value (LTV)** using the IBM Telco Customer Churn dataset.

Built over a 4-week internship sprint covering: data ingestion into PostgreSQL, EDA, feature engineering, multi-model training (Logistic Regression, Random Forest, XGBoost), SHAP explainability, FastAPI serving, Streamlit dashboard, and Docker containerisation.

---

## Architecture Overview

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  PostgreSQL  │◄───│ load_to_postgres  │◄───│  Raw CSV (data/raw) │
│  (churn_db)  │    │   .py (ingest)    │    └─────────────────────┘
└──────┬───────┘    └──────────────────┘
       │ SQL queries (optional analytics)
┌──────▼──────────────────────────────────────────┐
│               Notebooks (research layer)         │
│  01_EDA → 02_Preprocessing → 03_Training → 04_LTV│
└──────┬──────────────────────────────────────────┘
       │ saved models → models/
┌──────▼───────┐    ┌────────────────────┐
│  FastAPI     │    │  Streamlit         │
│  (port 8000) │    │  Dashboard         │
│  /predict/*  │    │  (port 8501)       │
└──────────────┘    └────────────────────┘
       └───────────── Docker Compose ──────────────┘
```

---

## Project Structure

```text
customer-churn-ltv-engine/
│
├── data/
│   ├── raw/                 # Original Telco CSV (not committed — add to data/raw/)
│   └── processed/           # Cleaned and split datasets for training
│
├── notebooks/
│   ├── 01_EDA.ipynb                # EDA + baseline analytics report (Week 1)
│   ├── 02_Data_Preprocessing.ipynb # Cleaning, encoding, feature engineering (Week 1-2)
│   ├── 03_Model_Training.ipynb     # LR + RF + XGBoost training + SHAP (Week 2)
│   └── 04_LTV_Prediction.ipynb     # LTV regression model (Week 3)
│
├── src/                     # Production Python package
│   ├── __init__.py
│   ├── data_loader.py       # CSV ingestion + schema validation
│   ├── preprocessing.py     # Cleaning, encoding, train/test split
│   ├── feature_engineering.py # tenure cohorts, charge ratio, service density
│   ├── train.py             # XGBoost churn + LTV training pipelines
│   ├── predict.py           # Inference (single + batch)
│   └── utils.py             # Logging, config loading, path resolution
│
├── scripts/
│   ├── init_db.sql          # PostgreSQL schema (auto-run by Docker on first start)
│   └── load_to_postgres.py  # Ingest Telco CSV → PostgreSQL customers table
│
├── app/
│   └── main.py              # FastAPI service (health, /predict/single, /predict/batch)
│
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard (churn risk + LTV segments)
│
├── models/                  # Serialised XGBoost models + preprocessing metadata
├── reports/                 # Plots, confusion matrices, baseline_analytics_report.csv
├── tests/                   # Unit tests
│
├── Dockerfile               # Single image: runs FastAPI or Streamlit
├── docker-compose.yml       # Orchestrates: postgres + api + dashboard
├── requirements.txt         # Python dependencies
├── config.yaml              # Centralised hyperparameters and paths
└── README.md
```

---

## Week-by-Week Implementation

### Week 1 — Data Ingestion & EDA
- **PostgreSQL** database set up via `docker-compose` (service: `postgres`).
- Schema created automatically via `scripts/init_db.sql` (customers + predictions tables, dashboard views).
- `scripts/load_to_postgres.py` — bulk-upserts the Telco CSV into `churn_db`.
- `notebooks/01_EDA.ipynb` — Pandas + Seaborn EDA: churn distribution, tenure/contract/internet correlations.
- Saves `reports/baseline_analytics_report.csv` with key dataset statistics.
- Missing value handling and categorical encoding prototyped in `notebooks/02_Data_Preprocessing.ipynb`.

### Week 2 — Feature Engineering & Predictive Modeling
- `src/feature_engineering.py` engineers:
  - `charge_ratio` = MonthlyCharges / TotalCharges (avg monthly usage vs base charge)
  - `charges_difference` = TotalCharges − (tenure × MonthlyCharges)
  - `tenure_group` — cohort bins (0-1 Year, 1-2 Years, …)
  - `total_services` — count of subscribed add-on services
- `notebooks/03_Model_Training.ipynb` trains **three classifiers**:
  1. **Logistic Regression** (pure numpy, gradient descent)
  2. **Random Forest** (bootstrap ensemble of decision stumps, pure numpy)
  3. **XGBoost** (with `xgb.cv()` hyperparameter tuning)
- Evaluation: Precision, Recall, F1-Score, ROC-AUC — all via custom numpy implementations.
- **SHAP values** computed for XGBoost model; summary plots saved to `reports/`.

### Week 3 — LTV Calculation & API Development
- `notebooks/04_LTV_Prediction.ipynb` — XGBRegressor trained on TotalCharges as LTV target.
- `src/train.py` — `train_ltv_pipeline()` and `train_churn_pipeline()`.
- `app/main.py` — FastAPI service:
  - `GET  /health` — liveness probe
  - `POST /predict/single` — single customer churn probability + LTV score
  - `POST /predict/batch` — batch scoring from JSON list (up to 5 000 records)
  - `POST /predict/batch/csv` — batch scoring from uploaded CSV (up to 10 000 rows)
  - `GET  /model/info` — loaded model metadata

### Week 4 — Visualization & Deployment
- **Metabase** (port `3000`) — **Primary BI dashboard** (spec requirement):
  - Connects directly to PostgreSQL `churn_db`
  - No-code question builder for creating charts and dashboards
  - Pre-built views available: `v_churn_by_contract`, `v_ltv_segments`
  - Every `/predict` API call is automatically logged to the `predictions` table → Metabase shows live data
- **Streamlit** (port `8501`) — Supplementary developer view (kept for rapid prototyping)
- `Dockerfile` + `docker-compose.yml` — containerises all **four** services: `postgres`, `api`, `metabase`, `dashboard`

---

## Quick Start

### Option A — Docker (Recommended)

```bash
# 1. Add your Telco CSV
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# 2. Start all services (PostgreSQL + FastAPI + Metabase + Streamlit)
docker-compose up --build

# 3. Load the dataset into PostgreSQL (run after postgres is healthy)
docker-compose exec api python scripts/load_to_postgres.py

# 4. Open services:
#    Metabase (PRIMARY dashboard):  http://localhost:3000
#    FastAPI docs:                  http://localhost:8000/docs
#    Streamlit (dev view):          http://localhost:8501
#    PostgreSQL:                    localhost:5432  (churn_user/churn_pass/churn_db)
```

### Option B — Local (Virtual Environment)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Add dataset
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# 4. Run notebooks in order: 01 → 02 → 03 → 04
jupyter notebook

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Start the dashboard (separate terminal)
streamlit run dashboard/app.py
```

---

## Metabase Setup Guide (First-Run, One-Time)

After `docker-compose up --build`, wait ~60 seconds for Metabase to initialise, then:

1. **Open** → `http://localhost:3000`
2. **Create admin account** — set your name, email, and password
3. **Add your database:**
   - Click **"Add your data"** → Select **PostgreSQL**
   - Fill in:
     | Field | Value |
     |---|---|
     | Display name | `Churn DB` |
     | Host | `postgres` |
     | Port | `5432` |
     | Database name | `churn_db` |
     | Username | `churn_user` |
     | Password | `churn_pass` |
   - Click **"Save"**

4. **Build dashboards** — Metabase auto-discovers all tables and views:

| Table / View | What to build |
|---|---|
| `customers` | Churn rate by contract type, tenure distribution, revenue analysis |
| `predictions` | Live churn risk feed, LTV prediction log |
| `v_churn_by_contract` | Pre-aggregated: churn rate % per contract type |
| `v_ltv_segments` | Pre-aggregated: avg LTV per risk tier (HIGH/MEDIUM/LOW) |

5. **Suggested dashboards to create:**
   - 📊 **Global Churn Risk** — pie chart from `v_churn_by_contract`, gauge for overall churn rate
   - 💰 **LTV Segmentation** — bar chart from `v_ltv_segments`, scatter of LTV vs tenure
   - 🔴 **High Risk Customers** — filtered table from `predictions` where `churn_risk_tier = 'HIGH'`
   - 📈 **Prediction Activity** — time-series of `predictions.predicted_at` (API usage tracking)

> **Note on Streamlit:** The `dashboard/app.py` Streamlit app is retained as a lightweight
> developer view for rapid prototyping. For the production/presentation BI layer,
> Metabase is the correct tool per the project specification.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe — model load status |
| GET | `/model/info` | Loaded model metadata + feature count |
| POST | `/predict/single` | Score one customer (JSON body) |
| POST | `/predict/batch` | Score a list of customers (JSON) |
| POST | `/predict/batch/csv` | Score customers from uploaded CSV |

Interactive API docs: `http://localhost:8000/docs`

---

## PostgreSQL Schema

| Table | Description |
|-------|-------------|
| `customers` | Raw Telco dataset (loaded by `load_to_postgres.py`) |
| `predictions` | Inference log — churn prob, LTV, risk tier per customer |
| `v_churn_by_contract` | View — churn rate grouped by contract type |
| `v_ltv_segments` | View — average LTV per risk tier |

---

## Configuration

All parameters live in [`config.yaml`](config.yaml):
- Dataset paths (`data.raw_path`, `data.processed_path`)
- Feature lists (`features.categorical`, `features.numerical`)
- Model hyperparameters (`model.churn.*`, `model.ltv.*`)
- Logging level and log file path

---

## Running Tests

```bash
pytest tests/ -v
# or run the integration test module:
python test_modules.py
```
