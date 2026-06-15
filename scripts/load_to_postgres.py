# scripts/load_to_postgres.py
"""
Week 1 requirement: Setup PostgreSQL database and load the Telco dataset.

This script reads the raw Telco CSV and bulk-loads it into the PostgreSQL
`customers` table that was created by scripts/init_db.sql.

Usage (local):
    pip install psycopg2-binary pandas pyyaml
    python scripts/load_to_postgres.py

Usage (Docker):
    docker-compose exec api python scripts/load_to_postgres.py

Environment variables (or defaults):
    DB_HOST      = localhost
    DB_PORT      = 5432
    DB_NAME      = churn_db
    DB_USER      = churn_user
    DB_PASSWORD  = churn_pass
"""

from __future__ import annotations

import io
import os
import sys
import csv
import logging

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Resolve project root so this script works from any CWD ───────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── DB connection parameters (env overrides) ──────────────────────────────────
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "5432"))
DB_NAME     = os.getenv("DB_NAME",     "churn_db")
DB_USER     = os.getenv("DB_USER",     "churn_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "churn_pass")

# ── Dataset path ──────────────────────────────────────────────────────────────
import yaml
_cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

RAW_CSV = _cfg["data"]["raw_path"]
if not os.path.isabs(RAW_CSV):
    RAW_CSV = os.path.join(PROJECT_ROOT, RAW_CSV)


# ── Column mapping: CSV header → DB column name ───────────────────────────────
COLUMN_MAP = {
    "customerID":       "customer_id",
    "gender":           "gender",
    "SeniorCitizen":    "senior_citizen",
    "Partner":          "partner",
    "Dependents":       "dependents",
    "tenure":           "tenure",
    "PhoneService":     "phone_service",
    "MultipleLines":    "multiple_lines",
    "InternetService":  "internet_service",
    "OnlineSecurity":   "online_security",
    "OnlineBackup":     "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport":      "tech_support",
    "StreamingTV":      "streaming_tv",
    "StreamingMovies":  "streaming_movies",
    "Contract":         "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod":    "payment_method",
    "MonthlyCharges":   "monthly_charges",
    "TotalCharges":     "total_charges",
    "Churn":            "churn",
}


def load_and_clean_csv(path: str) -> pd.DataFrame:
    """Load and clean the raw Telco CSV, returning a DB-ready DataFrame."""
    logger.info("Loading raw dataset: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # Fix TotalCharges (whitespace → NaN → 0 for new customers)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Map Churn to binary int
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Rename to DB columns
    df = df.rename(columns=COLUMN_MAP)

    # Keep only mapped columns
    db_cols = list(COLUMN_MAP.values())
    df = df[[c for c in db_cols if c in df.columns]]

    logger.info("Cleaned DataFrame: %d rows, columns: %s", len(df), list(df.columns))
    return df


def upsert_to_postgres(df: pd.DataFrame) -> None:
    """Bulk-upsert the cleaned DataFrame into the customers table."""
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        logger.error(
            "psycopg2 not installed. Run: pip install psycopg2-binary"
        )
        sys.exit(1)

    conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
    logger.info("Connecting to PostgreSQL: %s@%s:%d/%s", DB_USER, DB_HOST, DB_PORT, DB_NAME)

    try:
        conn = psycopg2.connect(conn_str)
        cur  = conn.cursor()
    except psycopg2.OperationalError as exc:
        logger.error("Could not connect to PostgreSQL: %s", exc)
        sys.exit(1)

    cols    = list(df.columns)
    values  = [tuple(row) for row in df.itertuples(index=False, name=None)]
    col_str = ", ".join(cols)
    upd_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "customer_id")

    sql = f"""
        INSERT INTO customers ({col_str})
        VALUES %s
        ON CONFLICT (customer_id) DO UPDATE SET {upd_str};
    """

    logger.info("Upserting %d rows into 'customers' table...", len(values))
    execute_values(cur, sql, values, page_size=500)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM customers;")
    total = cur.fetchone()[0]
    logger.info("Upsert complete. Total rows in 'customers': %d", total)

    cur.close()
    conn.close()


def main() -> None:
    if not os.path.exists(RAW_CSV):
        logger.error(
            "Dataset not found at '%s'. "
            "Download the IBM Telco Churn CSV and place it in data/raw/ first.",
            RAW_CSV,
        )
        sys.exit(1)

    df = load_and_clean_csv(RAW_CSV)
    upsert_to_postgres(df)
    logger.info("Done! The Telco dataset has been loaded into PostgreSQL.")


if __name__ == "__main__":
    main()
