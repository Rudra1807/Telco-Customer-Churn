-- scripts/init_db.sql
-- PostgreSQL schema initialisation for the Customer Churn & LTV Prediction Engine.
-- This file is auto-executed by the postgres Docker container on first start
-- (mounted to /docker-entrypoint-initdb.d/).
--
-- Week 1 requirement: Setup PostgreSQL database and load the Telco dataset.

-- ─────────────────────────────────────────────────────────────────────────────
-- Table: customers  (raw Telco dataset)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id         TEXT        PRIMARY KEY,
    gender              TEXT,
    senior_citizen      INTEGER     CHECK (senior_citizen IN (0, 1)),
    partner             TEXT,
    dependents          TEXT,
    tenure              INTEGER     CHECK (tenure >= 0),
    phone_service       TEXT,
    multiple_lines      TEXT,
    internet_service    TEXT,
    online_security     TEXT,
    online_backup       TEXT,
    device_protection   TEXT,
    tech_support        TEXT,
    streaming_tv        TEXT,
    streaming_movies    TEXT,
    contract            TEXT,
    paperless_billing   TEXT,
    payment_method      TEXT,
    monthly_charges     NUMERIC(10, 2),
    total_charges       NUMERIC(10, 2),
    churn               INTEGER     CHECK (churn IN (0, 1)),
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Table: predictions  (inference logs)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL      PRIMARY KEY,
    customer_id         TEXT        REFERENCES customers(customer_id) ON DELETE SET NULL,
    churn_probability   NUMERIC(6, 4),
    churn_prediction    INTEGER,
    churn_risk_tier     TEXT,
    ltv_prediction      NUMERIC(10, 2),
    model_version       TEXT,
    predicted_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_customers_churn      ON customers(churn);
CREATE INDEX IF NOT EXISTS idx_customers_contract   ON customers(contract);
CREATE INDEX IF NOT EXISTS idx_customers_tenure     ON customers(tenure);
CREATE INDEX IF NOT EXISTS idx_predictions_customer ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_tier     ON predictions(churn_risk_tier);

-- ─────────────────────────────────────────────────────────────────────────────
-- Useful views for the dashboard
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_churn_by_contract AS
    SELECT
        contract,
        COUNT(*)                                AS total_customers,
        SUM(churn)                              AS churned,
        ROUND(AVG(churn) * 100, 2)              AS churn_rate_pct,
        ROUND(AVG(monthly_charges), 2)          AS avg_monthly_charges
    FROM customers
    GROUP BY contract
    ORDER BY churn_rate_pct DESC;

CREATE OR REPLACE VIEW v_ltv_segments AS
    SELECT
        p.churn_risk_tier,
        COUNT(*)                                AS customers,
        ROUND(AVG(p.ltv_prediction), 2)         AS avg_ltv,
        ROUND(AVG(p.churn_probability) * 100, 2) AS avg_churn_prob_pct
    FROM predictions p
    GROUP BY p.churn_risk_tier
    ORDER BY avg_ltv DESC;
