# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile  —  Customer Churn & LTV Prediction Engine
# ─────────────────────────────────────────────────────────────────────────────
# Build:   docker build -t churn-ltv-engine .
# Run API: docker run -p 8000:8000 churn-ltv-engine uvicorn app.main:app --host 0.0.0.0 --port 8000
# Run UI:  docker run -p 8501:8501 churn-ltv-engine streamlit run dashboard/app.py --server.port 8501
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="Rudra Pratap Giri"
LABEL description="Customer Churn & LTV Prediction Engine — XGBoost + FastAPI + Streamlit"
LABEL version="2.0.0"

# Prevent Python buffering & .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── 1. Install OS-level dependencies ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── 2. Install Python dependencies ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── 3. Copy project source ───────────────────────────────────────────────────
COPY . .

# ── 4. Create necessary directories ──────────────────────────────────────────
RUN mkdir -p data/raw data/processed models reports

# ── 5. Expose ports ──────────────────────────────────────────────────────────
# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501

# ── 6. Default command: run the FastAPI service ───────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
