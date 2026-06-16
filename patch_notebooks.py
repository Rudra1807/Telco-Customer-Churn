"""
patch_notebooks.py  –  Fixes the gaps identified in the verification report:
  1. Add LR + RF training cells to notebooks/03_Model_Training.ipynb (after cell 1, before XGBoost)
  2. Add baseline analytics report save cell to notebooks/01_EDA.ipynb (after cell 16)
"""
import json, copy

# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────
def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }


# ════════════════════════════════════════════════════════════════════════════
# FIX 1 – notebooks/03_Model_Training.ipynb
#   Insert LR + RF training + comparison BEFORE the XGBoost cells (after cell 1)
# ════════════════════════════════════════════════════════════════════════════
NB03 = "notebooks/03_Model_Training.ipynb"
nb03 = json.load(open(NB03, encoding="utf-8"))

NEW_03_CELLS = [

    md_cell(
        "### Baseline Classifiers: Logistic Regression & Random Forest\n\n"
        "Per the project specification (Week 2, Day 4-6), we train and evaluate **three classifiers** — "
        "Logistic Regression, Random Forest, and XGBoost — and compare them using Precision, Recall, and F1-Score. "
        "This multi-model comparison lets us justify why XGBoost is selected as the champion model."
    ),

    code_cell(
        "# ── Logistic Regression (pure numpy baseline) ────────────────────────────────\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "# Sigmoid helper\n"
        "def sigmoid(z):\n"
        "    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))\n"
        "\n"
        "class LogisticRegressionNP:\n"
        "    \"\"\"\n"
        "    Logistic Regression trained via mini-batch gradient descent.\n"
        "    No sklearn dependency — pure numpy.\n"
        "    \"\"\"\n"
        "    def __init__(self, lr=0.01, n_iter=500, random_state=42):\n"
        "        self.lr = lr\n"
        "        self.n_iter = n_iter\n"
        "        self.random_state = random_state\n"
        "        self.weights = None\n"
        "        self.bias = None\n"
        "\n"
        "    def fit(self, X, y):\n"
        "        rng = np.random.default_rng(self.random_state)\n"
        "        n_samples, n_features = X.shape\n"
        "        self.weights = rng.normal(0, 0.01, n_features)\n"
        "        self.bias = 0.0\n"
        "        y = np.asarray(y, dtype=float)\n"
        "        for _ in range(self.n_iter):\n"
        "            z = X @ self.weights + self.bias\n"
        "            pred = sigmoid(z)\n"
        "            err = pred - y\n"
        "            self.weights -= self.lr * (X.T @ err) / n_samples\n"
        "            self.bias    -= self.lr * err.mean()\n"
        "        return self\n"
        "\n"
        "    def predict_proba(self, X):\n"
        "        return sigmoid(X @ self.weights + self.bias)\n"
        "\n"
        "    def predict(self, X, threshold=0.5):\n"
        "        return (self.predict_proba(X) >= threshold).astype(int)\n"
        "\n"
        "\n"
        "# ── Train ─────────────────────────────────────────────────────────────────────\n"
        "print('Training Logistic Regression (numpy, gradient descent)...')\n"
        "X_train_np = X_train.values.astype(float)\n"
        "X_test_np  = X_test.values.astype(float)\n"
        "y_train_np = y_train.values.astype(float)\n"
        "y_test_np  = y_test.values.astype(float)\n"
        "\n"
        "lr_model = LogisticRegressionNP(lr=0.05, n_iter=800, random_state=42)\n"
        "lr_model.fit(X_train_np, y_train_np)\n"
        "\n"
        "lr_pred  = lr_model.predict(X_test_np)\n"
        "lr_prob  = lr_model.predict_proba(X_test_np)\n"
        "lr_acc   = float((lr_pred == y_test_np).mean())\n"
        "\n"
        "print(f'  Logistic Regression Accuracy : {lr_acc:.4f}')\n"
        "print('  Done.')\n"
    ),

    md_cell(
        "### Logistic Regression — Precision, Recall & F1\n\n"
        "We compute per-class metrics manually (numpy only) consistent with the rest of this notebook."
    ),

    code_cell(
        "# ── LR Metrics ───────────────────────────────────────────────────────────────\n"
        "import numpy as np\n"
        "\n"
        "def precision_recall_f1_score(y_true, y_pred):\n"
        "    y_true = np.asarray(y_true, dtype=int)\n"
        "    y_pred = np.asarray(y_pred, dtype=int)\n"
        "    tp = int(((y_pred == 1) & (y_true == 1)).sum())\n"
        "    fp = int(((y_pred == 1) & (y_true == 0)).sum())\n"
        "    fn = int(((y_pred == 0) & (y_true == 1)).sum())\n"
        "    tn = int(((y_pred == 0) & (y_true == 0)).sum())\n"
        "    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0\n"
        "    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0\n"
        "    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0\n"
        "    acc   = (tp + tn) / len(y_true)\n"
        "    return dict(accuracy=round(acc,4), precision=round(prec,4),\n"
        "                recall=round(rec,4), f1=round(f1,4),\n"
        "                tp=tp, fp=fp, fn=fn, tn=tn)\n"
        "\n"
        "lr_metrics = precision_recall_f1_score(y_test_np, lr_pred)\n"
        "print('Logistic Regression Metrics')\n"
        "print(f\"  Accuracy  : {lr_metrics['accuracy']}\")\n"
        "print(f\"  Precision : {lr_metrics['precision']}\")\n"
        "print(f\"  Recall    : {lr_metrics['recall']}\")\n"
        "print(f\"  F1-Score  : {lr_metrics['f1']}\")\n"
    ),

    md_cell(
        "### Random Forest Classifier (pure numpy decision stump ensemble)\n\n"
        "A bagging ensemble of decision stumps trained on bootstrapped subsets — "
        "no sklearn dependency."
    ),

    code_cell(
        "# ── Random Forest (bootstrap + decision-stump ensemble) ──────────────────────\n"
        "import numpy as np\n"
        "\n"
        "class DecisionStump:\n"
        "    \"\"\"Single-feature binary decision stump (threshold split).\"\"\"\n"
        "    def __init__(self):\n"
        "        self.feature_idx = None\n"
        "        self.threshold   = None\n"
        "        self.polarity    = 1\n"
        "\n"
        "    def fit(self, X, y, rng):\n"
        "        n, p = X.shape\n"
        "        best_gini = float('inf')\n"
        "        # Sample a random subset of features (sqrt rule)\n"
        "        n_feats = max(1, int(np.sqrt(p)))\n"
        "        feat_ids = rng.choice(p, size=n_feats, replace=False)\n"
        "        for fi in feat_ids:\n"
        "            vals = X[:, fi]\n"
        "            thresholds = np.unique(vals)\n"
        "            for t in thresholds:\n"
        "                for pol in [1, -1]:\n"
        "                    pred = np.where(pol * vals >= pol * t, 1, 0)\n"
        "                    # Gini impurity\n"
        "                    for split_val in [0, 1]:\n"
        "                        mask = pred == split_val\n"
        "                        if mask.sum() == 0:\n"
        "                            continue\n"
        "                        p1 = y[mask].mean()\n"
        "                        gini = 2 * p1 * (1 - p1)\n"
        "                    total_gini = gini * mask.sum() / n\n"
        "                    if total_gini < best_gini:\n"
        "                        best_gini = total_gini\n"
        "                        self.feature_idx = fi\n"
        "                        self.threshold   = t\n"
        "                        self.polarity    = pol\n"
        "\n"
        "    def predict(self, X):\n"
        "        return np.where(\n"
        "            self.polarity * X[:, self.feature_idx] >= self.polarity * self.threshold,\n"
        "            1, 0\n"
        "        )\n"
        "\n"
        "\n"
        "class RandomForestNP:\n"
        "    \"\"\"\n"
        "    Bootstrap-aggregated ensemble of DecisionStumps.\n"
        "    Majority-vote prediction. No sklearn dependency.\n"
        "    \"\"\"\n"
        "    def __init__(self, n_estimators=100, random_state=42):\n"
        "        self.n_estimators = n_estimators\n"
        "        self.random_state = random_state\n"
        "        self.estimators   = []\n"
        "\n"
        "    def fit(self, X, y):\n"
        "        rng = np.random.default_rng(self.random_state)\n"
        "        y = np.asarray(y, dtype=int)\n"
        "        n = len(y)\n"
        "        for _ in range(self.n_estimators):\n"
        "            idx = rng.integers(0, n, size=n)\n"
        "            stump = DecisionStump()\n"
        "            stump.fit(X[idx], y[idx], rng)\n"
        "            self.estimators.append(stump)\n"
        "        return self\n"
        "\n"
        "    def predict_proba(self, X):\n"
        "        votes = np.stack([e.predict(X) for e in self.estimators], axis=1)\n"
        "        return votes.mean(axis=1)  # fraction voting class=1\n"
        "\n"
        "    def predict(self, X, threshold=0.5):\n"
        "        return (self.predict_proba(X) >= threshold).astype(int)\n"
        "\n"
        "\n"
        "print('Training Random Forest (100 stumps, bootstrap aggregating)...')\n"
        "rf_model = RandomForestNP(n_estimators=100, random_state=42)\n"
        "rf_model.fit(X_train_np, y_train_np)\n"
        "\n"
        "rf_pred = rf_model.predict(X_test_np)\n"
        "rf_metrics = precision_recall_f1_score(y_test_np, rf_pred)\n"
        "print(f\"  Accuracy  : {rf_metrics['accuracy']}\")\n"
        "print(f\"  Precision : {rf_metrics['precision']}\")\n"
        "print(f\"  Recall    : {rf_metrics['recall']}\")\n"
        "print(f\"  F1-Score  : {rf_metrics['f1']}\")\n"
        "print('  Done.')\n"
    ),

    md_cell(
        "### Model Comparison: LR vs Random Forest vs XGBoost\n\n"
        "Side-by-side comparison of all three classifiers on the held-out test set."
    ),

    code_cell(
        "# ── Model comparison table ────────────────────────────────────────────────────\n"
        "# (xgb metrics computed after XGBoost cell runs; placeholders until then)\n"
        "import pandas as pd\n"
        "\n"
        "comparison_data = {\n"
        "    'Model':     ['Logistic Regression', 'Random Forest (Stumps)'],\n"
        "    'Accuracy':  [lr_metrics['accuracy'],  rf_metrics['accuracy']],\n"
        "    'Precision': [lr_metrics['precision'], rf_metrics['precision']],\n"
        "    'Recall':    [lr_metrics['recall'],    rf_metrics['recall']],\n"
        "    'F1-Score':  [lr_metrics['f1'],        rf_metrics['f1']],\n"
        "}\n"
        "df_compare = pd.DataFrame(comparison_data)\n"
        "print('\\n=== Classifier Comparison (XGBoost metrics added after training) ===')\n"
        "print(df_compare.to_string(index=False))\n"
        "print('\\n→ XGBoost trained in next section — compare all three once complete.')\n"
    ),
]

# Insert after cell 1 (the data-loading cell)
cells_03 = nb03["cells"]
nb03["cells"] = cells_03[:2] + NEW_03_CELLS + cells_03[2:]

with open(NB03, "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=1, ensure_ascii=False)
print(f"[OK] Patched {NB03}: inserted LR + RF + comparison cells ({len(NEW_03_CELLS)} cells)")


# ════════════════════════════════════════════════════════════════════════════
# FIX 2 – notebooks/01_EDA.ipynb
#   Append baseline analytics report cell at the end
# ════════════════════════════════════════════════════════════════════════════
NB01 = "notebooks/01_EDA.ipynb"
nb01 = json.load(open(NB01, encoding="utf-8"))

BASELINE_CELLS = [
    md_cell(
        "### Step 8: Save Baseline Analytics Report\n\n"
        "Per the project specification (Week 1, Day 6-7), we save a baseline analytics report "
        "capturing key dataset statistics as a CSV artifact in `reports/`. "
        "This provides a reproducible reference point for tracking data drift and model performance over time."
    ),
    code_cell(
        "# ── Baseline Analytics Report ─────────────────────────────────────────────────\n"
        "import os, datetime\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "\n"
        "os.makedirs('../reports', exist_ok=True)\n"
        "\n"
        "if 'df_raw' in locals():\n"
        "    df_tmp = df_raw.copy()\n"
        "    df_tmp['TotalCharges'] = pd.to_numeric(df_tmp['TotalCharges'], errors='coerce')\n"
        "\n"
        "    churn_series = df_tmp['Churn'].map({'Yes': 1, 'No': 0})\n"
        "\n"
        "    # ── Key statistical metrics ────────────────────────────────────────────────\n"
        "    baseline_stats = {\n"
        "        'generated_at':            datetime.datetime.now().isoformat(),\n"
        "        'total_customers':         len(df_tmp),\n"
        "        'churn_count':             int(churn_series.sum()),\n"
        "        'churn_rate_pct':          round(churn_series.mean() * 100, 2),\n"
        "        'tenure_mean':             round(df_tmp['tenure'].mean(), 2),\n"
        "        'tenure_median':           round(df_tmp['tenure'].median(), 2),\n"
        "        'tenure_std':              round(df_tmp['tenure'].std(), 2),\n"
        "        'monthly_charges_mean':    round(df_tmp['MonthlyCharges'].mean(), 2),\n"
        "        'monthly_charges_median':  round(df_tmp['MonthlyCharges'].median(), 2),\n"
        "        'total_charges_mean':      round(df_tmp['TotalCharges'].mean(), 2),\n"
        "        'total_charges_median':    round(df_tmp['TotalCharges'].median(), 2),\n"
        "        'missing_values_total':    int(df_tmp.isnull().sum().sum()),\n"
        "        'contract_month_to_month': int((df_tmp['Contract'] == 'Month-to-month').sum()),\n"
        "        'contract_one_year':       int((df_tmp['Contract'] == 'One year').sum()),\n"
        "        'contract_two_year':       int((df_tmp['Contract'] == 'Two year').sum()),\n"
        "        'internet_fiber':          int((df_tmp['InternetService'] == 'Fiber optic').sum()),\n"
        "        'internet_dsl':            int((df_tmp['InternetService'] == 'DSL').sum()),\n"
        "        'internet_none':           int((df_tmp['InternetService'] == 'No').sum()),\n"
        "        'paperless_billing_pct':   round((df_tmp['PaperlessBilling'] == 'Yes').mean() * 100, 2),\n"
        "        'senior_citizen_pct':      round(df_tmp['SeniorCitizen'].mean() * 100, 2),\n"
        "        'partner_pct':             round((df_tmp['Partner'] == 'Yes').mean() * 100, 2),\n"
        "    }\n"
        "\n"
        "    # ── Save to CSV ────────────────────────────────────────────────────────────\n"
        "    df_report = pd.DataFrame([baseline_stats]).T.reset_index()\n"
        "    df_report.columns = ['metric', 'value']\n"
        "    report_path = '../reports/baseline_analytics_report.csv'\n"
        "    df_report.to_csv(report_path, index=False)\n"
        "\n"
        "    print('=== Baseline Analytics Report Saved ===')\n"
        "    print(f'  Path: {os.path.abspath(report_path)}')\n"
        "    print()\n"
        "    print(df_report.to_string(index=False))\n"
        "else:\n"
        "    print('df_raw not found — run the data ingestion cell first.')\n"
    ),
]

nb01["cells"] = nb01["cells"] + BASELINE_CELLS

with open(NB01, "w", encoding="utf-8") as f:
    json.dump(nb01, f, indent=1, ensure_ascii=False)
print(f"[OK] Patched {NB01}: appended baseline analytics report cells")

print("\nAll notebook patches applied successfully.")
