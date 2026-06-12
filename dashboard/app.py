# =============================================================================
# Telco Customer Churn Intelligence Dashboard  ·  dashboard/app.py
# Run from project root:  streamlit run dashboard/app.py
# =============================================================================

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── 1. PAGE CONFIG (must be the very first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Telco Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. GLOBAL CSS — single block ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── Page background (deep purple-black dark mode) ── */
    .stApp, body, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"], .main {
        background: linear-gradient(135deg, #1A0B2E 0%, #130924 40%, #11071F 100%) fixed !important;
        color: #E2D9F3 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(26,11,46,0.85) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(167,139,250,0.18) !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #C4B5FD !important;
    }
    [data-testid="stSidebar"] .stSlider > div { color: #C4B5FD !important; }
    [data-testid="stSidebar"] .stRadio label { color: #C4B5FD !important; }
    [data-testid="stSidebar"] .stCheckbox label { color: #C4B5FD !important; }
    [data-testid="stSidebar"] .stMultiSelect label { color: #C4B5FD !important; }
    [data-testid="stSidebar"] .block-container { padding-top: 0.5rem !important; }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── Block container ── */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
        max-width: 1440px !important;
    }

    /* ── Glassmorphism card (dark) ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(167,139,250,0.18);
        border-radius: 16px;
        box-shadow: 0 4px 32px rgba(108,99,255,0.18);
        padding: 18px 20px;
    }

    /* ── KPI card (dark) ── */
    .kpi-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(167,139,250,0.18);
        border-radius: 16px;
        box-shadow: 0 4px 32px rgba(108,99,255,0.18);
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        margin-bottom: 0;
    }
    .kpi-blob {
        position: absolute;
        top: -10px;
        right: -10px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        opacity: 0.12;
    }
    .kpi-label {
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #7C6FA0;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 4px;
    }
    .kpi-delta {
        font-size: 9px;
        color: #6B5E8A;
        margin-top: 2px;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #7C6FA0;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* ── Chart card wrapper (dark) ── */
    .chart-wrap {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(167,139,250,0.18);
        border-radius: 16px;
        box-shadow: 0 4px 32px rgba(108,99,255,0.18);
        padding: 18px 20px 8px 20px;
        margin-bottom: 0;
    }
    .chart-title {
        font-size: 13px;
        font-weight: 700;
        color: #E2D9F3;
        margin-bottom: 2px;
    }
    .chart-sub {
        font-size: 11px;
        color: #7C6FA0;
        margin-bottom: 10px;
    }

    /* ── Donut legend ── */
    .donut-legend {
        display: flex;
        gap: 16px;
        justify-content: center;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .legend-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
        vertical-align: middle;
    }
    .legend-item {
        font-size: 11px;
        color: #C4B5FD;
        font-weight: 500;
    }

    /* ── Risk factor rows (dark) ── */
    .risk-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 10px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #E2D9F3;
    }
    .risk-badge {
        font-size: 10px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
    }

    /* ── Header badge pills (dark) ── */
    .pill-total {
        background: rgba(139,92,246,0.25);
        color: #C4B5FD;
        border: 1px solid rgba(167,139,250,0.35);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
        margin-right: 8px;
    }
    .pill-churn {
        background: rgba(239,68,68,0.18);
        color: #FCA5A5;
        border: 1px solid rgba(239,68,68,0.3);
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
        display: inline-block;
    }

    /* ── Apply/Reset button styling ── */
    div[data-testid="stButton"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }

    /* ── Footer ── */
    .dash-footer {
        border-top: 1px solid #E5E7EB;
        text-align: center;
        padding-top: 12px;
        margin-top: 16px;
        font-size: 11px;
        color: #9CA3AF;
    }

    /* ── Sidebar button custom look ── */
    .btn-apply button {
        background: #6C63FF !important;
        color: white !important;
        border: none !important;
    }
    .btn-apply button:hover {
        background: #5b52f0 !important;
    }
    .btn-reset button {
        background: white !important;
        color: #6C63FF !important;
        border: 1px solid #6C63FF !important;
    }
    .btn-reset button:hover {
        background: #f5f3ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 3. DATA LOADING ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    return df


df_full = load_data()

# ── 4. SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / brand
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px; padding-bottom:16px; border-bottom:1px solid rgba(108,99,255,0.15);">
            <div style="width:38px; height:38px; border-radius:10px;
                        background:linear-gradient(135deg,#6C63FF,#A78BFA);
                        display:flex; align-items:center; justify-content:center;
                        font-size:18px; flex-shrink:0;">📡</div>
            <div>
                <div style="font-size:15px; font-weight:800; color:#1E1B4B; line-height:1.1;">TELCO</div>
                <div style="font-size:10px; color:#9CA3AF; line-height:1.2;">Churn Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filters label
    st.markdown("<div class='section-label'>FILTERS</div>", unsafe_allow_html=True)

    # Contract Type
    contract_opts = sorted(df_full["Contract"].unique().tolist())
    sel_contract = st.multiselect(
        "Contract Type",
        options=contract_opts,
        default=contract_opts,
        key="sel_contract",
    )

    # Internet Service
    internet_opts = sorted(df_full["InternetService"].unique().tolist())
    sel_internet = st.multiselect(
        "Internet Service",
        options=internet_opts,
        default=internet_opts,
        key="sel_internet",
    )

    # Tenure slider
    sel_tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=(0, 72),
        key="sel_tenure",
    )

    # Monthly Charges slider
    sel_charges = st.slider(
        "Monthly Charges ($)",
        min_value=18,
        max_value=119,
        value=(18, 119),
        key="sel_charges",
    )

    # Gender radio
    sel_gender = st.radio(
        "Gender",
        options=["All", "Male", "Female"],
        horizontal=True,
        key="sel_gender",
    )

    # Senior citizen checkbox
    sel_senior = st.checkbox("Only senior citizens", value=False, key="sel_senior")

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # Apply / Reset buttons
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        st.markdown("<div class='btn-apply'>", unsafe_allow_html=True)
        apply_clicked = st.button("Apply Filters", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with btn_col2:
        st.markdown("<div class='btn-reset'>", unsafe_allow_html=True)
        reset_clicked = st.button("Reset", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if reset_clicked:
        st.session_state.sel_contract = contract_opts
        st.session_state.sel_internet = internet_opts
        st.session_state.sel_tenure = (0, 72)
        st.session_state.sel_charges = (18, 119)
        st.session_state.sel_gender = "All"
        st.session_state.sel_senior = False
        st.rerun()

# ── 5. APPLY FILTERS ──────────────────────────────────────────────────────────
df = df_full.copy()

if sel_contract:
    df = df[df["Contract"].isin(sel_contract)]
if sel_internet:
    df = df[df["InternetService"].isin(sel_internet)]
df = df[(df["tenure"] >= sel_tenure[0]) & (df["tenure"] <= sel_tenure[1])]
df = df[(df["MonthlyCharges"] >= sel_charges[0]) & (df["MonthlyCharges"] <= sel_charges[1])]
if sel_gender != "All":
    df = df[df["gender"] == sel_gender]
if sel_senior:
    df = df[df["SeniorCitizen"] == 1]

if df.empty:
    st.warning("⚠️ No customers match the current filters. Please adjust your selections.")
    st.stop()

# ── 6. PRE-COMPUTE METRICS ────────────────────────────────────────────────────
n_total = len(df)
n_churned = int(df["Churn_Binary"].sum())
churn_rate = n_churned / n_total * 100
avg_charges = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

# ── 7. HEADER ─────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([2.5, 1])
with h_left:
    st.markdown(
        """
        <h1 style="font-size:26px; font-weight:700; color:#1E1B4B; margin-bottom:2px; margin-top:0;">
            Customer Churn Intelligence
        </h1>
        <p style="font-size:12px; color:#9CA3AF; margin-top:0; margin-bottom:0;">
            Understand behavior · Identify risks · Drive retention strategies
        </p>
        """,
        unsafe_allow_html=True,
    )
with h_right:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:flex-end; padding-top:6px; gap:8px; flex-wrap:wrap;">
            <span class="pill-total">👥 {n_total:,} customers</span>
            <span class="pill-churn">⚠ {churn_rate:.1f}% churn rate</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── 8. KPI ROW ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

kpi_specs = [
    (k1, "Total Customers", f"{n_total:,}", "#6C63FF", f"of {len(df_full):,} full dataset"),
    (k2, "Churn Rate", f"{churn_rate:.1f}%", "#EF4444", f"{n_churned:,} customers churned"),
    (k3, "Avg Monthly Charges", f"${avg_charges:.2f}", "#D97706", f"vs. ${df_full['MonthlyCharges'].mean():.2f} overall"),
    (k4, "Avg Tenure", f"{avg_tenure:.1f} mo", "#10B981", f"vs. {df_full['tenure'].mean():.1f} mo overall"),
]

for col, label, value, accent, delta in kpi_specs:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-blob" style="background:{accent};"></div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value" style="color:{accent};">{value}</div>
                <div class="kpi-delta">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

# ── 9. CHART ROW 1 — Donut + Contract Horizontal Bar ─────────────────────────
r1_left, r1_right = st.columns([1, 1.7])

# --- Donut ---
with r1_left:
    retained_n = n_total - n_churned
    fig_donut = go.Figure(data=[go.Pie(
        labels=["Retained", "Churned"],
        values=[retained_n, n_churned],
        hole=0.65,
        marker=dict(colors=["#6C63FF", "#EF4444"], line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        showlegend=False,
    )])
    fig_donut.add_annotation(
        text=f"<b>{n_total:,}</b><br><span style='font-size:10px'>total</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#E2D9F3", family="Inter"),
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C4B5FD"),
        margin=dict(l=10, r=10, t=10, b=0),
        height=240,
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Churn Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-sub'>Overall retained vs. churned customers</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
    churn_pct = churn_rate
    retain_pct = 100 - churn_pct
    st.markdown(
        f"""
        <div class="donut-legend">
            <span class="legend-item"><span class="legend-dot" style="background:#6C63FF;"></span>Retained {retain_pct:.1f}%</span>
            <span class="legend-item"><span class="legend-dot" style="background:#EF4444;"></span>Churned {churn_pct:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Horizontal Grouped Bar: Contract Type ---
with r1_right:
    contract_grp = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Count")
    )
    retained_data = contract_grp[contract_grp["Churn"] == "No"]
    churned_data  = contract_grp[contract_grp["Churn"] == "Yes"]
    all_contracts = sorted(contract_grp["Contract"].unique())

    def safe_get(sub, contract):
        row = sub[sub["Contract"] == contract]
        return int(row["Count"].values[0]) if len(row) else 0

    fig_contract = go.Figure()
    fig_contract.add_trace(go.Bar(
        name="Retained",
        y=all_contracts,
        x=[safe_get(retained_data, c) for c in all_contracts],
        orientation="h",
        marker=dict(color="#6C63FF", line=dict(width=0)),
        opacity=0.85,
        text=[safe_get(retained_data, c) for c in all_contracts],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b> — Retained: %{x:,}<extra></extra>",
    ))
    fig_contract.add_trace(go.Bar(
        name="Churned",
        y=all_contracts,
        x=[safe_get(churned_data, c) for c in all_contracts],
        orientation="h",
        marker=dict(color="#EF4444", line=dict(width=0)),
        opacity=0.85,
        text=[safe_get(churned_data, c) for c in all_contracts],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b> — Churned: %{x:,}<extra></extra>",
    ))
    fig_contract.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        font=dict(family="Inter, sans-serif", color="#C4B5FD", size=12),
        margin=dict(l=10, r=60, t=10, b=30),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="rgba(167,139,250,0.12)", gridwidth=1, zeroline=False,
                   linecolor="rgba(167,139,250,0.2)", tickfont=dict(size=11, color="#9E8EC5"), title="Customers"),
        yaxis=dict(showgrid=False, linecolor="rgba(167,139,250,0.2)", tickfont=dict(size=11, color="#9E8EC5"), title=""),
        hoverlabel=dict(bgcolor="#1A0B2E", font_size=12, font_family="Inter"),
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Churn by Contract Type</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-sub'>Customer count by contract and churn status</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_contract, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── 10. CHART ROW 2 — Histogram + Scatter + Internet Bar ─────────────────────
r2a, r2b, r2c = st.columns([1.2, 1, 1])

# --- Monthly Charges Histogram ---
with r2a:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df[df["Churn"] == "No"]["MonthlyCharges"],
        name="Retained",
        marker_color="#6C63FF",
        opacity=0.45,
        nbinsx=30,
        hovertemplate="$%{x:.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.add_trace(go.Histogram(
        x=df[df["Churn"] == "Yes"]["MonthlyCharges"],
        name="Churned",
        marker_color="#EF4444",
        opacity=0.45,
        nbinsx=30,
        hovertemplate="$%{x:.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
        font=dict(family="Inter, sans-serif", color="#C4B5FD", size=12),
        margin=dict(l=10, r=10, t=10, b=30),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="rgba(167,139,250,0.12)", linecolor="rgba(167,139,250,0.2)",
                   tickfont=dict(size=11, color="#9E8EC5"), title="Monthly Charges ($)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(167,139,250,0.12)", linecolor="rgba(167,139,250,0.2)",
                   tickfont=dict(size=11, color="#9E8EC5"), title="Count"),
        hoverlabel=dict(bgcolor="#1A0B2E", font_size=12, font_family="Inter"),
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Monthly Charges Distribution</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-sub'>Retained vs. churned by spend level</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tenure vs Monthly Charges Scatter ---
with r2b:
    sample_df = df.sample(min(len(df), 1200), random_state=42)
    fig_scatter = go.Figure()
    for label, color in [("No", "#6C63FF"), ("Yes", "#EF4444")]:
        sub = sample_df[sample_df["Churn"] == label]
        name = "Retained" if label == "No" else "Churned"
        fig_scatter.add_trace(go.Scatter(
            x=sub["tenure"],
            y=sub["MonthlyCharges"],
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.5, line=dict(width=0)),
            hovertemplate=f"<b>{name}</b><br>Tenure: %{{x}} mo<br>Charges: $%{{y:.2f}}<extra></extra>",
        ))
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C4B5FD", size=12),
        margin=dict(l=10, r=10, t=10, b=30),
        height=260,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(showgrid=False, linecolor="rgba(167,139,250,0.2)", tickfont=dict(size=11, color="#9E8EC5"), title="Tenure (months)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(167,139,250,0.12)", linecolor="rgba(167,139,250,0.2)",
                   tickfont=dict(size=11, color="#9E8EC5"), title="Monthly Charges ($)"),
        hoverlabel=dict(bgcolor="#1A0B2E", font_size=12, font_family="Inter"),
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Tenure vs Monthly Charges</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-sub'>Customer scatter by tenure and spend</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# --- Churn Rate by Internet Service (vertical bar) ---
with r2c:
    inet_churn = (
        df.groupby("InternetService")["Churn_Binary"]
        .mean()
        .reset_index()
        .rename(columns={"Churn_Binary": "ChurnRate"})
        .sort_values("ChurnRate")
    )
    inet_churn["ChurnPct"] = (inet_churn["ChurnRate"] * 100).round(1)

    color_map = {"No": "#A78BFA", "DSL": "#6C63FF", "Fiber optic": "#EF4444"}
    bar_colors = [color_map.get(v, "#6C63FF") for v in inet_churn["InternetService"]]

    fig_inet = go.Figure(go.Bar(
        x=inet_churn["InternetService"],
        y=inet_churn["ChurnPct"],
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in inet_churn["ChurnPct"]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>",
    ))
    fig_inet.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C4B5FD", size=12),
        margin=dict(l=10, r=10, t=10, b=30),
        height=260,
        showlegend=False,
        xaxis=dict(showgrid=False, linecolor="rgba(167,139,250,0.2)", tickfont=dict(size=11, color="#9E8EC5"), title=""),
        yaxis=dict(showgrid=True, gridcolor="rgba(167,139,250,0.12)", linecolor="rgba(167,139,250,0.2)",
                   tickfont=dict(size=11, color="#9E8EC5"), title="Churn Rate (%)",
                   range=[0, inet_churn["ChurnPct"].max() * 1.25]),
        hoverlabel=dict(bgcolor="#1A0B2E", font_size=12, font_family="Inter"),
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Churn Rate by Internet Service</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-sub'>Which service type drives the most churn?</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_inet, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── 11. CHART ROW 3 — Payment Method Bar + Risk Factors ──────────────────────
r3a, r3b = st.columns([1.5, 1])

# --- Churn Rate by Payment Method (horizontal bar with gradient color) ---
with r3a:
    pay_churn = (
        df.groupby("PaymentMethod")["Churn_Binary"]
        .mean()
        .reset_index()
        .rename(columns={"Churn_Binary": "ChurnRate"})
        .sort_values("ChurnRate", ascending=True)
    )
    pay_churn["ChurnPct"] = (pay_churn["ChurnRate"] * 100).round(1)
    n_pay = len(pay_churn)

    # Custom colorscale from #A78BFA (low) to #EF4444 (high)
    def lerp_color(t):
        r = int(167 + (239 - 167) * t)
        g = int(139 + (68  - 139) * t)
        b = int(250 + (68  - 250) * t)
        return f"rgb({r},{g},{b})"

    norm_vals = pay_churn["ChurnPct"].values
    v_min, v_max = norm_vals.min(), norm_vals.max()
    bar_colors_pay = [
        lerp_color((v - v_min) / (v_max - v_min + 1e-9))
        for v in norm_vals
    ]

    fig_pay = go.Figure(go.Bar(
        x=pay_churn["ChurnPct"],
        y=pay_churn["PaymentMethod"],
        orientation="h",
        marker=dict(color=bar_colors_pay, line=dict(width=0)),
        text=[f"  {v:.1f}%" for v in pay_churn["ChurnPct"]],
        textposition="outside",
        textfont=dict(size=11, color="#C4B5FD"),
        hovertemplate="<b>%{y}</b><br>Churn Rate: %{x:.1f}%<extra></extra>",
    ))
    max_x = pay_churn["ChurnPct"].max()
    fig_pay.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#C4B5FD", size=12),
        margin=dict(l=10, r=70, t=10, b=30),
        height=290,
        showlegend=False,
        xaxis=dict(showgrid=False, showline=False, zeroline=False,
                   tickfont=dict(size=11, color="#9E8EC5"), title="Churn Rate (%)",
                   range=[0, max_x * 1.3]),
        yaxis=dict(showgrid=False, showline=False, tickfont=dict(size=11, color="#9E8EC5"), title=""),
        hoverlabel=dict(bgcolor="#0F061A", font_size=12, font_family="Inter", font_color="#E2D9F3"),
    )
    st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
    st.markdown("<div class='chart-title'>Churn Rate by Payment Method</div>", unsafe_allow_html=True)
    st.markdown("<div class='chart-sub'>Sorted descending · purple = lower · red = higher risk</div>", unsafe_allow_html=True)
    st.plotly_chart(fig_pay, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# --- Risk Factors (pure HTML card) ---
with r3b:
    risk_factors = [
        {
            "name": "Month-to-month contract",
            "level": "HIGH",
            "row_bg": "rgba(239,68,68,0.12)",
            "badge_bg": "rgba(239,68,68,0.15)",
            "badge_color": "#FCA5A5",
            "border_color": "#EF4444",
        },
        {
            "name": "Fiber optic service",
            "level": "HIGH",
            "row_bg": "#FEF2F2",
            "badge_bg": "#FEF2F2",
            "badge_color": "#EF4444",
            "border_color": "#EF4444",
        },
        {
            "name": "Tenure 0–12 months",
            "level": "MED",
            "row_bg": "#FFFBEB",
            "badge_bg": "#FFFBEB",
            "badge_color": "#D97706",
            "border_color": "#F59E0B",
        },
        {
            "name": "Electronic check payment",
            "level": "MED",
            "row_bg": "#FFFBEB",
            "badge_bg": "#FFFBEB",
            "badge_color": "#D97706",
            "border_color": "#F59E0B",
        },
        {
            "name": "Senior citizen",
            "level": "LOW",
            "row_bg": "#F0FDF4",
            "badge_bg": "#F0FDF4",
            "badge_color": "#10B981",
            "border_color": "#10B981",
        },
    ]

    rows_html = ""
    for rf in risk_factors:
        rows_html += f"""
        <div class="risk-row" style="background:{rf['row_bg']}; border-left:3px solid {rf['border_color']};">
            <span style="font-size:12px; color:#E2D9F3;">{rf['name']}</span>
            <span class="risk-badge" style="background:{rf['badge_bg']}; color:{rf['badge_color']};">{rf['level']}</span>
        </div>
        """

    st.markdown(
        f"""
        <div class="chart-wrap" style="height:100%; min-height:290px;">
            <div style="font-size:14px; font-weight:700; color:#1E1B4B; margin-bottom:10px;">Risk Factors</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

# ── 12. DATA TABLE ────────────────────────────────────────────────────────────
st.markdown(
    "<div class='chart-wrap' style='margin-bottom:8px;'>",
    unsafe_allow_html=True,
)
st.markdown("<div class='chart-title'>Filtered Customer Data</div>", unsafe_allow_html=True)

display_cols = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "Contract", "InternetService", "MonthlyCharges", "TotalCharges",
    "PaymentMethod", "Churn",
]
display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(
    df[display_cols],
    use_container_width=True,
    height=220,
    hide_index=True,
)
st.markdown(
    f"<p style='font-size:11px; color:#9CA3AF; margin-top:6px;'>"
    f"Showing {n_total:,} of 7,043 records</p>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# ── 13. FOOTER ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="dash-footer">
        Telco Churn Analytics Dashboard · Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
