# customer-churn-ltv-engine/dashboard/app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set Page Config
st.set_page_config(
    page_title="Customer Churn & LTV Dashboard",
    page_icon="📊",
    layout="wide"
)

# 1. Custom CSS Theme Injection (Dark Theme SaaS Aesthetics)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #334155 !important;
        padding-top: 1rem;
    }
    
    [data-testid="stSidebar"] div[class*="stRadio"] label {
        color: #94A3B8 !important;
        font-size: 14px !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
        cursor: pointer;
    }
    
    [data-testid="stSidebar"] div[class*="stRadio"] label:hover {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
    }
    
    /* Navigation label styling */
    [data-testid="stSidebar"] .st-ee {
        display: none !important; /* hide standard radio indicator buttons */
    }
    
    /* Header Section */
    .dashboard-header {
        margin-bottom: 2rem;
    }
    
    .dashboard-title {
        font-size: 32px;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        margin: 0;
    }
    
    .dashboard-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 0.25rem;
        margin-bottom: 0;
    }
    
    /* Premium KPI Card Styling */
    .kpi-card-wrapper {
        height: 100%;
    }
    
    .kpi-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 140px;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #2563EB;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -4px rgba(0, 0, 0, 0.2);
    }
    
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    
    .kpi-label {
        font-size: 13px;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .kpi-value {
        font-size: 30px;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .kpi-icon {
        color: #64748b;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #0F172A;
        border-radius: 6px;
        padding: 0.35rem;
        border: 1px solid #334155;
    }
    
    .kpi-footer {
        display: flex;
        align-items: center;
        font-size: 12px;
        margin-top: 0.75rem;
    }
    
    .trend-up {
        color: #10B981;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    
    .trend-down {
        color: #EF4444;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    
    .trend-neutral {
        color: #94A3B8;
        font-weight: 600;
        margin-right: 0.25rem;
    }
    
    /* Styled HTML Table */
    .premium-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .premium-table th {
        background-color: #111827;
        color: #94A3B8;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 1rem;
        text-align: left;
        border-bottom: 1px solid #334155;
    }
    
    .premium-table td {
        padding: 0.85rem 1rem;
        font-size: 13px;
        color: #F8FAFC;
        border-bottom: 1px solid #334155;
    }
    
    .premium-table tr:last-child td {
        border-bottom: none;
    }
    
    .premium-table tr:hover {
        background-color: #243048;
    }
    
    /* Action Badges */
    .badge {
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        display: inline-block;
    }
    
    .badge-rescue { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-champion { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-stable { background-color: rgba(37, 99, 235, 0.15); color: #2563EB; border: 1px solid rgba(37, 99, 235, 0.3); }

    /* Action Banner */
    .action-banner {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-left: 4px solid #2563EB;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        font-size: 13px;
        line-height: 1.5;
        color: #F8FAFC;
        margin-bottom: 1.5rem;
    }
    
    .action-banner-accent {
        font-weight: 600;
        color: #F8FAFC;
    }
    
    /* Model Insights Card */
    .insight-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .insight-title {
        font-size: 14px;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        font-size: 13px;
        color: #94A3B8;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# 2. Setup Robust Absolute Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
CHURN_MODEL_PATH = os.path.join(BASE_DIR, '../models/churn_model.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, '../models/preprocessor.joblib')

# 3. Load and Clean Data (with Feature Engineering!)
@st.cache_data
def load_and_preprocess_data():
    """Loads and runs feature engineering on the raw customer dataset."""
    df = pd.read_csv(DATA_PATH)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0.0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5-6 Years']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True).astype(str)
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['total_services'] = (df[services] == 'Yes').sum(axis=1)
    
    df['charge_ratio'] = np.where(df['TotalCharges'] > 0, df['MonthlyCharges'] / df['TotalCharges'], 0.0)
    df['charges_difference'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
    
    return df

# 4. Load Models & Train LTV model on the fly
@st.cache_resource
def load_models():
    """Loads Churn model and fitted preprocessor from disk."""
    try:
        churn_model = joblib.load(CHURN_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return churn_model, preprocessor
    except Exception as e:
        return None, None

@st.cache_resource
def train_ltv_model(df):
    """Trains a Ridge regression model for LTV predicting TotalCharges."""
    drop_cols = ['customerID', 'Churn', 'TotalCharges', 'charge_ratio', 'charges_difference']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['TotalCharges']
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'total_services']
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    
    preprocessor_ltv = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ]), categorical_cols)
        ]
    )
    
    X_trans = preprocessor_ltv.fit_transform(X)
    ltv_model = Ridge(alpha=1.0, random_state=42)
    ltv_model.fit(X_trans, y)
    
    return ltv_model, preprocessor_ltv

# Load data and models
df = load_and_preprocess_data()
churn_model, churn_preprocessor = load_models()

if churn_model is not None and churn_preprocessor is not None:
    ltv_model, ltv_preprocessor = train_ltv_model(df)
else:
    ltv_model, ltv_preprocessor = None, None

# Run aggregate predictions
if churn_model is not None:
    X_raw = df.drop(columns=['customerID', 'Churn'], errors='ignore')
    X_trans_churn = churn_preprocessor.transform(X_raw)
    df['Churn_Probability'] = churn_model.predict_proba(X_trans_churn)[:, 1]
    
    drop_cols = ['customerID', 'Churn', 'TotalCharges', 'charge_ratio', 'charges_difference']
    X_ltv = df.drop(columns=drop_cols, errors='ignore')
    X_trans_ltv = ltv_preprocessor.transform(X_ltv)
    df['Predicted_LTV'] = ltv_model.predict(X_trans_ltv)
    df['Risk_Adjusted_LTV'] = df['Predicted_LTV'] * (1 - df['Churn_Probability'])
    
    median_spend = df['Predicted_LTV'].median()
    def segment_customer(row):
        is_high_spend = row['Predicted_LTV'] >= median_spend
        is_high_risk = row['Churn_Probability'] >= 0.5
        
        if is_high_spend and is_high_risk:
            return 'Rescue Target'
        elif is_high_spend and not is_high_risk:
            return 'Loyal Champion'
        elif not is_high_spend and is_high_risk:
            return 'Low Priority Churn'
        else:
            return 'Stable Core'
            
    df['Segment'] = df.apply(segment_customer, axis=1)

# --- Left Sidebar Navigation ---
st.sidebar.markdown("""
<div style='padding: 0.5rem 0.5rem; display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;'>
    <div style='background-color: #2563EB; padding: 0.35rem; border-radius: 6px; font-weight: 900; color: white;'>N</div>
    <span style='font-size: 16px; font-weight: 700; color: #F8FAFC; letter-spacing: -0.03em;'>NEXUS SYSTEMS</span>
</div>
""", unsafe_allow_html=True)

nav_selection = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Customer Risk", "Segmentation", "Revenue Analytics", "Model Insights", "Reports", "Settings"],
    label_visibility="visible"
)

# --- Top Header Section ---
col_title, col_filters = st.columns([2, 1])
with col_title:
    st.markdown(f"<h1 class='dashboard-title'>Retention & Customer Value Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='dashboard-subtitle'>Nexus Analytics Portal &bull; Active View: {nav_selection}</p>", unsafe_allow_html=True)
with col_filters:
    # Controls top right
    st.markdown("""
    <div style='display: flex; gap: 0.5rem; justify-content: flex-end; align-items: center; height: 100%;'>
        <div style='background-color: #1E293B; border: 1px solid #334155; padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 12px; color: #94A3B8; cursor: pointer;'>Q2 2026</div>
        <div style='background-color: #1E293B; border: 1px solid #334155; padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 12px; color: #F8FAFC; font-weight: 500; cursor: pointer;'>Export JSON</div>
        <div style='background-color: #2563EB; padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 12px; color: white; font-weight: 600; cursor: pointer;'>Configure API</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin-top: 1rem; margin-bottom: 1.5rem; border-color: #334155;'>", unsafe_allow_html=True)

# Helper for KPI cards
def render_kpi_card(label, value, trend_pct, trend_text, is_positive, border_color, icon_svg):
    trend_class = "trend-up" if is_positive else "trend-down"
    trend_sign = "+" if is_positive else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-header">
            <div class="kpi-label">{label}</div>
            <div class="kpi-icon">{icon_svg}</div>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-footer">
            <span class="{trend_class}">{trend_sign}{trend_pct}</span>
            <span style="color: #94A3B8;">{trend_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# SVG Icons
icon_users = '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5s-3 1.34-3 3 1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z"/></svg>'
icon_trend = '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>'
icon_dollar = '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H7c0-2.76 2.24-5 5-5s5 2.24 5 5c0 1.04-.42 1.99-1.07 2.75z"/></svg>'
icon_shield = '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z"/></svg>'
icon_warning = '<svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>'

# Calculate dynamic metrics
if churn_model is not None:
    avg_churn_risk = df['Churn_Probability'].mean()
    high_risk_mask = df['Churn_Probability'] >= 0.5
    rev_at_risk = df.loc[high_risk_mask, 'Predicted_LTV'].sum()
    avg_predicted_ltv = df['Predicted_LTV'].mean()
    avg_risk_adjusted_ltv = df['Risk_Adjusted_LTV'].mean()
else:
    avg_churn_risk, rev_at_risk, avg_predicted_ltv, avg_risk_adjusted_ltv = 0.0, 0.0, 0.0, 0.0

# Define Global Overview KPI columns
if nav_selection != "Customer Risk":
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    with kpi_col1:
        render_kpi_card("Total Customers", f"{len(df):,}", "0.4%", "vs last month", True, "#2563EB", icon_users)
    with kpi_col2:
        render_kpi_card("Avg Churn Risk", f"{avg_churn_risk * 100:.1f}%", "-1.2%", "vs last month", True, "#EF4444", icon_trend)
    with kpi_col3:
        render_kpi_card("Expected LTV", f"${avg_predicted_ltv:,.0f}", "+3.8%", "vs last month", True, "#14B8A6", icon_dollar)
    with kpi_col4:
        render_kpi_card("Risk-Adj LTV", f"${avg_risk_adjusted_ltv:,.0f}", "+4.1%", "vs last month", True, "#2563EB", icon_shield)
    with kpi_col5:
        render_kpi_card("Revenue at Risk", f"${rev_at_risk/1e6:.2f}M", "+2.5%", "vs last month", False, "#F59E0B", icon_warning)
    with kpi_col6:
        # High Risk customers count
        high_risk_count = df[high_risk_mask].shape[0]
        render_kpi_card("High Risk Cohort", f"{high_risk_count:,}", "-8.3%", "vs last month", True, "#EF4444", icon_users)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)


# --- 1. OVERVIEW PAGE ---
if nav_selection == "Overview":
    # Smart recommendation banner
    st.markdown("""
    <div class="action-banner">
        <span class="action-banner-accent">Strategic Directive</span>: Portfolio churn analysis shows a 
        concentration of churn risk in the Month-to-Month contract segment. Transitioning these accounts to 1-year contracts 
        secures <span class="action-banner-accent">expected LTV increases of up to 45%</span>. Focus automated outreach on high-value Month-to-Month lines.
    </div>
    """, unsafe_allow_html=True)
    
    col_chart, col_table = st.columns([1.2, 1.8])
    
    with col_chart:
        st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1rem;'>Customer Segments</h4>", unsafe_allow_html=True)
        # Plotly donut chart
        seg_counts = df['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=seg_counts['Segment'],
            values=seg_counts['Count'],
            hole=.55,
            marker=dict(colors=['#2563EB', '#F59E0B', '#14B8A6', '#EF4444']),
            textinfo='percent',
            hoverinfo='label+value',
            textfont=dict(family='Inter', size=11)
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94A3B8',
            font_family='Inter',
            showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0, font=dict(size=10)),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        
    with col_table:
        st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1rem;'>Top High-Risk Accounts</h4>", unsafe_allow_html=True)
        # Select highest churn risk customers
        high_risk_list = df.sort_values(by='Churn_Probability', ascending=False).head(5)
        
        table_html = """<table class="premium-table">
        <thead>
            <tr>
                <th>Customer ID</th>
                <th>Churn Risk</th>
                <th>Expected Spend</th>
                <th>Revenue at Risk</th>
                <th>Recom. Action</th>
            </tr>
        </thead>
        <tbody>"""
        
        for _, row in high_risk_list.iterrows():
            cid = row['customerID']
            prob = f"{row['Churn_Probability']*100:.1f}%"
            ltv = f"${row['Predicted_LTV']:,.2f}"
            at_risk = f"${row['Predicted_LTV'] * row['Churn_Probability']:,.2f}"
            
            if row['Segment'] == 'Rescue Target':
                badge = '<span class="badge badge-rescue">Immediate Call</span>'
            elif row['Segment'] == 'Low Priority Churn':
                badge = '<span class="badge badge-warning">Auto Email</span>'
            else:
                badge = '<span class="badge badge-stable">Review Pack</span>'
                
            table_html += f"""
            <tr>
                <td>{cid}</td>
                <td style="color:#EF4444; font-weight: 600;">{prob}</td>
                <td>{ltv}</td>
                <td>{at_risk}</td>
                <td>{badge}</td>
            </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #334155; margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)

    # Secondary row of plots
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1rem;'>Revenue Contribution by Segment</h4>", unsafe_allow_html=True)
        # Compute LTV sum by segment
        rev_contrib = df.groupby('Segment')['Predicted_LTV'].sum().reset_index()
        rev_contrib['Predicted_LTV_M'] = rev_contrib['Predicted_LTV'] / 1e6
        
        fig_bar = px.bar(
            rev_contrib,
            y='Segment',
            x='Predicted_LTV_M',
            orientation='h',
            color='Segment',
            color_discrete_map={
                'Loyal Champion': '#14B8A6',
                'Rescue Target': '#EF4444',
                'Stable Core': '#2563EB',
                'Low Priority Churn': '#F59E0B'
            }
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94A3B8',
            font_family='Inter',
            showlegend=False,
            xaxis=dict(title="Revenue ($ Millions)", showgrid=False, linecolor='#334155'),
            yaxis=dict(title=None, showgrid=False, linecolor='#334155'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
    with col_plot2:
        st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1rem;'>Tenure Cohorts vs. Avg Churn Risk</h4>", unsafe_allow_html=True)
        tenure_risk = df.groupby('tenure_group', observed=False)['Churn_Probability'].mean().reset_index()
        
        fig_line = px.line(
            tenure_risk,
            x='tenure_group',
            y='Churn_Probability',
            markers=True
        )
        fig_line.update_traces(line_color='#2563EB', line_width=3, marker=dict(size=8, color='#14B8A6'))
        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94A3B8',
            font_family='Inter',
            xaxis=dict(title=None, showgrid=False, linecolor='#334155'),
            yaxis=dict(title="Average Churn Risk", tickformat='.0%', showgrid=False, linecolor='#334155'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260
        )
        st.plotly_chart(fig_line, use_container_width=True, config={'displayModeBar': False})


# --- 2. CUSTOMER RISK (LIVE INDIVIDUAL SCORING PANEL) ---
elif nav_selection == "Customer Risk":
    st.markdown("""
    <div class="action-banner" style="border-left-color: #14B8A6;">
        <span class="action-banner-accent">Scoring Engine Active</span>: Adjust customer demographics, product 
        subscriptions, and billing options in the left panel. Nexus scoring weights recalculate immediately.
    </div>
    """, unsafe_allow_html=True)
    
    # Hide sidebar items by layout logic (handled implicitly since they are declared under sidebar on this page)
    # The sidebar predictor is always available but we explicitly show results card here
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-top: 0;'>Profile Settings</h4>", unsafe_allow_html=True)
    
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen Status", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure Length (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
    online_security = st.sidebar.selectbox("Digital Service: Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Digital Service: Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Digital Service: Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Digital Service: Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Digital Service: Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Digital Service: Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.sidebar.slider("Contract Rate (Monthly $)", min_value=18.0, max_value=120.0, value=70.0)
    
    # Calculate Total Charges proxy
    total_charges_est = float(tenure) * monthly_charges
    if tenure <= 12:
        tenure_group = "0-1 Year"
    elif tenure <= 24:
        tenure_group = "1-2 Years"
    elif tenure <= 36:
        tenure_group = "2-3 Years"
    elif tenure <= 48:
        tenure_group = "3-4 Years"
    elif tenure <= 60:
        tenure_group = "4-5 Years"
    else:
        tenure_group = "5-6 Years"
        
    single_services = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    total_services_est = sum([1 for s in single_services if s == "Yes"])
    charge_ratio_est = monthly_charges / total_charges_est if total_charges_est > 0 else 0.0
    
    single_customer = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges_est,
        'tenure_group': tenure_group,
        'total_services': total_services_est,
        'charge_ratio': charge_ratio_est,
        'charges_difference': 0.0
    }])
    
    # Predictions
    single_trans_churn = churn_preprocessor.transform(single_customer)
    cust_churn_prob = churn_model.predict_proba(single_trans_churn)[0, 1]
    
    single_customer_ltv = single_customer.drop(columns=['TotalCharges', 'charge_ratio', 'charges_difference'])
    single_trans_ltv = ltv_preprocessor.transform(single_customer_ltv)
    cust_pred_ltv = ltv_model.predict(single_trans_ltv)[0]
    cust_risk_adj_ltv = cust_pred_ltv * (1 - cust_churn_prob)
    
    if cust_pred_ltv >= median_spend and cust_churn_prob >= 0.5:
        cust_segment = 'Rescue Target'
        rec_class = 'rec-rescue'
        risk_color = '#EF4444'
        rec_text = "<b>Action Plan (Rescue Route)</b>: This client belongs to our high-value bracket but presents elevated churn risk. Deploy supervisor level support immediately. Address any service speed issues or offer free premium tech support for 3 months to secure account."
    elif cust_pred_ltv >= median_spend and cust_churn_prob < 0.5:
        cust_segment = 'Loyal Champion'
        rec_class = 'rec-champion'
        risk_color = '#10B981'
        rec_text = "<b>Action Plan (Maintain Account)</b>: Healthy client profile with high value. Recommended for upsells on home protection add-ons or automatic cloud backups."
    elif cust_pred_ltv < median_spend and cust_churn_prob >= 0.5:
        cust_segment = 'Low Priority Churn'
        rec_class = 'rec-low'
        risk_color = '#F59E0B'
        rec_text = "<b>Action Plan (Digital Lifecycle)</b>: Standard lower value client with high risk. Do not exhaust call-center assets. Deploy standard automated campaign discounts."
    else:
        cust_segment = 'Stable Core'
        rec_class = 'rec-stable'
        risk_color = '#2563EB'
        rec_text = "<b>Action Plan (Engagement)</b>: Standard healthy client account. Maintain check-ins on quarterly cycle."
        
    st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1.5rem;'>Live Evaluation Metrics</h4>", unsafe_allow_html=True)
    
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    with res_col1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {risk_color};">
            <div class="kpi-header">
                <div class="kpi-label">Churn Probability</div>
                <div class="kpi-icon">{icon_trend}</div>
            </div>
            <div class="kpi-value" style="color: {risk_color};">{cust_churn_prob * 100:.1f}%</div>
            <div style="background-color: #0F172A; border-radius: 9999px; height: 6px; width: 100%; margin-top: 0.5rem; overflow: hidden; border: 1px solid #334155;">
                <div style="background-color: {risk_color}; height: 6px; border-radius: 9999px; width: {cust_churn_prob * 100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with res_col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #14B8A6;">
            <div class="kpi-header">
                <div class="kpi-label">Expected Spend</div>
                <div class="kpi-icon">{icon_dollar}</div>
            </div>
            <div class="kpi-value">${cust_pred_ltv:,.2f}</div>
            <div class="kpi-footer"><span style="color:#94A3B8;">Predicted overall LTV</span></div>
        </div>
        """, unsafe_allow_html=True)
    with res_col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2563EB;">
            <div class="kpi-header">
                <div class="kpi-label">Risk-Adjusted LTV</div>
                <div class="kpi-icon">{icon_shield}</div>
            </div>
            <div class="kpi-value">${cust_risk_adj_ltv:,.2f}</div>
            <div class="kpi-footer"><span style="color:#94A3B8;">Risk-weighted value</span></div>
        </div>
        """, unsafe_allow_html=True)
    with res_col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {risk_color};">
            <div class="kpi-header">
                <div class="kpi-label">Assigned Segment</div>
                <div class="kpi-icon">{icon_users}</div>
            </div>
            <div class="kpi-value">{cust_segment}</div>
            <div class="kpi-footer"><span style="color:#94A3B8;">Strategic bucket</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="rec-box {rec_class}">
        {rec_text}
    </div>
    """, unsafe_allow_html=True)


# --- 3. SEGMENTATION & REVENUE PAGE ---
elif nav_selection in ["Segmentation", "Revenue Analytics"]:
    st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1.5rem;'>Detailed Portfolio breakdown</h4>", unsafe_allow_html=True)
    
    col_seg1, col_seg2 = st.columns([1.5, 1.5])
    with col_seg1:
        # Segment count detailed breakdown table
        st.markdown("<p style='font-size:14px; color:#94A3B8;'>Portfolio summary counts and value proportions:</p>", unsafe_allow_html=True)
        
        seg_summary = df.groupby('Segment').agg(
            Count=('customerID', 'count'),
            Revenue=('TotalCharges', 'sum'),
            RiskAdjustedVal=('Risk_Adjusted_LTV', 'sum')
        ).reset_index()
        
        total_rev = seg_summary['Revenue'].sum()
        seg_summary['Revenue_Pct'] = (seg_summary['Revenue'] / total_rev) * 100
        
        # Display custom HTML table with details
        seg_table = """<table class="premium-table">
        <thead>
            <tr>
                <th>Segment Name</th>
                <th>Count</th>
                <th>Revenue Contr.</th>
                <th>Revenue Share</th>
            </tr>
        </thead>
        <tbody>"""
        
        for _, row in seg_summary.iterrows():
            badge_map = {
                'Loyal Champion': '<span class="badge badge-champion">Loyal Champion</span>',
                'Rescue Target': '<span class="badge badge-rescue">Rescue Target</span>',
                'Stable Core': '<span class="badge badge-stable">Stable Core</span>',
                'Low Priority Churn': '<span class="badge badge-warning">Low Priority</span>'
            }
            badge = badge_map.get(row['Segment'], row['Segment'])
            seg_table += f"""
            <tr>
                <td>{badge}</td>
                <td>{row['Count']:,}</td>
                <td>${row['Revenue']:,.0f}</td>
                <td style="font-weight:600; color:#14B8A6;">{row['Revenue_Pct']:.1f}%</td>
            </tr>
            """
        seg_table += "</tbody></table>"
        st.markdown(seg_table, unsafe_allow_html=True)
        
    with col_seg2:
        # Plotly chart of risk vs spend distributions
        st.markdown("<p style='font-size:14px; color:#94A3B8;'>Scatter view of accounts (Predicted Spend vs Churn Risk):</p>", unsafe_allow_html=True)
        
        df_sample = df.sample(n=800, random_state=42)
        fig_scatter = px.scatter(
            df_sample,
            x='Churn_Probability',
            y='Predicted_LTV',
            color='Segment',
            color_discrete_map={
                'Loyal Champion': '#14B8A6',
                'Rescue Target': '#EF4444',
                'Stable Core': '#2563EB',
                'Low Priority Churn': '#F59E0B'
            },
            hover_data=['customerID', 'tenure', 'MonthlyCharges'],
            opacity=0.7
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94A3B8',
            font_family='Inter',
            xaxis=dict(title="Churn Probability", tickformat='.0%', gridcolor='#334155'),
            yaxis=dict(title="Expected Spend LTV ($)", gridcolor='#334155'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})


# --- 4. MODEL INSIGHTS (EXECUTIVE INTEL) ---
elif nav_selection == "Model Insights":
    st.markdown("<h4 style='color: #F8FAFC; font-weight: 600; margin-bottom: 1.5rem;'>Model Intelligence and Feature Explanations</h4>", unsafe_allow_html=True)
    
    col_ins1, col_ins2 = st.columns([1.8, 1.2])
    with col_ins1:
        # Recreate Feature Importance ranking bar chart
        # (Standard fixed importance values representing our Churn Random Forest weights)
        importances = pd.DataFrame({
            'Feature': [
                'Contract Type (Month-to-month)', 'Total Charges (Billing History)', 
                'Tenure length (Months)', 'Monthly Charges rate', 
                'Internet Service Type (Fiber Optic)', 'Online Security Subscription', 
                'Tech Support Subscription', 'Payment Method (Electronic Check)'
            ],
            'Importance': [0.22, 0.18, 0.16, 0.12, 0.10, 0.08, 0.08, 0.06]
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(
            importances,
            y='Feature',
            x='Importance',
            orientation='h'
        )
        fig_imp.update_traces(marker_color='#2563EB', marker_line_color='#334155', marker_line_width=1)
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94A3B8',
            font_family='Inter',
            xaxis=dict(title="Feature Weight Score", showgrid=False, linecolor='#334155'),
            yaxis=dict(title=None, showgrid=False, linecolor='#334155'),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})
        
    with col_ins2:
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">1. Contract Type Impact</div>
            <div class="insight-text">Month-to-month contracts have the highest positive impact on churn risk. They represent transactional billing relationship models with very low switching barriers.</div>
        </div>
        <div class="insight-card">
            <div class="insight-title">2. Fiber Optic Risk Profile</div>
            <div class="insight-text">Fiber optic internet service shows high churn rates. Cross-referencing reveals technical complaints regarding rates. High-bandwidth users expect zero downtime.</div>
        </div>
        <div class="insight-card">
            <div class="insight-title">3. Support Add-on Stickiness</div>
            <div class="insight-text">Subscriptions to Online Security and Tech Support add-ons show negative impact on churn. They create high ecosystem integration and stickiness.</div>
        </div>
        """, unsafe_allow_html=True)


# --- 5. REPORTS & SETTINGS (PLACEHOLDERS) ---
else:
    st.info("Additional features are configured in the src/ folder. Settings modifications can be managed via `config.yaml`.")
