from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CareerPulse | Placement Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM UI DESIGN
# ==========================================================

CUSTOM_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1020;
    color: #e8edf7;
}

[data-testid="stSidebar"] {
    background: #111a2e;
    border-right: 1px solid #26334e;
}

[data-testid="stHeader"] {
    background: rgba(11,16,32,0.88);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f7ff;
}

.hero {
    background: linear-gradient(
        120deg,
        #172849 0%,
        #18213d 52%,
        #27204b 100%
    );
    border: 1px solid #34466b;
    border-radius: 20px;
    padding: 25px 28px;
    margin-bottom: 18px;
    box-shadow: 0 14px 40px rgba(0,0,0,.18);
}

.hero-kicker {
    color: #8fb5ff;
    text-transform: uppercase;
    letter-spacing: .16em;
    font-size: .72rem;
    font-weight: 700;
}

.hero-title {
    font-family: 'Space Grotesk';
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin: 5px 0 4px 0;
}

.hero-sub {
    color: #b7c4dc;
    font-size: .96rem;
}

.metric-card {
    background: linear-gradient(145deg,#151f35,#121a2d);
    border: 1px solid #2a3957;
    border-radius: 15px;
    padding: 16px 18px;
    min-height: 112px;
}

.metric-label {
    color: #9aabc8;
    font-size: .78rem;
    font-weight: 600;
}

.metric-value {
    color: #f5f8ff;
    font-family: 'Space Grotesk';
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 5px;
}

.metric-note {
    color: #8c9bb7;
    font-size: .72rem;
    margin-top: 2px;
}

div[data-testid="stTabs"] button {
    color: #aebbd3;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #8fb5ff;
    border-bottom-color: #6d9eff;
}

.stDownloadButton button,
.stButton button {
    border-radius: 10px;
    border: 1px solid #5277b9;
    background: #284a83;
    color: white;
    font-weight: 600;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #2a3957;
    border-radius: 12px;
}

hr {
    border-color: #293753;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PALETTE = [
    "#78a9ff", "#45d6b5", "#b69cff", "#ffbd69",
    "#ff7e91", "#6ed0e8", "#d6e27a", "#f08bd4"
]

# ==========================================================
# DATA LOADING
# ==========================================================

DEFAULT_FILE = (
    Path(__file__).parent /
    "student_placement_career_success_dataset.csv"
)


@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


def clean_data(data):

    data = data.copy()

    data.columns = [
        str(c).strip()
        for c in data.columns
    ]

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].replace(
            r"^\s*$", np.nan, regex=True
        )

    for col in data.select_dtypes(include=np.number).columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    if "placement_status" in data.columns:
        data["placement_status"] = (
            data["placement_status"]
            .astype("string")
            .str.strip()
        )

    return data


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("## 🎓 CareerPulse")

    st.caption("Student placement intelligence")

    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"]
    )

    if uploaded is not None:

        raw = load_csv(uploaded.getvalue())
        source_name = uploaded.name

    elif DEFAULT_FILE.exists():

        raw = load_csv(DEFAULT_FILE.read_bytes())
        source_name = DEFAULT_FILE.name

    else:

        raw = None
        source_name = "No file selected"

    st.markdown("---")

    st.markdown("### Dashboard controls")

    st.caption("Filters apply across analytical pages.")

    st.markdown("---")

    st.caption(
        "Built with Streamlit · Plotly · Scikit-learn"
    )


if raw is None:

    st.markdown("""
    <div class="hero">

        <div class="hero-kicker">
            Career Intelligence Platform
        </div>

        <div class="hero-title">
            Student Placement & Career Success
        </div>

        <div class="hero-sub">
            Explore placement outcomes, salary patterns,
            skills and career readiness.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Place student_placement_career_success_dataset.csv "
        "beside app.py or upload it using the sidebar."
    )

    st.stop()


df = clean_data(raw)

if "placement_status" not in df.columns:

    st.error(
        "The uploaded dataset must contain "
        "a placement_status column."
    )

    st.stop()


# ==========================================================
# GLOBAL FILTERS
# ==========================================================

filter_cols = [
    "state",
    "branch",
    "degree",
    "gender",
    "college_tier",
    "city_tier",
    "placement_status",
    "company_type",
    "work_mode"
]

filter_cols = [
    c for c in filter_cols
    if c in df.columns
]

filtered = df.copy()

with st.sidebar:

    for col in filter_cols:

        vals = sorted(
            filtered[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if vals and len(vals) <= 80:

            chosen = st.multiselect(
                col.replace("_", " ").title(),
                vals,
                default=vals,
                key=f"filter_{col}"
            )

            if chosen:

                filtered = filtered[
                    filtered[col].astype(str).isin(chosen)
                ]

            else:

                filtered = filtered.iloc[0:0]

    if st.button(
        "Reset filters",
        use_container_width=True
    ):
        st.rerun()


if filtered.empty:

    st.warning(
        "No records match the selected filters. "
        "Reset or broaden your filters."
    )

    st.stop()


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def numeric(col, frame=None):

    if frame is None:
        frame = filtered

    if col in frame.columns:

        return pd.to_numeric(
            frame[col],
            errors="coerce"
        )

    return pd.Series(dtype=float)


def is_placed(frame):

    return (
        frame["placement_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("placed")
    )


def fmt_lpa(value):

    if pd.notna(value):
        return f"₹{value:,.2f} LPA"

    return "—"


def metric(label, value, note=""):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-note">
                {note}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def chart_layout(fig, height=360):

    fig.update_layout(

        height=height,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#dce6f7",
            family="DM Sans"
        ),

        margin=dict(
            l=18,
            r=18,
            t=48,
            b=18
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),

        title_font=dict(
            size=15,
            color="#f4f7ff"
        )
    )

    fig.update_xaxes(
        gridcolor="#26344f",
        zerolinecolor="#26344f"
    )

    fig.update_yaxes(
        gridcolor="#26344f",
        zerolinecolor="#26344f"
    )

    return fig


def show_chart(fig):

    st.plotly_chart(
        chart_layout(fig),
        use_container_width=True,
        theme=None
    )


# ==========================================================
# KPI CALCULATIONS
# ==========================================================

placed_mask = is_placed(filtered)

placement_rate = placed_mask.mean() * 100

placed_df = filtered.loc[placed_mask]

salary = numeric(
    "salary_lpa",
    placed_df
).dropna()

avg_salary = (
    salary.mean()
    if len(salary)
    else np.nan
)

median_salary = (
    salary.median()
    if len(salary)
    else np.nan
)

unique_students = (
    filtered["student_id"].nunique()
    if "student_id" in filtered.columns
    else len(filtered)
)


# ==========================================================
# DASHBOARD HEADER
# ==========================================================

st.markdown("""
<div class="hero">

    <div class="hero-kicker">
        Placement Intelligence · Data Analytics
    </div>

    <div class="hero-title">
        Student Placement & Career Success
    </div>

    <div class="hero-sub">
        Understand outcomes. Discover skill gaps.
        Turn student data into career strategy.
    </div>

</div>
""", unsafe_allow_html=True)


st.caption(
    f"Source: {source_name} · "
    f"Showing {len(filtered):,} of {len(df):,} records · "
    f"{filtered.shape[1]} fields"
)


# ==========================================================
# KPI CARDS
# ==========================================================

k1, k2, k3, k4, k5 = st.columns(5)

with k1:

    metric(
        "STUDENTS",
        f"{unique_students:,}",
        "Unique student records"
    )

with k2:

    metric(
        "PLACEMENT RATE",
        f"{placement_rate:.2f}%",
        f"{int(placed_mask.sum()):,} placed"
    )

with k3:

    metric(
        "AVERAGE SALARY",
        fmt_lpa(avg_salary),
        "Placed students only"
    )

with k4:

    metric(
        "MEDIAN SALARY",
        fmt_lpa(median_salary),
        "Placed students only"
    )

with k5:

    offers = numeric("offer_count").dropna()

    metric(
        "TOTAL OFFERS",
        f"{offers.sum():,.0f}" if len(offers) else "—",
        "Across filtered records"
    )


# ==========================================================
# DASHBOARD NAVIGATION
# ==========================================================

pages = st.tabs([

    "✨ Overview",

    "📌 Placement",

    "💸 Salary",

    "🧠 Skills & Academics",

    "🤖 AI & Wellbeing",

    "🔎 Relationships",

    "👤 Student Explorer",

    "🧪 ML Lab"

])
