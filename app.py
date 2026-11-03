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

# ==========================================================
# OVERVIEW PAGE
# ==========================================================

with pages[0]:

    left, right = st.columns([1.25, 1])

    with left:

        st.subheader("Placement outcome snapshot")

        counts = (
            filtered["placement_status"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Students")
        )

        fig = px.pie(
            counts,
            names="Outcome",
            values="Students",
            hole=0.68,
            color_discrete_sequence=[
                "#45d6b5",
                "#ff7e91",
                "#71809d"
            ],
            title="Student placement status"
        )

        fig.update_traces(
            textposition="outside",
            textinfo="percent+label"
        )

        show_chart(fig)

    with right:

        st.subheader("Student profile")

        for col, title in [
            ("branch", "Branch mix"),
            ("degree", "Degree mix"),
            ("college_tier", "College tier mix")
        ]:

            if col in filtered.columns:

                temp = (
                    filtered[col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .head(8)
                    .reset_index()
                )

                temp.columns = [title, "Students"]

                fig = px.bar(
                    temp,
                    x="Students",
                    y=title,
                    orientation="h",
                    color="Students",
                    color_continuous_scale="Tealgrn",
                    title=title
                )

                fig.update_layout(
                    showlegend=False,
                    coloraxis_showscale=False
                )

                show_chart(fig)

    # Quick insights

    st.subheader("Quick insights")

    insights = []

    if "branch" in filtered.columns:

        b = (
            filtered.assign(_placed=placed_mask)
            .groupby("branch")["_placed"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )

        if len(b):

            insights.append(
                f"Highest observed placement rate by branch: "
                f"**{b.index[0]} ({b.iloc[0]*100:.1f}%)**."
            )

    if len(salary):

        insights.append(
            f"Placed-student salary spans "
            f"**{salary.min():.2f}–{salary.max():.2f} LPA** "
            f"in the current selection."
        )

    if "internships_completed" in filtered.columns:

        med = (
            filtered
            .groupby(placed_mask)["internships_completed"]
            .median()
        )

        if True in med.index and False in med.index:

            insights.append(
                f"Median internships: "
                f"**{med.loc[True]:.1f}** among placed vs "
                f"**{med.loc[False]:.1f}** among not placed."
            )

    if not insights:

        insights.append(
            "Use the filters to compare student cohorts "
            "and explore the available measures."
        )

    for item in insights:

        st.markdown(f"- {item}")


# ==========================================================
# PLACEMENT ANALYTICS
# ==========================================================

with pages[1]:

    st.subheader("Placement performance")

    c1, c2 = st.columns(2)

    # Placement by branch

    with c1:

        if "branch" in filtered.columns:

            tmp = (
                filtered.assign(_placed=placed_mask)
                .groupby("branch")
                .agg(
                    students=("_placed", "size"),
                    rate=("_placed", "mean")
                )
                .reset_index()
            )

            tmp["rate"] *= 100

            sorted_tmp = tmp.sort_values("rate")

            fig = px.bar(
                sorted_tmp,
                x="rate",
                y="branch",
                orientation="h",
                color="rate",
                color_continuous_scale="Tealgrn",
                text=sorted_tmp["rate"].map(
                    lambda x: f"{x:.1f}%"
                ),
                title="Placement rate by branch"
            )

            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Placement rate (%)",
                yaxis_title=""
            )

            show_chart(fig)

    # Placement by college tier

    with c2:

        group_col = (
            "college_tier"
            if "college_tier" in filtered.columns
            else "degree"
        )

        if group_col in filtered.columns:

            tmp = (
                filtered.assign(_placed=placed_mask)
                .groupby(group_col)["_placed"]
                .mean()
                .mul(100)
                .reset_index(name="Placement rate")
            )

            fig = px.bar(
                tmp,
                x=group_col,
                y="Placement rate",
                color=group_col,
                color_discrete_sequence=PALETTE,
                title=f"Placement rate by {group_col.replace('_',' ')}"
            )

            show_chart(fig)

    # State and company analytics

    c3, c4 = st.columns(2)

    with c3:

        if "state" in filtered.columns:

            tmp = (
                filtered.assign(_placed=placed_mask)
                .groupby("state")
                .agg(
                    students=("_placed", "size"),
                    rate=("_placed", "mean")
                )
                .reset_index()
            )

            tmp["rate"] *= 100

            tmp = tmp.sort_values(
                "students",
                ascending=False
            ).head(12)

            fig = px.bar(
                tmp,
                x="state",
                y="rate",
                hover_data=["students"],
                title="Placement rate across leading states",
                color="rate",
                color_continuous_scale="Blues"
            )

            fig.update_layout(
                xaxis_tickangle=-35,
                coloraxis_showscale=False,
                yaxis_title="Placement rate (%)"
            )

            show_chart(fig)

    with c4:

        if "company_type" in filtered.columns:

            tmp = (
                filtered[
                    filtered["company_type"].notna()
                ]
                .groupby("company_type")
                .size()
                .reset_index(name="Students")
            )

            fig = px.pie(
                tmp,
                names="company_type",
                values="Students",
                hole=0.55,
                title="Company type distribution",
                color_discrete_sequence=PALETTE
            )

            show_chart(fig)

    st.markdown("**Outcome counts**")

    st.dataframe(
        filtered["placement_status"]
        .value_counts(dropna=False)
        .rename_axis("Placement status")
        .reset_index(name="Students"),
        use_container_width=True,
        hide_index=True
    )


# ==========================================================
# SALARY ANALYTICS
# ==========================================================

with pages[2]:

    st.subheader("Compensation intelligence")

    if "salary_lpa" not in filtered.columns:

        st.info("No salary_lpa field in this dataset.")

    else:

        salary_df = filtered.loc[placed_mask].copy()

        salary_df["salary_lpa"] = pd.to_numeric(
            salary_df["salary_lpa"],
            errors="coerce"
        )

        salary_df = salary_df[
            salary_df["salary_lpa"] > 0
        ]

        if salary_df.empty:

            st.info(
                "No positive salary records in the current selection."
            )

        else:

            c1, c2 = st.columns(2)

            # Salary distribution

            with c1:

                fig = px.histogram(
                    salary_df,
                    x="salary_lpa",
                    nbins=35,
                    marginal="box",
                    title="Salary distribution (placed students)",
                    color_discrete_sequence=["#78a9ff"]
                )

                fig.update_xaxes(
                    title="Salary (LPA)"
                )

                show_chart(fig)

            # Salary by category

            with c2:

                cat = next(
                    (
                        c for c in [
                            "company_type",
                            "work_mode",
                            "branch",
                            "degree"
                        ]
                        if c in salary_df.columns
                    ),
                    None
                )

                if cat:

                    fig = px.box(
                        salary_df,
                        x=cat,
                        y="salary_lpa",
                        color=cat,
                        title=f"Salary spread by {cat.replace('_',' ')}",
                        color_discrete_sequence=PALETTE
                    )

                    fig.update_layout(
                        showlegend=False,
                        xaxis_tickangle=-25
                    )

                    show_chart(fig)

            # Internship vs salary

            c3, c4 = st.columns(2)

            with c3:

                if "internships_completed" in salary_df.columns:

                    tmp = (
                        salary_df
                        .groupby("internships_completed")[
                            "salary_lpa"
                        ]
                        .median()
                        .reset_index()
                    )

                    fig = px.line(
                        tmp,
                        x="internships_completed",
                        y="salary_lpa",
                        markers=True,
                        title="Median salary vs. internships",
                        color_discrete_sequence=["#45d6b5"]
                    )

                    show_chart(fig)

            # CGPA vs salary

            with c4:

                if "cgpa" in salary_df.columns:

                    fig = px.scatter(
                        salary_df,
                        x="cgpa",
                        y="salary_lpa",
                        color=(
                            "company_type"
                            if "company_type" in salary_df.columns
                            else None
                        ),
                        opacity=0.65,
                        trendline="ols",
                        title="CGPA vs. salary"
                    )

                    show_chart(fig)

            st.caption(
                "Salary analysis excludes unplaced students and "
                "zero/missing salary values. Association is not "
                "proof of causation."
            )

# ==========================================================
# SKILLS & ACADEMIC ANALYTICS
# ==========================================================

with pages[3]:

    st.subheader("Academic and employability signals")

    skill_cols = [
        c for c in [
            "cgpa",
            "attendance_percentage",
            "DSA_problems_solved",
            "GitHub_repos",
            "hackathons_participated",
            "development_projects_count",
            "AI_ML_projects",
            "internships_completed",
            "resume_score",
            "communication_skills",
            "aptitude_score",
            "mock_interview_score",
            "adaptability_score"
        ]
        if c in filtered.columns
    ]

    if skill_cols:

        c1, c2 = st.columns([1.1, 1])

        # Skill distribution

        with c1:

            chosen_metric = st.selectbox(
                "Explore a metric",
                skill_cols,
                index=0
            )

            tmp = filtered.copy()

            tmp["_outcome"] = np.where(
                placed_mask,
                "Placed",
                "Not Placed"
            )

            fig = px.histogram(
                tmp,
                x=chosen_metric,
                color="_outcome",
                barmode="overlay",
                opacity=0.72,
                marginal="box",
                title=(
                    f"{chosen_metric.replace('_',' ').title()} "
                    "by placement outcome"
                ),
                color_discrete_map={
                    "Placed": "#45d6b5",
                    "Not Placed": "#ff7e91"
                }
            )

            show_chart(fig)

        # Skill correlation

        with c2:

            corr_cols = [
                c for c in skill_cols
                if pd.api.types.is_numeric_dtype(filtered[c])
            ]

            if len(corr_cols) >= 2:

                corr = filtered[corr_cols].corr(
                    numeric_only=True
                )

                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Academic & skill correlation"
                )

                show_chart(fig)

        # Cohort comparison

        st.markdown("**Cohort comparison**")

        means = (
            filtered.assign(
                _outcome=np.where(
                    placed_mask,
                    "Placed",
                    "Not Placed"
                )
            )
            .groupby("_outcome")[skill_cols]
            .mean(numeric_only=True)
            .T
            .reset_index()
        )

        means = means.rename(
            columns={"index": "Metric"}
        )

        st.dataframe(
            means.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No recognized academic or skills columns were found."
        )


# ==========================================================
# AI READINESS & WELLBEING
# ==========================================================

with pages[4]:

    st.subheader("AI readiness and student wellbeing")

    ai_cols = [
        c for c in [
            "AI_tool_usage_frequency",
            "prompt_engineering_skill",
            "AI_fear_score",
            "adaptability_score",
            "AI_ML_projects"
        ]
        if c in filtered.columns
    ]

    well_cols = [
        c for c in [
            "sleep_hours",
            "screen_time",
            "gaming_hours",
            "study_hours_daily",
            "stress_level",
            "burnout_score",
            "gym_frequency",
            "motivation_level",
            "self_learning_hours"
        ]
        if c in filtered.columns
    ]

    c1, c2 = st.columns(2)

    # AI readiness

    with c1:

        if ai_cols:

            selected_ai = st.selectbox(
                "AI readiness metric",
                ai_cols
            )

            tmp = filtered.assign(
                _outcome=np.where(
                    placed_mask,
                    "Placed",
                    "Not Placed"
                )
            )

            fig = px.violin(
                tmp,
                x="_outcome",
                y=selected_ai,
                box=True,
                points="outliers",
                color="_outcome",
                title=(
                    f"{selected_ai.replace('_',' ').title()} "
                    "by outcome"
                ),
                color_discrete_map={
                    "Placed": "#78a9ff",
                    "Not Placed": "#ffbd69"
                }
            )

            show_chart(fig)

        else:

            st.info(
                "AI-related columns not available."
            )

    # Wellbeing

    with c2:

        if well_cols:

            selected_well = st.selectbox(
                "Wellbeing / lifestyle metric",
                well_cols
            )

            tmp = filtered.assign(
                _outcome=np.where(
                    placed_mask,
                    "Placed",
                    "Not Placed"
                )
            )

            fig = px.box(
                tmp,
                x="_outcome",
                y=selected_well,
                color="_outcome",
                title=(
                    f"{selected_well.replace('_',' ').title()} "
                    "by outcome"
                ),
                color_discrete_sequence=[
                    "#45d6b5",
                    "#b69cff"
                ]
            )

            show_chart(fig)

        else:

            st.info(
                "Wellbeing columns not available."
            )

    # AI summary

    if ai_cols:

        st.markdown("**AI readiness summary**")

        st.dataframe(
            filtered[ai_cols].describe().T.round(2),
            use_container_width=True
        )

    # Wellbeing summary

    if well_cols:

        st.markdown("**Wellbeing summary**")

        st.dataframe(
            filtered[well_cols].describe().T.round(2),
            use_container_width=True
        )

    st.caption(
        "Wellbeing metrics are descriptive indicators, "
        "not medical assessments or causal explanations of placement."
    )


# ==========================================================
# CORRELATION & RELATIONSHIP ANALYSIS
# ==========================================================

with pages[5]:

    st.subheader("Explore relationships between variables")

    numeric_cols = (
        filtered
        .select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    excluded = {"student_id"}

    numeric_cols = [
        c for c in numeric_cols
        if c not in excluded
    ]

    if len(numeric_cols) >= 2:

        chosen = st.multiselect(
            "Variables for correlation heatmap",
            numeric_cols,
            default=numeric_cols[
                :min(16, len(numeric_cols))
            ]
        )

        if len(chosen) >= 2:

            corr = filtered[chosen].corr(
                numeric_only=True
            )

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Pearson correlation matrix"
            )

            show_chart(fig)

        else:

            st.info(
                "Select at least two numeric variables."
            )

        # Interactive scatter plot

        c1, c2 = st.columns(2)

        with c1:

            xvar = st.selectbox(
                "X-axis",
                numeric_cols,
                index=0,
                key="rel_x"
            )

        with c2:

            y_options = [
                c for c in numeric_cols
                if c != xvar
            ]

            yvar = st.selectbox(
                "Y-axis",
                y_options,
                index=0,
                key="rel_y"
            )

        color_col = st.selectbox(
            "Color grouping",
            ["None"] + [
                c for c in [
                    "placement_status",
                    "branch",
                    "gender",
                    "company_type"
                ]
                if c in filtered.columns
            ]
        )

        fig = px.scatter(
            filtered,
            x=xvar,
            y=yvar,
            color=(
                None if color_col == "None"
                else color_col
            ),
            opacity=0.65,
            trendline="ols",
            title=f"{xvar} vs. {yvar}"
        )

        show_chart(fig)

    else:

        st.info(
            "At least two numeric columns are required."
        )


# ==========================================================
# STUDENT EXPLORER
# ==========================================================

with pages[6]:

    st.subheader("Student-level exploration")

    display_cols = [
        c for c in [
            "student_id",
            "placement_status",
            "branch",
            "degree",
            "state",
            "college_tier",
            "cgpa",
            "DSA_problems_solved",
            "internships_completed",
            "resume_score",
            "mock_interview_score",
            "company_type",
            "work_mode",
            "salary_lpa",
            "offer_count"
        ]
        if c in filtered.columns
    ]

    search = st.text_input(
        "Search student ID / branch / state",
        placeholder="e.g. STU00001 or Computer Science"
    )

    view = filtered[display_cols].copy()

    if search:

        mask = pd.Series(
            False,
            index=view.index
        )

        for c in [
            "student_id",
            "branch",
            "state"
        ]:

            if c in view.columns:

                mask |= (
                    view[c]
                    .astype(str)
                    .str.contains(
                        search,
                        case=False,
                        na=False
                    )
                )

        view = view[mask]

    st.write(
        f"**{len(view):,} students** in this view"
    )

    st.dataframe(
        view,
        use_container_width=True,
        hide_index=True,
        height=420
    )

    # Export filtered records

    st.download_button(
        "⬇ Download filtered student data",

        data=filtered.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="filtered_student_placement_data.csv",

        mime="text/csv"
    )

    st.caption(
        "Avoid publishing identifiable student-level records "
        "in a public repository."
    )
    # ==========================================================
# MACHINE LEARNING LAB
# ==========================================================

with pages[7]:

    st.subheader("Placement prediction laboratory")

    st.warning(
        "This is an exploratory prediction model, not a "
        "hiring or admissions decision tool. Predictions "
        "should not be used to judge individual students."
    )

    if filtered["placement_status"].nunique(dropna=True) < 2:

        st.info(
            "At least two placement outcome classes are "
            "needed to train a classifier. Broaden your filters."
        )

    else:

        target = "placement_status"

        # Remove identifiers and features that reveal
        # information available only after placement.

        leakage = {
            "student_id",
            "salary_lpa",
            "company_type",
            "work_mode",
            "interview_rounds_cleared",
            "offer_count",
            "joining_delay_months",
            "layoffs_risk_score"
        }

        candidate_features = [
            c for c in filtered.columns
            if c not in leakage | {target}
        ]

        candidate_features = [
            c for c in candidate_features
            if filtered[c].notna().any()
        ]

        defaults = [
            c for c in candidate_features
            if c not in {"gender", "state"}
        ]

        # Feature selection

        selected_features = st.multiselect(
            "Model input features",
            candidate_features,
            default=defaults[:min(22, len(defaults))]
        )

        test_size = st.slider(
            "Test set (%)",
            15,
            35,
            20,
            5
        )

        # Train model

        if st.button(
            "Train placement model",
            type="primary"
        ):

            if not selected_features:

                st.error(
                    "Select at least one feature."
                )

            else:

                model_data = filtered[
                    selected_features + [target]
                ].dropna(
                    subset=[target]
                ).copy()

                X = model_data[selected_features]

                y = model_data[target].astype(str)

                if y.value_counts().min() < 2:

                    st.error(
                        "Each outcome class needs at least "
                        "two rows. Broaden your filters."
                    )

                else:

                    # Separate numerical and categorical features

                    numeric_features = (
                        X.select_dtypes(include=np.number)
                        .columns
                        .tolist()
                    )

                    categorical_features = [
                        c for c in selected_features
                        if c not in numeric_features
                    ]

                    # Numerical preprocessing

                    numeric_pipe = Pipeline([

                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),

                        (
                            "scaler",
                            StandardScaler()
                        )

                    ])

                    # Categorical preprocessing

                    categorical_pipe = Pipeline([

                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )

                    ])

                    transformers = []

                    if numeric_features:

                        transformers.append(
                            (
                                "num",
                                numeric_pipe,
                                numeric_features
                            )
                        )

                    if categorical_features:

                        transformers.append(
                            (
                                "cat",
                                categorical_pipe,
                                categorical_features
                            )
                        )

                    preprocessor = ColumnTransformer(
                        transformers
                    )

                    # Random Forest classifier

                    classifier = RandomForestClassifier(

                        n_estimators=350,

                        min_samples_leaf=2,

                        class_weight="balanced",

                        random_state=42,

                        n_jobs=-1

                    )

                    pipeline = Pipeline([

                        (
                            "preprocess",
                            preprocessor
                        ),

                        (
                            "model",
                            classifier
                        )

                    ])

                    # Train-test split

                    stratify = (
                        y
                        if y.value_counts().min() >= 2
                        else None
                    )

                    X_train, X_test, y_train, y_test = (
                        train_test_split(

                            X,

                            y,

                            test_size=test_size / 100,

                            random_state=42,

                            stratify=stratify

                        )
                    )

                    # Fit model

                    with st.spinner(
                        "Training Random Forest model..."
                    ):

                        pipeline.fit(
                            X_train,
                            y_train
                        )

                        pred = pipeline.predict(
                            X_test
                        )

                    # Prediction probabilities

                    labels = list(
                        pipeline
                        .named_steps["model"]
                        .classes_
                    )

                    positive_label = (
                        "Placed"
                        if "Placed" in labels
                        else labels[-1]
                    )

                    positive_idx = labels.index(
                        positive_label
                    )

                    probs = pipeline.predict_proba(
                        X_test
                    )[:, positive_idx]

                    # Calculate ROC AUC

                    auc = np.nan

                    try:

                        if len(labels) == 2:

                            y_binary = (
                                y_test == positive_label
                            ).astype(int)

                            auc = roc_auc_score(
                                y_binary,
                                probs
                            )

                    except Exception:

                        pass

                    # ==================================================
                    # MODEL PERFORMANCE CARDS
                    # ==================================================

                    st.markdown(
                        "### Model performance"
                    )

                    m1, m2, m3, m4, m5 = st.columns(5)

                    with m1:

                        metric(
                            "ACCURACY",
                            f"{accuracy_score(y_test, pred)*100:.1f}%",
                            "Held-out test set"
                        )

                    with m2:

                        metric(
                            "PRECISION",
                            f"{precision_score(y_test, pred, pos_label=positive_label, zero_division=0)*100:.1f}%",
                            positive_label
                        )

                    with m3:

                        metric(
                            "RECALL",
                            f"{recall_score(y_test, pred, pos_label=positive_label, zero_division=0)*100:.1f}%",
                            positive_label
                        )

                    with m4:

                        metric(
                            "F1 SCORE",
                            f"{f1_score(y_test, pred, pos_label=positive_label, zero_division=0):.3f}",
                            "Positive class"
                        )

                    with m5:

                        metric(
                            "ROC AUC",
                            f"{auc:.3f}" if pd.notna(auc) else "N/A",
                            "Binary classification"
                        )

                    # ==================================================
                    # CONFUSION MATRIX
                    # ==================================================

                    st.markdown(
                        "### Confusion matrix"
                    )

                    cm = confusion_matrix(
                        y_test,
                        pred,
                        labels=labels
                    )

                    fig = px.imshow(

                        cm,

                        text_auto=True,

                        x=labels,

                        y=labels,

                        labels={
                            "x": "Predicted",
                            "y": "Actual",
                            "color": "Students"
                        },

                        color_continuous_scale="Blues",

                        title="Held-out confusion matrix"

                    )

                    show_chart(fig)

                    # ==================================================
                    # CLASSIFICATION REPORT
                    # ==================================================

                    st.markdown(
                        "### Classification report"
                    )

                    report = classification_report(

                        y_test,

                        pred,

                        output_dict=True,

                        zero_division=0

                    )

                    report_df = pd.DataFrame(
                        report
                    ).T

                    st.dataframe(
                        report_df.round(3),
                        use_container_width=True
                    )

                    # ==================================================
                    # FEATURE IMPORTANCE
                    # ==================================================

                    st.markdown(
                        "### Important placement predictors"
                    )

                    try:

                        feature_names = (
                            pipeline
                            .named_steps["preprocess"]
                            .get_feature_names_out()
                        )

                        importances = (
                            pipeline
                            .named_steps["model"]
                            .feature_importances_
                        )

                        imp = pd.DataFrame({

                            "Feature": feature_names,

                            "Importance": importances

                        })

                        imp = (
                            imp
                            .sort_values(
                                "Importance",
                                ascending=False
                            )
                            .head(15)
                        )

                        fig = px.bar(

                            imp.sort_values("Importance"),

                            x="Importance",

                            y="Feature",

                            orientation="h",

                            title="Top model feature importances",

                            color="Importance",

                            color_continuous_scale="Tealgrn"

                        )

                        fig.update_layout(
                            coloraxis_showscale=False
                        )

                        show_chart(fig)

                    except Exception:

                        st.caption(
                            "Feature importance chart could not "
                            "be generated for this feature combination."
                        )

                    # ==================================================
                    # MODEL SUMMARY
                    # ==================================================

                    st.success(
                        "Model training completed successfully."
                    )

                    st.caption(
                        "Metrics are for the selected filtered sample "
                        "and random train-test split only."
                    )

                    st.info(
                        "The dataset is highly imbalanced. "
                        "Interpret recall, precision, F1, ROC AUC "
                        "and the confusion matrix together. "
                        "Accuracy alone can be misleading."
                    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#7f90ad;
        font-size:.78rem;
    ">
        CareerPulse · Student Placement & Career Success Analytics
        <br>
        Built with Python · Streamlit · Plotly · Scikit-learn
        <br>
        Exploratory analytics only
    </div>
    """,
    unsafe_allow_html=True
)
