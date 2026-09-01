# app.py — Auto-Visualizer Streamlit App
#
# Automatically detects column types from any DataFrame and renders
# the most appropriate chart for each feature / feature pair.
#
# Run:  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Visualizer",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DUMMY DATA SOURCES
#    Each dataset exercises a different mix of column types so the
#    visualizer has something interesting to detect automatically.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_sales_data() -> pd.DataFrame:
    """Monthly sales figures across product categories and regions."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=24, freq="ME")
    categories = ["Electronics", "Clothing", "Food", "Sports"]
    regions    = ["North", "South", "East", "West"]

    rows = []
    for d in dates:
        for cat in categories:
            for reg in regions:
                rows.append({
                    "date":     d,
                    "category": cat,
                    "region":   reg,
                    "revenue":  rng.integers(5_000, 50_000),
                    "units":    rng.integers(50, 500),
                    "profit":   rng.integers(500, 15_000),
                    "returned": rng.integers(0, 30),
                })
    return pd.DataFrame(rows)


@st.cache_data
def load_employee_data() -> pd.DataFrame:
    """HR snapshot: mix of numerical, categorical, and boolean columns."""
    rng = np.random.default_rng(7)
    n   = 200
    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    levels      = ["Junior", "Mid", "Senior", "Lead", "Manager"]

    return pd.DataFrame({
        "age":        rng.integers(22, 60, n),
        "salary":     rng.integers(30_000, 150_000, n),
        "experience": rng.integers(0, 35, n),
        "department": rng.choice(departments, n),
        "level":      rng.choice(levels, n),
        "remote":     rng.choice([True, False], n),
        "rating":     np.round(rng.uniform(1.0, 5.0, n), 1),
        "tenure":     rng.integers(0, 15, n),
    })


@st.cache_data
def load_sensor_data() -> pd.DataFrame:
    """IoT sensor readings: pure time-series with multiple signals."""
    t = pd.date_range("2024-01-01", periods=500, freq="h")
    noise = np.random.default_rng(99)
    return pd.DataFrame({
        "timestamp":   t,
        "temperature": 20 + 8 * np.sin(np.linspace(0, 8 * np.pi, 500))
                       + noise.normal(0, 0.5, 500),
        "humidity":    60 + 15 * np.cos(np.linspace(0, 6 * np.pi, 500))
                       + noise.normal(0, 1.0, 500),
        "pressure":    1013 + 5 * np.sin(np.linspace(0, 4 * np.pi, 500))
                       + noise.normal(0, 0.3, 500),
        "vibration":   np.abs(noise.normal(0, 0.8, 500)),
        "status":      np.where(noise.random(500) > 0.97, "ALERT", "OK"),
    })


DATASETS = {
    "Sales (categorical + time-series + numerical)": load_sales_data,
    "Employees (numerical + categorical + boolean)": load_employee_data,
    "Sensor (time-series + numerical + status)":     load_sensor_data,
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. COLUMN TYPE DETECTOR
#    Returns one of: "datetime", "boolean", "categorical", "numerical"
# ─────────────────────────────────────────────────────────────────────────────
def detect_col_type(series: pd.Series) -> str:
    """Infer the semantic type of a single column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    # Treat low-cardinality object columns as categorical
    return "categorical"


def profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Build a type profile for every column.
    Returns: { col_name: type_string, ... }
    """
    return {col: detect_col_type(df[col]) for col in df.columns}


# ─────────────────────────────────────────────────────────────────────────────
# 4. INDIVIDUAL CHART BUILDERS
#    Each function receives the full DataFrame + relevant column name(s)
#    and returns a Plotly figure.
# ─────────────────────────────────────────────────────────────────────────────

def chart_distribution(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram + KDE overlay for a numerical column."""
    fig = px.histogram(
        df, x=col,
        nbins=30,
        marginal="violin",         # adds mini violin on top
        title=f"Distribution — {col}",
        color_discrete_sequence=["#636EFA"],
        opacity=0.75,
    )
    fig.update_layout(bargap=0.05)
    return fig


def chart_bar_categorical(df: pd.DataFrame, col: str) -> go.Figure:
    """Sorted bar chart of value counts for a categorical column."""
    counts = (
        df[col].value_counts()
                .reset_index()
                .rename(columns={"index": col, "count": "count"})
    )
    # value_counts() already returns two columns in newer pandas
    counts.columns = [col, "count"]
    fig = px.bar(
        counts, x=col, y="count",
        title=f"Value Counts — {col}",
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    return fig


def chart_boolean(df: pd.DataFrame, col: str) -> go.Figure:
    """Pie chart for boolean columns (True / False split)."""
    counts = df[col].value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index.astype(str),
        title=f"Boolean Split — {col}",
        color_discrete_sequence=["#00CC96", "#EF553B"],
        hole=0.35,
    )
    return fig


def chart_time_series(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    """Line chart for a numerical column over time."""
    tmp = df[[date_col, value_col]].dropna().sort_values(date_col)
    fig = px.line(
        tmp, x=date_col, y=value_col,
        title=f"{value_col} over time",
        markers=False,
        color_discrete_sequence=["#AB63FA"],
    )
    fig.update_traces(line_width=1.8)
    return fig


def chart_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                  color_col: str = None) -> go.Figure:
    """Scatter plot between two numerical columns, optionally coloured."""
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col,
        title=f"{y_col} vs {x_col}",
        opacity=0.65,
        trendline="ols",           # linear regression line
        trendline_color_override="#FF6692",
    )
    return fig


def chart_box(df: pd.DataFrame, num_col: str, cat_col: str) -> go.Figure:
    """Box plot: numerical column broken down by a categorical column."""
    fig = px.box(
        df, x=cat_col, y=num_col,
        title=f"{num_col} by {cat_col}",
        color=cat_col,
        points="outliers",
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_heatmap_corr(df: pd.DataFrame, num_cols: list) -> go.Figure:
    """Correlation heatmap for all numerical columns."""
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Matrix",
        aspect="auto",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUTO-LAYOUT ENGINE
#    Decides which charts to render based on the column-type profile.
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame, profile: dict) -> None:
    """Top-level summary metrics (rows, columns, missing values)."""
    total_missing = df.isnull().sum().sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing", f"{total_missing:,}")
    c4.metric("Memory",  f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    with st.expander("Column type profile", expanded=False):
        profile_df = pd.DataFrame(
            {"Column": profile.keys(), "Detected Type": profile.values()}
        )
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    with st.expander("Raw data preview", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)


def render_univariate(df: pd.DataFrame, profile: dict) -> None:
    """One chart per column — appropriate type chosen automatically."""
    st.subheader("📋 Univariate — Each Column")

    num_cols  = [c for c, t in profile.items() if t == "numerical"]
    cat_cols  = [c for c, t in profile.items() if t == "categorical"]
    bool_cols = [c for c, t in profile.items() if t == "boolean"]

    # Numerical: histograms in a 2-column grid
    if num_cols:
        st.markdown("**Numerical columns**")
        pairs = list(zip(num_cols[::2], num_cols[1::2]))
        solo  = [num_cols[-1]] if len(num_cols) % 2 else []

        for left, right in pairs:
            c1, c2 = st.columns(2)
            c1.plotly_chart(chart_distribution(df, left),  use_container_width=True)
            c2.plotly_chart(chart_distribution(df, right), use_container_width=True)
        for col in solo:
            st.plotly_chart(chart_distribution(df, col), use_container_width=True)

    # Categorical: bar charts in a 2-column grid
    if cat_cols:
        st.markdown("**Categorical columns**")
        pairs = list(zip(cat_cols[::2], cat_cols[1::2]))
        solo  = [cat_cols[-1]] if len(cat_cols) % 2 else []

        for left, right in pairs:
            c1, c2 = st.columns(2)
            c1.plotly_chart(chart_bar_categorical(df, left),  use_container_width=True)
            c2.plotly_chart(chart_bar_categorical(df, right), use_container_width=True)
        for col in solo:
            st.plotly_chart(chart_bar_categorical(df, col), use_container_width=True)

    # Boolean: pie charts in a 3-column grid
    if bool_cols:
        st.markdown("**Boolean columns**")
        cols = st.columns(min(len(bool_cols), 3))
        for i, col in enumerate(bool_cols):
            cols[i % 3].plotly_chart(chart_boolean(df, col), use_container_width=True)


def render_time_series(df: pd.DataFrame, profile: dict) -> None:
    """Time-series section — only shown when a datetime column exists."""
    date_cols = [c for c, t in profile.items() if t == "datetime"]
    num_cols  = [c for c, t in profile.items() if t == "numerical"]

    if not date_cols or not num_cols:
        return  # nothing to plot

    st.subheader("📈 Time-Series")
    date_col = date_cols[0]   # use the first datetime column as x-axis

    # Let user pick which signals to show
    selected = st.multiselect(
        "Select metrics to plot over time",
        options=num_cols,
        default=num_cols[:3],
        key="ts_select",
    )

    for col in selected:
        st.plotly_chart(
            chart_time_series(df, date_col, col),
            use_container_width=True,
        )


def render_bivariate(df: pd.DataFrame, profile: dict) -> None:
    """Scatter plots, box plots, and correlation heatmap."""
    st.subheader("🔗 Bivariate Relationships")

    num_cols = [c for c, t in profile.items() if t == "numerical"]
    cat_cols = [c for c, t in profile.items() if t == "categorical"]

    # ── Correlation heatmap (needs ≥2 numeric cols) ─────────────────────────
    if len(num_cols) >= 2:
        st.plotly_chart(chart_heatmap_corr(df, num_cols), use_container_width=True)

    # ── User-driven scatter plot ─────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown("**Scatter plot builder**")
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X axis",    num_cols, index=0, key="sc_x")
        y_col = c2.selectbox("Y axis",    num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        color = c3.selectbox("Color by",  ["(none)"] + cat_cols, key="sc_c")
        color_col = None if color == "(none)" else color

        st.plotly_chart(
            chart_scatter(df, x_col, y_col, color_col),
            use_container_width=True,
        )

    # ── Box plots: each numerical × first categorical ────────────────────────
    if num_cols and cat_cols:
        cat_col = cat_cols[0]
        # Only plot box if cardinality is manageable
        if df[cat_col].nunique() <= 15:
            st.markdown(f"**Box plots grouped by `{cat_col}`**")
            for col in num_cols[:4]:   # cap at 4 to avoid clutter
                st.plotly_chart(
                    chart_box(df, col, cat_col),
                    use_container_width=True,
                )


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Auto Visualizer")
    st.caption(
        "Upload any CSV or pick a built-in dataset. "
        "Column types are detected automatically and the best charts are selected."
    )

    # ── Sidebar: data source selection ──────────────────────────────────────
    with st.sidebar:
        st.header("Data Source")
        source = st.radio(
            "Choose source",
            ["Built-in dataset", "Upload CSV"],
            index=0,
        )

        df = None

        if source == "Built-in dataset":
            chosen = st.selectbox("Dataset", list(DATASETS.keys()))
            df = DATASETS[chosen]()
            st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")

        else:
            uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded, parse_dates=True, infer_datetime_format=True)
                # Try to coerce any column that looks like a date
                for col in df.select_dtypes("object").columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        pass
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
            else:
                st.info("Waiting for file…")

        st.divider()
        st.markdown("**Sections to display**")
        show_uni  = st.checkbox("Univariate",  value=True)
        show_ts   = st.checkbox("Time-Series", value=True)
        show_bi   = st.checkbox("Bivariate",   value=True)

    # ── Guard: nothing to show yet ───────────────────────────────────────────
    if df is None:
        st.info("👈 Select a dataset or upload a CSV to get started.")
        return

    # ── Detect types ─────────────────────────────────────────────────────────
    profile = profile_dataframe(df)

    # ── Render sections ──────────────────────────────────────────────────────
    render_overview(df, profile)
    st.divider()

    if show_uni:
        render_univariate(df, profile)
        st.divider()

    if show_ts:
        render_time_series(df, profile)
        st.divider()

    if show_bi:
        render_bivariate(df, profile)


if __name__ == "__main__":
    main()
