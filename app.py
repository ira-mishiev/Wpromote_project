import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# ---------- Helpers ----------
def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    return (numer / denom).astype(float)


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ctr"] = safe_div(df["clicks"], df["impressions"])
    df["cpc"] = safe_div(df["spend"], df["clicks"])
    df["cvr"] = safe_div(df["conversions"], df["clicks"])
    df["cpa"] = safe_div(df["spend"], df["conversions"])
    df["roas"] = safe_div(df["revenue"], df["spend"])
    return df


def rolling_zscore(series: pd.Series, window: int = 7) -> pd.Series:
    mean = series.shift(1).rolling(window).mean()
    std = series.shift(1).rolling(window).std(ddof=0).replace(0, np.nan)
    return (series - mean) / std


def detect_alerts(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    df = df.copy()

    df["roas_z"] = df.groupby(["channel", "campaign"])["roas"].transform(
        lambda s: rolling_zscore(s, window)
    )
    df["cpa_z"] = df.groupby(["channel", "campaign"])["cpa"].transform(
        lambda s: rolling_zscore(s, window)
    )

    df["alert_roas_drop"] = df["roas_z"] <= -2
    df["alert_cpa_spike"] = df["cpa_z"] >= 2

    ctr_baseline = df.groupby(["channel", "campaign"])["ctr"].transform(
        lambda s: s.shift(1).rolling(window).mean()
    )
    df["alert_ctr_drop"] = df["ctr"] < (0.8 * ctr_baseline)

    def label_row(r):
        reasons = []
        if bool(r["alert_roas_drop"]):
            reasons.append("ROAS_drop")
        if bool(r["alert_cpa_spike"]):
            reasons.append("CPA_spike")
        if bool(r["alert_ctr_drop"]):
            reasons.append("CTR_drop")
        return ",".join(reasons)

    df["alert_reasons"] = df.apply(label_row, axis=1)
    df["has_alert"] = df["alert_reasons"].str.len() > 0

    return df


def plot_trend(df: pd.DataFrame, metric: str, title: str):
    d = df[["date", metric]].dropna().sort_values("date")
    fig = plt.figure()
    plt.plot(d["date"].to_numpy(), d[metric].to_numpy())
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


# ---------- App ----------
st.set_page_config(page_title="Marketing KPI Monitor", layout="wide")
st.title("Marketing KPI Monitor (CTR / CPA / ROAS) + Alerts")

st.write(
    "Upload a campaign CSV (date, channel, campaign, creative_id, impressions, clicks, conversions, revenue, spend). "
    "The app calculates KPIs and flags unusual changes using a rolling baseline."
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
default_path = "campaign_daily.csv"

if uploaded is None:
    st.info(f"No file uploaded. Trying to load local file: `{default_path}`")
    try:
        df = pd.read_csv(default_path)
    except Exception as e:
        st.error(
            "Could not load default CSV. Either upload a file, or ensure `campaign_daily.csv` exists.\n\n"
            f"Error: {e}"
        )
        st.stop()
else:
    df = pd.read_csv(uploaded)

required_cols = {
    "date", "channel", "campaign", "creative_id",
    "impressions", "clicks", "conversions", "revenue", "spend"
}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["channel", "campaign", "date"])

st.sidebar.header("Filters")
channels = ["All"] + sorted(df["channel"].unique().tolist())
campaigns = ["All"] + sorted(df["campaign"].unique().tolist())

channel_sel = st.sidebar.selectbox("Channel", channels)
campaign_sel = st.sidebar.selectbox("Campaign", campaigns)
window = st.sidebar.slider("Rolling window (days)", 3, 14, 7)

f = df.copy()
if channel_sel != "All":
    f = f[f["channel"] == channel_sel]
if campaign_sel != "All":
    f = f[f["campaign"] == campaign_sel]

f = add_kpis(f)
f = detect_alerts(f, window=window)

st.subheader("KPI Summary (Filtered)")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("CTR (avg)", f"{f['ctr'].mean():.2%}" if len(f) else "—")
col2.metric("CPC (avg)", f"${f['cpc'].mean():.2f}" if len(f) else "—")
col3.metric("CVR (avg)", f"{f['cvr'].mean():.2%}" if len(f) else "—")
col4.metric("CPA (avg)", f"${f['cpa'].mean():.2f}" if len(f) else "—")
col5.metric("ROAS (avg)", f"{f['roas'].mean():.2f}x" if len(f) else "—")

st.divider()

st.subheader("Trends")
left, right = st.columns(2)

with left:
    plot_trend(f, "ctr", "CTR Trend")
with right:
    plot_trend(f, "roas", "ROAS Trend")

left2, right2 = st.columns(2)
with left2:
    plot_trend(f, "cpa", "CPA Trend")
with right2:
    plot_trend(f, "spend", "Spend Trend")

st.divider()

st.subheader("Alerts")
alerts = f[f["has_alert"]].copy()

if alerts.empty:
    st.success("No alerts triggered for the current filters.")
else:
    show_cols = [
        "date", "channel", "campaign", "creative_id",
        "spend", "revenue", "ctr", "cpa", "roas",
        "roas_z", "cpa_z", "alert_reasons"
    ]
    alerts_display = alerts[show_cols].copy()

    alerts_display["ctr"] = alerts_display["ctr"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    alerts_display["cpa"] = alerts_display["cpa"].map(lambda x: f"${x:.2f}" if pd.notna(x) else "")
    alerts_display["roas"] = alerts_display["roas"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "")
    alerts_display["spend"] = alerts_display["spend"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    alerts_display["revenue"] = alerts_display["revenue"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    alerts_display["roas_z"] = alerts_display["roas_z"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    alerts_display["cpa_z"] = alerts_display["cpa_z"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")

    st.dataframe(alerts_display, use_container_width=True)

    st.download_button(
        "Download alerts as CSV",
        data=alerts.to_csv(index=False).encode("utf-8"),
        file_name="alerts.csv",
        mime="text/csv",
    )

st.divider()

st.subheader("Data Preview")
st.write("First 10 rows:")
st.dataframe(f.head(10), use_container_width=True)
