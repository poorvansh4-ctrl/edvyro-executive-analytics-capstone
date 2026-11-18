"""Interactive Streamlit dashboard for the IBM Telco churn sample."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "customer_churn_cleaned.csv"
VISUALS = ROOT / "visuals"

st.set_page_config(
    page_title="Customer Churn | EdVyro Internship Project",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
      background: #f4f7fb; border: 1px solid #e3e9f2;
      padding: 1rem 1.1rem; border-radius: 12px;
    }
    .source-note {font-size: .88rem; color: #65738a;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA_PATH)
    frame["tenure_band"] = pd.cut(
        frame["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"],
    )
    return frame


def rate_table(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[dimension, "Customers", "Churned", "Churn rate"])
    result = (
        frame.groupby(dimension, observed=False)
        .agg(Customers=("customerID", "size"), Churned=("Churn", lambda values: values.eq("Yes").sum()))
        .reset_index()
    )
    result = result.loc[result["Customers"].gt(0)].copy()
    result["Churn rate"] = result["Churned"] / result["Customers"]
    return result.sort_values("Churn rate", ascending=False)


def show_chart(filename: str, height: int = 500) -> None:
    chart_path = VISUALS / filename
    if not chart_path.exists():
        st.warning(f"Chart file is missing: {filename}")
        return
    st.iframe(chart_path, width="stretch", height=height, alt=filename)


data = load_data()
st.title("Customer Churn Analysis")
st.caption("Data quality audit · exploratory analysis · interactive customer segments")
st.markdown(
    '<p class="source-note">Public IBM Telco sample used as an alternative dataset because EdVyro resource files were unavailable. '
    'This is not the official EdVyro-provided dataset.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Explore a customer segment")
    contracts = st.multiselect(
        "Contract type",
        sorted(data["Contract"].dropna().unique().tolist()),
        default=sorted(data["Contract"].dropna().unique().tolist()),
    )
    payment_methods = st.multiselect(
        "Payment method",
        sorted(data["PaymentMethod"].dropna().unique().tolist()),
        default=sorted(data["PaymentMethod"].dropna().unique().tolist()),
    )
    internet_services = st.multiselect(
        "Internet service",
        sorted(data["InternetService"].dropna().unique().tolist()),
        default=sorted(data["InternetService"].dropna().unique().tolist()),
    )
    churn_labels = st.multiselect("Churn status", ["No", "Yes"], default=["No", "Yes"])
    tenure_range = st.slider(
        "Tenure range (months)",
        min_value=int(data["tenure"].min()),
        max_value=int(data["tenure"].max()),
        value=(int(data["tenure"].min()), int(data["tenure"].max())),
    )
    segmentation = st.selectbox(
        "Segment table",
        ["Tenure band", "Contract", "Payment method", "Internet service"],
    )

filtered = data[
    data["Contract"].isin(contracts)
    & data["PaymentMethod"].isin(payment_methods)
    & data["InternetService"].isin(internet_services)
    & data["Churn"].isin(churn_labels)
    & data["tenure"].between(tenure_range[0], tenure_range[1])
].copy()

if filtered.empty:
    st.warning("No customers match the current filters. Adjust the sidebar selections.")
    total = churned = retained = 0
    rate = 0.0
else:
    total = int(len(filtered))
    churned = int(filtered["Churn"].eq("Yes").sum())
    retained = total - churned
    rate = churned / total

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers in segment", f"{total:,}")
k2.metric("Churned", f"{churned:,}")
k3.metric("Retained", f"{retained:,}")
k4.metric("Churn rate", f"{rate:.1%}" if total else "—")

st.divider()
left, right = st.columns([1.3, 1])
with left:
    st.subheader(f"Segment breakdown · {segmentation}")
    dimension = {
        "Tenure band": "tenure_band",
        "Contract": "Contract",
        "Payment method": "PaymentMethod",
        "Internet service": "InternetService",
    }[segmentation]
    table = rate_table(filtered, dimension)
    if not table.empty:
        table["Churn rate"] = table["Churn rate"].map(lambda value: f"{value:.1%}")
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.info("No segment data for the current selection.")

with right:
    st.subheader("Business insights")
    if filtered.empty:
        st.write("Choose filters that include at least one customer.")
    else:
        contract_summary = rate_table(filtered, "Contract")
        payment_summary = rate_table(filtered, "PaymentMethod")
        if not contract_summary.empty:
            lead = contract_summary.iloc[0]
            st.markdown(
                f"**Highest observed contract churn rate:** {lead['Contract']} — "
                f"{lead['Churn rate']:.1%} ({int(lead['Churned']):,} of "
                f"{int(lead['Customers']):,} customers)."
            )
        if not payment_summary.empty:
            lead = payment_summary.iloc[0]
            st.markdown(
                f"**Highest observed payment-method churn rate:** {lead['PaymentMethod']} — "
                f"{lead['Churn rate']:.1%} ({int(lead['Churned']):,} of "
                f"{int(lead['Customers']):,} customers)."
            )
        means = filtered.groupby("Churn")["MonthlyCharges"].mean()
        if {"Yes", "No"}.issubset(means.index):
            st.markdown(
                f"Mean MonthlyCharges are {means['Yes']:.2f} for churned customers and "
                f"{means['No']:.2f} for retained customers (reported charge units)."
            )
        st.caption(
            "These are descriptive sample patterns, not causal effects. Rates use the "
            "currently filtered customers in each segment."
        )

st.divider()
st.subheader("Full-sample visual context")
st.caption(
    "The charts below describe the complete cleaned IBM sample and remain fixed as a "
    "reference. Sidebar filters update the KPIs, segment breakdown, and business insights."
)
show_chart("churn_distribution_notion_agent_chart.html", height=430)
show_chart("contract_churn_notion_agent_chart.html", height=430)
show_chart("segment_analysis_notion_agent_chart.html", height=570)

st.divider()
st.subheader("Filtered customer preview")
preview_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
st.dataframe(filtered[preview_columns].head(100), width="stretch", hide_index=True)
st.caption("Preview is limited to 100 rows; customer identifiers are omitted.")

with st.expander("Data source and method"):
    st.markdown(
        """
        - Dataset: [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
        - Source repository: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
        - Cleaning: whitespace trimming, numeric conversion for `TotalCharges`, and a documented zero imputation for 11 blank cumulative-charge values associated with zero tenure.
        - Validation: unique customer IDs, no missing values after cleaning, expected categorical domains, and nonnegative tenure/charges.
        - Limitation: this public sample is an alternative dataset, not the official EdVyro resource.
        """
    )