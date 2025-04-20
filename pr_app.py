import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore

st.set_page_config(page_title="Procurement Compliance & Sustainability Analyzer", layout="wide")
st.title("🛠 Procurement Compliance & Sustainability Analyzer")

# Sidebar: Data Upload
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload Procurement Data CSV", type=["csv"])
uploaded_emission_file = st.sidebar.file_uploader(
    "Upload CPVS→Emission Mapping CSV (optional)", type=["csv"],
    help="CSV with columns: CPVS, emission_factor"
)

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()

    # Auto-mapping helper
    def auto_map(options):
        for col in cols:
            for opt in options:
                if opt.lower() in col.lower():
                    return col
        return None

    # Default mappings
    mapping = {
        "TenderID": auto_map(["tender_id", "pr_id", "id"]),
        "Value": auto_map(["value", "contract_value", "amount", "price"]),
        "BidCount": auto_map(["bid", "bidcount", "recordedbidscount"]),
        "ContractDate": auto_map(["date", "award_date", "contract_date", "contractsignaturedate"]),
        "CPVS": auto_map(["cpvs", "cpv"]),
        "Vendor": auto_map(["vendor", "supplier", "buyer"]),
    }

    # User-confirmable mapping
    st.sidebar.subheader("🔧 Column Mapping")
    for std, default in mapping.items():
        mapping[std] = st.sidebar.selectbox(
            f"Select {std} column", options=[""] + cols, index=(cols.index(default)+1 if default in cols else 0)
        )

    # Rename dataframe columns
    rename_dict = {mapping[k]: k for k in mapping if mapping[k]}
    df = df.rename(columns=rename_dict)

    # Convert to numeric
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    if mapping["BidCount"]:
        df["BidCount"] = pd.to_numeric(df["BidCount"], errors="coerce")

    # Drop rows missing required fields
    df = df.dropna(subset=["TenderID", "Value"])
    st.success("✅ Data loaded and columns mapped successfully!")

    # --- Analytics 2: Anomaly Detection ---
    st.header("🚨 Anomaly Detection")
    df["z_score"] = zscore(df["Value"].fillna(0))
    outliers = df[df["z_score"] > 3]
    st.subheader("High Value Outliers (Z-score > 3)")
    st.dataframe(outliers[["TenderID", "Vendor", "Value", "z_score"]])

    iso = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iso.fit_predict(df[["Value"]].fillna(0))
    anomalies = df[df["anomaly"] == -1]
    st.subheader("Isolation Forest Anomalies")
    st.dataframe(anomalies[["TenderID", "Vendor", "Value", "anomaly"]])

    # --- Analytics 3: Vendor Concentration ---
    st.header("📊 Vendor Concentration Analysis")
    if mapping["Vendor"]:
        spend_by_vendor = df.groupby("Vendor")["Value"].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots()
        spend_by_vendor.head(10).plot(kind="bar", ax=ax1)
        ax1.set_title("Top 10 Vendors by Spend")
        ax1.set_ylabel("Total Spend")
        st.pyplot(fig1)

        vendor_share = spend_by_vendor / spend_by_vendor.sum()
        high_dep = vendor_share[vendor_share > 0.3]
        st.subheader("High Dependency Vendors (>30% of total spend)")
        st.dataframe(high_dep)
    else:
        st.warning("Vendor column not mapped; skipping vendor concentration analysis.")

    # --- Analytics 4: Carbon Footprint Estimation ---
    st.header("🌱 Carbon Footprint Estimation")
    if uploaded_emission_file:
        ef = pd.read_csv(uploaded_emission_file)
        # Expect columns: CPVS, emission_factor
        df = df.merge(ef, on="CPVS", how="left")
        df["CO2e"] = df["Value"] * df["emission_factor"]
        st.subheader("Top Carbon Impact Tenders")
        st.dataframe(df.nlargest(10, "CO2e")[["TenderID", "Vendor", "CPVS", "Value", "CO2e"]])

        cpvs_group = df.groupby("CPVS")["CO2e"].sum().sort_values(ascending=False)
        st.subheader("CO₂e by CPVS Category")
        st.bar_chart(cpvs_group.head(10))

        # Download carbon report
        st.sidebar.download_button(
            "Download Carbon Report CSV", df.to_csv(index=False),
            file_name="carbon_report.csv", mime="text/csv"
        )
    else:
        st.info("Upload CPVS→Emission mapping file to enable carbon footprint estimation.")

    # --- Download Reports ---
    st.sidebar.subheader("📥 Download Reports")
    st.sidebar.download_button(
        "Download Outliers CSV", outliers.to_csv(index=False),
        file_name="outliers.csv", mime="text/csv"
    )
    st.sidebar.download_button(
        "Download Anomalies CSV", anomalies.to_csv(index=False),
        file_name="anomalies.csv", mime="text/csv"
    )

else:
    st.info("Please upload a procurement CSV file to begin analysis.")
