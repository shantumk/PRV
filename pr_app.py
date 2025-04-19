import streamlit as st
import pandas as pd
import requests
import fitz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Hugging Face API Setup ---
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN = st.secrets["api"]["hf_token"]
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Extract plain text from uploaded PDF policy ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

# --- LLM call for PR validation ---
def llm_validate(row, policy_text):
    prompt = (
        f"Policy:\n{policy_text}\n\n"
        f"PR Details:\n"
        f"- PR ID: {row['PR_ID']}\n"
        f"- Vendor: {row['VendorName']}\n"
        f"- Value: {row['ContractValue']}\n"
        f"- Item: {row['ItemDescription']}\n\n"
        "Is this PR compliant with the policy? If not, explain why."
    )
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0].get("generated_text", "").strip()
    else:
        return f"Error: {response.status_code}"

# --- Attempt auto column matching ---
def auto_match_columns(headers):
    match_dict = {
        'PR_ID': ['tender_no', 'pr_number', 'reference'],
        'VendorName': ['supplier', 'vendor', 'supplier_name'],
        'ContractValue': ['awarded_amt', 'contract_value', 'amount_awarded'],
        'ItemDescription': ['description', 'details', 'item_description'],
        'Agency': ['agency', 'department', 'org'],
        'ContractDate': ['award_date', 'contract_date']
    }
    result = {}
    for std, options in match_dict.items():
        for col in headers:
            if any(opt.lower() in col.lower() for opt in options):
                result[std] = col
                break
    return result

# --- Streamlit App ---
st.title("🧾 PR Validation & Spend Opportunity Dashboard")

st.sidebar.header("📂 Upload Files")
uploaded_csv = st.sidebar.file_uploader("Step 1: Upload PR Data (CSV)", type="csv")
uploaded_policy = st.sidebar.file_uploader("Step 2: Upload Policy Document (PDF)", type="pdf")

if uploaded_csv and uploaded_policy:
    st.success("Files uploaded. Analyzing your data...")

    df = pd.read_csv(uploaded_csv)
    policy_text = extract_text_from_pdf(uploaded_policy)

    # Column Mapping
    st.subheader("🧩 Column Mapping")
    auto_map = auto_match_columns(df.columns.tolist())

    PR_ID = st.selectbox("Select PR ID column", df.columns, index=df.columns.get_loc(auto_map.get("PR_ID", df.columns[0])))
    Vendor = st.selectbox("Select Vendor column", df.columns, index=df.columns.get_loc(auto_map.get("VendorName", df.columns[1])))
    Value = st.selectbox("Select Contract Value column", df.columns, index=df.columns.get_loc(auto_map.get("ContractValue", df.columns[2])))
    Item = st.selectbox("Select Item Description column", df.columns, index=df.columns.get_loc(auto_map.get("ItemDescription", df.columns[3])))
    Agency = st.selectbox("Select Agency column", df.columns, index=df.columns.get_loc(auto_map.get("Agency", df.columns[4])))
    Date = st.selectbox("Select Contract Date column", df.columns, index=df.columns.get_loc(auto_map.get("ContractDate", df.columns[5])))

    df = df.rename(columns={
        PR_ID: 'PR_ID',
        Vendor: 'VendorName',
        Value: 'ContractValue',
        Item: 'ItemDescription',
        Agency: 'Agency',
        Date: 'ContractDate'
    })

    df['ContractValue'] = pd.to_numeric(df['ContractValue'], errors='coerce')
    df = df.dropna(subset=['ContractValue'])
    df['ContractDate'] = pd.to_datetime(df['ContractDate'], dayfirst=True, errors='coerce')

    st.info("🔄 Running compliance validation in the background. Scroll down for spend insights.")
    validation_placeholder = st.empty()

    def run_llm_validation(df, policy_text):
        output = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(llm_validate, row, policy_text): idx for idx, row in df.iterrows()}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = str(e)
                output.append((idx, result))
        return output

    # Spend Analysis
    st.header("💰 Spend Opportunity Dashboard")
    spend_by_vendor = df.groupby('VendorName')['ContractValue'].sum().sort_values(ascending=False)
    st.subheader("Top Vendors by Spend")
    st.bar_chart(spend_by_vendor.head(10))

    st.subheader("Top Agencies by Spend")
    spend_by_agency = df.groupby('Agency')['ContractValue'].sum().sort_values(ascending=False)
    st.bar_chart(spend_by_agency.head(10))

    # Quarterly Spend
    time_series = df.set_index('ContractDate').resample('QE-DEC')['ContractValue'].sum()
    st.subheader("Spend Over Time (Quarterly)")
    st.line_chart(time_series)

    # Z-score based Outliers
    st.subheader("High Value Outliers")
    df['z_score'] = zscore(df['ContractValue'].fillna(0))
    outliers = df[df['z_score'] > 3]
    st.dataframe(outliers[['PR_ID', 'VendorName', 'ContractValue', 'z_score']])

    # Anomaly Detection
    st.subheader("🔍 Anomaly Detection (Isolation Forest)")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = iso_forest.fit_predict(df[['ContractValue']].fillna(0))
    anomalies = df[df['anomaly_score'] == -1]
    st.write("The following PRs are flagged as anomalous:")
    st.dataframe(anomalies[['PR_ID', 'VendorName', 'ContractValue', 'ItemDescription']])

    # Visualizing Anomalies
    st.subheader("📈 Anomaly Score Visualization")
    anomaly_viz = df[['ContractValue']].copy()
    anomaly_viz['Anomaly Score'] = iso_forest.decision_function(df[['ContractValue']].fillna(0))
    anomaly_viz['Is Anomaly'] = df['anomaly_score'] == -1

    fig, ax = plt.subplots()
    sns.scatterplot(data=anomaly_viz, x=np.arange(len(anomaly_viz)), y='ContractValue', hue='Is Anomaly', palette='coolwarm', ax=ax)
    ax.set_title("Contract Value vs Anomaly Flag")
    st.pyplot(fig)

    # Supplier Consolidation
    st.subheader("💡 Supplier Consolidation Opportunities")
    grouped_items = df.groupby(['ItemDescription', 'VendorName'])['ContractValue'].sum().reset_index()
    consolidation_pivot = grouped_items.pivot_table(index='ItemDescription', columns='VendorName', values='ContractValue', fill_value=0)
    consolidation_candidates = consolidation_pivot[consolidation_pivot.astype(bool).sum(axis=1) > 1]

    if not consolidation_candidates.empty:
        st.write("Items procured from multiple vendors:")
        st.dataframe(consolidation_candidates)

        st.subheader("💰 Estimated Savings")
        savings_opportunities = []
        for item, row in consolidation_candidates.iterrows():
    total_spend = row.sum()
    positive_values = row[row > 0]
    if not positive_values.empty:
        best_vendor_cost = positive_values.min()
        savings = total_spend - best_vendor_cost
        savings_opportunities.append({
            'Item': item,
            'Total Spend': total_spend,
            'Best Vendor Cost': best_vendor_cost,
            'Estimated Savings': savings
        })
savings_df = pd.DataFrame(savings_opportunities).sort_values(by='Estimated Savings', ascending=False).sort_values(by='Estimated Savings', ascending=False)
        st.dataframe(savings_df)

    # Spend Categorization
    st.subheader("🧠 Spend Category Insights")
    def categorize_item(description):
        d = description.lower()
        if 'software' in d or 'license' in d:
            return 'IT & Software'
        elif 'consult' in d or 'advisory' in d:
            return 'Consulting Services'
        elif 'chair' in d or 'desk' in d or 'furniture' in d:
            return 'Office Furniture'
        elif 'printer' in d or 'stationery' in d:
            return 'Office Supplies'
        elif 'training' in d or 'course' in d:
            return 'Training & Development'
        else:
            return 'Uncategorized'

    df['SpendCategory'] = df['ItemDescription'].apply(categorize_item)
    category_summary = df.groupby('SpendCategory')['ContractValue'].sum().sort_values(ascending=False)
    st.bar_chart(category_summary)

    # Finish validation
    with st.spinner("Finalizing LLM Validation..."):
        results = run_llm_validation(df, policy_text)
        validation_map = {idx: result for idx, result in results}
        df['LLM_Check'] = df.index.map(lambda i: validation_map.get(i, "Pending"))

    st.success("✅ PR validation complete!")
    validation_placeholder.dataframe(df[['PR_ID', 'VendorName', 'ContractValue', 'ItemDescription', 'LLM_Check']])

    # Download section
    st.subheader("📥 Download Reports")
    st.download_button("Download Anomalies CSV", data=anomalies.to_csv(index=False), file_name="anomalies.csv", mime="text/csv")
    st.download_button("Download Outliers CSV", data=outliers.to_csv(index=False), file_name="outliers.csv", mime="text/csv")
    st.sidebar.download_button(
        label="📥 Download Validated PR Data",
        data=df.to_csv(index=False),
        file_name="validated_pr_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload both the PR CSV and policy PDF to begin.")
