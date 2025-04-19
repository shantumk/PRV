# Full updated version of the app with:
# - Hybrid column mapping (auto + manual override)
# - Parallel LLM PR validation using threadpool
# - Spend analysis shown while validation happens

import streamlit as st
import pandas as pd
import requests
import fitz
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Hugging Face API ---
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN = "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"  
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Function to extract policy text from PDF ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

# --- LLM validation function ---
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

# --- Attempt automatic column matching ---
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

    # Column matching section
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

    # --- Start async validation ---
    st.info("🔄 Running compliance validation in the background. Scroll down for spend analysis...")
    validation_placeholder = st.empty()
    validation_results = []

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

    # --- Spend Dashboard ---
    st.header("💰 Spend Opportunity Dashboard")

    st.subheader("Top Vendors by Spend")
    spend_by_vendor = df.groupby('VendorName')['ContractValue'].sum().sort_values(ascending=False)
    st.bar_chart(spend_by_vendor.head(10))

    st.subheader("Top Agencies by Spend")
    spend_by_agency = df.groupby('Agency')['ContractValue'].sum().sort_values(ascending=False)
    st.bar_chart(spend_by_agency.head(10))

    st.subheader("Spend Over Time")
    time_series = df.set_index('ContractDate').resample('Q')['ContractValue'].sum()
    st.line_chart(time_series)

    st.subheader("High Value Outliers")
    df['z_score'] = zscore(df['ContractValue'].fillna(0))
    outliers = df[df['z_score'] > 3]
    st.dataframe(outliers[['PR_ID', 'VendorName', 'ContractValue', 'z_score']])

    st.subheader("Vendor Spend Clusters")
    vendor_df = spend_by_vendor.reset_index().rename(columns={'ContractValue': 'TotalSpend'})
    scaler = StandardScaler()
    vendor_df['ScaledSpend'] = scaler.fit_transform(vendor_df[['TotalSpend']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    vendor_df['Cluster'] = kmeans.fit_predict(vendor_df[['ScaledSpend']])
    st.dataframe(vendor_df)

    # --- Finish validation and show ---
    with st.spinner("Merging validation results..."):
        results = run_llm_validation(df, policy_text)
        validation_map = {idx: result for idx, result in results}
        df['LLM_Check'] = df.index.map(lambda i: validation_map.get(i, "Pending"))

    st.success("✅ PR validation complete!")
    validation_placeholder.dataframe(df[['PR_ID', 'VendorName', 'ContractValue', 'ItemDescription', 'LLM_Check']])

    # Allow download
    st.sidebar.download_button(
        label="📥 Download Validated PR Data",
        data=df.to_csv(index=False),
        file_name="validated_pr_data.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload both the PR CSV and policy PDF to begin.")
