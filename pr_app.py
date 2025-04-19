import streamlit as st
import pandas as pd
import requests
import fitz 
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- Hugging Face LLM setup ---
# We’ll use a pre-trained language model to validate PRs based on policy
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN = "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"  
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# Function to extract plain text from uploaded PDF policy file
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

# Function that sends each PR to the LLM for validation
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

# --- STREAMLIT APP STARTS HERE ---
st.title("🧾 PR Validation & Spend Opportunity Dashboard")

st.sidebar.header("📂 Upload Your Files")
uploaded_csv = st.sidebar.file_uploader("Step 1: Upload PR Data (CSV)", type=["csv"])
uploaded_policy = st.sidebar.file_uploader("Step 2: Upload Policy Document (PDF)", type=["pdf"])

# Once both files are uploaded, begin processing\if uploaded_csv and uploaded_policy:
# Once both files are uploaded, begin processing
if uploaded_csv and uploaded_policy:
    st.success("All files uploaded successfully! Processing your data now...")

    # Load the CSV PR data
    df = pd.read_csv(uploaded_csv)

    # Extract plain text from the uploaded policy PDF
    policy_text = extract_text_from_pdf(uploaded_policy)

    # Standardize column names so we can use them easily in code
    df = df.rename(columns={
        'tender_no.': 'PR_ID',
        'supplier_name': 'VendorName',
        'awarded_amt': 'ContractValue',
        'tender_description': 'ItemDescription',
        'agency': 'Agency',
        'award_date': 'ContractDate'
    })

    # Convert contract value to numbers just in case it's read as text
    df['ContractValue'] = pd.to_numeric(df['ContractValue'], errors='coerce')
    df = df.dropna(subset=['ContractValue'])

    # ---- PR COMPLIANCE CHECK ----
    st.subheader("✅ Compliance Check (LLM-Based)")
    st.write("We're using a language model to read each PR and validate it against your uploaded policy. This may take a few seconds...")

    with st.spinner("Validating all PRs... hang tight!"):
        df['LLM_Check'] = df.apply(lambda row: llm_validate(row, policy_text), axis=1)

    st.success("All PRs have been validated!")
    st.dataframe(df[['PR_ID', 'VendorName', 'ContractValue', 'ItemDescription', 'LLM_Check']])

    # ---- SPEND ANALYSIS SECTION ----
    st.header("💰 Spend Analysis & Opportunity Dashboard")

    # Top vendors by spend
    st.subheader("🔝 Top Vendors by Spend")
    spend_by_vendor = df.groupby('VendorName')['ContractValue'].sum().sort_values(ascending=False)
    st.bar_chart(spend_by_vendor.head(10))

    # Spend by agency
    st.subheader("🏢 Spend by Agency")
    spend_by_agency = df.groupby('Agency')['ContractValue'].sum().sort_values(ascending=False)
    st.bar_chart(spend_by_agency.head(10))

    # Trend over time
    st.subheader("📈 Spend Over Time")
    df['ContractDate'] = pd.to_datetime(df['ContractDate'], dayfirst=True, errors='coerce')
    time_series = df.set_index('ContractDate').resample('Q')['ContractValue'].sum()
    st.line_chart(time_series)

    # Outlier detection using z-score
    st.subheader("⚠️ High Value Outliers")
    from scipy.stats import zscore
    df['z_score'] = zscore(df['ContractValue'].fillna(0))
    outliers = df[df['z_score'] > 3]
    st.write("These contracts are significantly above average in value and might be worth a closer look:")
    st.dataframe(outliers[['PR_ID', 'VendorName', 'ContractValue', 'z_score']])

    # Optional: Vendor spend clustering
    st.subheader("📊 Vendor Tier Clustering")
    vendor_df = spend_by_vendor.reset_index().rename(columns={'ContractValue': 'TotalSpend'})
    scaler = StandardScaler()
    vendor_df['ScaledSpend'] = scaler.fit_transform(vendor_df[['TotalSpend']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    vendor_df['Cluster'] = kmeans.fit_predict(vendor_df[['ScaledSpend']])
    st.write("Vendors grouped into 3 spend tiers (Low, Medium, High):")
    st.dataframe(vendor_df.sort_values(by='TotalSpend', ascending=False))

    # Option to download final results
    st.sidebar.download_button(
        label="📥 Download Validated PR Data",
        data=df.to_csv(index=False),
        file_name="validated_pr_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload both the PR data CSV and a policy document PDF to begin.")
