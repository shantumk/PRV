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
import logging

# --- Enable debugging logs ---
logging.basicConfig(level=logging.DEBUG)
st.write("🔧 Debug: Streamlit app started")

# --- Hugging Face API Setup ---
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
HF_TOKEN = st.secrets["api"]["hf_token"]
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- Extract plain text from uploaded PDF policy ---
def extract_text_from_pdf(uploaded_file):
    st.write("🔧 Debug: Extracting text from PDF")
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

# --- LLM call for PR validation ---
def llm_validate(row, policy_text):
    st.write(f"🔧 Debug: Validating PR ID {row['PR_ID']}")
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
        st.error(f"🔧 API Error {response.status_code}: {response.text}")
        return f"Error: {response.status_code}"
