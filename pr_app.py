import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import RidgeCV

# Optional LLM import
try:
    from huggingface_hub import InferenceClient
    hf_hub_installed = True
except ImportError:
    hf_hub_installed = False

# --- Page Config ---
st.set_page_config(page_title="🛡️ Corruption Risk Analyzer for Procurement", layout="wide")

# --- Intro Section ---
st.title("🛡️ Corruption Risk Analyzer for Procurement")
st.markdown("""
This app helps procurement officers, auditors, and analysts **detect high-risk tenders** in public or private procurement data using explainable machine learning.

**Upload your data → Automatically map → Clean and visualize → Predict risk → Prioritize supplier audits.**

### 📌 Objective:
To identify corruption-prone tenders using a data-driven approach, highlight high-risk vendors, and assist in prioritizing audit efforts using ML-based risk scores and domain context.
""")

# --- LLM Setup ---
hf_ready = False
if hf_hub_installed:
    try:
        hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            hf_infer = InferenceClient(model="google/flan-t5-small", token=hf_token)
            hf_ready = True
        else:
            st.warning("⚠️ LLM token missing; descriptive summaries disabled.")
    except Exception:
        st.warning("⚠️ Error initializing LLM; summaries disabled.")
else:
    st.warning("⚠️ huggingface_hub not installed; LLM features disabled.")

# --- Cached Functions ---
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return pd.DataFrame()

@st.cache_data
def clean_and_engineer(df, mapping):
    df = df.rename(columns={mapping[k]: k for k in mapping if mapping[k]})

    required_fields = ['TenderID', 'Value', 'BidCount', 'ProcedureType', 'CRI', 'Vendor']
    missing_cols = [col for col in required_fields if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Required columns missing from uploaded file: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), 0, 0, []

    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['BidCount'] = pd.to_numeric(df['BidCount'], errors='coerce')
    df['CRI'] = pd.to_numeric(df['CRI'], errors='coerce')

    before = len(df)
    df = df.dropna(subset=required_fields).reset_index(drop=True)
    after = len(df)

    if after == 0:
        st.error("❌ All rows were dropped during cleaning. Ensure your data has values in the required fields.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), before, after, []

    df['high_risk'] = (df['CRI'] >= 0.5).astype(int)
    X = pd.DataFrame({
        'Value_log': np.log1p(df['Value']),
        'BidCount_log': np.log1p(df['BidCount'])
    })
    proc = pd.get_dummies(df['ProcedureType'].astype(str), prefix='ptype', dummy_na=True)
    X = pd.concat([X, proc], axis=1)
    corr_cols = [c for c in df.columns if c.startswith('corr_')]
    for c in corr_cols:
        X[c] = df[c].fillna(0)
    y = df['high_risk']
    return df, X, y, before, after, corr_cols

# Describe LLM helper
def describe_field(text):
    if not hf_ready:
        return 'LLM unavailable'
    try:
        return hf_infer.text_generation(prompt=text, max_new_tokens=100).strip()
    except Exception as e:
        return f'LLM error: {e}'

# --- Sidebar: Upload & Map ---
st.sidebar.header("📂 Upload Your Procurement CSV")
file = st.sidebar.file_uploader("Upload CSV", type="csv")
if not file:
    st.info("Please upload your procurement CSV file.")
    st.stop()

raw = load_data(file)
cols = raw.columns.tolist()
st.sidebar.success(f"Loaded {len(raw):,} rows")

# --- Auto-mapping ---
def am(opts):
    for c in cols:
        for o in opts:
            if o.lower() in c.lower():
                return c
    return None

default_map = {
    'TenderID': am(['tender_id', 'id']),
    'Value': am(['value', 'amount']),
    'BidCount': am(['bidcount', 'bids']),
    'ProcedureType': am(['procedure', 'type']),
    'CRI': am(['cri', 'risk']),
    'Vendor': am(['vendor', 'supplier']),
}

mapping = {}
st.sidebar.subheader("🔧 Confirm Column Mapping")
for key, default in default_map.items():
    mapping[key] = st.sidebar.selectbox(key, [''] + cols, index=(cols.index(default)+1 if default in cols else 0))

df, X, y, before, after, corr_cols = clean_and_engineer(raw, mapping)
if df.empty or X.empty or y.empty:
    st.stop()

# --- Tabs ---
tabs = st.tabs(["📊 EDA", "🤖 Modeling", "📝 Prediction", "🚩 Suppliers", "📋 Summary", "🧮 Predict CRI"])

# --- Tab 1: EDA ---
with tabs[0]:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("""
    **Purpose:** To understand data quality, distribution, and spot potential risk signals.
    """)
    if hf_ready and st.button("💡 Generate Dataset Summary"):
        insight = describe_field(f"Summarize a procurement dataset with {len(df)} rows and {df.shape[1]} fields.")
        st.info(insight)

    st.subheader("1. Field Availability (% non-null values)")
    avail = df.notnull().mean() * 100
    st.bar_chart(avail)

    st.markdown("This helps check for missing data and evaluate field reliability.")

    st.subheader("2. Spend vs Bid Count with Risk Coloring")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="BidCount", y="Value", hue="high_risk", ax=ax)
    ax.set_xscale("log"); ax.set_yscale("log")
    st.pyplot(fig)
    st.markdown("This identifies low-competition, high-value tenders—often at higher risk.")

    st.subheader("3. CRI Distribution by Risk Label")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="high_risk", y="CRI", ax=ax)
    st.pyplot(fig)
    st.markdown("This shows that high-risk tenders typically have higher CRI scores.")

# --- Tab 2: Modeling ---
with tabs[1]:
    st.header("🤖 Model Training")
    n_trees = st.slider("Random Forest Trees", 10, 300, 100)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    metrics = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [accuracy_score(y_test, lr.predict(X_test)), accuracy_score(y_test, rf.predict(X_test))],
        "AUC": [roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]), roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])]
    }).set_index("Model")

    st.subheader("Model Comparison (Accuracy & AUC)")
    st.dataframe(metrics.style.format("{:.3f}"))

    st.subheader("Confusion Matrix (Random Forest)")
    cm = confusion_matrix(y_test, rf.predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    st.markdown("Use confusion matrix to understand False Positives (unnecessary audits) and False Negatives (missed risks).")

# --- Tab 6: Predict CRI using Regression ---
with tabs[5]:
    st.header("🧮 Predict CRI Score Based on Other Features")
    target_cols = [c for c in df.columns if c not in ['CRI', 'high_risk', 'Vendor', 'TenderID'] and df[c].dtype in [np.float64, np.int64]]
    if len(target_cols) < 2:
        st.warning("Not enough numerical features to build CRI regression model.")
    else:
        st.write("Training on features:", target_cols)
        X_reg = df[target_cols].dropna()
        y_reg = df.loc[X_reg.index, 'CRI']

        X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        model = RidgeCV(alphas=np.logspace(-3, 3, 7))
        model.fit(X_train, y_train)

        st.subheader("Model Coefficients")
        coef_df = pd.DataFrame({"Feature": target_cols, "Coefficient": model.coef_})
        st.dataframe(coef_df)

        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_squared_error, r2_score
        st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
        st.write(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual CRI"); ax.set_ylabel("Predicted CRI")
        ax.set_title("CRI Prediction vs Actual")
        st.pyplot(fig)

        st.markdown("This helps explore whether CRI can be reliably estimated from the features. Strong R² would indicate internal consistency.")
