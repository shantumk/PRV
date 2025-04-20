import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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
def clean_and_engineer(df, mapping, threshold=0.45):
    df = df.rename(columns={mapping[k]: k for k in mapping if mapping[k]})

    # Drop columns below threshold
    availability = df.notnull().mean()
    df = df[availability[availability >= threshold].index.tolist()]

    required_fields = ['TenderID', 'Value', 'BidCount', 'ProcedureType', 'CRI', 'Vendor']
    missing_cols = [col for col in required_fields if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Required columns missing: {missing_cols}")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), 0, 0, []

    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['BidCount'] = pd.to_numeric(df['BidCount'], errors='coerce')
    df['CRI'] = pd.to_numeric(df['CRI'], errors='coerce')

    before = len(df)
    df = df.dropna(subset=required_fields).reset_index(drop=True)
    after = len(df)

    if after == 0:
        st.error("❌ All rows dropped. Check required fields.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(), before, after, []

    df['high_risk'] = (df['CRI'] >= 0.5).astype(int)
    # Feature matrix
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

# Threshold Slider
threshold = st.sidebar.slider("Minimum Column Availability (%)", 10, 100, 45) / 100

# --- Auto-mapping ---
def am(opts):
    for c in cols:
        for o in opts:
            if o.lower() in c.lower():
                return c
    return None

default_map = {
    'TenderID': am(['tender_id','id']),
    'Value': am(['value','amount']),
    'BidCount': am(['bidcount','bids']),
    'ProcedureType': am(['procedure','type']),
    'CRI': am(['cri','risk']),
    'Vendor': am(['vendor','supplier']),
}

mapping = {}
st.sidebar.subheader("🔧 Confirm Column Mapping")
for key, default in default_map.items():
    mapping[key] = st.sidebar.selectbox(key, ['']+cols, index=(cols.index(default)+1 if default in cols else 0))

# Clean and prepare data
df, X, y, before, after, corr_cols = clean_and_engineer(raw, mapping, threshold)
if df.empty or X.empty or y.empty:
    st.stop()

# --- Tabs ---
tabs = st.tabs(["📊 EDA","🤖 Modeling","📝 Prediction","🚩 Suppliers","📋 Summary","🧮 Predict CRI"])

# Tab 1: EDA
with tabs[0]:
    st.header("📊 Exploratory Data Analysis")
    st.markdown("**Purpose:** Understand data quality and risk signals.")
    if hf_ready and st.button("💡 Generate Dataset Summary"):
        insight = describe_field(f"Summarize a procurement dataset with {len(df)} rows and {df.shape[1]} fields.")
        st.info(insight)
    # Availability
    st.subheader("Field Availability (% non-null)")
    st.bar_chart(df.notnull().mean()*100)
    # Spend vs Bid Count
    st.subheader("Spend vs Bid Count")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='BidCount', y='Value', hue='high_risk', ax=ax)
    ax.set_xscale('log'); ax.set_yscale('log')
    st.pyplot(fig)
    # CRI distribution
    st.subheader("CRI Distribution by Risk")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='high_risk', y='CRI', ax=ax)
    st.pyplot(fig)

# Tab 2: Modeling
with tabs[1]:
    st.header("🤖 Model Training & Evaluation")
    n_trees = st.slider("Random Forest Trees", 10, 300, 100)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, stratify=y, random_state=42)
    lr = LogisticRegression(max_iter=500); rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    lr.fit(X_train,y_train); rf.fit(X_train,y_train)
    # Metrics
    p_lr, pr_lr = lr.predict(X_test), lr.predict_proba(X_test)[:,1]
    p_rf, pr_rf = rf.predict(X_test), rf.predict_proba(X_test)[:,1]
    metrics = pd.DataFrame({'Model':['LR','RF'], 'Accuracy':[accuracy_score(y_test,p_lr),accuracy_score(y_test,p_rf)], 'AUC':[roc_auc_score(y_test,pr_lr),roc_auc_score(y_test,pr_rf)]}).set_index('Model')
    st.subheader("Model Comparison")
    st.dataframe(metrics.style.format('{:.3f}'))
    # Confusion Matrix
    st.subheader("Confusion Matrix (Random Forest)")
    cm = confusion_matrix(y_test, rf.predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

# Tab 3: Prediction
with tabs[2]:
    st.header("📝 Predict Risk on New Data")
    new_file = st.file_uploader("Upload New Tenders CSV", type="csv", key="pred")
    if new_file:
        nd = load_data(new_file)
        nd.rename(columns={mapping[k]:k for k in mapping if mapping[k]}, inplace=True)
        X_new = pd.DataFrame(0, index=nd.index, columns=X.columns)
        X_new['Value_log'] = np.log1p(pd.to_numeric(nd.get('Value',0),errors='coerce'))
        X_new['BidCount_log'] = np.log1p(pd.to_numeric(nd.get('BidCount',0),errors='coerce'))
        proc_new = pd.get_dummies(nd.get('ProcedureType',pd.Series()), prefix='ptype', dummy_na=True)
        for col in proc_new.columns:
            if col in X_new.columns: X_new[col] = proc_new[col]
        for c in corr_cols: X_new[c] = nd.get(c,0)
        nd['risk_prob']=rf.predict_proba(X_new)[:,1]; nd['risk_label']=(nd['risk_prob']>=0.5).astype(int)
        st.dataframe(nd[['TenderID','Vendor','risk_prob','risk_label']])
        st.download_button("Download Predictions", nd.to_csv(index=False), file_name="predictions.csv")

# Tab 4: Suppliers
with tabs[3]:
    st.header("🚩 High-Risk Suppliers")
    hr = df[df['high_risk']==1]
    if hr.empty:
        st.info("No high-risk suppliers found.")
    else:
        spend = hr.groupby('Vendor')['Value'].sum().nlargest(10)
        fig, ax = plt.subplots()
        spend.plot(kind='bar', color='red', ax=ax)
        ax.set_title("Top 10 High-Risk Vendors by Spend")
        ax.set_ylabel("Total Spend")
        st.pyplot(fig)

# Tab 5: Summary
with tabs[4]:
    st.header("📋 Executive Summary")
    high_risk_pct = y.mean()*100
    auc_full = roc_auc_score(y, rf.predict_proba(X)[:,1])
    st.markdown(f"- **Records:** {len(df)}\n- **High-risk:** {y.sum()} ({high_risk_pct:.1f}%)\n- **RF AUC (full data):** {auc_full:.3f}")
    st.markdown("_View modeling tab for train/test evaluation._")

# Tab 6: Predict CRI
with tabs[5]:
    st.header("🧮 Predict CRI Score from Features")
    target_cols = [c for c in df.columns if c not in ['CRI','high_risk','TenderID','Vendor'] and df[c].dtype in [np.float64,np.int64]]
    if len(target_cols)<2:
        st.warning("Not enough numerical features for CRI regression.")
    else:
        X_reg = df[target_cols].dropna()
        y_reg = df.loc[X_reg.index,'CRI']
        X_tr, X_te, y_tr, y_te = train_test_split(X_reg,y_reg,test_size=0.2, random_state=42)
        model = RidgeCV(alphas=np.logspace(-3,3,7)); model.fit(X_tr,y_tr)
        coef_df = pd.DataFrame({'Feature':target_cols,'Coefficient':model.coef_})
        st.subheader("Regression Coefficients")
        st.dataframe(coef_df)
        from sklearn.metrics import r2_score, mean_squared_error
        y_pred = model.predict(X_te)
        st.write(f"R²: {r2_score(y_te,y_pred):.3f}, MSE: {mean_squared_error(y_te,y_pred):.3f}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_te, y=y_pred, ax=ax)
        ax.set_xlabel("Actual CRI"); ax.set_ylabel("Predicted CRI")
        st.pyplot(fig)
    
    st.balloons()
