import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Optional LLM import
try:
    from huggingface_hub import InferenceApi
    hf_hub_installed = True
except ImportError:
    hf_hub_installed = False
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
# Optional SHAP import for explainability
try:
    import shap
    shap_installed = True
except ImportError:
    shap_installed = False

# --- Page Config ---
st.set_page_config(page_title="Procurement Corruption-Risk Prediction", layout="wide")

# --- LLM Setup ---
hf_ready = False
if hf_hub_installed:
    try:
        hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
        if hf_token:
            hf_infer = InferenceApi(repo_id="google/flan-t5-small", token=hf_token)
            hf_ready = True
        else:
            st.warning("⚠️ LLM token missing; descriptive summaries disabled.")
    except Exception:
        st.warning("⚠️ Error initializing LLM; summaries disabled.")
else:
    st.warning("⚠️ huggingface_hub not installed; LLM features disabled.")

# --- Cached Functions ---
@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner=False)
def clean_and_engineer(df, mapping):
    # Rename
    df = df.rename(columns={mapping[k]:k for k in mapping if mapping[k]})
    # Convert
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["BidCount"] = pd.to_numeric(df["BidCount"], errors="coerce")
    df["CRI"] = pd.to_numeric(df["CRI"], errors="coerce")
    # Clean & report
    before = len(df)
    df = df.dropna(subset=["TenderID","Value","BidCount","ProcedureType","CRI","Vendor"]).reset_index(drop=True)
    after = len(df)
    # Label
    df["high_risk"] = (df["CRI"] >= 0.5).astype(int)
    # Feature engineering
    X = pd.DataFrame({
        "Value_log": np.log1p(df["Value"]),
        "BidCount_log": np.log1p(df["BidCount"])
    })
    proc = pd.get_dummies(df["ProcedureType"], prefix="ptype", dummy_na=True)
    X = pd.concat([X, proc], axis=1)
    corr_cols = [c for c in df.columns if c.startswith("corr_")]
    for c in corr_cols:
        X[c] = df[c].fillna(0)
    y = df["high_risk"]
    return df, X, y, before, after, corr_cols

# Describe helper
def describe_field(field):
    if not hf_ready:
        return "LLM unavailable"
    prompt = (f"You are a procurement expert. Explain what `{field}` represents and its relevance to corruption risk in 2-3 sentences.")
    try:
        resp = hf_infer(inputs=prompt, parameters={"max_new_tokens":100})
        if isinstance(resp, list) and "generated_text" in resp[0]:
            return resp[0]["generated_text"].strip()
        if isinstance(resp, str):
            return resp.strip()
        return str(resp)
    except Exception as e:
        return f"LLM error: {e}"

# --- Sidebar & Data Upload ---
st.title("🔍 Procurement Corruption-Risk Prediction Dashboard")
st.sidebar.header("1. Upload & Map Data")
file = st.sidebar.file_uploader("Procurement CSV (<200MB)", type="csv")
if not file:
    st.info("Please upload your procurement CSV to proceed.")
    st.stop()
raw = load_data(file)
st.sidebar.success(f"Loaded {len(raw):,} rows")
cols = raw.columns.tolist()

# Auto-mapping helper
def am(opts):
    for c in cols:
        for o in opts:
            if o.lower() in c.lower(): return c
    return None

def_map = {
    "TenderID": am(["tender_id","id","ref"]),
    "Value": am(["value","amount"]),
    "BidCount": am(["bidcount","bids"]),
    "ProcedureType": am(["procedure","type"]),
    "CRI": am(["cri","risk"]),
    "Vendor": am(["vendor","supplier"])
}

mapping = {}
st.sidebar.subheader("Confirm Mapping")
for k, d in def_map.items():
    mapping[k] = st.sidebar.selectbox(k, [""]+cols, index=(cols.index(d)+1 if d in cols else 0))

# Clean & feature
try:
    df, X, y, before, after, corr_cols = clean_and_engineer(raw, mapping)
except KeyError:
    st.error("Mapping incomplete or columns missing. Please adjust.")
    st.stop()
st.sidebar.write(f"Rows before: {before}, after cleaning: {after}")

# --- Tabs ---
tabs = st.tabs(["📈 EDA","🤖 Modeling","📝 Prediction","🚩 Suppliers","📋 Summary"])

# EDA
with tabs[0]:
    st.header("Exploratory Data Analysis")
    # LLM summary
    if hf_ready and st.button("Generate Dataset Insight"):
        stats = {"records": len(df), "fields": df.shape[1], "dropped_pct": round((before-after)/before*100,2)}
        prompt = (f"Dataset has {stats['records']} records, {stats['fields']} fields, "
                  f"and {stats['dropped_pct']}% rows dropped. Provide a high-level insight.")
        st.write(describe_field(prompt))
    # Data availability
    avail = df.notnull().mean()*100
    choice = st.radio("Availability view", ["Top 10","Bottom 10","Full Table"], horizontal=True)
    if choice != "Full Table":
        sel = avail.nlargest(10) if choice=="Top 10" else avail.nsmallest(10)
        fig, ax = plt.subplots(figsize=(6,4))
        sel.sort_values().plot.barh(ax=ax, color='skyblue'); ax.set_xlabel("% Non-null"); st.pyplot(fig)
    else:
        st.dataframe(avail.sort_values(ascending=False).to_frame("Availability (%)"))
    # Scatter & box
    fig, ax = plt.subplots(); sns.scatterplot(df, x="BidCount", y="Value", hue="high_risk", ax=ax); ax.set(xscale='log',yscale='log'); st.pyplot(fig)
    fig, ax = plt.subplots(); sns.boxplot(x="high_risk", y="CRI", data=df, ax=ax); st.pyplot(fig)

# Modeling
with tabs[1]:
    st.header("Model Training & Evaluation")
    exp = st.expander("Settings")
    with exp:
        n_trees = st.number_input("RF trees",10,500,100)
        test_frac = st.slider("Test size",0.1,0.5,0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=test_frac, stratify=y, random_state=42)
    lr = LogisticRegression(max_iter=500); rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr)
    if shap_installed:
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_tr)
        st.subheader("SHAP Summary")
        shap.summary_plot(shap_vals[1], X_tr, show=False)
        fig_shap = plt.gcf()
        st.pyplot(fig_shap)
    else:
        st.warning("⚠️ SHAP library not installed; skipping SHAP explanation.")
    p_lr, pr_lr = lr.predict(X_te), lr.predict_proba(X_te)[:,1]
    p_rf, pr_rf = rf.predict(X_te), rf.predict_proba(X_te)[:,1]
    met = pd.DataFrame({"Model":["LR","RF"],"AUC":[roc_auc_score(y_te,pr_lr),roc_auc_score(y_te,pr_rf)],"Acc":[accuracy_score(y_te,p_lr),accuracy_score(y_te,p_rf)]}).set_index("Model")
    st.dataframe(met)

# Prediction
with tabs[2]:
    st.header("Model Prediction")
    st.info("Upload a new CSV under the '📝 Prediction' tab to get risk predictions for unseen tenders.")
    new_file = st.file_uploader("Upload new tenders CSV", type="csv", key="pred")
    if new_file:
        nd = load_data(new_file)
        try:
            nd.rename(columns={mapping[k]:k for k in mapping if mapping[k]}, inplace=True)
            # Feature engineering for new data
            X_new = pd.DataFrame({"Value_log": np.log1p(pd.to_numeric(nd.get('Value',0),errors='coerce')),
                                   "BidCount_log": np.log1p(pd.to_numeric(nd.get('BidCount',0),errors='coerce'))})
            proc_new = pd.get_dummies(nd.get('ProcedureType', pd.Series()), prefix="ptype", dummy_na=True)
            X_new = pd.concat([X_new, proc.reindex(proc_new.index, fill_value=0)], axis=1)
            for c in corr_cols:
                X_new[c] = nd.get(c,0)
            probs = rf.predict_proba(X_new)[:,1]
            nd['risk_prob'] = probs; nd['risk_label'] = (probs>=0.5).astype(int)
            st.dataframe(nd[["TenderID","Vendor","risk_prob","risk_label"]])
            st.download_button("Download Predictions", nd.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Suppliers
with tabs[3]:
    st.header("High-Risk Suppliers")
    hr = df[df["high_risk"]==1]; spend = hr.groupby("Vendor")["Value"].sum().nlargest(10)
    fig, ax = plt.subplots(); spend.plot.pie(ax=ax, autopct='%1.1f%%'); st.pyplot(fig)

# Summary
with tabs[4]:
    st.header("Executive Summary")
    insights = [f"Total: {len(df)} records", f"High-risk: {int(y.sum())} ({y.mean()*100:.1f}%)", f"RF AUC: {roc_auc_score(y_te,pr_rf):.3f}"]
    for i in insights: st.write("- "+i)
    st.balloons()
