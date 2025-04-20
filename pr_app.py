import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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

# --- Page Config & Styling ---
st.set_page_config(page_title="🛡️ Corruption Risk Analyzer", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #F5F7FA; }
    .stApp { color: #0E1117; }
    .metric-value { font-size: 1.5rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True
)

# --- Header ---
st.markdown("# 🛡️ Corruption Risk Analyzer for Procurement")
st.markdown("""
Optimize audit prioritization by spotting high-risk tenders in your procurement data using explainable ML.
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
            st.warning("⚠️ LLM token missing; summaries disabled.")
    except Exception:
        st.warning("⚠️ Error initializing LLM; summaries disabled.")
else:
    st.warning("⚠️ huggingface_hub not installed; LLM features disabled.")

# --- Cached Functions ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def clean_and_engineer(df, mapping, threshold=0.45):
    df = df.rename(columns={mapping[k]: k for k in mapping if mapping[k]})
    avail = df.notnull().mean()
    df = df[avail[avail >= threshold].index]
    req = ['TenderID','Value','BidCount','ProcedureType','CRI','Vendor']
    if any(c not in df.columns for c in req):
        return None
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['BidCount'] = pd.to_numeric(df['BidCount'], errors='coerce')
    df['CRI'] = pd.to_numeric(df['CRI'], errors='coerce')
    before = len(df)
    df.dropna(subset=req, inplace=True)
    after = len(df)
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
st.sidebar.header("📂 Upload & Map")
file = st.sidebar.file_uploader("Upload Procurement CSV", type='csv')
if not file:
    st.info("Upload a CSV to get started.")
    st.stop()
raw = load_data(file)
cols = raw.columns.tolist()
st.sidebar.success(f"{len(raw):,} rows loaded")
thresh = st.sidebar.slider("Min column availability (%)", 10, 100, 45) / 100

def am(opts):
    for col in cols:
        if any(o.lower() in col.lower() for o in opts):
            return col
    return ''
def_map = {
    'TenderID': am(['tender_id','id']),
    'Value': am(['value','amount']),
    'BidCount': am(['bidcount','bids']),
    'ProcedureType': am(['procedure','type']),
    'CRI': am(['cri','risk']),
    'Vendor': am(['vendor','supplier']),
}
st.sidebar.subheader("🔧 Confirm Mapping")
mapping = {k: st.sidebar.selectbox(k, ['']+cols, index=(cols.index(v)+1 if v in cols else 0))
           for k, v in def_map.items()}
res = clean_and_engineer(raw, mapping, thresh)
if not res:
    st.error("Mapping or threshold issue — adjust inputs.")
    st.stop()
df, X, y, before, after, corr_cols = res
risk_pct = y.mean() * 100

# --- Top Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw Rows", f"{len(raw):,}")
col2.metric("Kept Rows", f"{len(df):,}", delta=f"-{before - after:,}")
col3.metric("High-Risk %", f"{risk_pct:.1f}%")
auc_full = roc_auc_score(y, RandomForestClassifier(n_estimators=100).fit(X, y).predict_proba(X)[:,1])
col4.metric("Model AUC", f"{auc_full:.3f}")

# --- Tabs ---
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["📊 EDA", "🤖 Modeling", "📝 Prediction", "🚩 Suppliers", "📋 Summary", "🧮 Predict CRI"]
)

# Tab 1: EDA
with t1:
    st.header("📊 Exploratory Data Analysis")
    if hf_ready and st.button("💡 Generate Insight"):
        insight = describe_field(
            f"Procurement dataset with {len(df)} rows and {df.shape[1]} columns."
        )
        st.info(insight)
    st.subheader("Field Availability")
    fig1 = px.bar(x=df.notnull().mean() * 100, labels={'x': '% Non-null', 'index': 'Field'})
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader("Spend vs Bid Count")
    fig2 = px.scatter(df, x='BidCount', y='Value', color=df['high_risk'].map({0:'Low',1:'High'}), log_x=True, log_y=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("CRI Distribution by Risk")
    fig3 = px.box(df, x='high_risk', y='CRI', labels={'high_risk':'Risk Label', 'CRI':'CRI Score'})
    st.plotly_chart(fig3, use_container_width=True)

# Tab 2: Modeling
with t2:
    st.header("🤖 Model Training & Evaluation")
    n_trees = st.slider("RF Trees", 10, 300, 100)
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    met = pd.DataFrame({
        'Model': ['LR', 'RF'],
        'Accuracy': [accuracy_score(y_test, lr.predict(X_test)), accuracy_score(y_test, rf.predict(X_test))],
        'AUC': [roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]), roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])]
    }).set_index('Model')
    st.subheader("Metrics")
    st.dataframe(met.style.format("{:.3f}"))
    st.subheader("Confusion Matrix (RF)")
    cm = confusion_matrix(y_test, rf.predict(X_test))
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.pyplot(fig_cm)

# Tab 3: Prediction
with t3:
    st.header("📝 Predict Risk on New Data")
    nf = st.file_uploader("Upload New CSV", type='csv', key='pred')
    if nf:
        nd = load_data(nf)
        nd.rename(columns={mapping[k]:k for k in mapping if mapping[k]}, inplace=True)
        X_new = pd.DataFrame(0, index=nd.index, columns=X.columns)
        X_new['Value_log'] = np.log1p(pd.to_numeric(nd['Value'],errors='coerce'))
        X_new['BidCount_log'] = np.log1p(pd.to_numeric(nd['BidCount'],errors='coerce'))
        proc = pd.get_dummies(nd['ProcedureType'].astype(str), prefix='ptype', dummy_na=True)
        for c in proc.columns:
            if c in X_new: X_new[c] = proc[c]
        for c in corr_cols:
            X_new[c] = nd.get(c, 0)
        probs = rf.predict_proba(X_new)[:,1]
        nd['risk_prob'] = probs
        nd['risk_label'] = (probs>=0.5).astype(int)
        st.dataframe(nd[['TenderID','Vendor','risk_prob','risk_label']])
        st.download_button("Download Predictions", nd.to_csv(index=False), "preds.csv")

# Tab 4: Suppliers
with t4:
    st.header("🚩 High-Risk Suppliers")
    hr = df[df['high_risk']==1]
    top_sup = hr.groupby('Vendor')['Value'].sum().nlargest(10)
    fig4 = px.pie(values=top_sup.values, names=top_sup.index, hole=0.4)
    st.plotly_chart(fig4, use_container_width=True)

# Tab 5: Summary
with t5:
    st.header("📋 Executive Summary")
    st.markdown(f"- **Kept Records:** {len(df)}  ")
    st.markdown(f"- **High-Risk Tenders:** {int(y.sum())} ({risk_pct:.1f}%)  ")
    st.markdown(f"- **Full Model AUC:** {auc_full:.3f}  ")
    st.balloons()

# Tab 6: Predict CRI
with t6:
    st.header("🧮 Predict CRI Score")
    num_feats = [c for c in df.columns if df[c].dtype in [np.float64,int] and c not in ['CRI','high_risk']]
    if len(num_feats)>1:
        Xr = df[num_feats].dropna()
        yr = df.loc[Xr.index,'CRI']
        Xtr, Xte, ytr, yte = train_test_split(Xr,yr,test_size=0.2,random_state=42)
        model = RidgeCV(alphas=np.logspace(-3,3,7))
        model.fit(Xtr,ytr)
        coefs = pd.DataFrame({'Feature':num_feats,'Coef':model.coef_})
        st.subheader("Coefficients")
        st.dataframe(coefs)
        preds = model.predict(Xte)
        st.write(f"R²: {r2_score(yte,preds):.3f}")
        fig5 = px.scatter(x=yte, y=preds, labels={'x':'Actual CRI','y':'Predicted CRI'})
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Not enough features to predict CRI.")
