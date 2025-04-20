import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
# Optional LLM import
try:
    from huggingface_hub import InferenceApi
    hf_hub_installed = True
except ImportError:
    hf_hub_installed = False

# --- Page Config ---
st.set_page_config(page_title="Procurement Corruption-Risk Prediction", layout="wide")

# --- LLM Setup ---
hf_ready = False
if hf_hub_installed:
    try:
        hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
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
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def clean_and_engineer(df, mapping):
    # Rename columns
    df = df.rename(columns={mapping[k]:k for k in mapping if mapping[k]})
    # Convert numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['BidCount'] = pd.to_numeric(df['BidCount'], errors='coerce')
    df['CRI'] = pd.to_numeric(df['CRI'], errors='coerce')
    # Drop incomplete
    before = len(df)
    df = df.dropna(subset=['TenderID','Value','BidCount','ProcedureType','CRI','Vendor']).reset_index(drop=True)
    after = len(df)
    # Label
    df['high_risk'] = (df['CRI'] >= 0.5).astype(int)
    # Base features
    X = pd.DataFrame({
        'Value_log': np.log1p(df['Value']),
        'BidCount_log': np.log1p(df['BidCount'])
    })
    # ProcedureType dummies
    proc = pd.get_dummies(df['ProcedureType'], prefix='ptype', dummy_na=True)
    X = pd.concat([X, proc], axis=1)
    # Any corr_ columns
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
        resp = hf_infer(inputs=text, parameters={'max_new_tokens':100})
        if isinstance(resp, list) and 'generated_text' in resp[0]:
            return resp[0]['generated_text'].strip()
        if isinstance(resp, str):
            return resp.strip()
        return str(resp)
    except Exception as e:
        return f'LLM error: {e}'

# --- Sidebar: Upload & Map ---
st.title('🔍 Procurement Corruption-Risk Prediction')
st.sidebar.header('1. Upload & Map Data')
file = st.sidebar.file_uploader('Procurement CSV (<200MB)', type='csv')
if not file:
    st.info('Upload a CSV to begin analysis.')
    st.stop()
raw = load_data(file)
st.sidebar.success(f'Loaded {len(raw):,} rows')
cols = raw.columns.tolist()

# Auto-mapping defaults
def am(opts):
    for c in cols:
        for o in opts:
            if o.lower() in c.lower(): return c
    return None

def_map = {
    'TenderID': am(['tender_id','id','ref']),
    'Value': am(['value','amount']),
    'BidCount': am(['bidcount','bids']),
    'ProcedureType': am(['procedure','type']),
    'CRI': am(['cri','risk']),
    'Vendor': am(['vendor','supplier'])
}
mapping = {}
st.sidebar.subheader('Confirm Mappings')
for k, d in def_map.items():
    mapping[k] = st.sidebar.selectbox(k, ['']+cols, index=(cols.index(d)+1 if d in cols else 0))

# Clean & engineer
try:
    df, X, y, before, after, corr_cols = clean_and_engineer(raw, mapping)
except KeyError:
    st.error('Mapping error: please adjust your column selections.')
    st.stop()
st.sidebar.write(f'Rows before: {before}, after clean: {after}')

# --- Tabs Setup ---
tabs = st.tabs(['📈 EDA','🤖 Modeling','📝 Prediction','🚩 Suppliers','📋 Summary'])

# EDA Tab
with tabs[0]:
    st.header('Exploratory Data Analysis')
    # LLM insight
    if hf_ready and st.button('Generate Insight'):
        stats = f'Dataset: {len(df)} records, {df.shape[1]} fields, dropped {round((before-after)/before*100,2)}% rows.'
        st.write(describe_field(stats))
    # Availability plot
    avail = df.notnull().mean()*100
    view = st.radio('Availability view', ['Top 10','Bottom 10','All'], horizontal=True)
    if view != 'All':
        sel = avail.nlargest(10) if view=='Top 10' else avail.nsmallest(10)
        fig, ax = plt.subplots(); sel.sort_values().plot.barh(ax=ax); ax.set_xlabel('% non-null'); st.pyplot(fig)
    else:
        st.dataframe(avail.sort_values(ascending=False).to_frame('% non-null'))
    # Scatter
    fig, ax = plt.subplots(); sns.scatterplot(df, x='BidCount', y='Value', hue='high_risk', ax=ax); ax.set(xscale='log', yscale='log'); st.pyplot(fig)
    # Box
    fig, ax = plt.subplots(); sns.boxplot(x='high_risk', y='CRI', data=df, ax=ax); st.pyplot(fig)

# Modeling Tab
with tabs[1]:
    st.header('Model Training & Evaluation')
    exp = st.expander('Settings')
    with exp:
        n_trees = st.number_input('RF trees', 10,500,100)
        test_frac = st.slider('Test size', 0.1,0.5,0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=42)
    lr = LogisticRegression(max_iter=500); rf = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    lr.fit(X_tr, y_tr); rf.fit(X_tr, y_tr)
    # Metrics
    p_lr, pr_lr = lr.predict(X_te), lr.predict_proba(X_te)[:,1]
    p_rf, pr_rf = rf.predict(X_te), rf.predict_proba(X_te)[:,1]
    met = pd.DataFrame({'Model':['LR','RF'],'AUC':[roc_auc_score(y_te,pr_lr),roc_auc_score(y_te,pr_rf)],'Acc':[accuracy_score(y_te,p_lr),accuracy_score(y_te,p_rf)]}).set_index('Model')
    st.table(met)

# Prediction Tab
with tabs[2]:
    st.header('Model Prediction')
    new_file = st.file_uploader('Upload new tenders CSV', type='csv', key='pred')
    if new_file:
        nd = load_data(new_file)
        nd.rename(columns={mapping[k]:k for k in mapping if mapping[k]}, inplace=True)
        # Build X_new with same columns as X
        X_new = pd.DataFrame(0, index=nd.index, columns=X.columns)
        X_new['Value_log'] = np.log1p(pd.to_numeric(nd['Value'], errors='coerce'))
        X_new['BidCount_log'] = np.log1p(pd.to_numeric(nd['BidCount'], errors='coerce'))
        proc_new = pd.get_dummies(nd['ProcedureType'], prefix='ptype', dummy_na=True)
        for col in proc_new.columns:
            if col in X_new.columns:
                X_new[col] = proc_new[col]
        for c in corr_cols:
            X_new[c] = nd.get(c, 0)
        probs = rf.predict_proba(X_new)[:,1]
        nd['risk_prob'] = probs; nd['risk_label'] = (probs>=0.5).astype(int)
        st.dataframe(nd[['TenderID','Vendor','risk_prob','risk_label']])
        st.download_button('Download Predictions', nd.to_csv(index=False), 'predictions.csv')

# Suppliers Tab
with tabs[3]:
    st.header('High-Risk Suppliers')
    hr = df[df['high_risk']==1]; spend = hr.groupby('Vendor')['Value'].sum().nlargest(10)
    fig, ax = plt.subplots(); spend.plot.pie(ax=ax, autopct='%1.1f%%'); st.pyplot(fig)

# Summary Tab
with tabs[4]:
    st.header('Executive Summary')
    vals = f"Total {len(df)} records, {int(y.sum())} high-risk ({y.mean()*100:.1f}%), RF AUC {roc_auc_score(y_te,pr_rf):.3f}"
    st.write(vals)
    st.balloons()
