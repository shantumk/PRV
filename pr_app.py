import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import InferenceApi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# --- Page Configuration ---
st.set_page_config(page_title="Procurement Corruption-Risk Prediction", layout="wide")
st.title("🔍 Procurement Corruption-Risk Prediction Dashboard")

# --- Hugging Face LLM Setup ---
try:
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    hf_infer = InferenceApi(
        repo_id="google/flan-t5-small",
        token=hf_token,
    )
    hf_ready = True
except Exception:
    hf_ready = False
    st.warning(
        "⚠️ Hugging Face token not found. Field descriptions disabled. "
        "Add HUGGINGFACEHUB_API_TOKEN in Secrets."
    )

def describe_field_with_llm(field_name: str) -> str:
    if not hf_ready:
        return "🚫 LLM unavailable (missing token)."
    prompt = (
        f"You are a procurement data expert. "
        f"Explain what the feature `{field_name}` represents in a tender dataset "
        f"and why it matters for assessing corruption risk in 2–3 sentences."
    )
    try:
        resp = hf_infer(inputs=prompt, parameters={"max_new_tokens":100})
        if isinstance(resp, list) and "generated_text" in resp[0]:
            return resp[0]["generated_text"].strip()
        if isinstance(resp, str):
            return resp.strip()
        return str(resp)
    except Exception as e:
        return f"❌ LLM Error: {e}"

# --- Sidebar: Data Upload ---
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Procurement CSV (<200MB)", type=["csv"]
)

if uploaded_file:
    # Load and preview
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data loaded successfully!")
    cols = df.columns.tolist()

    # --- Auto-mapping helper ---
    def auto_map(options):
        for col in cols:
            for opt in options:
                if opt.lower() in col.lower():
                    return col
        return None

    # --- Default Column Mappings ---
    mapping = {
        "TenderID": auto_map(["tender_id","id","ref"]),
        "Value": auto_map(["value","amount","contract"]),
        "BidCount": auto_map(["bidcount","bids","recordedbidscount"]),
        "ProcedureType": auto_map(["procedure","method","type"]),
        "CRI": auto_map(["cri","risk_index","corruption"]),
        "Vendor": auto_map(["vendor","supplier","buyer"]),
    }
    st.sidebar.subheader("🔧 Confirm Column Mapping")
    for key, default in mapping.items():
        mapping[key] = st.sidebar.selectbox(
            f"Select {key} column", [""]+cols,
            index=(cols.index(default)+1 if default in cols else 0)
        )

    # Rename for consistency
    rename_dict = {mapping[k]:k for k in mapping if mapping[k]}
    df = df.rename(columns=rename_dict)

        # Convert types & clean
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["BidCount"] = pd.to_numeric(df["BidCount"], errors="coerce")
    df["CRI"] = pd.to_numeric(df["CRI"], errors="coerce")
    # Report row count before & after dropping missing key fields
    before = len(df)
    df = df.dropna(subset=["TenderID","Value","BidCount","ProcedureType","CRI","Vendor"]).reset_index(drop=True)
    after = len(df)
    st.sidebar.write(f"Rows before cleaning: {before}, after cleaning: {after}")

    # High-risk label
    df["high_risk"] = (df["CRI"] >= 0.5).astype(int)
    corr_cols = [c for c in df.columns if c.startswith("corr_")]

    # Feature engineering
    X_base = pd.DataFrame({
        "Value_log": np.log1p(df["Value"]),
        "BidCount_log": np.log1p(df["BidCount"])
    })
    proc_dummies = pd.get_dummies(df["ProcedureType"], prefix="ptype", dummy_na=True)
    X_base = pd.concat([X_base, proc_dummies], axis=1)
    for c in corr_cols:
        X_base[c] = df[c].fillna(0)
    y_base = df["high_risk"]

    # Tabs for workflow
    tab_eda, tab_model, tab_pred, tab_summary, tab_suppliers = st.tabs([
        "EDA","Modeling","Prediction","Summary","High-Risk Suppliers"
    ])

    # --- EDA Tab ---
    with tab_eda:
        st.header("📊 Exploratory Data Analysis")
        if st.checkbox("📝 Show Field Descriptions"):
            fields = st.multiselect(
                "Select fields to explain", options=list(X_base.columns)+["CRI"],
                default=["Value_log","BidCount_log","CRI"]
            )
            if st.button("Describe selected fields"):
                with st.spinner("Fetching descriptions…"):
                    for f in fields:
                        st.subheader(f)
                        st.write(describe_field_with_llm(f))
        st.markdown("---")
        # Data availability section
        st.subheader("Data Availability")
        avail = df.notnull().mean()*100
        view_mode = st.radio("Availability view:",["Top/Bottom 10","Full Table"], key='avail')
        if view_mode=="Top/Bottom 10":
            tb = st.selectbox("Show:",["Most Complete","Most Missing"], key='tb')
            subset = (avail.sort_values(ascending=False).head(10)
                      if tb=="Most Complete" else avail.sort_values().head(10))
            fig, ax = plt.subplots(figsize=(6, max(2, len(subset)*0.4)))
            subset.plot.barh(ax=ax,color='skyblue')
            ax.set_xlabel("% Non-Null")
            st.pyplot(fig)
        else:
            st.dataframe(avail.sort_values(ascending=False).to_frame("Availability (%)"))
        # Scatterplot
        st.subheader("Value vs. BidCount by Risk")
        fig_s, ax_s = plt.subplots()
        sns.scatterplot(df, x="BidCount", y="Value", hue="high_risk", palette=['green','red'], alpha=0.6, ax=ax_s)
        ax_s.set_xscale('log'); ax_s.set_yscale('log')
        ax_s.set_xlabel("Bid Count (log)"); ax_s.set_ylabel("Value (log)")
        st.pyplot(fig_s)
        # Boxplot CRI
        st.subheader("CRI Distribution by Risk")
        fig_b, ax_b = plt.subplots()
        sns.boxplot(x="high_risk", y="CRI", data=df, ax=ax_b)
        ax_b.set_xticklabels(['Low Risk','High Risk'])
        st.pyplot(fig_b)
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr_df = pd.concat([X_base, df[['CRI']]], axis=1)
        fig_h, ax_h = plt.subplots(figsize=(12,10))
        sns.heatmap(corr_df.corr(), cmap='vlag', center=0, ax=ax_h)
        st.pyplot(fig_h)

    # --- Modeling Tab ---
    with tab_model:
        st.header("🤖 Model Training & Evaluation")
        test_size = st.slider("Test fraction", 0.1,0.5,0.2,0.05)
        X_train, X_test, y_train, y_test = train_test_split(X_base, y_base, stratify=y_base, test_size=test_size, random_state=42)
        st.write(f"Train: {len(X_train)}, Test: {len(X_test)}")
        # Train
        lr = LogisticRegression(max_iter=1000); rf = RandomForestClassifier(n_estimators=100,random_state=42)
        lr.fit(X_train,y_train); rf.fit(X_train,y_train)
        # Feature importances
        st.subheader("Feature Importance (RF)")
        imp = pd.Series(rf.feature_importances_, index=X_base.columns).nlargest(10)
        fig_i, ax_i = plt.subplots(figsize=(8,6))
        imp.sort_values().plot.barh(ax=ax_i, color='teal'); ax_i.set_xlabel('Importance')
        st.pyplot(fig_i)
        # Metrics
        y_pred_lr = lr.predict(X_test); y_prob_lr = lr.predict_proba(X_test)[:,1]
        y_pred_rf = rf.predict(X_test); y_prob_rf = rf.predict_proba(X_test)[:,1]
        metrics = pd.DataFrame({ 'Model':['LogReg','RandomForest'],
                                 'AUC':[roc_auc_score(y_test,y_prob_lr),roc_auc_score(y_test,y_prob_rf)],
                                 'Accuracy':[accuracy_score(y_test,y_pred_lr),accuracy_score(y_test,y_pred_rf)]
                               }).set_index('Model')
        st.dataframe(metrics)
        # ROC
        st.subheader("ROC Curves")
        fpr_lr,tpr_lr,_ = roc_curve(y_test,y_prob_lr)
        fpr_rf,tpr_rf,_ = roc_curve(y_test,y_prob_rf)
        fig_r, ax_r = plt.subplots()
        ax_r.plot(fpr_lr,tpr_lr,label=f"LR AUC={metrics.loc['LogReg','AUC']:.2f}")
        ax_r.plot(fpr_rf,tpr_rf,label=f"RF AUC={metrics.loc['RandomForest','AUC']:.2f}")
        ax_r.plot([0,1],[0,1],'--',color='gray')
        ax_r.set_xlabel('FPR'); ax_r.set_ylabel('TPR'); ax_r.legend(); st.pyplot(fig_r)

    # --- Prediction Tab ---
    with tab_pred:
        st.header("📝 Predictions")
        src = st.radio("Prediction source",["Held-out Test","New File"])
        model = rf if st.selectbox("Model",["RandomForest","LogReg"])=='RandomForest' else lr
        if src=='Held-out Test':
            pr = model.predict(X_test); pp = model.predict_proba(X_test)[:,1]
            st.metric("Accuracy",f"{accuracy_score(y_test,pr):.3f}")
            st.metric("AUC",f"{roc_auc_score(y_test,pp):.3f}")
        else:
            nf = st.file_uploader("Upload new CSV", type='csv', key='new')
            if nf:
                nd = pd.read_csv(nf).rename(columns=rename_dict)
                Xn = pd.DataFrame({"Value_log":np.log1p(pd.to_numeric(nd['Value'],errors='coerce')),
                                   "BidCount_log":np.log1p(pd.to_numeric(nd['BidCount'],errors='coerce'))})
                pdum = pd.get_dummies(nd['ProcedureType'],prefix='ptype',dummy_na=True)
                Xn = pd.concat([Xn, proc_dummies.reindex(pdum.index,fill_value=0)],axis=1)
                for c in corr_cols: Xn[c]=nd.get(c,0)
                nd['risk_prob']=model.predict_proba(Xn)[:,1]
                nd['risk_label']=(nd['risk_prob']>=0.5).astype(int)
                st.dataframe(nd[['TenderID','Vendor','risk_prob','risk_label']])
                st.download_button("Download Predictions", nd.to_csv(index=False), file_name='predictions.csv')

    # --- High-Risk Suppliers Tab ---
    with tab_suppliers:
        st.header("🚩 High-Risk Suppliers Dashboard")
        hr = df[df['high_risk']==1]
        if not hr.empty:
            spend = hr.groupby('Vendor')['Value'].sum().nlargest(10)
            fig_p, ax_p = plt.subplots()
            ax_p.pie(spend, labels=spend.index, autopct='%.1f%%'); ax_p.axis('equal')
            st.subheader('Top 10 Spend-at-Risk Suppliers'); st.pyplot(fig_p)
            flags = hr[corr_cols].sum()
            fig_b2, ax_b2 = plt.subplots(figsize=(8,5))
            flags.plot.bar(ax=ax_b2, color='coral'); ax_b2.set_ylabel('Count'); plt.xticks(rotation=45)
            st.subheader('Common Corruption Flags'); st.pyplot(fig_b2)
            sel = st.selectbox('Drill-down Supplier', spend.index)
            sdata = hr[hr['Vendor']==sel]
            st.write(f"Tenders: {len(sdata)}, Spend-at-risk: {sdata['Value'].sum():,.2f}")
            st.dataframe(sdata[['TenderID','ProcedureType','Value','CRI']])
        else:
            st.info('No high-risk suppliers detected.')

    # --- Summary Tab ---
    with tab_summary:
        st.header('📋 Analysis Summary')
        st.markdown(f"- Total records: **{df.shape[0]}**")
        st.markdown(f"- High-risk tenders: **{int(df['high_risk'].sum())}** ({df['high_risk'].mean()*100:.1f}%)")
        st.markdown(f"- RF AUC: **{roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.3f}** | Acc: **{accuracy_score(y_test, rf.predict(X_test)):.3f}**")
        st.markdown(f"- LR AUC: **{roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]):.3f}** | Acc: **{accuracy_score(y_test, lr.predict(X_test)):.3f}**")
        st.markdown('## Key Insights')
        st.markdown('- Top spend-at-risk suppliers highlighted for review.')
        st.markdown('- Feature importances reveal key risk drivers.')
        st.markdown('- Field descriptions provide context for each metric.')

else:
    st.info("Please upload a procurement CSV (<200MB) to begin analysis.")
