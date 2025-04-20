import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import requests

# --- Hugging Face LLM Setup (Field Descriptions) ---
HF_MODEL = "google/flan-t5-small"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = st.secrets.get("hf_token", "")  # store your HF token under secrets.toml as hf_token
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}  

def describe_field_with_llm(field_name, context="procurement risk analysis"):
    prompt = f"Describe the meaning and relevance of the '{field_name}' field in the context of {context}. Provide why it's important for corruption-risk prediction."
    payload = {"inputs": prompt}
    try:
        r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        if r.status_code == 200:
            # assume model returns a list of dicts with 'generated_text'
            resp = r.json()
            if isinstance(resp, list) and 'generated_text' in resp[0]:
                return resp[0]['generated_text'].strip()
            return resp
        else:
            return f"LLM Error {r.status_code}: {r.text}"  
    except Exception as e:
        return f"Request failed: {e}"
import requests

# --- Hugging Face LLM Setup (Field Descriptions) ---
HF_MODEL = "google/flan-t5-small"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HF_TOKEN = st.secrets.get("hf_token", "")  # store your HF token under secrets.toml as hf_token
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}  

def describe_field_with_llm(field_name, context="procurement risk analysis"):
    prompt = f"Describe the meaning and relevance of the '{field_name}' field in the context of {context}. Provide why it's important for corruption-risk prediction."
    payload = {"inputs": prompt}
    try:
        r = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        if r.status_code == 200:
            # assume model returns a list of dicts with 'generated_text'
            resp = r.json()
            if isinstance(resp, list) and 'generated_text' in resp[0]:
                return resp[0]['generated_text'].strip()
            return resp
        else:
            return f"LLM Error {r.status_code}: {r.text}"  
    except Exception as e:
        return f"Request failed: {e}"

# --- Page Configuration ---
st.set_page_config(page_title="Procurement Corruption-Risk Prediction", layout="wide")
st.title("🔍 Procurement Corruption-Risk Prediction Dashboard")

# --- Sidebar: Data Upload ---
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload trimmed DIB CSV (<200MB)", type=["csv"])

if uploaded_file:
    # Load dataset
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

    # --- Default column mapping ---
    mapping = {
        "TenderID": auto_map(["tender_id", "tenderref", "id"]),
        "Value": auto_map(["tender_value", "value", "amount"]),
        "BidCount": auto_map(["recordedbidscount", "bidcount", "bids"]),
        "ProcedureType": auto_map(["tender_proceduretype", "proceduretype"]),
        "CRI": auto_map(["cri", "corruption_risk_index"]),
        "Vendor": auto_map(["vendor", "supplier", "buyer"])
    }

    st.sidebar.subheader("🔧 Confirm Column Mapping")
    for key, default in mapping.items():
        mapping[key] = st.sidebar.selectbox(
            f"Select column for {key}", [""] + cols,
            index=(cols.index(default) + 1 if default in cols else 0)
        )

    # Rename dataframe columns for consistency
    rename_dict = {mapping[k]: k for k in mapping if mapping[k]}
    df = df.rename(columns=rename_dict)

    # --- Type conversions and cleaning ---
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["BidCount"] = pd.to_numeric(df["BidCount"], errors="coerce")
    df["CRI"] = pd.to_numeric(df["CRI"], errors="coerce")
    df = df.dropna(subset=["TenderID", "Value", "BidCount", "ProcedureType", "CRI", "Vendor"]).reset_index(drop=True)

    # --- Derive risk label ---
    df["high_risk"] = (df["CRI"] >= 0.5).astype(int)

    # Identify corruption flag columns
    corr_cols = [c for c in df.columns if c.startswith("corr_")]

    # --- Feature engineering for modeling ---
    X_base = pd.DataFrame({
        "Value_log": np.log1p(df["Value"]),
        "BidCount_log": np.log1p(df["BidCount"])
    })
    proc_dummies = pd.get_dummies(df["ProcedureType"], prefix="ptype", dummy_na=True)
    X_base = pd.concat([X_base, proc_dummies], axis=1)
    for c in corr_cols:
        X_base[c] = df[c].fillna(0)
    y_base = df["high_risk"]

    # --- Create application tabs ---
    tab_eda, tab_model, tab_pred, tab_summary, tab_suppliers = st.tabs([
        "EDA", "Modeling", "Prediction", "Summary", "High-Risk Suppliers"
    ])

    # --- EDA Tab ---
    with tab_eda:
        st.header("📊 Exploratory Data Analysis")

        # ___ Field Descriptions via LLM ___
        if st.checkbox("📝 Show Field Descriptions"):
            fields = st.multiselect(
                "Select fields to explain", 
                options=list(X_base.columns) + ['CRI'],
                default=['Value_log','BidCount_log','CRI']
            )
            if st.button("Describe selected fields"):
                with st.spinner("Fetching descriptions..."):
                    for f in fields:
                        st.subheader(f"**{f}**")
                        st.write(describe_field_with_llm(f))
        st.markdown("---")

        # Data availability with toggle options
        st.subheader("Data Availability")
        avail = df.notnull().mean() * 100
        view_mode = st.radio(
            "Choose availability display:",
            ["Top/Bottom 10", "Full Data Table"], key='avail_view'
        )  
        if view_mode == "Top/Bottom 10":
            top_bottom = st.selectbox(
                "Show (by availability):", ["Most Complete", "Most Missing"], index=0, key='tb'
            )
            if top_bottom == "Most Complete":
                subset = avail.sort_values(ascending=False).head(10)
            else:
                subset = avail.sort_values(ascending=True).head(10)
            fig_sub, ax_sub = plt.subplots(figsize=(6, max(2, len(subset)*0.4)))
            subset.plot(kind="barh", ax=ax_sub, color='skyblue')
            ax_sub.set_xlabel("% Non-Null")
            st.pyplot(fig_sub)
        else:
            avail_df = avail.sort_values(ascending=False).to_frame("Availability (%)")
            st.dataframe(avail_df)

        # Scatter Value vs BidCount colored by risk
        st.subheader("Value vs. BidCount (colored by risk)")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(
            data=df, x="BidCount", y="Value", hue="high_risk", 
            palette=['green','red'], alpha=0.6, ax=ax_scatter
        )
        ax_scatter.set_yscale('log')
        ax_scatter.set_xscale('log')
        ax_scatter.set_xlabel("Bid Count (log scale)")
        ax_scatter.set_ylabel("Value (log scale)")
        st.pyplot(fig_scatter)

        # CRI distribution by risk label
        st.subheader("CRI Distribution by Risk Label")
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x="high_risk", y="CRI", data=df, ax=ax_box)
        ax_box.set_xticklabels(['Low Risk','High Risk'])
        st.pyplot(fig_box)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap of Key Features and Risk")
        corr_features = list(X_base.columns) + ['CRI']
        corr_df = pd.concat([X_base, df[['CRI']]], axis=1)[corr_features]
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_df.corr(), cmap="vlag", center=0, annot=False, ax=ax_corr)
        st.pyplot(fig_corr)

# --- Modeling Tab ---
    with tab_model:
        st.header("🤖 Model Training & Evaluation")
        test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X_base, y_base, stratify=y_base, test_size=test_size, random_state=42
        )
        st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

        # Fit models
        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        # Feature importance for RF
        st.subheader("Feature Importance (Random Forest)")
        importances = pd.Series(rf.feature_importances_, index=X_base.columns).sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
        importances.head(10).plot(kind='barh', ax=ax_imp, color='teal')
        ax_imp.invert_yaxis()
        ax_imp.set_xlabel("Importance")
        st.pyplot(fig_imp)

        # Evaluate models
        y_pred_lr = lr.predict(X_test); y_prob_lr = lr.predict_proba(X_test)[:,1]
        y_pred_rf = rf.predict(X_test); y_prob_rf = rf.predict_proba(X_test)[:,1]
        auc_lr = roc_auc_score(y_test, y_prob_lr); auc_rf = roc_auc_score(y_test, y_prob_rf)
        acc_lr = accuracy_score(y_test, y_pred_lr); acc_rf = accuracy_score(y_test, y_pred_rf)

        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            'Model': ['Logistic Reg','Random Forest'],
            'AUC': [auc_lr, auc_rf],
            'Accuracy': [acc_lr, acc_rf]
        }).set_index('Model')
        st.dataframe(metrics_df)

        # ROC curves
        st.subheader("ROC Curves")
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc_lr:.2f})")
        ax_roc.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.2f})")
        ax_roc.plot([0,1],[0,1],'--',color='gray')
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(); st.pyplot(fig_roc)

    # --- Prediction Tab ---
    with tab_pred:
        st.header("📝 Predictions")
        choice = st.radio("Prediction Source", ["Held-out Test","New Upload"])
        if choice == "Held-out Test":
            model = rf if st.selectbox("Select model", ["Random Forest","Logistic Reg"]) == "Random Forest" else lr
            acc = accuracy_score(y_test, model.predict(X_test))
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            st.metric("Test Accuracy", f"{acc:.3f}")
            st.metric("Test AUC", f"{auc:.3f}")
        else:
            new_file = st.file_uploader("Upload new tenders CSV", type=["csv"], key="new_up")
            if new_file:
                new_df = pd.read_csv(new_file).rename(columns=rename_dict)
                # build features
                Xn = pd.DataFrame({
                    "Value_log": np.log1p(pd.to_numeric(new_df["Value"], errors="coerce")),
                    "BidCount_log": np.log1p(pd.to_numeric(new_df["BidCount"], errors="coerce"))
                })
                proc_n = pd.get_dummies(new_df["ProcedureType"], prefix="ptype", dummy_na=True)
                Xn = pd.concat([Xn, proc_dummies.reindex(proc_n.index, fill_value=0)], axis=1)
                for c in corr_cols: Xn[c] = new_df.get(c, 0)
                model = rf if st.selectbox("Model", ["Random Forest","Logistic Reg"]) == "Random Forest" else lr
                new_df["risk_prob"] = model.predict_proba(Xn)[:,1]
                new_df["risk_label"] = (new_df["risk_prob"] >= 0.5).astype(int)
                st.dataframe(new_df[["TenderID","Vendor","risk_prob","risk_label"]])
                st.download_button("Download Predictions", new_df.to_csv(index=False), file_name="predictions.csv")

    # --- High-Risk Suppliers Tab ---
    with tab_suppliers:
        st.header("🚩 High-Risk Suppliers Dashboard")
        high_df = df[df["high_risk"] == 1]
        if not high_df.empty:
            # Spend by supplier (top 10)
            spend = high_df.groupby("Vendor")["Value"].sum().nlargest(10)
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(spend, labels=spend.index, autopct="%.1f%%", startangle=140)
            ax_pie.axis('equal')
            st.subheader("Spend at Risk by Top 10 Suppliers")
            st.pyplot(fig_pie)

            # Spend trend over risk flag count
            flag_counts = high_df[corr_cols].sum()
            fig_bar, ax_bar = plt.subplots(figsize=(8,5))
            flag_counts.plot(kind='bar', ax=ax_bar, color='coral')
            ax_bar.set_ylabel("Count of Flags")
            plt.xticks(rotation=45)
            st.subheader("Common Corruption Flags")
            st.pyplot(fig_bar)

            # Interactive supplier selector
            supplier = st.selectbox("Select Supplier to drill down", spend.index)
            sup_df = high_df[high_df["Vendor"] == supplier]
            st.subheader(f"Details for {supplier}")
            st.write(f"Number of high-risk tenders: {len(sup_df)}")
            st.write(f"Total spend at risk: {sup_df['Value'].sum():,.2f}")
            st.dataframe(sup_df[["TenderID","ProcedureType","Value","CRI"]])
        else:
            st.info("No high-risk suppliers detected.")

    # --- Summary Tab ---
    with tab_summary:
        st.header("📋 Analysis Summary Report")
        st.subheader("Dataset Overview")
        st.markdown(f"- Total records: **{df.shape[0]}**")
        st.markdown(f"- High-risk tenders: **{int(df['high_risk'].sum())}** ({df['high_risk'].mean()*100:.1f}%)")
        st.subheader("Model Performance")
        st.markdown(f"- Random Forest: AUC {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.3f}, Acc {accuracy_score(y_test, rf.predict(X_test)):.3f}")
        st.markdown(f"- Logistic Reg:  AUC {roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]):.3f}, Acc {accuracy_score(y_test, lr.predict(X_test)):.3f}")
        st.subheader("Key Insights")
        st.markdown("- Top suppliers by spend at risk highlight areas for compliance review.")
        st.markdown("- Feature importance shows which indicators drive risk predictions.")
        st.markdown("- Visualizations across tabs give comprehensive procurement risk insights.")

else:
    st.info("Upload a procurement CSV file to begin your corruption-risk analysis.")
