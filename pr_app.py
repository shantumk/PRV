import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, accuracy_score

# --- Page Config ---
st.set_page_config(page_title="Procurement Corruption-Risk Prediction", layout="wide")
st.title("🔍 Procurement Corruption-Risk Prediction Dashboard")

# --- Sidebar: Data Upload ---
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload trimmed DIB CSV (<200MB)", type=["csv"])

if uploaded_file:
    # Load data and basic cleanup
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

    # --- Default mappings ---
    mapping = {
        "TenderID": auto_map(["tender_id", "tenderref", "id"]),
        "Value": auto_map(["tender_value", "value", "amount"]),
        "BidCount": auto_map(["recordedbidscount", "bidcount", "bids"]),
        "ProcedureType": auto_map(["tender_proceduretype", "proceduretype"]),
        "CRI": auto_map(["cri", "corruption_risk_index"])
    }

    st.sidebar.subheader("🔧 Confirm Column Mapping")
    for key, default in mapping.items():
        mapping[key] = st.sidebar.selectbox(
            f"Select column for {key}", [""] + cols,
            index=(cols.index(default) + 1 if default in cols else 0)
        )

    # Rename columns for consistency
    rename_dict = {mapping[k]: k for k in mapping if mapping[k]}
    df = df.rename(columns=rename_dict)

    # Convert types
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["BidCount"] = pd.to_numeric(df["BidCount"], errors="coerce")
    df["CRI"] = pd.to_numeric(df["CRI"], errors="coerce")
    df = df.dropna(subset=["TenderID", "Value", "BidCount", "ProcedureType", "CRI"]).reset_index(drop=True)

    # Derive high-risk label
    df["high_risk"] = (df["CRI"] >= 0.5).astype(int)

    # Identify corruption flag columns
    corr_cols = [c for c in df.columns if c.startswith("corr_")]

    # Define feature matrix and label once
    X_base = pd.DataFrame({
        "Value_log": np.log1p(df["Value"]),
        "BidCount_log": np.log1p(df["BidCount"])  
    })
    proc_dummies = pd.get_dummies(df["ProcedureType"], prefix="ptype", dummy_na=True)
    X_base = pd.concat([X_base, proc_dummies], axis=1)
    for c in corr_cols:
        X_base[c] = df[c].fillna(0)
    y_base = df["high_risk"]

    # Tabs for organization
    tab_eda, tab_model, tab_pred, tab_summary = st.tabs(["EDA", "Modeling", "Prediction", "Summary"])

    with tab_eda:
        st.header("📊 Exploratory Data Analysis")
        # Data availability
        st.subheader("Data Availability (%)")
        avail = df.notnull().mean() * 100
        fig_avail, ax = plt.subplots(figsize=(10, 4))
        avail.plot(kind="bar", ax=ax)
        ax.set_ylabel("% Available")
        plt.xticks(rotation=90)
        st.pyplot(fig_avail)

        # Distribution of key fields
        st.subheader("Tender Value Distribution (log scale)")
        fig1, ax1 = plt.subplots()
        sns.histplot(np.log1p(df["Value"]), bins=50, ax=ax1)
        ax1.set_xlabel("log1p(Value)")
        st.pyplot(fig1)

        st.subheader("CRI Score Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["CRI"], bins=50, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Correlation Heatmap")
        num_cols = ["Value", "BidCount", "CRI"] + corr_cols
        corr_mat = df[num_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_mat, cmap="coolwarm", center=0, ax=ax3)
        st.pyplot(fig3)

    with tab_model:
        st.header("🤖 Model Training & Evaluation")
        # Train/test size selection
        test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X_base, y_base, stratify=y_base, test_size=test_size, random_state=42)
        st.write(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

        # Fit models
        lr = LogisticRegression(max_iter=1000)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        lr.fit(X_train, y_train); rf.fit(X_train, y_train)

        # Predict & evaluate
        y_pred_lr = lr.predict(X_test); y_prob_lr = lr.predict_proba(X_test)[:,1]
        y_pred_rf = rf.predict(X_test); y_prob_rf = rf.predict_proba(X_test)[:,1]
        auc_lr = roc_auc_score(y_test, y_prob_lr); auc_rf = roc_auc_score(y_test, y_prob_rf)
        acc_lr = accuracy_score(y_test, y_pred_lr); acc_rf = accuracy_score(y_test, y_pred_rf)

        st.subheader("Performance Metrics")
        st.markdown(f"- Logistic Regression: AUC = {auc_lr:.3f}, Accuracy = {acc_lr:.3f}")
        st.markdown(f"- Random Forest:       AUC = {auc_rf:.3f}, Accuracy = {acc_rf:.3f}")

        # Display confusion matrices
        fig_cm1, ax_cm1 = plt.subplots(); sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", ax=ax_cm1)
        ax_cm1.set_title("Confusion Matrix - LR"); st.pyplot(fig_cm1)
        fig_cm2, ax_cm2 = plt.subplots(); sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", ax=ax_cm2)
        ax_cm2.set_title("Confusion Matrix - RF"); st.pyplot(fig_cm2)

        # ROC curves
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc_lr:.3f})")
        ax_roc.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.3f})")
        ax_roc.plot([0,1],[0,1],'--',color='gray'); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend()
        st.pyplot(fig_roc)

    with tab_pred:
        st.header("📝 Prediction on New Data or Original Split")
        st.markdown("You can either upload a new tender file or use a portion of the original data as a test set.")
        choice = st.radio("Prediction Source", ["Upload New CSV", "Use Held-out Test Set"], index=1)
        if choice == "Upload New CSV":
            new_file = st.file_uploader("Upload new tenders CSV", type=["csv"], key="new")
            if new_file:
                new_df = pd.read_csv(new_file)
                new_df = new_df.rename(columns=rename_dict)
                # Build features as before
                X_new = pd.DataFrame({
                    "Value_log": np.log1p(pd.to_numeric(new_df["Value"], errors="coerce")),
                    "BidCount_log": np.log1p(pd.to_numeric(new_df["BidCount"], errors="coerce"))
                })
                proc_new = pd.get_dummies(new_df["ProcedureType"], prefix="ptype", dummy_na=True)
                X_new = pd.concat([X_new, proc_dummies.reindex(proc_new.index, fill_value=0)], axis=1)
                for c in corr_cols: X_new[c] = new_df.get(c, 0)
                # Choose model
                model = lr if st.selectbox("Select model", ["Logistic Regression","Random Forest"]) == "Logistic Regression" else rf
                new_df["risk_prob"] = model.predict_proba(X_new)[:,1]
                new_df["risk_label"] = (new_df["risk_prob"] >= 0.5).astype(int)
                st.dataframe(new_df[["TenderID","risk_prob","risk_label"]])
                st.download_button("Download Predictions", new_df.to_csv(index=False), file_name="predictions.csv")
        else:
            # Use test split
            st.markdown(f"Using the held-out {len(X_test)} samples from original data.")
            model = lr if st.selectbox("Select model for evaluation", ["LR","RF"]) == "LR" else rf
            X_eval, y_eval = X_test, y_test
            y_eval_prob = model.predict_proba(X_eval)[:,1]
            y_eval_pred = model.predict(X_eval)
            st.markdown(f"**Evaluation Accuracy:** {accuracy_score(y_eval, y_eval_pred):.3f}")
            st.markdown(f"**Evaluation AUC:** {roc_auc_score(y_eval, y_eval_prob):.3f}")

    with tab_summary:
        st.header("📋 Analysis Summary Report")
        st.subheader("Dataset Summary")
        st.markdown(f"- Total records: **{df.shape[0]}**")
        st.markdown(f"- Total features: **{df.shape[1]}**")
        st.markdown(f"- High-risk tenders: **{int(df['high_risk'].sum())}** ({(df['high_risk'].mean()*100):.1f}%)")

        st.subheader("Model Summary")
        st.markdown(f"- Logistic Regression: AUC {auc_lr:.3f}, Accuracy {acc_lr:.3f}")
        st.markdown(f"- Random Forest: AUC {auc_rf:.3f}, Accuracy {acc_rf:.3f}")

        st.subheader("Next Steps & Insights")
        st.markdown(
            "- Investigate items with highest predicted risk for manual review."
        )
        st.markdown(
            "- Tune models with additional features (timing, vendor history)."
        )
        st.markdown(
            "- Deploy this dashboard to support procurement decision-making."
        )

else:
    st.info("Upload a procurement CSV file to begin your corruption-risk analysis.")
