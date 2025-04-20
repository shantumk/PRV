import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve

# --- Page Config ---
st.set_page_config(page_title="Procurement Corruption-Risk Prediction", layout="wide")
st.title("🔍 Procurement Corruption-Risk Prediction Dashboard")

# --- Sidebar: Data Upload ---
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload trimmed DIB CSV (<200MB)", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    cols = df.columns.tolist()

    # Auto-mapping helper
    def auto_map(options):
        for col in cols:
            for opt in options:
                if opt.lower() in col.lower():
                    return col
        return None

    # Default column mapping
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

    # Rename columns
    rename_dict = {mapping[k]: k for k in mapping if mapping[k]}
    df = df.rename(columns=rename_dict)

    # Convert types and filter
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["BidCount"] = pd.to_numeric(df["BidCount"], errors="coerce")
    df["CRI"] = pd.to_numeric(df["CRI"], errors="coerce")
    df = df.dropna(subset=["TenderID", "Value", "BidCount", "ProcedureType", "CRI"])

    # Derive label
    df["high_risk"] = (df["CRI"] >= 0.5).astype(int)

    # Identify corruption flag columns
    corr_cols = [col for col in df.columns if col.startswith("corr_")]

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["EDA", "Modeling", "Prediction"])

    with tab1:
        st.header("📊 Exploratory Data Analysis")

        st.subheader("Data Overview")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.dataframe(df.head())

        st.subheader("Missing Values (%)")
        missing = df.isnull().mean() * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        missing.plot(kind="bar", ax=ax)
        ax.set_ylabel("Percent Missing")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.subheader("Tender Value Distribution (log scale)")
        fig, ax = plt.subplots()
        sns.histplot(np.log1p(df["Value"]), bins=50, ax=ax)
        ax.set_xlabel("log1p(Value)")
        st.pyplot(fig)

        st.subheader("CRI Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["CRI"], bins=50, ax=ax)
        st.pyplot(fig)

        st.subheader("Correlation Matrix")
        num_cols = ["Value", "BidCount", "CRI"] + corr_cols
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

        st.subheader("CRI by Procedure Type")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x="ProcedureType", y="CRI", data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with tab2:
        st.header("🤖 Model Training & Evaluation")

        # Feature engineering
        X = pd.DataFrame()
        X["Value_log"] = np.log1p(df["Value"])
        X["BidCount_log"] = np.log1p(df["BidCount"])
        proc_dummies = pd.get_dummies(df["ProcedureType"], prefix="ptype", dummy_na=True)
        X = pd.concat([X, proc_dummies], axis=1)
        for col in corr_cols:
            X[col] = df[col].fillna(0)
        y = df["high_risk"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        auc_lr = roc_auc_score(y_test, y_prob_lr)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        auc_rf = roc_auc_score(y_test, y_prob_rf)

        st.subheader("Model Performance")
        st.write(f"Logistic Regression ROC-AUC: {auc_lr:.3f}")
        st.write(f"Random Forest ROC-AUC: {auc_rf:.3f}")

        st.subheader("Classification Report - LR")
        report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
        st.dataframe(pd.DataFrame(report_lr).transpose())

        st.subheader("Classification Report - RF")
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        st.dataframe(pd.DataFrame(report_rf).transpose())

        st.subheader("Confusion Matrix - LR")
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig, ax = plt.subplots()
        sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Confusion Matrix - RF")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("ROC Curves")
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        fig, ax = plt.subplots()
        ax.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc_lr:.3f})")
        ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.3f})")
        ax.plot([0,1], [0,1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.header("📝 Predict New Tenders")
        new_file = st.file_uploader("Upload CSV of new tenders", type=["csv"], key="pred")
        model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])
        if new_file:
            new_df = pd.read_csv(new_file)
            new_df = new_df.rename(columns=rename_dict)
            # Validate
            required_cols = ["Value", "BidCount", "ProcedureType"]
            if not all(col in new_df.columns for col in required_cols):
                st.error("Uploaded file missing required columns. Please map accordingly.")
            else:
                X_new = pd.DataFrame()
                X_new["Value_log"] = np.log1p(pd.to_numeric(new_df["Value"], errors="coerce"))
                X_new["BidCount_log"] = np.log1p(pd.to_numeric(new_df["BidCount"], errors="coerce"))
                proc_new = pd.get_dummies(new_df["ProcedureType"], prefix="ptype", dummy_na=True)
                X_new = pd.concat([X_new, proc_dummies.reindex(proc_new.index, fill_value=0)], axis=1)
                for col in corr_cols:
                    X_new[col] = new_df.get(col, 0)
                if model_choice == "Logistic Regression":
                    new_df["risk_prob"] = lr.predict_proba(X_new)[:, 1]
                else:
                    new_df["risk_prob"] = rf.predict_proba(X_new)[:, 1]
                new_df["risk_label"] = (new_df["risk_prob"] >= 0.5).astype(int)
                st.dataframe(new_df[["TenderID", "risk_prob", "risk_label"]])
                st.download_button("Download Predictions", new_df.to_csv(index=False), file_name="predictions.csv")
else:
    st.info("Upload a procurement CSV file to begin.")
