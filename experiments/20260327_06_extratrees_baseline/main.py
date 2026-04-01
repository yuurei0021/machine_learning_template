"""
ExtraTrees Baseline (~40 features)
Experiment: 20260327_06_extratrees_baseline

基本特徴量 + ORIG_proba single + Arithmetic + Service Counts (~44特徴量)
ランダム分割によりXGBoost/LGBMと異なる誤差パターンを生成。
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================
class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent.parent / "data"
    INPUT_DIR = DATA_DIR / "raw"

    TRAIN_DATA = INPUT_DIR / "train.csv"
    TEST_DATA = INPUT_DIR / "test.csv"
    ORIG_DATA = INPUT_DIR / "orig" / "Telco_customer_churn.xlsx"

    TARGET_COL = "Churn"
    ID_COL = "id"

    PRED_DIR = BASE_DIR / "predictions"
    N_SPLITS = 5
    RANDOM_STATE = 42

    CAT_COLS = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]
    NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
    SERVICE_COLS = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    ORIG_PROBA_CATS = [
        "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
        "TechSupport", "OnlineBackup", "DeviceProtection", "PaperlessBilling",
        "StreamingMovies", "StreamingTV", "Partner", "Dependents",
    ]

    ET_PARAMS = {
        "n_estimators": 1000,
        "max_depth": None,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }


config = Config()


def load_data():
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


def load_orig_data():
    orig = pd.read_excel(config.ORIG_DATA)
    col_map = {
        "Gender": "gender", "Senior Citizen": "SeniorCitizen",
        "Partner": "Partner", "Dependents": "Dependents",
        "Tenure Months": "tenure", "Phone Service": "PhoneService",
        "Multiple Lines": "MultipleLines", "Internet Service": "InternetService",
        "Online Security": "OnlineSecurity", "Online Backup": "OnlineBackup",
        "Device Protection": "DeviceProtection", "Tech Support": "TechSupport",
        "Streaming TV": "StreamingTV", "Streaming Movies": "StreamingMovies",
        "Contract": "Contract", "Paperless Billing": "PaperlessBilling",
        "Payment Method": "PaymentMethod", "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges", "Churn Label": "Churn",
    }
    orig = orig.rename(columns=col_map)
    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)
    orig["Churn_binary"] = (orig["Churn"] == "Yes").astype(int)
    keep_cols = config.CAT_COLS + config.NUM_COLS + ["SeniorCitizen", "Churn_binary"]
    orig = orig[[c for c in keep_cols if c in orig.columns]]
    print(f"Orig: {orig.shape}")
    return orig


def extract_features(train_df, test_df, orig_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    dfs = [train_df, test_df]

    # Charge_Difference
    for df in dfs:
        df["Charge_Difference"] = (df["TotalCharges"] - df["MonthlyCharges"] * df["tenure"]).astype("float32")

    # Arithmetic
    for df in dfs:
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")

    # Service Counts
    for df in dfs:
        df["service_count"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

    # ORIG_proba single
    for col in config.ORIG_PROBA_CATS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    for col in config.NUM_COLS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")

    # Label Encoding
    for col in config.CAT_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    exclude_cols = {config.TARGET_COL, config.ID_COL, "Churn_binary"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Features: {len(feature_cols)}")
    return train_df[feature_cols], test_df[feature_cols]


def main():
    config.PRED_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("ExtraTrees Baseline (~40 features)")
    print("=" * 60)

    train_df, test_df = load_data()
    orig_df = load_orig_data()

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    X, X_test = extract_features(train_df, test_df, orig_df)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_val = y[tr_idx], y[va_idx]

        model = ExtraTreesClassifier(**config.ET_PARAMS)
        model.fit(X_tr, y_tr)

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / config.N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        print(f"  AUC: {fold_auc:.6f}")

        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\nOOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")

    pd.DataFrame({"id": train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(config.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": test_preds}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_preds}).to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {config.PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
