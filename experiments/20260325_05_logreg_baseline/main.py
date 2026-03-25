"""
Logistic Regression Baseline
Experiment: 20260325_05_logreg_baseline

目的: One-Hot Encoding + StandardScaler + LogisticRegression（追加特徴量なし）
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, log_loss

warnings.filterwarnings("ignore")

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent.parent / "data"
    TRAIN_DATA = DATA_DIR / "raw" / "train.csv"
    TEST_DATA = DATA_DIR / "raw" / "test.csv"
    TARGET_COL = "Churn"
    ID_COL = "id"
    PRED_DIR = BASE_DIR / "predictions"
    MODEL_DIR = BASE_DIR / "model"
    N_SPLITS = 5
    RANDOM_STATE = 42

config = Config()

def main():
    print("=" * 80)
    print("Logistic Regression Baseline")
    print("=" * 80)

    train_df = pd.read_csv(config.TRAIN_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    # Feature engineering: One-Hot + Charge_Difference
    exclude = [config.TARGET_COL, config.ID_COL]
    cat_cols = [c for c in train_df.columns if pd.api.types.is_string_dtype(train_df[c]) and c not in exclude]
    num_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c]) and c not in exclude]

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["Charge_Difference"] = train_df["TotalCharges"] - train_df["MonthlyCharges"] * train_df["tenure"]
    test_df["Charge_Difference"] = test_df["TotalCharges"] - test_df["MonthlyCharges"] * test_df["tenure"]
    num_cols.append("Charge_Difference")

    # Ensure string dtype for get_dummies
    for col in cat_cols:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    train_cat = pd.get_dummies(train_df[cat_cols], drop_first=True)
    test_cat = pd.get_dummies(test_df[cat_cols], drop_first=True)
    for col in set(train_cat.columns) - set(test_cat.columns):
        test_cat[col] = 0
    test_cat = test_cat[train_cat.columns]

    X = pd.concat([train_df[num_cols].reset_index(drop=True), train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[num_cols].reset_index(drop=True), test_cat.reset_index(drop=True)], axis=1)
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    print(f"Features: {X.shape[1]}")

    # CV
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=config.RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr_s, y[tr_idx])

        val_proba = model.predict_proba(X_va_s)[:, 1]
        oof_preds[va_idx] = val_proba
        test_preds += model.predict_proba(X_te_s)[:, 1] / config.N_SPLITS

        auc = roc_auc_score(y[va_idx], val_proba)
        fold_scores.append(auc)
        print(f"  AUC: {auc:.6f}")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nOOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {log_loss(y, oof_preds):.6f}")

    # Save
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(config.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({config.ID_COL: test_ids, "prob": test_preds}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: test_preds}).to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"Saved to {config.PRED_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
