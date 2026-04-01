"""
Gaussian Naive Bayes Baseline
Experiment: 20260327_04_gnb_baseline

基本20特徴量（Label Encoding + Charge_Difference）でGNBを実行。
アンサンブル多様性確保が目的（ツリー/NN系と直交する誤差パターン）。
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
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

    TARGET_COL = "Churn"
    ID_COL = "id"

    PRED_DIR = BASE_DIR / "predictions"
    N_SPLITS = 5
    RANDOM_STATE = 42


config = Config()


def load_data():
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


def extract_features(train_df, test_df):
    exclude_cols = [config.TARGET_COL, config.ID_COL]

    cat_cols = [c for c in train_df.columns
                if pd.api.types.is_string_dtype(train_df[c]) and c not in exclude_cols]
    num_cols = [c for c in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[c]) and c not in exclude_cols]

    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    train_df["Charge_Difference"] = train_df["TotalCharges"] - train_df["MonthlyCharges"] * train_df["tenure"]
    test_df["Charge_Difference"] = test_df["TotalCharges"] - test_df["MonthlyCharges"] * test_df["tenure"]

    feature_cols = num_cols + cat_cols + ["Charge_Difference"]
    print(f"Features: {len(feature_cols)}")
    return train_df[feature_cols], test_df[feature_cols]


def main():
    config.PRED_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Gaussian Naive Bayes Baseline (20 features)")
    print("=" * 60)

    train_df, test_df = load_data()
    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    X, X_test = extract_features(train_df, test_df)

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_val = y[tr_idx], y[va_idx]

        model = GaussianNB()
        model.fit(X_tr, y_tr)

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / config.N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        print(f"  Fold {fold+1}: AUC={fold_auc:.6f}")

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
