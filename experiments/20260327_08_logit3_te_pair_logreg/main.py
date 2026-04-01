"""
Logit3 TE-Pair Logistic Regression
Experiment: 20260327_08_logit3_te_pair_logreg

Chris Deotte's "Logistic Regression with Target Aggregation":
  1. All pairwise category combinations -> Target Encoding
  2. Logit transform: z = log(p / (1-p))
  3. Polynomial expansion: [z, z^2, z^3]
  4. StandardScaler -> LogisticRegression (L2)

Reference: https://www.kaggle.com/code/cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178
Reported CV: 0.9160
"""

import warnings
from pathlib import Path
from itertools import combinations
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from scipy.special import logit

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
    INNER_FOLDS = 5
    RANDOM_STATE = 42

    # All columns to use for pairwise combinations
    CAT_COLS = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    LR_C = 0.5


config = Config()


# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


# ============================================================================
# Create pairwise combination columns
# ============================================================================
def create_pair_columns(train_df, test_df):
    """Create all pairwise combinations of CAT + NUM_AS_CAT columns."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Convert numericals to string categories
    all_cols = list(config.CAT_COLS)
    for col in config.NUM_COLS:
        cat_name = f"CAT_{col}"
        for df in [train_df, test_df]:
            df[cat_name] = df[col].astype(str)
        all_cols.append(cat_name)

    # SeniorCitizen is already in CAT_COLS as numeric, convert to string
    for df in [train_df, test_df]:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Create all pairs
    pair_cols = []
    pairs = list(combinations(all_cols, 2))
    for c1, c2 in pairs:
        pair_name = f"PAIR_{c1}_{c2}"
        for df in [train_df, test_df]:
            df[pair_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        pair_cols.append(pair_name)

    print(f"  All columns for pairing: {len(all_cols)}")
    print(f"  Pairwise combinations: {len(pair_cols)}")

    return train_df, test_df, pair_cols


# ============================================================================
# Target Encoding for pairs (inner K-fold, leakage-safe)
# ============================================================================
def apply_pair_te(X_tr, y_tr, X_val, X_te, pair_cols, inner_folds, seed):
    """Leakage-safe target encoding for pair columns."""
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    te_names = [f"TE_{col}" for col in pair_cols]
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    # Initialize
    for c in te_names:
        X_tr[c] = np.float32(0.5)

    # Inner fold TE for train
    target_col = "__target__"
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr].copy()
        X_tr2[target_col] = y_tr[in_tr]
        for pair_col, te_col in zip(pair_cols, te_names):
            mapping = X_tr2.groupby(pair_col)[target_col].mean()
            vals = X_tr.iloc[in_va][pair_col].map(mapping)
            X_tr.iloc[in_va, X_tr.columns.get_loc(te_col)] = vals.fillna(0.5).astype("float32").values

    # Full-fold TE for val/test
    temp = X_tr.copy()
    temp[target_col] = y_tr
    for pair_col, te_col in zip(pair_cols, te_names):
        mapping = temp.groupby(pair_col)[target_col].mean()
        X_val[te_col] = X_val[pair_col].map(mapping).fillna(0.5).astype("float32")
        X_te[te_col] = X_te[pair_col].map(mapping).fillna(0.5).astype("float32")

    return X_tr[te_names], X_val[te_names], X_te[te_names]


# ============================================================================
# Logit3 transform
# ============================================================================
def logit3_transform(te_values):
    """Apply logit transform + polynomial expansion [z, z^2, z^3]."""
    # Clip to avoid inf in logit
    clipped = np.clip(te_values, 1e-6, 1 - 1e-6)
    z = logit(clipped).astype("float32")

    # Stack [z, z^2, z^3]
    n_samples, n_features = z.shape
    result = np.empty((n_samples, n_features * 3), dtype="float32")
    result[:, :n_features] = z
    result[:, n_features:2*n_features] = z ** 2
    result[:, 2*n_features:] = z ** 3

    # Column names
    cols = te_values.columns.tolist()
    out_cols = (
        [f"{c}_z1" for c in cols] +
        [f"{c}_z2" for c in cols] +
        [f"{c}_z3" for c in cols]
    )
    return pd.DataFrame(result, index=te_values.index, columns=out_cols)


# ============================================================================
# Main
# ============================================================================
def main():
    config.PRED_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Logit3 TE-Pair Logistic Regression")
    print(f"  C={config.LR_C}, {config.N_SPLITS}-fold CV")
    print("=" * 60)

    train_df, test_df = load_data()

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print("\nCreating pairwise combinations...")
    train_df, test_df, pair_cols = create_pair_columns(train_df, test_df)

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

        # TE for pairs
        print("  Target Encoding...")
        te_tr, te_val, te_te = apply_pair_te(
            train_df.iloc[tr_idx], y[tr_idx],
            train_df.iloc[va_idx], test_df,
            pair_cols, config.INNER_FOLDS, config.RANDOM_STATE,
        )

        # Logit3 transform
        print("  Logit3 transform...")
        X_tr = logit3_transform(te_tr)
        X_val = logit3_transform(te_val)
        X_te = logit3_transform(te_te)

        if fold == 0:
            print(f"  Features: {X_tr.shape[1]} ({len(pair_cols)} pairs x 3)")

        # StandardScaler
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr.fillna(0))
        X_val_scaled = scaler.transform(X_val.fillna(0))
        X_te_scaled = scaler.transform(X_te.fillna(0))

        # Logistic Regression
        model = LogisticRegression(C=config.LR_C, max_iter=1000, random_state=config.RANDOM_STATE)
        model.fit(X_tr_scaled, y[tr_idx])

        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        oof_preds[va_idx] = val_proba
        test_preds += model.predict_proba(X_te_scaled)[:, 1] / config.N_SPLITS

        fold_auc = roc_auc_score(y[va_idx], val_proba)
        print(f"  AUC: {fold_auc:.6f}")

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"OOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")
    print(f"{'='*60}")

    pd.DataFrame({"id": train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(config.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": test_preds}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_preds}).to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {config.PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
