"""
Stacking LGBM: raw features + TabM/RealMLP/Ridge OOF predictions
Experiment: 20260331_01_stacking_lgbm

Raw features (Label Encoded categoricals + numericals + Charge_Difference)
+ OOF predictions from TabM, RealMLP, Ridge as stacking features.
LGBM with default parameters, 5-fold CV.
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data" / "raw"
EXPERIMENTS_DIR = BASE_DIR.parent

TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

TARGET_COL = "Churn"
ID_COL = "id"
PRED_DIR = BASE_DIR / "predictions"

N_SPLITS = 5
RANDOM_STATE = 42

# Stacking features: OOF predictions from other models
STACK_MODELS = {
    "tabm_pred":    ("20260330_01_tabm_nb1feat_20fold", "oof.csv", "test_proba.csv"),
    "realmlp_pred": ("20260325_04_realmlp_orig_data_reference", "oof.csv", "test_proba.csv"),
    "ridge_pred":   ("20260328_04_ridge_201feat_oof", "oof.csv", "test_proba.csv"),
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "n_jobs": -1,
}

NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100


# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    print("Loading data...")
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    print(f"  Train: {train.shape}, Test: {test.shape}")
    return train, test


def load_stacking_features():
    """Load OOF and test predictions from stacking models."""
    print("Loading stacking features...")
    oof_features = {}
    test_features = {}

    for feat_name, (exp_name, oof_file, test_file) in STACK_MODELS.items():
        oof_path = EXPERIMENTS_DIR / exp_name / "predictions" / oof_file
        test_path = EXPERIMENTS_DIR / exp_name / "predictions" / test_file

        oof_df = pd.read_csv(oof_path)
        test_df = pd.read_csv(test_path)

        oof_features[feat_name] = oof_df["prob"].values
        test_features[feat_name] = test_df["prob"].values

        print(f"  {feat_name}: OOF {len(oof_df)}, Test {len(test_df)}")

    return oof_features, test_features


def extract_features(train_df, test_df):
    """Raw features: Label Encoding + Charge_Difference."""
    exclude_cols = [TARGET_COL, ID_COL]

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
    print(f"  Raw features: {len(feature_cols)}")
    return train_df[feature_cols], test_df[feature_cols], feature_cols


# ============================================================================
# Main
# ============================================================================
def main():
    PRED_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Stacking LGBM: raw features + TabM/RealMLP/Ridge predictions")
    print("=" * 60)

    train_df, test_df = load_data()

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[TARGET_COL])
    train_ids = train_df[ID_COL].values
    test_ids = test_df[ID_COL].values

    X_train, X_test, feature_cols = extract_features(train_df, test_df)

    # Add stacking features
    oof_features, test_features = load_stacking_features()
    stack_cols = []
    for feat_name, oof_vals in oof_features.items():
        X_train[feat_name] = oof_vals
        X_test[feat_name] = test_features[feat_name]
        stack_cols.append(feat_name)

    all_features = feature_cols + stack_cols
    print(f"  Total features: {len(all_features)} (raw {len(feature_cols)} + stack {len(stack_cols)})")

    # CV
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y[tr_idx], y[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(200),
            ],
        )

        val_proba = model.predict(X_val)
        oof_preds[va_idx] = val_proba
        test_preds += model.predict(X_test) / N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        print(f"  AUC: {fold_auc:.6f}, Best iter: {model.best_iteration}")

        # Feature importance (fold 0 only)
        if fold == 0:
            imp = pd.Series(model.feature_importance(importance_type="gain"), index=all_features)
            print(f"\n  Top features (gain):")
            for name, val in imp.sort_values(ascending=False).head(10).items():
                marker = " [STACK]" if name in stack_cols else ""
                print(f"    {name:<25s} {val:12.1f}{marker}")

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"OOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")
    print(f"{'='*60}")

    # Save
    pd.DataFrame({"id": train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": test_preds}).to_csv(PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_preds}).to_csv(PRED_DIR / "test.csv", index=False)
    with open(PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
