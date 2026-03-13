"""
Bartz (Bayesian Additive Regression Trees) Baseline
Experiment: 20260313_06_bartz_baseline

目的: GBDTとは異なるアプローチであるBartzを試す。アンサンブルの多様性に貢献する。
アプローチ: Target Encoding + Bartz (MCMC) + 5-Fold Stratified CV
参考: Chris Deotte's Bartz starter notebook
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import warnings
import gc
from pathlib import Path
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from bartz.BART import gbart

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
    MODEL_DIR = BASE_DIR / "model"

    N_SPLITS = 5
    RANDOM_STATE = 42

    CATS = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Bartz MCMC parameters (from Chris Deotte's notebook)
    BARTZ_PARAMS = {
        "maxdepth": 6,
        "ntree": 400,
        "k": 5,
        "sigdf": 3,
        "sigquant": 0.9,
    }
    NDPOST = 1000
    NSKIP = 200
    KEEPEVERY = 2

config = Config()

# ============================================================================
# Target Encoder (from Chris Deotte's notebook)
# ============================================================================
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_encode, aggs=["mean"], cv=5, smooth="auto", drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df["target"] = y
        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)
        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)["target"].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f"TE_{col}_{agg_func}"
                map_series = self.mappings_[col][agg_func]
                X_transformed[new_col_name] = X[col].map(map_series)
                X_transformed[new_col_name] = X_transformed[new_col_name].fillna(
                    self.global_stats_[agg_func]
                )
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train["target"] = y_train
            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f"TE_{col}_{agg_func}"
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)["target"].agg(agg_func)
                    if agg_func == "mean":
                        counts = temp_df_train.groupby(col)["target"].count()
                        m = self.smooth
                        if self.smooth == "auto":
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)["target"].var().mean()
                            if pd.notna(variance_between) and variance_between > 0:
                                m = avg_variance_within / variance_between
                                if pd.isna(m):
                                    m = 0
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(
                        fold_global_stat
                    )
        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]
        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

# ============================================================================
# Helpers
# ============================================================================
def factorize_together(train_s, val_s, test_s):
    combined = pd.concat([train_s, val_s, test_s], axis=0, ignore_index=True)
    codes, _ = pd.factorize(combined.astype(str))
    n1, n2 = len(train_s), len(val_s)
    return (
        pd.Series(codes[:n1], index=train_s.index, dtype="int32"),
        pd.Series(codes[n1:n1+n2], index=val_s.index, dtype="int32"),
        pd.Series(codes[n1+n2:], index=test_s.index, dtype="int32"),
    )

# ============================================================================
# Data Loading & Feature Engineering
# ============================================================================
def load_data():
    print("Loading data...")
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    return train, test

def prepare_features(train_df, test_df):
    """Prepare base features before CV loop."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Target
    train_df[config.TARGET_COL] = train_df[config.TARGET_COL].map({"Yes": 1, "No": 0}).astype("int8")

    # Numeric cleanup
    for df in [train_df, test_df]:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("float32")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").astype("float32")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").astype("float32")
        fill_val = df["tenure"] * df["MonthlyCharges"]
        df["TotalCharges"] = df["TotalCharges"].fillna(fill_val).astype("float32")

    # Categorical cleanup
    for df in [train_df, test_df]:
        for c in config.CATS:
            df[c] = df[c].astype(str).fillna("Missing")

    # Charge_Difference
    for df in [train_df, test_df]:
        df["Charge_Difference"] = df["TotalCharges"] - df["MonthlyCharges"] * df["tenure"]

    # Base features
    base_cols = [c for c in train_df.columns if c not in [config.TARGET_COL, config.ID_COL]]

    # Interaction features (pairwise combinations of categorical features)
    inter_cols = []
    cat_for_inter = [c for c in config.CATS if c in base_cols]
    print(f"  Building pairwise interaction features from {len(cat_for_inter)} categoricals...")
    for col1, col2 in combinations(cat_for_inter, 2):
        new_col = f"{col1}_{col2}"
        inter_cols.append(new_col)
        for df in [train_df, test_df]:
            df[new_col] = df[col1].astype(str) + "_" + df[col2].astype(str)
    print(f"  {len(inter_cols)} interaction features created.")

    feature_cols = base_cols + inter_cols
    return train_df, test_df, feature_cols, inter_cols, base_cols

# ============================================================================
# Visualization
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title("Confusion Matrix (OOF)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_roc_curve(y_true, y_score, save_path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OOF) - Bartz")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# ============================================================================
# Save
# ============================================================================
def save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices):
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)

    oof_df = pd.DataFrame({
        config.ID_COL: train_ids,
        "prob": oof_preds,
        "predicted": (oof_preds > 0.5).astype(int),
        "true": y,
    })
    oof_df.to_csv(config.PRED_DIR / "oof.csv", index=False)

    test_proba_df = pd.DataFrame({
        config.ID_COL: test_ids,
        "prob": test_preds,
    })
    test_proba_df.to_csv(config.PRED_DIR / "test_proba.csv", index=False)

    submit_df = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: test_preds,
    })
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False)

    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Predictions saved to {config.PRED_DIR}")

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("Bartz (Bayesian Additive Regression Trees) Baseline")
    print("=" * 80)

    np.random.seed(config.RANDOM_STATE)

    # Load data
    train_df, test_df = load_data()
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    # Prepare features
    print(f"\n{'='*80}")
    print("Feature Engineering")
    print(f"{'='*80}")
    train_df, test_df, feature_cols, inter_cols, base_cols = prepare_features(train_df, test_df)

    y = train_df[config.TARGET_COL].values
    print(f"  Churn rate: {y.mean():.4f}")
    print(f"  Total feature columns: {len(feature_cols)}")

    # Target encoding columns
    base_te_cols = config.NUMS + ["Charge_Difference"]

    # CV
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation (Bartz)")
    print(f"  Bartz params: {config.BARTZ_PARAMS}")
    print(f"  NDPOST={config.NDPOST}, NSKIP={config.NSKIP}, KEEPEVERY={config.KEEPEVERY}")
    print(f"{'='*80}")

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    fold_indices = {"train": [], "val": []}
    fold_scores = []

    X_all = train_df[feature_cols].copy()
    y_series = pd.Series(y)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y), 1):
        print(f"\n--- Fold {fold}/{config.N_SPLITS} ---")

        X_train = X_all.iloc[train_idx].copy()
        X_val = X_all.iloc[val_idx].copy()
        y_train = y_series.iloc[train_idx].copy()
        y_val = y_series.iloc[val_idx].copy()
        X_test = test_df[feature_cols].copy()

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

        # Target Encode interaction features (drop originals)
        te_inter = TargetEncoder(
            cols_to_encode=inter_cols,
            cv=5, smooth="auto", aggs=["mean"],
            drop_original=True,
        )
        X_train = te_inter.fit_transform(X_train, y_train)
        X_val = te_inter.transform(X_val)
        X_test = te_inter.transform(X_test)

        # Target Encode numeric base cols (keep originals)
        te_base = TargetEncoder(
            cols_to_encode=base_te_cols,
            cv=5, smooth="auto", aggs=["mean"],
            drop_original=False,
        )
        X_train = te_base.fit_transform(X_train, y_train)
        X_val = te_base.transform(X_val)
        X_test = te_base.transform(X_test)

        # Factorize categorical columns
        for c in config.CATS:
            if c in X_train.columns:
                X_train[c], X_val[c], X_test[c] = factorize_together(
                    X_train[c], X_val[c], X_test[c]
                )

        # Factorize any remaining non-numeric columns
        for c in X_train.columns:
            if not pd.api.types.is_numeric_dtype(X_train[c]):
                X_train[c], X_val[c], X_test[c] = factorize_together(
                    X_train[c], X_val[c], X_test[c]
                )

        print(f"  Features after encoding: {X_train.shape[1]}")

        # Train Bartz (expects features x samples)
        print("  Training Bartz...")
        model = gbart(
            X_train.to_numpy(dtype=np.float32).T,
            y_train.to_numpy(dtype=np.float32),
            **config.BARTZ_PARAMS,
            ndpost=config.NDPOST,
            nskip=config.NSKIP,
            keepevery=config.KEEPEVERY,
        )

        # Predict (clip to [0,1] as Bartz can output values outside this range)
        val_proba = model.predict(X_val.to_numpy(dtype=np.float32).T).mean(axis=0)
        val_proba = np.clip(val_proba, 1e-7, 1 - 1e-7)
        oof_preds[val_idx] = val_proba.astype(np.float32)

        fold_auc = roc_auc_score(y_val, val_proba)
        fold_logloss = log_loss(y_val, val_proba)
        fold_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))

        fold_scores.append({
            "fold": fold,
            "auc": fold_auc,
            "logloss": fold_logloss,
            "accuracy": fold_acc,
        })
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_logloss:.6f}, Accuracy: {fold_acc:.6f}")

        # Test predictions
        fold_test = model.predict(X_test.to_numpy(dtype=np.float32).T).mean(axis=0)
        test_preds += fold_test.astype(np.float32) / config.N_SPLITS

        fold_indices["train"].append(train_idx)
        fold_indices["val"].append(val_idx)

        del X_train, X_val, X_test, model, val_proba, fold_test
        gc.collect()

    # Overall OOF
    oof_preds = np.clip(oof_preds, 0, 1)
    test_preds = np.clip(test_preds, 0, 1)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_logloss = log_loss(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))

    print(f"\n{'='*80}")
    print("CV Results:")
    for s in fold_scores:
        print(f"  Fold {s['fold']}: AUC={s['auc']:.6f}, LogLoss={s['logloss']:.6f}, Acc={s['accuracy']:.6f}")
    print(f"\nOOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_logloss:.6f}")
    print(f"OOF Accuracy: {oof_acc:.6f}")
    print(f"{'='*80}")

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)

    # Visualization
    oof_pred_labels = (oof_preds > 0.5).astype(int)
    plot_confusion_matrix(y, oof_pred_labels, config.BASE_DIR / "confusion_matrix.png")
    plot_roc_curve(y, oof_preds, config.BASE_DIR / "roc_curve.png")

    # Classification report
    report = classification_report(y, oof_pred_labels, target_names=["No", "Yes"])
    report_path = config.BASE_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
