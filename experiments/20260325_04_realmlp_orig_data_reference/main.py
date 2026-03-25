"""
Original Data Reference Distribution + RealMLP
Experiment: 20260325_04_realmlp_orig_data_reference

目的: RealMLP (PyTabKit) で元データ参照特徴量を使い、アンサンブル多様性を確保
アプローチ: 参照Notebook (yekenot) のFE + 実験01のORIG_proba特徴量 + RealMLP_TD_Classifier
参考: https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit/notebook
      https://www.kaggle.com/competitions/playground-series-s6e3/discussion/683316
"""

import os
import random
import warnings
from pathlib import Path
from itertools import combinations
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

import torch
from pytabkit import RealMLP_TD_Classifier

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
    MODEL_DIR = BASE_DIR / "model"

    N_SPLITS = 5
    RANDOM_STATE = 42

    CAT_COLS = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]
    NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
    ORIG_PROBA_CATS = [
        "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
        "TechSupport", "OnlineBackup", "DeviceProtection", "PaperlessBilling",
        "StreamingMovies", "StreamingTV", "Partner", "Dependents",
    ]
    ORIG_PROBA_CROSS_CATS = [
        "Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
    ]

    # RealMLP params (from yekenot notebook + discussion insights)
    REALMLP_PARAMS = {
        "random_state": 42,
        "verbosity": 2,
        "val_metric_name": "1-auc_ovr",

        "n_ens": 8,
        "n_epochs": 3,
        "batch_size": 256,
        "use_early_stopping": True,
        "early_stopping_additive_patience": 10,
        "early_stopping_multiplicative_patience": 1,

        "lr": 0.075,
        "wd": 0.0236,
        "sq_mom": 0.988,
        "lr_sched": "flat_anneal",
        "first_layer_lr_factor": 0.25,

        "add_front_scale": False,

        "embedding_size": 6,
        "max_one_hot_cat_size": 18,
        "hidden_sizes": [512, 256, 128],
        "act": "silu",
        "p_drop": 0.05,
        "p_drop_sched": "expm4t",

        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_act_name": "gelu",
        "plr_lr_factor": 0.1151,
        "plr_sigma": 2.33,

        "ls_eps": 0.01,
        "ls_eps_sched": "sqrt_cos",

        "tfms": ["one_hot", "median_center", "robust_scale",
                 "smooth_clip", "embedding", "l2_normalize"],
    }


config = Config()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    print("Loading data...")
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    return train, test


def load_orig_data():
    print("Loading original data...")
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
    if orig["SeniorCitizen"].dtype == object:
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    orig["Churn_binary"] = (orig["Churn"] == "Yes").astype(int)
    keep_cols = [config.ID_COL] if config.ID_COL in orig.columns else []
    keep_cols += config.CAT_COLS + config.NUM_COLS + ["SeniorCitizen", "Churn", "Churn_binary"]
    orig = orig[[c for c in keep_cols if c in orig.columns]]
    print(f"  Original: {orig.shape}")
    return orig


# ============================================================================
# Feature Engineering
# ============================================================================
def add_features(train_df, test_df, orig_df):
    """Feature engineering combining yekenot notebook FE + ORIG_proba features."""
    dfs = [train_df, test_df]

    # --- Notebook FE: Arithmetic interactions ---
    print("  Arithmetic interactions...")
    for df in dfs:
        df["_MC_div_TC"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1e-6)).astype("float32")
        df["_TC_div_tenure"] = (df["TotalCharges"] / (df["tenure"] + 1e-6)).astype("float32")
        df["_MC_to_avg_ratio"] = (df["MonthlyCharges"] / (df["_TC_div_tenure"] + 1e-6)).astype("float32")
        df["_TC_div_MC"] = (df["TotalCharges"] / (df["MonthlyCharges"] + 1e-6)).astype("float32")
        df["_tenure_sq"] = (df["tenure"] ** 2).astype("float32")
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

    # --- Notebook FE: Digit/modular features ---
    print("  Digit/modular features...")
    for df in dfs:
        df["_TC_mod100"] = (np.floor(df["TotalCharges"]) % 100).astype("float32")
        df["_TC_mod1000"] = (np.floor(df["TotalCharges"]) % 1000).astype("float32")
        df["TC_is_mult10_"] = (np.floor(df["TotalCharges"]) % 10 == 0).astype("category")
        df["TC_d_m3_"] = ((df["TotalCharges"] * 1e-3) % 10).astype("int8").astype("category")
        df["is_loyal_"] = (df["tenure"] >= 24).astype("category")

    # --- Notebook FE: KBinsDiscretizer ---
    print("  KBins discretization...")
    bin_config = {"TotalCharges": [4000, 500], "MonthlyCharges": [200, 100]}
    category_map = {}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_bin_"
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
            train_df[bin_name] = kb.fit_transform(train_df[[col]]).ravel().astype("int32")
            test_df[bin_name] = kb.transform(test_df[[col]]).ravel().astype("int32")
            train_df[bin_name] = train_df[bin_name].astype("category")
            test_df[bin_name] = test_df[bin_name].astype("category")
            category_map[bin_name] = kb

    # --- Notebook FE: Categorize numericals ---
    print("  Categorize numericals...")
    for col in config.NUM_COLS:
        cat_name = f"{col}_cat_"
        round_flag = col == "TotalCharges"
        series_tr = train_df[col].round(0) if round_flag else train_df[col]
        codes_tr, uniques = series_tr.factorize()
        train_df[cat_name] = codes_tr
        train_df[cat_name] = train_df[cat_name].astype("category")

        series_te = test_df[col].round(0) if round_flag else test_df[col]
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes_te = series_te.map(code_map).fillna(-1).astype("int32")
        test_df[cat_name] = codes_te
        test_df[cat_name] = test_df[cat_name].astype("category")

    # --- Notebook FE: 3-way combo ---
    print("  Categorical combos...")
    important_combos = [("Contract", "InternetService", "PaymentMethod")]
    combo_names = []
    for col1, col2, col3 in important_combos:
        combo_name = f"{col1}_{col2}_{col3}_"
        combo_names.append(combo_name)
        combo_tr = train_df[col1].astype(str) + "_" + train_df[col2].astype(str) + "_" + train_df[col3].astype(str)
        codes_tr, uniques = combo_tr.factorize()
        train_df[combo_name] = codes_tr
        train_df[combo_name] = train_df[combo_name].astype("category")

        combo_te = test_df[col1].astype(str) + "_" + test_df[col2].astype(str) + "_" + test_df[col3].astype(str)
        code_map = {cat: i for i, cat in enumerate(uniques)}
        test_df[combo_name] = combo_te.map(code_map).fillna(-1).astype("int32")
        test_df[combo_name] = test_df[combo_name].astype("category")

    # --- ORIG_proba features (from experiment 01) ---
    print("  ORIG_proba single...")
    for col in config.ORIG_PROBA_CATS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"_ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    for col in config.NUM_COLS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"_ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")

    print("  ORIG_proba cross...")
    cross_pairs = list(combinations(config.ORIG_PROBA_CROSS_CATS, 2))
    for c1, c2 in cross_pairs:
        mapping = orig_df.groupby([c1, c2])["Churn_binary"].mean()
        name = f"_ORIG_proba_{c1}_{c2}"
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[name] = idx.map(mapping).fillna(0.5).values.astype("float32")

    # Factorize original categorical columns for RealMLP (integer codes + category dtype)
    print("  Encoding cat cols as integer category...")
    for col in config.CAT_COLS:
        codes, uniques = train_df[col].factorize()
        train_df[col] = codes
        train_df[col] = train_df[col].astype("category")
        code_map = {cat: i for i, cat in enumerate(uniques)}
        test_df[col] = test_df[col].map(code_map).fillna(-1).astype("int32")
        test_df[col] = test_df[col].astype("category")

    return train_df, test_df, combo_names


# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y, X_test, combo_names):
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation (RealMLP)")
    print(f"{'='*80}")

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}
    fold_scores = []

    skf = StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE,
    )

    te_cols = combo_names

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")

        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        X_tst = X_test.copy()
        y_tr = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(f"  Train: {len(X_tr)}, Val: {len(X_val)}")

        # Target encoding for combo columns within fold
        if te_cols:
            te = TargetEncoder(cv=config.N_SPLITS, smooth="auto", shuffle=True, random_state=config.RANDOM_STATE)
            tr_enc = te.fit_transform(X_tr[te_cols], y_tr)
            val_enc = te.transform(X_val[te_cols])
            tst_enc = te.transform(X_tst[te_cols])
            te_names = [f"_{col}TE" for col in te_cols]
            X_tr[te_names] = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc

        if fold == 0:
            print(f"  Features: {X_tr.shape[1]}")

        model = RealMLP_TD_Classifier(**config.REALMLP_PARAMS)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        test_preds += model.predict_proba(X_tst)[:, 1] / config.N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        fold_logloss = log_loss(y_val, val_proba)
        fold_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))

        fold_scores.append({
            "fold": fold + 1, "auc": fold_auc, "logloss": fold_logloss,
            "accuracy": fold_acc,
        })
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_logloss:.6f}, Accuracy: {fold_acc:.6f}")

        fold_indices["train"].append(train_idx)
        fold_indices["val"].append(val_idx)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    return oof_preds, test_preds, fold_indices, fold_scores


# ============================================================================
# Visualization & Save
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
    plt.title("ROC Curve (OOF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices):
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({
        config.ID_COL: train_ids, "prob": oof_preds,
        "predicted": (oof_preds > 0.5).astype(int), "true": y.values,
    })
    oof_df.to_csv(config.PRED_DIR / "oof.csv", index=False)
    test_proba_df = pd.DataFrame({config.ID_COL: test_ids, "prob": test_preds})
    test_proba_df.to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    submit_df = pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: test_preds})
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"Predictions saved to {config.PRED_DIR}")


# ============================================================================
# Main
# ============================================================================
def main():
    seed_everything(config.RANDOM_STATE)

    print("=" * 80)
    print("Original Data Reference Distribution + RealMLP")
    print("=" * 80)

    train_df, test_df = load_data()
    orig_df = load_orig_data()

    # Prepare target
    train_df[config.TARGET_COL] = train_df[config.TARGET_COL].map({"Yes": 1, "No": 0})
    y = train_df[config.TARGET_COL]
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print(f"\nChurn rate: {y.mean():.4f}")

    # Feature engineering
    print(f"\n{'='*80}")
    print("Feature Engineering")
    print(f"{'='*80}")
    train_df, test_df, combo_names = add_features(train_df, test_df, orig_df)

    # Prepare X (drop ID and target)
    drop_cols = [config.ID_COL, config.TARGET_COL]
    X = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=[config.ID_COL])

    print(f"\n  X shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Train
    oof_preds, test_preds, fold_indices, fold_scores = train_with_cv(
        X, y, X_test, combo_names,
    )

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)

    # Visualization
    oof_pred_labels = (oof_preds > 0.5).astype(int)
    plot_confusion_matrix(y.values, oof_pred_labels, config.BASE_DIR / "confusion_matrix.png")
    plot_roc_curve(y.values, oof_preds, config.BASE_DIR / "roc_curve.png")

    report = classification_report(y.values, oof_pred_labels, target_names=["No", "Yes"])
    report_path = config.BASE_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
