"""
TabM + Original Data Reference Features
Experiment: 20260325_10_tabm_orig_data_reference

目的: TabM (PyTabKit) で元データ参照特徴量を使い、アンサンブル多様性を確保
アプローチ: yekenot FE + ORIG_proba特徴量 + TabM_D_Classifier (5-Fold Stratified CV)
実行: Kaggle GPU (T4 x2) で実行。ローカルではVRAM不足のため実行不可。
      kaggle_kernel/ のスクリプトと同一ロジック。パスのみ異なる。

ローカル実行コマンド（GPU 30GB+ 必要）:
    uv run python experiments/20260325_10_tabm_orig_data_reference/main.py

Kaggle実行:
    uv run kaggle kernels push -p experiments/20260325_10_tabm_orig_data_reference/kaggle_kernel/
"""

import os, random, warnings
from pathlib import Path
from itertools import combinations
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, TargetEncoder
from sklearn.metrics import roc_auc_score, log_loss

import torch
from pytabkit import TabM_D_Classifier

warnings.filterwarnings("ignore")

# ============================================================================
# Config (local paths)
# ============================================================================
class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent.parent / "data"
    TRAIN_DATA = DATA_DIR / "raw" / "train.csv"
    TEST_DATA = DATA_DIR / "raw" / "test.csv"
    ORIG_DATA = DATA_DIR / "raw" / "orig" / "Telco_customer_churn.xlsx"
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
    ORIG_PROBA_CROSS_CATS = [
        "Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
    ]

    # TabM params (same as kaggle_kernel/tabm_kernel.py)
    TABM_PARAMS = {
        "random_state": 42,
        "verbosity": 2,
        "val_metric_name": "1-auc_ovr",
        "n_cv": 1,
        "n_refit": 0,
        "n_epochs": 50,
        "patience": 10,
        "batch_size": 256,
        "lr": 0.002,
        "weight_decay": 1e-5,
        "d_block": 512,
        "n_blocks": 3,
        "dropout": 0.1,
        "num_emb_type": "pbld",
        "num_emb_n_bins": 64,
        "d_embedding": 16,
        "tabm_k": 32,
        "tfms": ["one_hot", "median_center", "robust_scale",
                 "smooth_clip", "l2_normalize"],
    }

config = Config()

# ============================================================================
# Helpers (identical to kaggle_kernel)
# ============================================================================
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")

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
    if orig["SeniorCitizen"].dtype == object:
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    orig["Churn_binary"] = (orig["Churn"] == "Yes").astype(int)
    keep_cols = config.CAT_COLS + config.NUM_COLS + ["SeniorCitizen", "Churn", "Churn_binary"]
    orig = orig[[c for c in keep_cols if c in orig.columns]]
    return orig

# ============================================================================
# Feature Engineering (identical to kaggle_kernel)
# ============================================================================
def add_features(train_df, test_df, orig_df):
    dfs = [train_df, test_df]

    print("  Arithmetic interactions...")
    for df in dfs:
        df["_MC_div_TC"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1e-6)).astype("float32")
        df["_TC_div_tenure"] = (df["TotalCharges"] / (df["tenure"] + 1e-6)).astype("float32")
        df["_MC_to_avg_ratio"] = (df["MonthlyCharges"] / (df["_TC_div_tenure"] + 1e-6)).astype("float32")
        df["_TC_div_MC"] = (df["TotalCharges"] / (df["MonthlyCharges"] + 1e-6)).astype("float32")
        df["_tenure_sq"] = (df["tenure"] ** 2).astype("float32")
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

    print("  Digit/modular features...")
    for df in dfs:
        df["_TC_mod100"] = (np.floor(df["TotalCharges"]) % 100).astype("float32")
        df["_TC_mod1000"] = (np.floor(df["TotalCharges"]) % 1000).astype("float32")
        df["TC_is_mult10_"] = (np.floor(df["TotalCharges"]) % 10 == 0).astype("category")
        df["TC_d_m3_"] = ((df["TotalCharges"] * 1e-3) % 10).astype("int8").astype("category")
        df["is_loyal_"] = (df["tenure"] >= 24).astype("category")

    print("  KBins discretization...")
    bin_config = {"TotalCharges": [4000, 500], "MonthlyCharges": [200, 100]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_bin_"
            kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
            train_df[bin_name] = kb.fit_transform(train_df[[col]]).ravel().astype("int32")
            test_df[bin_name] = kb.transform(test_df[[col]]).ravel().astype("int32")
            train_df[bin_name] = train_df[bin_name].astype("category")
            test_df[bin_name] = test_df[bin_name].astype("category")

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
        test_df[cat_name] = series_te.map(code_map).fillna(-1).astype("int32").astype("category")

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
        test_df[combo_name] = combo_te.map(code_map).fillna(-1).astype("int32").astype("category")

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
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[f"_ORIG_proba_{c1}_{c2}"] = idx.map(mapping).fillna(0.5).values.astype("float32")

    print("  Encoding cat cols as integer category...")
    for col in config.CAT_COLS:
        codes, uniques = train_df[col].factorize()
        train_df[col] = codes
        train_df[col] = train_df[col].astype("category")
        code_map = {cat: i for i, cat in enumerate(uniques)}
        test_df[col] = test_df[col].map(code_map).fillna(-1).astype("int32").astype("category")

    return train_df, test_df, combo_names

# ============================================================================
# Main (identical logic to kaggle_kernel, different paths)
# ============================================================================
def main():
    seed_everything(config.RANDOM_STATE)
    print("=" * 80)
    print("TabM + Original Data Reference Features")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    train_df = pd.read_csv(config.TRAIN_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
    orig_df = load_orig_data()
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Orig: {orig_df.shape}")

    train_df[config.TARGET_COL] = train_df[config.TARGET_COL].map({"Yes": 1, "No": 0})
    y = train_df[config.TARGET_COL]
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values
    print(f"\nChurn rate: {y.mean():.4f}")

    print("\nFeature Engineering...")
    train_df, test_df, combo_names = add_features(train_df, test_df, orig_df)

    drop_cols = [config.ID_COL, config.TARGET_COL]
    X = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=[config.ID_COL])
    print(f"\n  X shape: {X.shape}, X_test shape: {X_test.shape}")

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    te_cols = combo_names

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[va_idx].copy()
        X_tst = X_test.copy()
        y_tr = y.iloc[tr_idx]
        y_val = y.iloc[va_idx]

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

        model = TabM_D_Classifier(**config.TABM_PARAMS)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[va_idx] = val_proba
        test_preds += model.predict_proba(X_tst)[:, 1] / config.N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        fold_ll = log_loss(y_val, val_proba)
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_ll:.6f}")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)
        torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\n{'='*80}")
    print(f"OOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_ll:.6f}")
    print(f"{'='*80}")

    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y.values}).to_csv(config.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({config.ID_COL: test_ids, "prob": test_preds}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: test_preds}).to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"Saved to {config.PRED_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
