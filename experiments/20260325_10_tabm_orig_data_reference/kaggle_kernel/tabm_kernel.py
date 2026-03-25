"""
TabM + Original Data Reference Features (Kaggle GPU execution)
Experiment: 20260325_10_tabm_orig_data_reference

This is the Kaggle kernel version of main.py with path adjustments only.
Preprocessing: StandardScaler for numericals, OrdinalEncoder for categoricals,
following the reference notebook approach for TabM.
"""

import subprocess, sys
# Install pytabkit without replacing Kaggle's pre-installed PyTorch (P100 compatible)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "pytabkit"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "delu", "rtdl_num_embeddings"])

import gc, os, random, time, warnings
from itertools import combinations
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

import torch
from pytabkit import TabM_D_Classifier

warnings.filterwarnings("ignore")

# ============================================================================
# Kaggle paths
# ============================================================================
COMP_DIR = "/kaggle/input/competitions/playground-series-s6e3"
ORIG_DIR = "/kaggle/input/telco-customer-churn"
OUTPUT_DIR = "/kaggle/working"

TRAIN_DATA = f"{COMP_DIR}/train.csv"
TEST_DATA = f"{COMP_DIR}/test.csv"
ORIG_DATA = f"{ORIG_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv"

TARGET_COL = "Churn"
ID_COL = "id"
N_SPLITS = 5
INNER_FOLDS = 5
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
TOP_CATS_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]

TABM_PARAMS = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "random_state": 42,
    "verbosity": 2,
    "arch_type": "tabm-mini-normal",
    "tabm_k": 32,
    "num_emb_type": "pwl",
    "d_embedding": 24,
    "batch_size": 512,
    "lr": 1e-3,
    "n_epochs": 50,
    "dropout": 0.2,
    "d_block": 256,
    "n_blocks": 3,
    "patience": 10,
    "weight_decay": 1e-3,
}

# ============================================================================
# Helpers
# ============================================================================
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")

# ============================================================================
# Data Loading
# ============================================================================
def load_orig_data():
    orig = pd.read_csv(ORIG_DATA)
    if "customerID" in orig.columns:
        orig.drop(columns=["customerID"], inplace=True)
    if orig["SeniorCitizen"].dtype == object:
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    orig[TARGET_COL] = orig[TARGET_COL].map({"Yes": 1, "No": 0})
    return orig

# ============================================================================
# Feature Engineering
# ============================================================================
def add_features(train_df, test_df, orig_df):
    """All engineered features are numerical (float32)."""
    dfs = [train_df, test_df]
    new_nums = []

    # 1. Frequency Encoding
    print("  [1/7] Frequency Encoding...")
    for col in NUM_COLS:
        freq = pd.concat([train_df[col], orig_df[col], test_df[col]]).value_counts(normalize=True)
        fname = f"FREQ_{col}"
        for df in dfs + [orig_df]:
            df[fname] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(fname)

    # 2. Arithmetic Features
    print("  [2/7] Arithmetic Features...")
    for df in dfs + [orig_df]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
    new_nums += ["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges"]

    # 3. Service Counts
    print("  [3/7] Service Counts...")
    for df in dfs + [orig_df]:
        df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
    new_nums += ["service_count", "has_internet", "has_phone"]

    # 4. ORIG_proba (single + cross)
    print("  [4/7] ORIG_proba mapping...")
    for col in CAT_COLS + NUM_COLS:
        mapping = orig_df.groupby(col)[TARGET_COL].mean()
        fname = f"ORIG_proba_{col}"
        for df in dfs:
            df[fname] = df[col].map(mapping).fillna(0.5).astype("float32")
        new_nums.append(fname)

    cross_pairs = list(combinations(ORIG_PROBA_CROSS_CATS, 2))
    for c1, c2 in cross_pairs:
        mapping = orig_df.groupby([c1, c2])[TARGET_COL].mean()
        fname = f"ORIG_proba_{c1}_{c2}"
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[fname] = idx.map(mapping).fillna(0.5).values.astype("float32")
        new_nums.append(fname)

    # 5. Distribution Features
    print("  [5/7] Distribution Features...")
    orig_ch_tc = orig_df.loc[orig_df[TARGET_COL] == 1, "TotalCharges"].values
    orig_nc_tc = orig_df.loc[orig_df[TARGET_COL] == 0, "TotalCharges"].values
    orig_tc = orig_df["TotalCharges"].values
    orig_is_mc = orig_df.groupby("InternetService")["MonthlyCharges"].mean()

    for df in dfs:
        tc = df["TotalCharges"].values
        df["pctrank_churner_TC"] = pctrank_against(tc, orig_ch_tc)
        df["pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nc_tc)
        df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
        df["pctrank_churn_gap_TC"] = (pctrank_against(tc, orig_ch_tc) - pctrank_against(tc, orig_nc_tc)).astype("float32")
        df["zscore_nonchurner_TC"] = zscore_against(tc, orig_nc_tc)
        df["zscore_churn_gap_TC"] = (np.abs(zscore_against(tc, orig_ch_tc)) - np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
        df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc).fillna(0)).astype("float32")
        for cat_col, out_col in [("InternetService", "cond_pctrank_IS_TC"), ("Contract", "cond_pctrank_C_TC")]:
            vals = np.zeros(len(df), dtype="float32")
            for cv in orig_df[cat_col].unique():
                mask = df[cat_col] == cv
                ref = orig_df.loc[orig_df[cat_col] == cv, "TotalCharges"].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
            df[out_col] = vals
    new_nums += [
        "pctrank_churner_TC", "pctrank_nonchurner_TC", "pctrank_orig_TC",
        "pctrank_churn_gap_TC", "zscore_nonchurner_TC", "zscore_churn_gap_TC",
        "resid_IS_MC", "cond_pctrank_IS_TC", "cond_pctrank_C_TC",
    ]

    # 6. Quantile Distance Features
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        for df in dfs:
            df[f"dist_ch_TC_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype("float32")
            df[f"dist_nc_TC_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype("float32")
            df[f"qdist_gap_TC_{q_label}"] = (df[f"dist_nc_TC_{q_label}"] - df[f"dist_ch_TC_{q_label}"]).astype("float32")
        new_nums += [f"dist_ch_TC_{q_label}", f"dist_nc_TC_{q_label}", f"qdist_gap_TC_{q_label}"]

    # 7. Digit Features (all numerical)
    print("  [6/7] Digit Features...")
    for df in dfs:
        t_str = df["tenure"].astype(str)
        mc_str = df["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
        tc_str = df["TotalCharges"].astype(str).str.replace(".", "", regex=False)

        df["tenure_first_digit"] = t_str.str[0].astype(int).astype("float32")
        df["tenure_last_digit"] = t_str.str[-1].astype(int).astype("float32")
        df["tenure_mod10"] = (df["tenure"] % 10).astype("float32")
        df["tenure_mod12"] = (df["tenure"] % 12).astype("float32")
        df["tenure_months_in_year"] = (df["tenure"] % 12).astype("float32")
        df["tenure_years"] = (df["tenure"] // 12).astype("float32")

        df["mc_first_digit"] = mc_str.str[0].astype(int).astype("float32")
        df["mc_fractional"] = (df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])).astype("float32")
        df["mc_rounded_10"] = (np.round(df["MonthlyCharges"] / 10) * 10).astype("float32")
        df["mc_dev_from_round10"] = np.abs(df["MonthlyCharges"] - df["mc_rounded_10"]).astype("float32")
        df["mc_is_multiple_10"] = (np.floor(df["MonthlyCharges"]) % 10 == 0).astype("float32")

        df["tc_first_digit"] = tc_str.str[0].astype(int).astype("float32")
        df["tc_fractional"] = (df["TotalCharges"] - np.floor(df["TotalCharges"])).astype("float32")
        df["tc_rounded_100"] = (np.round(df["TotalCharges"] / 100) * 100).astype("float32")
        df["tc_dev_from_round100"] = np.abs(df["TotalCharges"] - df["tc_rounded_100"]).astype("float32")
        df["tc_is_multiple_100"] = (np.floor(df["TotalCharges"]) % 100 == 0).astype("float32")

    new_nums += [
        "tenure_first_digit", "tenure_last_digit", "tenure_mod10", "tenure_mod12",
        "tenure_months_in_year", "tenure_years",
        "mc_first_digit", "mc_fractional", "mc_rounded_10", "mc_dev_from_round10", "mc_is_multiple_10",
        "tc_first_digit", "tc_fractional", "tc_rounded_100", "tc_dev_from_round100", "tc_is_multiple_100",
    ]

    # 8. N-gram columns (for target encoding in CV loop)
    print("  [7/7] N-gram interactions...")
    bigram_cols = []
    for c1, c2 in combinations(TOP_CATS_NGRAM, 2):
        name = f"BG_{c1}_{c2}"
        for df in dfs:
            df[name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        bigram_cols.append(name)

    trigram_cols = []
    top4 = TOP_CATS_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        name = f"TG_{c1}_{c2}_{c3}"
        for df in dfs:
            df[name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        trigram_cols.append(name)

    ngram_cols = bigram_cols + trigram_cols

    # Numericals as categories for TE
    num_as_cat = []
    for col in NUM_COLS:
        name = f"CAT_{col}"
        for df in dfs:
            df[name] = df[col].astype(str)
        num_as_cat.append(name)

    te_columns = num_as_cat + CAT_COLS

    all_num_cols = NUM_COLS + ["SeniorCitizen"] + new_nums
    print(f"  Numerical features: {len(all_num_cols)}")
    print(f"  Categorical features: {len(CAT_COLS)}")
    print(f"  N-gram columns: {len(ngram_cols)}")
    print(f"  TE columns: {len(te_columns)}")

    return train_df, test_df, all_num_cols, ngram_cols, te_columns

# ============================================================================
# Main
# ============================================================================
def main():
    seed_everything(RANDOM_STATE)
    print("=" * 80)
    print("TabM + Original Data Reference Features (Kaggle GPU)")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    orig_df = load_orig_data()
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Orig: {orig_df.shape}")

    train_df[TARGET_COL] = train_df[TARGET_COL].map({"Yes": 1, "No": 0})
    y = train_df[TARGET_COL].values
    train_ids = train_df[ID_COL].values
    test_ids = test_df[ID_COL].values

    print(f"\nChurn rate: {y.mean():.4f}")
    print("\nFeature Engineering...")
    train_df, test_df, all_num_cols, ngram_cols, te_columns = add_features(train_df, test_df, orig_df)

    # CV
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    t0_all = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
        t0 = time.time()
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

        X_tr = train_df.iloc[tr_idx].reset_index(drop=True).copy()
        y_tr = y[tr_idx]
        X_val = train_df.iloc[va_idx].reset_index(drop=True).copy()
        y_val = y[va_idx]
        X_te = test_df.copy()

        # --- Target Encoding (inner K-fold, leakage-safe) ---
        te_feat_names = [f"TE1_{col}_mean" for col in te_columns]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.0

        X_tr[TARGET_COL] = y_tr
        for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in te_columns:
                tmp = X_tr2.groupby(col)[TARGET_COL].mean().rename(f"TE1_{col}_mean")
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how="left")
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = merged[f"TE1_{col}_mean"].values

        for col in te_columns:
            tmp = X_tr.groupby(col)[TARGET_COL].mean().rename(f"TE1_{col}_mean")
            X_val[f"TE1_{col}_mean"] = X_val[[col]].merge(tmp, on=col, how="left")[f"TE1_{col}_mean"].values
            X_te[f"TE1_{col}_mean"] = X_te[[col]].merge(tmp, on=col, how="left")[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[TARGET_COL], inplace=True)

        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(0.5).astype("float32")

        # --- N-gram TE (full-fold mean) ---
        ng_te_feat_names = [f"TE_ng_{col}" for col in ngram_cols]
        X_tr[TARGET_COL] = y_tr
        for col in ngram_cols:
            ng_te = X_tr.groupby(col)[TARGET_COL].mean()
            ng_n = f"TE_ng_{col}"
            X_tr[ng_n] = X_tr[col].map(ng_te).fillna(0.5).astype("float32")
            X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype("float32")
            X_te[ng_n] = X_te[col].map(ng_te).fillna(0.5).astype("float32")
        X_tr.drop(columns=[TARGET_COL], inplace=True)

        # --- Prepare final features ---
        all_nums_final = all_num_cols + te_feat_names + ng_te_feat_names

        if fold == 0:
            print(f"  Numeric features: {len(all_nums_final)}")
            print(f"  Categorical features: {len(CAT_COLS)}")

        # OrdinalEncoder for categoricals
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(X_tr[CAT_COLS].astype(str))
        X_tr_cat = encoder.transform(X_tr[CAT_COLS].astype(str))
        X_val_cat = encoder.transform(X_val[CAT_COLS].astype(str))
        X_te_cat = encoder.transform(X_te[CAT_COLS].astype(str))

        # StandardScaler for numericals
        for df in [X_tr, X_val, X_te]:
            df[all_nums_final] = df[all_nums_final].fillna(0).astype("float32")
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[all_nums_final])
        X_val_num = scaler.transform(X_val[all_nums_final])
        X_te_num = scaler.transform(X_te[all_nums_final])

        # Combine
        all_cols = all_nums_final + CAT_COLS
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=all_cols)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=all_cols)
        X_te_final = pd.DataFrame(np.hstack([X_te_num, X_te_cat]), columns=all_cols)

        for c in CAT_COLS:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_te_final[c] = X_te_final[c].astype(int)

        # --- Train TabM ---
        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_tr_final, y_tr, X_val=X_val_final, y_val=y_val, cat_col_names=CAT_COLS)

        val_proba = model.predict_proba(X_val_final)[:, 1]
        oof_preds[va_idx] = val_proba
        test_preds += model.predict_proba(X_te_final)[:, 1] / N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        fold_ll = log_loss(y_val, val_proba)
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_ll:.6f}, Time: {(time.time()-t0)/60:.1f}min")

        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

        del model, X_tr_final, X_val_final, X_te_final, X_tr, X_val, X_te
        gc.collect()
        torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\n{'='*80}")
    print(f"OOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_ll:.6f}")
    print(f"Total time: {(time.time()-t0_all)/60:.1f}min")
    print(f"{'='*80}")

    # Save
    pd.DataFrame({ID_COL: train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(f"{OUTPUT_DIR}/oof.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, "prob": test_preds}).to_csv(f"{OUTPUT_DIR}/test_proba.csv", index=False)
    pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds}).to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    with open(f"{OUTPUT_DIR}/fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"Saved to {OUTPUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
