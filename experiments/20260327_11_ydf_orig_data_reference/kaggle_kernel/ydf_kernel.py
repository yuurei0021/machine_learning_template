"""
YDF (Yggdrasil Decision Forests) + Original Data Reference
Experiment: 20260327_11_ydf_orig_data_reference

Kaggle CPU execution (YDF not available on Windows).
Discussion 003: max_depth=2, BEST_FIRST_GLOBAL, categorical_algorithm=RANDOM
FE pipeline same as 20260323_01_orig_data_reference.
"""

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ydf"])

import warnings
from pathlib import Path
from itertools import combinations
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import ydf

warnings.filterwarnings("ignore")

# ============================================================================
# Kaggle paths (same as tabm_kernel.py)
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

# ============================================================================
# Helpers
# ============================================================================
def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")

# ============================================================================
# Data Loading (same pattern as tabm_kernel.py)
# ============================================================================
def load_orig_data():
    import os, glob
    # Find original data file with fallback
    orig_path = ORIG_DATA
    if not os.path.exists(orig_path):
        # Try alternative paths
        candidates = (
            glob.glob("/kaggle/input/*/WA_Fn*.csv") +
            glob.glob("/kaggle/input/*/*.csv")
        )
        print(f"  Primary path not found, scanning: {candidates[:10]}")
        for c in candidates:
            if "churn" in c.lower() and "train" not in c.lower() and "test" not in c.lower():
                orig_path = c
                print(f"  Using: {orig_path}")
                break
    orig = pd.read_csv(orig_path)
    if "customerID" in orig.columns:
        orig.drop(columns=["customerID"], inplace=True)
    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())
    orig["Churn_binary"] = (orig["Churn"] == "Yes").astype(int)
    keep_cols = CAT_COLS + NUM_COLS + ["SeniorCitizen", "Churn", "Churn_binary"]
    orig = orig[[c for c in keep_cols if c in orig.columns]]
    return orig

# ============================================================================
# Feature Engineering (same as 20260323_01)
# ============================================================================
def add_pre_cv_features(train_df, test_df, orig_df):
    dfs = [train_df, test_df]

    for col in NUM_COLS:
        freq = pd.concat([train_df[col], orig_df[col], test_df[col]]).value_counts(normalize=True)
        for df in dfs + [orig_df]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")

    for df in dfs + [orig_df]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
        df["is_first_month"] = (df["tenure"] == 1).astype("float32")
        df["dev_is_zero"] = (df["charges_deviation"] == 0).astype("float32")
        df["dev_sign"] = np.sign(df["charges_deviation"]).astype("float32")

    for df in dfs + [orig_df]:
        df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

    for col in ORIG_PROBA_CATS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    for col in NUM_COLS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")

    cross_pairs = list(combinations(ORIG_PROBA_CROSS_CATS, 2))
    for c1, c2 in cross_pairs:
        mapping = orig_df.groupby([c1, c2])["Churn_binary"].mean()
        name = f"ORIG_proba_{c1}_{c2}"
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[name] = idx.map(mapping).fillna(0.5).values.astype("float32")

    orig_ch_tc = orig_df.loc[orig_df["Churn_binary"] == 1, "TotalCharges"].values
    orig_nc_tc = orig_df.loc[orig_df["Churn_binary"] == 0, "TotalCharges"].values
    orig_tc = orig_df["TotalCharges"].values
    orig_ch_mc = orig_df.loc[orig_df["Churn_binary"] == 1, "MonthlyCharges"].values
    orig_nc_mc = orig_df.loc[orig_df["Churn_binary"] == 0, "MonthlyCharges"].values
    orig_ch_t = orig_df.loc[orig_df["Churn_binary"] == 1, "tenure"].values
    orig_nc_t = orig_df.loc[orig_df["Churn_binary"] == 0, "tenure"].values
    orig_is_mc_mean = orig_df.groupby("InternetService")["MonthlyCharges"].mean()

    for df in dfs:
        tc, mc, t = df["TotalCharges"].values, df["MonthlyCharges"].values, df["tenure"].values
        df["pctrank_ch_TC"] = pctrank_against(tc, orig_ch_tc)
        df["pctrank_nc_TC"] = pctrank_against(tc, orig_nc_tc)
        df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
        df["pctrank_gap_TC"] = (pctrank_against(tc, orig_ch_tc) - pctrank_against(tc, orig_nc_tc)).astype("float32")
        df["zscore_ch_TC"] = zscore_against(tc, orig_ch_tc)
        df["zscore_nc_TC"] = zscore_against(tc, orig_nc_tc)
        df["zscore_gap_TC"] = (np.abs(zscore_against(tc, orig_ch_tc)) - np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
        df["pctrank_ch_MC"] = pctrank_against(mc, orig_ch_mc)
        df["pctrank_nc_MC"] = pctrank_against(mc, orig_nc_mc)
        df["pctrank_gap_MC"] = (pctrank_against(mc, orig_ch_mc) - pctrank_against(mc, orig_nc_mc)).astype("float32")
        df["pctrank_ch_T"] = pctrank_against(t, orig_ch_t)
        df["pctrank_nc_T"] = pctrank_against(t, orig_nc_t)
        df["pctrank_gap_T"] = (pctrank_against(t, orig_ch_t) - pctrank_against(t, orig_nc_t)).astype("float32")
        vals_is = np.zeros(len(df), dtype="float32")
        vals_c = np.zeros(len(df), dtype="float32")
        for cat_val in orig_df["InternetService"].unique():
            mask = df["InternetService"] == cat_val
            ref = orig_df.loc[orig_df["InternetService"] == cat_val, "TotalCharges"].values
            if len(ref) > 0 and mask.sum() > 0:
                vals_is[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
        for cat_val in orig_df["Contract"].unique():
            mask = df["Contract"] == cat_val
            ref = orig_df.loc[orig_df["Contract"] == cat_val, "TotalCharges"].values
            if len(ref) > 0 and mask.sum() > 0:
                vals_c[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
        df["cond_pctrank_IS_TC"] = vals_is
        df["cond_pctrank_C_TC"] = vals_c
        df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0)).astype("float32")

    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q_tc, nc_q_tc = np.quantile(orig_ch_tc, q_val), np.quantile(orig_nc_tc, q_val)
        ch_q_t, nc_q_t = np.quantile(orig_ch_t, q_val), np.quantile(orig_nc_t, q_val)
        for df in dfs:
            df[f"dist_ch_TC_{q_label}"] = np.abs(df["TotalCharges"] - ch_q_tc).astype("float32")
            df[f"dist_nc_TC_{q_label}"] = np.abs(df["TotalCharges"] - nc_q_tc).astype("float32")
            df[f"qdist_gap_TC_{q_label}"] = (df[f"dist_nc_TC_{q_label}"] - df[f"dist_ch_TC_{q_label}"]).astype("float32")
            df[f"dist_ch_T_{q_label}"] = np.abs(df["tenure"] - ch_q_t).astype("float32")
            df[f"dist_nc_T_{q_label}"] = np.abs(df["tenure"] - nc_q_t).astype("float32")
            df[f"qdist_gap_T_{q_label}"] = (df[f"dist_nc_T_{q_label}"] - df[f"dist_ch_T_{q_label}"]).astype("float32")

    for df in dfs:
        t_str = df["tenure"].astype(str)
        mc_str = df["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
        tc_str = df["TotalCharges"].astype(str).str.replace(".", "", regex=False)
        df["tenure_mod10"] = (df["tenure"] % 10).astype("float32")
        df["tenure_mod12"] = (df["tenure"] % 12).astype("float32")
        df["tenure_years"] = (df["tenure"] // 12).astype("float32")
        df["tenure_months_in_year"] = (df["tenure"] % 12).astype("float32")
        df["tenure_is_multiple_12"] = ((df["tenure"] % 12) == 0).astype("float32")
        df["tenure_first_digit"] = t_str.str[0].astype(int).astype("float32")
        df["tenure_last_digit"] = t_str.str[-1].astype(int).astype("float32")
        df["mc_fractional"] = (df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])).astype("float32")
        df["mc_rounded_10"] = (np.round(df["MonthlyCharges"] / 10) * 10).astype("float32")
        df["mc_dev_from_round10"] = np.abs(df["MonthlyCharges"] - df["mc_rounded_10"]).astype("float32")
        df["mc_is_multiple_10"] = ((np.floor(df["MonthlyCharges"]) % 10) == 0).astype("float32")
        df["mc_first_digit"] = mc_str.str[0].astype(int).astype("float32")
        df["tc_fractional"] = (df["TotalCharges"] - np.floor(df["TotalCharges"])).astype("float32")
        df["tc_rounded_100"] = (np.round(df["TotalCharges"] / 100) * 100).astype("float32")
        df["tc_dev_from_round100"] = np.abs(df["TotalCharges"] - df["tc_rounded_100"]).astype("float32")
        df["tc_is_multiple_100"] = ((np.floor(df["TotalCharges"]) % 100) == 0).astype("float32")
        df["tc_first_digit"] = tc_str.str[0].astype(int).astype("float32")

    return train_df, test_df

# ============================================================================
# Main
# ============================================================================
def main():
    t0 = time.time()

    print("=" * 60)
    print("YDF + Original Data Reference Features")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_DATA)
    test_df = pd.read_csv(TEST_DATA)
    orig_df = load_orig_data()
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Orig: {orig_df.shape}")

    train_df[TARGET_COL] = train_df[TARGET_COL].map({"Yes": "Yes", "No": "No"})
    y_binary = (train_df[TARGET_COL] == "Yes").astype(int).values
    train_ids = train_df[ID_COL].values
    test_ids = test_df[ID_COL].values

    print(f"\nChurn rate: {y_binary.mean():.4f}")
    print("\nFeature Engineering...")
    train_df, test_df = add_pre_cv_features(train_df, test_df, orig_df)

    # YDF handles categoricals natively - keep string columns
    exclude_cols = {ID_COL, "Churn_binary"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and c != TARGET_COL]
    print(f"  Features: {len(feature_cols)}")

    # CV
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_binary)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

        tr_data = train_df.iloc[tr_idx][feature_cols + [TARGET_COL]].copy()
        va_data = train_df.iloc[va_idx][feature_cols].copy()
        te_data = test_df[feature_cols].copy()

        learner = ydf.GradientBoostedTreesLearner(
            label=TARGET_COL,
            task=ydf.Task.CLASSIFICATION,
            num_trees=5000,
            shrinkage=0.05,
            max_depth=2,
            growing_strategy="BEST_FIRST_GLOBAL",
            categorical_algorithm="RANDOM",
            early_stopping="LOSS_INCREASE",
            early_stopping_num_trees_look_ahead=200,
        )
        model = learner.train(tr_data)

        val_proba = model.predict(va_data)
        oof_preds[va_idx] = val_proba
        test_preds += model.predict(te_data) / N_SPLITS

        fold_auc = roc_auc_score(y_binary[va_idx], val_proba)
        print(f"  AUC: {fold_auc:.6f}")

    oof_auc = roc_auc_score(y_binary, oof_preds)
    oof_ll = log_loss(y_binary, oof_preds)
    print(f"\n{'='*60}")
    print(f"OOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")
    print(f"Time: {(time.time()-t0)/60:.1f}min")
    print(f"{'='*60}")

    # Save
    pd.DataFrame({"id": train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y_binary}).to_csv(f"{OUTPUT_DIR}/oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": test_preds}).to_csv(f"{OUTPUT_DIR}/test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_preds}).to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    with open(f"{OUTPUT_DIR}/fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {OUTPUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
