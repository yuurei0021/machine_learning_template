"""
TabM + NB1 Features (20-fold)
Experiment: 20260330_01_tabm_nb1feat_20fold

Reference notebook FE (blamerx, CV 0.91898) with 20-fold CV.
- CATS includes SeniorCitizen (16 cols)
- Digit features (35)
- No ORIG_proba cross (matching reference)
- 20-fold CV (reference used 10-fold)
"""

import subprocess, sys
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

TARGET = "Churn"
N_FOLDS = 10
INNER_FOLDS = 5
SEED = 42

# SeniorCitizen included as categorical (matching reference)
CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

TOP_CATS_FOR_NGRAM = [
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
    return (np.zeros(len(values), dtype="float32") if sigma == 0
            else ((values - mu) / sigma).astype("float32"))

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    import glob
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)

    orig_path = ORIG_DATA
    if not os.path.exists(orig_path):
        print(f"  Primary path not found: {orig_path}")
        print(f"  /kaggle/input/ contents: {os.listdir('/kaggle/input/')}")
        for d in os.listdir("/kaggle/input/"):
            full = f"/kaggle/input/{d}"
            if os.path.isdir(full):
                sub = os.listdir(full)
                print(f"    {full}/: {sub[:5]}")
                for s in sub:
                    sf = f"{full}/{s}"
                    if os.path.isdir(sf):
                        print(f"      {sf}/: {os.listdir(sf)[:5]}")
        candidates = glob.glob("/kaggle/input/**/*.csv", recursive=True)
        print(f"  Scanning {len(candidates)} CSV files (recursive)...")
        for c in candidates:
            basename = os.path.basename(c).lower()
            # Match original IBM Telco churn data
            if ("churn" in basename or "telco" in basename or "s6e" in basename) \
               and "train" not in basename and "test" not in basename and "sample" not in basename and "submission" not in basename:
                orig_path = c
                print(f"  Using orig: {orig_path}")
                break
    orig = pd.read_csv(orig_path)
    if "customerID" in orig.columns:
        orig.drop(columns=["customerID"], inplace=True)

    train[TARGET] = train[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    orig[TARGET] = orig[TARGET].map({"No": 0, "Yes": 1}).astype(int)

    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)

    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)

    # SeniorCitizen as string for categorical treatment
    for df in [train, test, orig]:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")
    return train, test, orig

# ============================================================================
# Feature Engineering (reference notebook exact reproduction)
# ============================================================================
def feature_engineering(train, test, orig):
    NEW_NUMS = []

    # 1. Frequency Encoding
    print("  [1/7] Frequency Encoding...")
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        NEW_NUMS.append(f"FREQ_{col}")

    # 2. Arithmetic Features
    print("  [2/7] Arithmetic Features...")
    for df in [train, test, orig]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
    NEW_NUMS += ["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges"]

    # 3. Service Counts
    print("  [3/7] Service Counts...")
    SVC = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
           "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for df in [train, test, orig]:
        df["service_count"] = (df[SVC] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
    NEW_NUMS += ["service_count", "has_internet", "has_phone"]

    # 4. ORIG_proba mapping
    print("  [4/7] ORIG_proba mapping...")
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train_merged = train.merge(tmp.rename(_name), on=col, how="left")
        train[_name] = train_merged[_name].fillna(0.5).astype("float32")
        test_merged = test.merge(tmp.rename(_name), on=col, how="left")
        test[_name] = test_merged[_name].fillna(0.5).astype("float32")
        NEW_NUMS.append(_name)

    # 5. Distribution Features
    print("  [5/7] Distribution Features...")
    orig_ch_tc = orig.loc[orig[TARGET] == 1, "TotalCharges"].values
    orig_nc_tc = orig.loc[orig[TARGET] == 0, "TotalCharges"].values
    orig_tc = orig["TotalCharges"].values
    orig_is_mc = orig.groupby("InternetService")["MonthlyCharges"].mean()

    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nc_tc)
        df["pctrank_churner_TC"] = pctrank_against(tc, orig_ch_tc)
        df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
        df["zscore_churn_gap_TC"] = (np.abs(zscore_against(tc, orig_ch_tc)) - np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
        df["zscore_nonchurner_TC"] = zscore_against(tc, orig_nc_tc)
        df["pctrank_churn_gap_TC"] = (pctrank_against(tc, orig_ch_tc) - pctrank_against(tc, orig_nc_tc)).astype("float32")
        df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc).fillna(0)).astype("float32")
        for cat_col, out_col in [("InternetService", "cond_pctrank_IS_TC"), ("Contract", "cond_pctrank_C_TC")]:
            vals = np.zeros(len(df), dtype="float32")
            for cv in orig[cat_col].unique():
                mask = df[cat_col] == cv
                ref = orig.loc[orig[cat_col] == cv, "TotalCharges"].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
            df[out_col] = vals

    NEW_NUMS += [
        "pctrank_nonchurner_TC", "zscore_churn_gap_TC", "pctrank_churn_gap_TC",
        "resid_IS_MC", "cond_pctrank_IS_TC", "zscore_nonchurner_TC",
        "pctrank_orig_TC", "pctrank_churner_TC", "cond_pctrank_C_TC",
    ]

    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        for df in [train, test]:
            df[f"dist_To_ch_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype("float32")
            df[f"dist_To_nc_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype("float32")
            df[f"qdist_gap_To_{q_label}"] = (df[f"dist_To_nc_{q_label}"] - df[f"dist_To_ch_{q_label}"]).astype("float32")
    NEW_NUMS += [
        "qdist_gap_To_q50", "dist_To_ch_q50", "dist_To_nc_q50",
        "dist_To_nc_q25", "qdist_gap_To_q25",
        "dist_To_nc_q75", "dist_To_ch_q75", "qdist_gap_To_q75",
    ]

    # 6. Digit Features (35 features, matching reference)
    print("  [6/7] Digit Features...")
    DIGIT_FEATURES = [
        "tenure_first_digit", "tenure_last_digit", "tenure_second_digit",
        "tenure_mod10", "tenure_mod12", "tenure_num_digits",
        "tenure_is_multiple_10", "tenure_rounded_10", "tenure_dev_from_round10",
        "mc_first_digit", "mc_last_digit", "mc_second_digit",
        "mc_mod10", "mc_mod100", "mc_num_digits",
        "mc_is_multiple_10", "mc_is_multiple_50",
        "mc_rounded_10", "mc_fractional", "mc_dev_from_round10",
        "tc_first_digit", "tc_last_digit", "tc_second_digit",
        "tc_mod10", "tc_mod100", "tc_num_digits",
        "tc_is_multiple_10", "tc_is_multiple_100",
        "tc_rounded_100", "tc_fractional", "tc_dev_from_round100",
        "tenure_years", "tenure_months_in_year", "mc_per_digit", "tc_per_digit",
    ]
    for df in [train, test]:
        t_str = df["tenure"].astype(str)
        df["tenure_first_digit"] = t_str.str[0].astype(int)
        df["tenure_last_digit"] = t_str.str[-1].astype(int)
        df["tenure_second_digit"] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["tenure_mod10"] = df["tenure"] % 10
        df["tenure_mod12"] = df["tenure"] % 12
        df["tenure_num_digits"] = t_str.str.len()
        df["tenure_is_multiple_10"] = (df["tenure"] % 10 == 0).astype("float32")
        df["tenure_rounded_10"] = np.round(df["tenure"] / 10) * 10
        df["tenure_dev_from_round10"] = np.abs(df["tenure"] - df["tenure_rounded_10"])

        mc_str = df["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
        df["mc_first_digit"] = mc_str.str[0].astype(int)
        df["mc_last_digit"] = mc_str.str[-1].astype(int)
        df["mc_second_digit"] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["mc_mod10"] = np.floor(df["MonthlyCharges"]) % 10
        df["mc_mod100"] = np.floor(df["MonthlyCharges"]) % 100
        df["mc_num_digits"] = np.floor(df["MonthlyCharges"]).astype(int).astype(str).str.len()
        df["mc_is_multiple_10"] = (np.floor(df["MonthlyCharges"]) % 10 == 0).astype("float32")
        df["mc_is_multiple_50"] = (np.floor(df["MonthlyCharges"]) % 50 == 0).astype("float32")
        df["mc_rounded_10"] = np.round(df["MonthlyCharges"] / 10) * 10
        df["mc_fractional"] = df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])
        df["mc_dev_from_round10"] = np.abs(df["MonthlyCharges"] - df["mc_rounded_10"])

        tc_str = df["TotalCharges"].astype(str).str.replace(".", "", regex=False)
        df["tc_first_digit"] = tc_str.str[0].astype(int)
        df["tc_last_digit"] = tc_str.str[-1].astype(int)
        df["tc_second_digit"] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df["tc_mod10"] = np.floor(df["TotalCharges"]) % 10
        df["tc_mod100"] = np.floor(df["TotalCharges"]) % 100
        df["tc_num_digits"] = np.floor(df["TotalCharges"]).astype(int).astype(str).str.len()
        df["tc_is_multiple_10"] = (np.floor(df["TotalCharges"]) % 10 == 0).astype("float32")
        df["tc_is_multiple_100"] = (np.floor(df["TotalCharges"]) % 100 == 0).astype("float32")
        df["tc_rounded_100"] = np.round(df["TotalCharges"] / 100) * 100
        df["tc_fractional"] = df["TotalCharges"] - np.floor(df["TotalCharges"])
        df["tc_dev_from_round100"] = np.abs(df["TotalCharges"] - df["tc_rounded_100"])

        df["tenure_years"] = df["tenure"] // 12
        df["tenure_months_in_year"] = df["tenure"] % 12
        df["mc_per_digit"] = df["MonthlyCharges"] / (df["mc_num_digits"] + 0.001)
        df["tc_per_digit"] = df["TotalCharges"] / (df["tc_num_digits"] + 0.001)

        for c in DIGIT_FEATURES:
            df[c] = df[c].astype("float32")

    NEW_NUMS += DIGIT_FEATURES

    # 7. N-gram
    print("  [7/7] N-gram Features...")
    BIGRAM_COLS = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        BIGRAM_COLS.append(col_name)

    TRIGRAM_COLS = []
    TOP4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        TRIGRAM_COLS.append(col_name)

    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS

    NUM_AS_CAT = []
    for col in NUMS:
        _new = f"CAT_{col}"
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS

    all_num_cols = NUMS + NEW_NUMS
    print(f"  Numerical: {len(all_num_cols)}, Categorical: {len(CATS)}, N-gram: {len(NGRAM_COLS)}")

    return train, test, all_num_cols, NGRAM_COLS, TE_COLUMNS

# ============================================================================
# Main
# ============================================================================
def main():
    seed_everything(SEED)
    t0_all = time.time()

    print("=" * 70)
    print(f"TabM + NB1 Features ({N_FOLDS}-fold CV)")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    train, test, orig = load_data()
    train_ids = train["id"].values
    test_ids = test["id"].values
    y = train[TARGET].values

    print("\nFeature Engineering...")
    train, test, all_num_cols, ngram_cols, te_columns = feature_engineering(train, test, orig)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y)):
        t0 = time.time()
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---")

        X_tr = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr = y[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y[val_idx]
        X_te = test.copy()

        # TE1 (mean, inner K-fold)
        te_feat_names = [f"TE1_{col}_mean" for col in te_columns]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.0

        X_tr[TARGET] = y_tr
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in te_columns:
                tmp = X_tr2.groupby(col)[TARGET].mean().rename(f"TE1_{col}_mean")
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how="left")
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = merged[f"TE1_{col}_mean"].values

        for col in te_columns:
            tmp = X_tr.groupby(col)[TARGET].mean().rename(f"TE1_{col}_mean")
            X_val[f"TE1_{col}_mean"] = X_val[[col]].merge(tmp, on=col, how="left")[f"TE1_{col}_mean"].values
            X_te[f"TE1_{col}_mean"] = X_te[[col]].merge(tmp, on=col, how="left")[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[TARGET], inplace=True)

        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(0.5).astype("float32")

        # N-gram TE (full-fold mean)
        ng_te_feat_names = [f"TE_ng_{col}" for col in ngram_cols]
        X_tr[TARGET] = y_tr
        for col in ngram_cols:
            ng_te = X_tr.groupby(col)[TARGET].mean()
            ng_n = f"TE_ng_{col}"
            X_tr[ng_n] = X_tr[col].map(ng_te).fillna(0.5).astype("float32")
            X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype("float32")
            X_te[ng_n] = X_te[col].map(ng_te).fillna(0.5).astype("float32")
        X_tr.drop(columns=[TARGET], inplace=True)

        # Prepare final features
        ALL_NUMS_FINAL = all_num_cols + te_feat_names + ng_te_feat_names
        ALL_CATS_FINAL = CATS

        if fold_i == 0:
            print(f"  Numeric features: {len(ALL_NUMS_FINAL)}")
            print(f"  Categorical features: {len(ALL_CATS_FINAL)}")

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(X_tr[ALL_CATS_FINAL].astype(str))
        X_tr_cat = encoder.transform(X_tr[ALL_CATS_FINAL].astype(str))
        X_val_cat = encoder.transform(X_val[ALL_CATS_FINAL].astype(str))
        X_te_cat = encoder.transform(X_te[ALL_CATS_FINAL].astype(str))

        for df in [X_tr, X_val, X_te]:
            df[ALL_NUMS_FINAL] = df[ALL_NUMS_FINAL].fillna(0).astype("float32")
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[ALL_NUMS_FINAL])
        X_val_num = scaler.transform(X_val[ALL_NUMS_FINAL])
        X_te_num = scaler.transform(X_te[ALL_NUMS_FINAL])

        ALL_COLS = ALL_NUMS_FINAL + ALL_CATS_FINAL
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=ALL_COLS)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=ALL_COLS)
        X_te_final = pd.DataFrame(np.hstack([X_te_num, X_te_cat]), columns=ALL_COLS)

        for c in ALL_CATS_FINAL:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_te_final[c] = X_te_final[c].astype(int)

        # Train TabM
        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_tr_final, y_tr, X_val=X_val_final, y_val=y_val, cat_col_names=ALL_CATS_FINAL)

        val_proba = model.predict_proba(X_val_final)[:, 1]
        oof[val_idx] = val_proba
        pred += model.predict_proba(X_te_final)[:, 1] / N_FOLDS

        fold_auc = roc_auc_score(y_val, val_proba)
        fold_scores.append(fold_auc)
        print(f"  AUC: {fold_auc:.6f}, Time: {(time.time()-t0)/60:.1f}min")

        del model, X_tr_final, X_val_final, X_te_final
        gc.collect()
        torch.cuda.empty_cache()

    oof_auc = roc_auc_score(y, oof)
    oof_ll = log_loss(y, oof)
    print(f"\n{'='*70}")
    print(f"OOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")
    print(f"Total time: {(time.time()-t0_all)/60:.1f}min")
    print(f"{'='*70}")

    pd.DataFrame({"id": train_ids, "prob": oof, "predicted": (oof > 0.5).astype(int), "true": y}).to_csv(f"{OUTPUT_DIR}/oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": pred}).to_csv(f"{OUTPUT_DIR}/test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, TARGET: pred}).to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

    print(f"Saved to {OUTPUT_DIR}")
    print("Done!")

if __name__ == "__main__":
    main()
