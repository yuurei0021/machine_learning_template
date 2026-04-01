"""
Ridge OOF Generation (201 features, 20-fold)
Experiment: 20260328_04_ridge_201feat_oof

201 features + Nested TE -> Ridge prediction.
Output: oof.csv (train ridge predictions), test_proba.csv (test ridge predictions)
These are used as input features for downstream XGB/LGB stacking experiments.
"""

import warnings
from pathlib import Path
from itertools import combinations
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import (
    LabelEncoder, TargetEncoder, StandardScaler, OneHotEncoder,
)
from sklearn.linear_model import Ridge
from scipy import sparse

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

    N_SPLITS = 20
    INNER_FOLDS = 5
    RANDOM_STATE = 42
    RIDGE_ALPHA = 10.0

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


config = Config()

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
# Data Loading
# ============================================================================
def load_data():
    print("Loading data...")
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)

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
    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)
    orig[config.TARGET_COL] = (orig[config.TARGET_COL] == "Yes").astype(int)
    for c in ["customerID", "Customer ID"]:
        if c in orig.columns:
            orig.drop(columns=[c], inplace=True)

    train[config.TARGET_COL] = train[config.TARGET_COL].map({"No": 0, "Yes": 1}).astype(int)
    print(f"  Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")
    return train, test, orig

# ============================================================================
# Feature Engineering (same as 20260326_05)
# ============================================================================
def add_features(train_df, test_df, orig_df):
    dfs = [train_df, test_df]
    new_nums = []
    TARGET = config.TARGET_COL

    print("  [1/8] Frequency Encoding...")
    for col in config.NUM_COLS:
        freq = pd.concat([train_df[col], orig_df[col]]).value_counts(normalize=True)
        fname = f"FREQ_{col}"
        for df in dfs + [orig_df]:
            df[fname] = df[col].map(freq).fillna(0).astype("float32")
        new_nums.append(fname)

    print("  [2/8] Arithmetic Features...")
    for df in dfs + [orig_df]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
        df["is_first_month"] = (df["tenure"] == 1).astype("float32")
        df["dev_is_zero"] = (df["charges_deviation"] == 0).astype("float32")
        df["dev_sign"] = np.sign(df["charges_deviation"]).astype("float32")
    new_nums += ["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges",
                 "is_first_month", "dev_is_zero", "dev_sign"]

    print("  [3/8] Service Counts...")
    for df in dfs + [orig_df]:
        df["service_count"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
    new_nums += ["service_count", "has_internet", "has_phone"]

    print("  [4/8] ORIG_proba single...")
    for col in config.CAT_COLS + config.NUM_COLS:
        mapping = orig_df.groupby(col)[TARGET].mean()
        fname = f"ORIG_proba_{col}"
        for df in dfs:
            df[fname] = df[col].map(mapping).fillna(0.5).astype("float32")
        new_nums.append(fname)

    print("  [5/8] ORIG_proba cross...")
    cross_pairs = list(combinations(config.ORIG_PROBA_CROSS_CATS, 2))
    for c1, c2 in cross_pairs:
        mapping = orig_df.groupby([c1, c2])[TARGET].mean()
        fname = f"ORIG_proba_{c1}_{c2}"
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[fname] = idx.map(mapping).fillna(0.5).values.astype("float32")
        new_nums.append(fname)

    print("  [6/8] Distribution Features...")
    orig_ch_tc = orig_df.loc[orig_df[TARGET] == 1, "TotalCharges"].values
    orig_nc_tc = orig_df.loc[orig_df[TARGET] == 0, "TotalCharges"].values
    orig_tc = orig_df["TotalCharges"].values
    orig_is_mc = orig_df.groupby("InternetService")["MonthlyCharges"].mean()
    orig_ch_mc = orig_df.loc[orig_df[TARGET] == 1, "MonthlyCharges"].values
    orig_nc_mc = orig_df.loc[orig_df[TARGET] == 0, "MonthlyCharges"].values
    orig_ch_t = orig_df.loc[orig_df[TARGET] == 1, "tenure"].values
    orig_nc_t = orig_df.loc[orig_df[TARGET] == 0, "tenure"].values

    for df in dfs:
        tc, mc, t = df["TotalCharges"].values, df["MonthlyCharges"].values, df["tenure"].values
        df["pctrank_ch_TC"] = pctrank_against(tc, orig_ch_tc)
        df["pctrank_nc_TC"] = pctrank_against(tc, orig_nc_tc)
        df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
        df["pctrank_gap_TC"] = (pctrank_against(tc, orig_ch_tc) - pctrank_against(tc, orig_nc_tc)).astype("float32")
        df["zscore_ch_TC"] = zscore_against(tc, orig_ch_tc)
        df["zscore_nc_TC"] = zscore_against(tc, orig_nc_tc)
        df["zscore_gap_TC"] = (np.abs(zscore_against(tc, orig_ch_tc)) - np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
        df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc).fillna(0)).astype("float32")
        df["pctrank_ch_MC"] = pctrank_against(mc, orig_ch_mc)
        df["pctrank_nc_MC"] = pctrank_against(mc, orig_nc_mc)
        df["pctrank_gap_MC"] = (pctrank_against(mc, orig_ch_mc) - pctrank_against(mc, orig_nc_mc)).astype("float32")
        df["pctrank_ch_T"] = pctrank_against(t, orig_ch_t)
        df["pctrank_nc_T"] = pctrank_against(t, orig_nc_t)
        df["pctrank_gap_T"] = (pctrank_against(t, orig_ch_t) - pctrank_against(t, orig_nc_t)).astype("float32")
        for cat_col, out_col in [("InternetService", "cond_pctrank_IS_TC"), ("Contract", "cond_pctrank_C_TC")]:
            vals = np.zeros(len(df), dtype="float32")
            for cv in orig_df[cat_col].unique():
                mask = df[cat_col] == cv
                ref = orig_df.loc[orig_df[cat_col] == cv, "TotalCharges"].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
            df[out_col] = vals

    new_nums += [
        "pctrank_ch_TC", "pctrank_nc_TC", "pctrank_orig_TC", "pctrank_gap_TC",
        "zscore_ch_TC", "zscore_nc_TC", "zscore_gap_TC", "resid_IS_MC",
        "pctrank_ch_MC", "pctrank_nc_MC", "pctrank_gap_MC",
        "pctrank_ch_T", "pctrank_nc_T", "pctrank_gap_T",
        "cond_pctrank_IS_TC", "cond_pctrank_C_TC",
    ]

    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q_tc = np.quantile(orig_ch_tc, q_val)
        nc_q_tc = np.quantile(orig_nc_tc, q_val)
        for df in dfs:
            df[f"dist_ch_TC_{q_label}"] = np.abs(df["TotalCharges"] - ch_q_tc).astype("float32")
            df[f"dist_nc_TC_{q_label}"] = np.abs(df["TotalCharges"] - nc_q_tc).astype("float32")
            df[f"qdist_gap_TC_{q_label}"] = (df[f"dist_nc_TC_{q_label}"] - df[f"dist_ch_TC_{q_label}"]).astype("float32")
        new_nums += [f"dist_ch_TC_{q_label}", f"dist_nc_TC_{q_label}", f"qdist_gap_TC_{q_label}"]

    print("  [7/8] Digit Features...")
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
        df["tenure_is_multiple_12"] = ((df["tenure"] % 12) == 0).astype("float32")
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
    DIGIT_FEATURES = [
        "tenure_first_digit", "tenure_last_digit", "tenure_mod10", "tenure_mod12",
        "tenure_months_in_year", "tenure_years", "tenure_is_multiple_12",
        "mc_first_digit", "mc_fractional", "mc_rounded_10", "mc_dev_from_round10", "mc_is_multiple_10",
        "tc_first_digit", "tc_fractional", "tc_rounded_100", "tc_dev_from_round100", "tc_is_multiple_100",
    ]
    new_nums += DIGIT_FEATURES

    print("  [8/8] N-gram features...")
    bigram_cols = []
    for c1, c2 in combinations(config.TOP_CATS_NGRAM, 2):
        name = f"BG_{c1}_{c2}"
        for df in dfs:
            df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        bigram_cols.append(name)
    trigram_cols = []
    top4 = config.TOP_CATS_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        name = f"TG_{c1}_{c2}_{c3}"
        for df in dfs:
            df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")
        trigram_cols.append(name)
    ngram_cols = bigram_cols + trigram_cols

    num_as_cat = []
    for col in config.NUM_COLS:
        name = f"CAT_{col}"
        num_as_cat.append(name)
        for df in dfs:
            df[name] = df[col].astype(str).astype("category")

    all_num_cols = config.NUM_COLS + ["SeniorCitizen"] + new_nums
    te_columns = num_as_cat + config.CAT_COLS

    print(f"  Numerical: {len(all_num_cols)}, Cat: {len(config.CAT_COLS)}, N-gram: {len(ngram_cols)}")
    return train_df, test_df, all_num_cols, ngram_cols, te_columns, DIGIT_FEATURES


# ============================================================================
# Main
# ============================================================================
def main():
    config.PRED_DIR.mkdir(exist_ok=True)
    t0_all = time.time()

    print("=" * 70)
    print("Ridge OOF Generation (201 features, 20-fold)")
    print(f"  N_SPLITS={config.N_SPLITS}, RIDGE_ALPHA={config.RIDGE_ALPHA}")
    print("=" * 70)

    train, test, orig = load_data()
    train_ids = train[config.ID_COL].values
    test_ids = test[config.ID_COL].values
    y = train[config.TARGET_COL].values

    print("\nFeature Engineering...")
    train, test, all_num_cols, ngram_cols, te_columns, digit_features = add_features(train, test, orig)

    CATS = config.CAT_COLS
    STATS = ["std", "min", "max"]
    TE_MEAN_COLS = [f"TE_{col}" for col in te_columns]

    skf_outer = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    skf_inner = StratifiedKFold(n_splits=config.INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    ridge_oof = np.zeros(len(train))
    ridge_pred = np.zeros(len(test))
    ridge_fold_scores = []
    fold_indices = {"train": [], "val": []}

    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, y)):
        t0 = time.time()
        print(f"\n--- Fold {i+1}/{config.N_SPLITS} ---")
        fold_indices["train"].append(train_idx)
        fold_indices["val"].append(val_idx)

        X_tr = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr = y[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y[val_idx]
        X_te = test.reset_index(drop=True).copy()

        # --- TE (std/min/max) ---
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr].copy()
            for col in te_columns:
                tmp = X_tr2.groupby(col, observed=False)[config.TARGET_COL].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[X_tr.index[in_va], c] = merged[c].values.astype("float32")

        for col in te_columns:
            tmp = X_tr.groupby(col, observed=False)[config.TARGET_COL].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)

        # --- N-gram TE ---
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr].copy()
            for col in ngram_cols:
                ng_te = X_tr2.groupby(col, observed=False)[config.TARGET_COL].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_tr.iloc[in_va][col].astype(str).map(ng_te)
                X_tr.loc[X_tr.index[in_va], ng_name] = pd.to_numeric(mapped, errors="coerce").fillna(0.5).astype("float32").values

        for col in ngram_cols:
            ng_te = X_tr.groupby(col, observed=False)[config.TARGET_COL].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
            X_te[ng_name] = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors="coerce").fillna(0.5).astype("float32")
            else:
                X_tr[ng_name] = 0.5

        # sklearn TargetEncoder (mean)
        te_enc = TargetEncoder(cv=config.INNER_FOLDS, shuffle=True, smooth="auto", target_type="binary", random_state=config.RANDOM_STATE)
        X_tr[TE_MEAN_COLS] = te_enc.fit_transform(X_tr[te_columns], y_tr)
        X_val[TE_MEAN_COLS] = te_enc.transform(X_val[te_columns])
        X_te[TE_MEAN_COLS] = te_enc.transform(X_te[te_columns])

        # === RIDGE ===
        te1_cols = [c for c in X_tr.columns if c.startswith("TE1_")]
        te_ng_cols = [c for c in X_tr.columns if c.startswith("TE_ng_")]
        ridge_numeric_features = all_num_cols + digit_features + te1_cols + te_ng_cols + TE_MEAN_COLS
        ridge_numeric_features = list(dict.fromkeys(ridge_numeric_features))

        scaler = StandardScaler()
        X_tr_rn = scaler.fit_transform(X_tr[ridge_numeric_features].fillna(0))
        X_val_rn = scaler.transform(X_val[ridge_numeric_features].fillna(0))
        X_te_rn = scaler.transform(X_te[ridge_numeric_features].fillna(0))

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_tr_rc = ohe.fit_transform(X_tr[CATS].astype(str))
        X_val_rc = ohe.transform(X_val[CATS].astype(str))
        X_te_rc = ohe.transform(X_te[CATS].astype(str))

        X_tr_ridge = sparse.hstack([X_tr_rn, X_tr_rc]).tocsr()
        X_val_ridge = sparse.hstack([X_val_rn, X_val_rc]).tocsr()
        X_te_ridge = sparse.hstack([X_te_rn, X_te_rc]).tocsr()

        if i == 0:
            print(f"  Ridge features: {X_tr_ridge.shape[1]}")

        ridge = Ridge(alpha=config.RIDGE_ALPHA, random_state=config.RANDOM_STATE)
        ridge.fit(X_tr_ridge, y_tr)

        ridge_val_pred = np.clip(ridge.predict(X_val_ridge), 0, 1)
        ridge_te_pred = np.clip(ridge.predict(X_te_ridge), 0, 1)

        ridge_oof[val_idx] = ridge_val_pred
        ridge_pred += ridge_te_pred / config.N_SPLITS

        fold_auc = roc_auc_score(y_val, ridge_val_pred)
        ridge_fold_scores.append(fold_auc)
        print(f"  Ridge AUC: {fold_auc:.6f}, Time: {(time.time()-t0)/60:.1f}min")

    oof_auc = roc_auc_score(y, ridge_oof)
    oof_ll = log_loss(y, ridge_oof)
    print(f"\n{'='*70}")
    print(f"Ridge OOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")
    print(f"Total time: {(time.time()-t0_all)/60:.1f}min")
    print(f"{'='*70}")

    # Save
    pd.DataFrame({"id": train_ids, "prob": ridge_oof, "predicted": (ridge_oof > 0.5).astype(int), "true": y}).to_csv(config.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": ridge_pred}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": ridge_pred}).to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {config.PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
