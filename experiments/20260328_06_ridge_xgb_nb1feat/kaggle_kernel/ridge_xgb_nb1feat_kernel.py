"""
Ridge -> XGB (NB1 features, ridge from exp05, our Optuna XGB params, 20-fold)
Experiment: 20260328_06_ridge_xgb_nb1feat

NB1 FE + ridge_pred (from 20260328_05_ridge_nb1feat_oof) + our Optuna XGB params.
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import pickle
from pathlib import Path
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration (matches original notebook exactly)
# ============================================================================
class CFG:
    TARGET = "Churn"
    N_FOLDS = 20
    INNER_FOLDS = 5
    RANDOM_SEED = 42
    COMP_DIR = "/kaggle/input/competitions/playground-series-s6e3"
    ORIG_DIR = "/kaggle/input/telco-customer-churn"
    RIDGE_DIR = "/kaggle/input/datasets/nyhalcyon/ridge-nb1feat-oof"

    TRAIN_PATH = f"{COMP_DIR}/train.csv"
    TEST_PATH = f"{COMP_DIR}/test.csv"
    ORIG_PATH = f"{ORIG_DIR}/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    RIDGE_OOF = f"{RIDGE_DIR}/ridge_oof.csv"
    RIDGE_TEST = f"{RIDGE_DIR}/ridge_test_proba.csv"

    PRED_DIR = Path("/kaggle/working")
    MODEL_DIR = Path("/kaggle/working/model")

TOP_CATS_FOR_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]

# Our Optuna-optimized XGB params (from 20260323_02)
XGB_PARAMS = {
    "n_estimators": 50000,
    "learning_rate": 0.004787126983706307,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.7797316213800588,
    "colsample_bytree": 0.27036930195333186,
    "reg_alpha": 0.0023599517797442456,
    "reg_lambda": 9.90372658125681,
    "gamma": 1.1966979929346213,
    "random_state": CFG.RANDOM_SEED,
    "early_stopping_rounds": 500,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
    "verbosity": 0,
    "n_jobs": -1,
}

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    print("Loading datasets...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)

    import os, glob
    orig_path = CFG.ORIG_PATH
    if not os.path.exists(orig_path):
        candidates = glob.glob("/kaggle/input/*/*.csv")
        for c in candidates:
            if "churn" in c.lower() and "train" not in c.lower() and "test" not in c.lower() and "sample" not in c.lower():
                orig_path = c
                print(f"  Using orig: {orig_path}")
                break
    orig = pd.read_csv(orig_path)
    if "customerID" in orig.columns:
        orig.drop(columns=["customerID"], inplace=True)

    train[CFG.TARGET] = train[CFG.TARGET].map({"No": 0, "Yes": 1}).astype(int)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({"No": 0, "Yes": 1}).astype(int)

    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)

    train_ids = train["id"].copy()
    test_ids = test["id"].copy()

    print(f"  Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")
    return train, test, orig, train_ids, test_ids


# ============================================================================
# Helpers
# ============================================================================
def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype="float32") if sigma == 0
            else ((values - mu) / sigma).astype("float32"))


# ============================================================================
# Feature Engineering (matches original notebook)
# ============================================================================
def feature_engineering(train, test, orig):
    CATS = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
    NEW_NUMS = []
    NUM_AS_CAT = []

    # 1. Frequency Encoding (train + orig only, no test)
    print("  [1/7] Frequency Encoding...")
    for col in NUMS:
        freq = pd.concat([train[col], orig[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
        NEW_NUMS.append(f"FREQ_{col}")

    # 2. Arithmetic Interactions
    print("  [2/7] Arithmetic Interactions...")
    for df in [train, test, orig]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
    NEW_NUMS += ["charges_deviation", "monthly_to_total_ratio", "avg_monthly_charges"]

    # 3. Service Counts
    print("  [3/7] Service Counts...")
    SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for df in [train, test, orig]:
        df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
    NEW_NUMS += ["service_count", "has_internet", "has_phone"]

    # 4. ORIG_proba Mapping
    print("  [4/7] ORIG_proba mapping...")
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype("float32")
        NEW_NUMS.append(_name)

    # 5. Distribution Features
    print("  [5/7] Distribution Features...")
    orig_churner_tc = orig.loc[orig[CFG.TARGET] == 1, "TotalCharges"].values
    orig_nonchurner_tc = orig.loc[orig[CFG.TARGET] == 0, "TotalCharges"].values
    orig_tc = orig["TotalCharges"].values
    orig_is_mc_mean = orig.groupby("InternetService")["MonthlyCharges"].mean()

    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_nonchurner_TC"] = pctrank_against(tc, orig_nonchurner_tc)
        df["pctrank_churner_TC"] = pctrank_against(tc, orig_churner_tc)
        df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
        df["zscore_churn_gap_TC"] = (np.abs(zscore_against(tc, orig_churner_tc)) -
                                     np.abs(zscore_against(tc, orig_nonchurner_tc))).astype("float32")
        df["zscore_nonchurner_TC"] = zscore_against(tc, orig_nonchurner_tc)
        df["pctrank_churn_gap_TC"] = (pctrank_against(tc, orig_churner_tc) -
                                      pctrank_against(tc, orig_nonchurner_tc)).astype("float32")
        df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0)).astype("float32")

        for cat_col, out_col in [("InternetService", "cond_pctrank_IS_TC"), ("Contract", "cond_pctrank_C_TC")]:
            vals = np.zeros(len(df), dtype="float32")
            for cat_val in orig[cat_col].unique():
                mask = df[cat_col] == cat_val
                ref = orig.loc[orig[cat_col] == cat_val, "TotalCharges"].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, "TotalCharges"].values, ref)
            df[out_col] = vals

    DIST_FEATURES = [
        "pctrank_nonchurner_TC", "zscore_churn_gap_TC", "pctrank_churn_gap_TC",
        "resid_IS_MC", "cond_pctrank_IS_TC", "zscore_nonchurner_TC",
        "pctrank_orig_TC", "pctrank_churner_TC", "cond_pctrank_C_TC",
    ]
    NEW_NUMS += DIST_FEATURES

    # 6. Quantile Distance Features
    print("  [6/7] Quantile Distance...")
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f"dist_To_ch_{q_label}"] = np.abs(df["TotalCharges"] - ch_q).astype("float32")
            df[f"dist_To_nc_{q_label}"] = np.abs(df["TotalCharges"] - nc_q).astype("float32")
            df[f"qdist_gap_To_{q_label}"] = (df[f"dist_To_nc_{q_label}"] - df[f"dist_To_ch_{q_label}"]).astype("float32")
    QDIST_FEATURES = [
        "qdist_gap_To_q50", "dist_To_ch_q50", "dist_To_nc_q50",
        "dist_To_nc_q25", "qdist_gap_To_q25",
        "dist_To_nc_q75", "dist_To_ch_q75", "qdist_gap_To_q75",
    ]
    NEW_NUMS += QDIST_FEATURES

    # 7. Numericals as Categories
    print("  [7/7] Numericals as Categories...")
    for col in NUMS:
        _new = f"CAT_{col}"
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype("category")

    # Digit Features
    print("  Digit Features...")
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
        df["tenure_dev_from_round10"] = abs(df["tenure"] - df["tenure_rounded_10"])

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
        df["mc_dev_from_round10"] = abs(df["MonthlyCharges"] - df["mc_rounded_10"])

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
        df["tc_dev_from_round100"] = abs(df["TotalCharges"] - df["tc_rounded_100"])

        df["tenure_years"] = df["tenure"] // 12
        df["tenure_months_in_year"] = df["tenure"] % 12
        df["mc_per_digit"] = df["MonthlyCharges"] / (df["mc_num_digits"] + 0.001)
        df["tc_per_digit"] = df["TotalCharges"] / (df["tc_num_digits"] + 0.001)

    NEW_NUMS += DIGIT_FEATURES

    # N-gram features
    print("  N-gram features...")
    BIGRAM_COLS = []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        BIGRAM_COLS.append(col_name)

    TRIGRAM_COLS = []
    TOP4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")
        TRIGRAM_COLS.append(col_name)

    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS

    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS = NUM_AS_CAT + CATS
    TO_REMOVE = NUM_AS_CAT + CATS + NGRAM_COLS
    STATS = ["std", "min", "max"]

    print(f"  Total features: {len(FEATURES)}")

    return (train, test, FEATURES, CATS, NUMS, NEW_NUMS, NUM_AS_CAT,
            NGRAM_COLS, TE_COLUMNS, TO_REMOVE, STATS, DIGIT_FEATURES)


# ============================================================================
# Main
# ============================================================================
def load_ridge_predictions():
    print("Loading Ridge predictions from experiment 05...")
    import os, glob
    ridge_oof_path = CFG.RIDGE_OOF
    ridge_test_path = CFG.RIDGE_TEST
    if not os.path.exists(ridge_oof_path):
        # Fallback: search in dataset dirs
        for c in glob.glob("/kaggle/input/*/ridge_oof.csv"):
            ridge_oof_path = c
            ridge_test_path = c.replace("ridge_oof.csv", "ridge_test_proba.csv")
            print(f"  Found ridge at: {ridge_oof_path}")
            break
    ridge_oof = pd.read_csv(ridge_oof_path)
    ridge_test = pd.read_csv(ridge_test_path)
    print(f"  Ridge OOF: {ridge_oof.shape}, Ridge Test: {ridge_test.shape}")
    print(f"  Ridge OOF AUC: {roc_auc_score(ridge_oof['true'], ridge_oof['prob']):.6f}")
    return ridge_oof["prob"].values, ridge_test["prob"].values


def main():
    CFG.PRED_DIR.mkdir(exist_ok=True)
    CFG.MODEL_DIR.mkdir(exist_ok=True)
    t0_all = time.time()

    print("=" * 70)
    print("Ridge -> XGB (NB1 features + ridge_pred from exp05, 20-fold)")
    print(f"  N_FOLDS={CFG.N_FOLDS}")
    print("=" * 70)

    train, test, orig, train_ids, test_ids = load_data()
    ridge_oof_vals, ridge_test_vals = load_ridge_predictions()

    print("\nFeature Engineering...")
    (train, test, FEATURES, CATS, NUMS, NEW_NUMS, NUM_AS_CAT,
     NGRAM_COLS, TE_COLUMNS, TO_REMOVE, STATS, DIGIT_FEATURES) = feature_engineering(train, test, orig)

    # Add ridge_pred
    train["ridge_pred"] = ridge_oof_vals.astype("float32")
    test["ridge_pred"] = ridge_test_vals.astype("float32")

    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    xgb_oof = np.zeros(len(train))
    xgb_pred = np.zeros(len(test))
    xgb_fold_scores = []
    fold_indices = {"train": [], "val": []}
    TE_MEAN_COLS = [f"TE_{col}" for col in TE_COLUMNS]

    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        t0 = time.time()
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        fold_indices["train"].append(train_idx)
        fold_indices["val"].append(val_idx)

        X_tr = train.loc[train_idx, FEATURES + [CFG.TARGET, "ridge_pred"]].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES + ["ridge_pred"]].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te = test[FEATURES + ["ridge_pred"]].reset_index(drop=True).copy()

        # --- Inner KFold TE for original categoricals (std/min/max) ---
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[X_tr.index[in_va], c] = merged[c].values.astype("float32")

        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)

        # --- Inner KFold TE for N-gram categoricals (mean) ---
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr].copy()
            for col in NGRAM_COLS:
                ng_te = X_tr2.groupby(col, observed=False)[CFG.TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_tr.iloc[in_va][col].astype(str).map(ng_te)
                X_tr.loc[X_tr.index[in_va], ng_name] = pd.to_numeric(mapped, errors="coerce").fillna(0.5).astype("float32").values

        for col in NGRAM_COLS:
            ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
            X_te[ng_name] = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors="coerce").fillna(0.5).astype("float32")
            else:
                X_tr[ng_name] = 0.5

        # sklearn TargetEncoder (mean)
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth="auto", target_type="binary", random_state=CFG.RANDOM_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])

        # Drop raw categoricals/ngrams, keep ridge_pred
        for df in [X_tr, X_val, X_te]:
            df.drop(columns=[c for c in TO_REMOVE if c in df.columns], inplace=True, errors="ignore")
        X_tr.drop(columns=[CFG.TARGET], inplace=True, errors="ignore")
        COLS_XGB = X_tr.columns

        if i == 0:
            print(f"  XGB features: {len(COLS_XGB)} (includes ridge_pred)")

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1000)

        xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
        xgb_fold_scores.append(fold_auc)
        xgb_pred += model.predict_proba(X_te[COLS_XGB])[:, 1] / CFG.N_FOLDS

        print(f"  XGB AUC: {fold_auc:.6f}, Time: {(time.time()-t0)/60:.1f}min")

        del X_tr, X_val, X_te, model
        gc.collect()

    xgb_overall = roc_auc_score(train[CFG.TARGET], xgb_oof)

    print(f"\n{'='*70}")
    print(f"XGB OOF AUC: {xgb_overall:.6f} (mean fold: {np.mean(xgb_fold_scores):.6f})")
    print(f"Total time: {(time.time()-t0_all)/60:.1f}min")
    print(f"{'='*70}")

    pd.DataFrame({"id": train_ids, "prob": xgb_oof, "predicted": (xgb_oof > 0.5).astype(int), "true": train[CFG.TARGET].values}).to_csv(CFG.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": xgb_pred}).to_csv(CFG.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": xgb_pred}).to_csv(CFG.PRED_DIR / "test.csv", index=False)
    with open(CFG.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {CFG.PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
