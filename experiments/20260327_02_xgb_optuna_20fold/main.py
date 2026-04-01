"""
XGBoost Optuna Best Params with 20-fold CV
Experiment: 20260327_02_xgb_optuna_20fold

目的: 実験02のOptuna最適パラメータを20-fold CVで実行し、fold数増加の効果を検証
ベース: 20260323_02_xgb_optuna_tuning (5-fold OOF AUC 0.91893)
"""

import warnings
import json
import time
from pathlib import Path
from itertools import combinations
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
import xgboost as xgb
import optuna

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

    RANDOM_STATE = 42
    INNER_FOLDS = 5

    # CV settings
    SEARCH_N_SPLITS = 20  # not used (no Optuna search)
    FINAL_N_SPLITS = 20

    # Categorical / Numeric column definitions
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
# Data Loading (same as experiment 01)
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
    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
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
# Feature Engineering (same as experiment 01)
# ============================================================================
def add_pre_cv_features(train_df, test_df, orig_df):
    dfs = [train_df, test_df]

    print("  Group 1: Frequency Encoding...")
    for col in config.NUM_COLS:
        freq = pd.concat([train_df[col], orig_df[col], test_df[col]]).value_counts(normalize=True)
        for df in dfs + [orig_df]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")

    print("  Group 2: Arithmetic Interactions...")
    for df in dfs + [orig_df]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
        df["is_first_month"] = (df["tenure"] == 1).astype("float32")
        df["dev_is_zero"] = (df["charges_deviation"] == 0).astype("float32")
        df["dev_sign"] = np.sign(df["charges_deviation"]).astype("float32")

    print("  Group 3: Service Counts...")
    for df in dfs + [orig_df]:
        df["service_count"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

    print("  Group 4: ORIG_proba single...")
    for col in config.ORIG_PROBA_CATS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    for col in config.NUM_COLS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")

    print("  Group 5: ORIG_proba cross...")
    cross_pairs = list(combinations(config.ORIG_PROBA_CROSS_CATS, 2))
    for c1, c2 in cross_pairs:
        mapping = orig_df.groupby([c1, c2])["Churn_binary"].mean()
        name = f"ORIG_proba_{c1}_{c2}"
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[name] = idx.map(mapping).fillna(0.5).values.astype("float32")

    print("  Group 6: Distribution Features...")
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

    print("  Group 7: Quantile Distance Features...")
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

    print("  Group 8: Digit/Modular Features...")
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


def prepare_base_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    label_encoders = {}
    for col in config.CAT_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le
    exclude_cols = {config.TARGET_COL, config.ID_COL, "Churn_binary"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    print(f"  Base features: {X_train.shape[1]}")
    return X_train, X_test, label_encoders


def create_ngram_columns(train_df, test_df):
    bigram_cols = []
    for c1, c2 in combinations(config.TOP_CATS_NGRAM, 2):
        name = f"BG_{c1}_{c2}"
        for df in [train_df, test_df]:
            df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
        bigram_cols.append(name)
    trigram_cols = []
    top4 = config.TOP_CATS_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        name = f"TG_{c1}_{c2}_{c3}"
        for df in [train_df, test_df]:
            df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str))
        trigram_cols.append(name)
    num_as_cat_cols = []
    for col in config.NUM_COLS:
        name = f"CAT_{col}"
        for df in [train_df, test_df]:
            df[name] = df[col].astype(str)
        num_as_cat_cols.append(name)
    return bigram_cols, trigram_cols, num_as_cat_cols


# ============================================================================
# Target Encoding (same as experiment 01)
# ============================================================================
def apply_target_encoding(X_tr, y_tr, X_val, X_te,
                          te_columns, ngram_columns, inner_folds, seed):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    # Layer A: TE with std/min/max
    stats = ["std", "min", "max"]
    te1_cols = [f"TE1_{col}_{s}" for col in te_columns for s in stats]
    for c in te1_cols:
        X_tr[c] = np.float32(0)
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr]
        temp_df = X_tr2.copy()
        temp_df["__target__"] = y_tr[in_tr]
        for col in te_columns:
            agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
            agg.columns = [f"TE1_{col}_{s}" for s in stats]
            for c in agg.columns:
                vals = pd.to_numeric(X_tr.iloc[in_va][col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values
                X_tr.iloc[in_va, X_tr.columns.get_loc(c)] = vals
    temp_df = X_tr.copy()
    temp_df["__target__"] = y_tr
    for col in te_columns:
        agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
        agg.columns = [f"TE1_{col}_{s}" for s in stats]
        for c in agg.columns:
            X_val[c] = pd.to_numeric(X_val[col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values
            X_te[c] = pd.to_numeric(X_te[col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values

    # Layer B: TE mean for ngram columns
    ng_te_cols = [f"TE_ng_{col}" for col in ngram_columns]
    for c in ng_te_cols:
        X_tr[c] = np.float32(0.5)
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr]
        temp_df = X_tr2.copy()
        temp_df["__target__"] = y_tr[in_tr]
        for col in ngram_columns:
            ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
            te_col = f"TE_ng_{col}"
            vals = pd.to_numeric(X_tr.iloc[in_va][col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
            X_tr.iloc[in_va, X_tr.columns.get_loc(te_col)] = vals
    temp_df = X_tr.copy()
    temp_df["__target__"] = y_tr
    for col in ngram_columns:
        ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
        te_col = f"TE_ng_{col}"
        X_val[te_col] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
        X_te[te_col] = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values

    # Layer C: sklearn TargetEncoder
    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te_encoder = TargetEncoder(cv=inner_folds, shuffle=True, smooth="auto", target_type="binary", random_state=seed)
    X_tr[te_mean_cols] = te_encoder.fit_transform(X_tr[te_columns], y_tr)
    X_val[te_mean_cols] = te_encoder.transform(X_val[te_columns])
    X_te[te_mean_cols] = te_encoder.transform(X_te[te_columns])

    # Drop raw ngram/num-as-cat string columns
    drop_cols = [c for c in ngram_columns if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_cols)
    X_val = X_val.drop(columns=drop_cols)
    X_te = X_te.drop(columns=drop_cols)

    return X_tr, X_val, X_te


# ============================================================================
# TE Cache: pre-compute Target Encoding and save to disk
# ============================================================================
CACHE_DIR = Config.BASE_DIR / "te_cache"


def precompute_te_cache(X, y, X_test_base, te_columns, ngram_columns, n_splits):
    """
    Pre-compute Target Encoding for each CV fold and save to parquet.
    Only 1 fold's data is in memory at a time during computation.
    During Optuna trials, each fold is loaded from disk as needed.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pre-computing TE cache for {n_splits} folds (saving to {CACHE_DIR})...")

    fold_meta = []
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_val = y[val_idx]
        X_te = X_test_base.copy()

        X_tr, X_val, X_te = apply_target_encoding(
            X_tr, y_tr, X_val, X_te,
            te_columns, ngram_columns,
            config.INNER_FOLDS, config.RANDOM_STATE,
        )

        # Save to parquet
        X_tr.to_parquet(CACHE_DIR / f"fold{fold}_X_tr.parquet")
        X_val.to_parquet(CACHE_DIR / f"fold{fold}_X_val.parquet")
        X_te.to_parquet(CACHE_DIR / f"fold{fold}_X_te.parquet")
        np.save(CACHE_DIR / f"fold{fold}_y_tr.npy", y_tr)
        np.save(CACHE_DIR / f"fold{fold}_y_val.npy", y_val)

        fold_meta.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "n_features": X_tr.shape[1],
        })
        print(f"  Fold {fold + 1}: {time.time() - t0:.1f}s, features={X_tr.shape[1]}")

    # Save fold indices
    with open(CACHE_DIR / "fold_meta.pkl", "wb") as f:
        pickle.dump(fold_meta, f)

    print("TE cache saved to disk.")
    return fold_meta


def load_fold(fold, load_test=False):
    """Load one fold's data from disk."""
    data = {
        "X_tr": pd.read_parquet(CACHE_DIR / f"fold{fold}_X_tr.parquet"),
        "y_tr": np.load(CACHE_DIR / f"fold{fold}_y_tr.npy"),
        "X_val": pd.read_parquet(CACHE_DIR / f"fold{fold}_X_val.parquet"),
        "y_val": np.load(CACHE_DIR / f"fold{fold}_y_val.npy"),
    }
    if load_test:
        data["X_te"] = pd.read_parquet(CACHE_DIR / f"fold{fold}_X_te.parquet")
    return data


# ============================================================================
# Optuna Objective (loads 1 fold at a time from disk)
# ============================================================================
def create_objective(y, fold_meta):
    """Create Optuna objective function. Loads cached TE data per fold."""

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }

        print(f"\n  [Trial {trial.number}] params: {params}", flush=True)
        oof_preds = np.zeros(len(y))

        for fold, meta in enumerate(fold_meta):
            t0 = time.time()
            fd = load_fold(fold, load_test=False)

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                learning_rate=0.1,
                n_estimators=10000,
                early_stopping_rounds=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                **params,
            )

            model.fit(
                fd["X_tr"], fd["y_tr"],
                eval_set=[(fd["X_val"], fd["y_val"])],
                verbose=500,
            )

            val_proba = model.predict_proba(fd["X_val"])[:, 1]
            oof_preds[meta["val_idx"]] = val_proba

            fold_auc = roc_auc_score(fd["y_val"], val_proba)
            elapsed = time.time() - t0
            print(f"    Fold {fold+1}: AUC={fold_auc:.6f}, best_iter={model.best_iteration}, {elapsed:.0f}s", flush=True)

            del model, fd
            import gc; gc.collect()

            trial.report(fold_auc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        oof_auc = roc_auc_score(y, oof_preds)
        print(f"  [Trial {trial.number}] OOF AUC={oof_auc:.6f}", flush=True)
        return oof_auc

    return objective


# ============================================================================
# Final Training (loads 1 fold at a time from disk)
# ============================================================================
def train_final(y, fold_meta, best_params):
    print(f"\n{'='*80}")
    print(f"Final Training: {len(fold_meta)}-Fold CV with Best Params")
    print(f"{'='*80}")
    print(f"Best params: {json.dumps(best_params, indent=2)}")

    oof_preds = np.zeros(len(y))
    test_preds_sum = None
    models = []
    fold_indices = {"train": [], "val": []}
    fold_scores = []

    for fold, meta in enumerate(fold_meta):
        print(f"\n--- Fold {fold + 1}/{len(fold_meta)} ---")

        fd = load_fold(fold, load_test=True)

        if fold == 0:
            test_preds_sum = np.zeros(len(fd["X_te"]))

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=50000,
            early_stopping_rounds=500,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **best_params,
        )

        model.fit(
            fd["X_tr"], fd["y_tr"],
            eval_set=[(fd["X_tr"], fd["y_tr"]), (fd["X_val"], fd["y_val"])],
            verbose=500,
        )

        val_proba = model.predict_proba(fd["X_val"])[:, 1]
        oof_preds[meta["val_idx"]] = val_proba
        test_preds_sum += model.predict_proba(fd["X_te"])[:, 1]

        fold_auc = roc_auc_score(fd["y_val"], val_proba)
        fold_logloss = log_loss(fd["y_val"], val_proba)
        fold_acc = accuracy_score(fd["y_val"], (val_proba > 0.5).astype(int))
        fold_scores.append({
            "fold": fold + 1, "auc": fold_auc, "logloss": fold_logloss,
            "accuracy": fold_acc, "best_iteration": model.best_iteration,
        })
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_logloss:.6f}, Best iter: {model.best_iteration}")

        models.append(model)
        fold_indices["train"].append(meta["train_idx"])
        fold_indices["val"].append(meta["val_idx"])

    test_preds = test_preds_sum / len(fold_meta)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_logloss = log_loss(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))

    print(f"\n{'='*80}")
    print("Final CV Results:")
    for s in fold_scores:
        print(f"  Fold {s['fold']}: AUC={s['auc']:.6f}, LogLoss={s['logloss']:.6f}, Iter={s['best_iteration']}")
    print(f"\nOOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_logloss:.6f}")
    print(f"OOF Accuracy: {oof_acc:.6f}")
    print(f"{'='*80}")

    return models, oof_preds, test_preds, fold_indices, fold_scores


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


def plot_feature_importance(models, feature_names, save_path, top_n=30):
    importance = np.zeros(len(feature_names))
    for model in models:
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        for i, fname in enumerate(feature_names):
            importance[i] += scores.get(fname, 0)
    importance /= len(models)
    top_n = min(top_n, len(feature_names))
    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    plt.barh(range(len(idx)), importance[idx], color="#2196F3")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Importance (Gain)")
    plt.title(f"Top {top_n} Feature Importance (XGBoost, Optuna-tuned)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


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


def plot_optuna_history(study, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    best_values = [max(values[:i+1]) for i in range(len(values))]
    axes[0].plot(range(len(values)), values, "o", alpha=0.5, label="Trial AUC")
    axes[0].plot(range(len(values)), best_values, "-r", lw=2, label="Best AUC")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("OOF AUC-ROC")
    axes[0].set_title("Optimization History")
    axes[0].legend()

    importances = optuna.importance.get_param_importances(study)
    names = list(importances.keys())
    vals = list(importances.values())
    axes[1].barh(range(len(names)), vals, color="#FF9800")
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Hyperparameter Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices):
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({
        config.ID_COL: train_ids, "prob": oof_preds,
        "predicted": (oof_preds > 0.5).astype(int), "true": y,
    })
    oof_df.to_csv(config.PRED_DIR / "oof.csv", index=False)

    test_proba_df = pd.DataFrame({config.ID_COL: test_ids, "prob": test_preds})
    test_proba_df.to_csv(config.PRED_DIR / "test_proba.csv", index=False)

    submit_df = pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: test_preds})
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False)

    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"Predictions saved to {config.PRED_DIR}")


def save_models(models):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fold, model in enumerate(models):
        model.save_model(config.MODEL_DIR / f"fold_{fold}.json")
    print(f"Models saved to {config.MODEL_DIR}")


# ============================================================================
# Main
# ============================================================================
def main():
    start_time = time.time()

    # Best params from experiment 20260323_02_xgb_optuna_tuning
    best_params = {
        "learning_rate": 0.004787126983706307,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.7797316213800588,
        "colsample_bytree": 0.27036930195333186,
        "reg_alpha": 0.0023599517797442456,
        "reg_lambda": 9.90372658125681,
        "gamma": 1.1966979929346213,
    }

    print("=" * 80)
    print("XGBoost Optuna Best Params with 20-fold CV")
    print(f"  CV: {config.FINAL_N_SPLITS}-fold")
    print(f"  Params from: 20260323_02_xgb_optuna_tuning (5-fold OOF AUC 0.91893)")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()
    orig_df = load_orig_data()

    # Prepare target
    le = LabelEncoder()
    y = le.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values
    print(f"\nChurn rate: {y.mean():.4f}")

    # Feature engineering (pre-CV)
    print(f"\n{'='*80}")
    print("Feature Engineering (pre-CV)")
    print(f"{'='*80}")
    train_df, test_df = add_pre_cv_features(train_df, test_df, orig_df)

    print("  Group 9: N-gram columns...")
    bigram_cols, trigram_cols, num_as_cat_cols = create_ngram_columns(train_df, test_df)
    ngram_columns = num_as_cat_cols + bigram_cols + trigram_cols

    X, X_test, label_encoders = prepare_base_features(train_df, test_df)
    te_columns = config.CAT_COLS + num_as_cat_cols

    # Pre-compute TE cache
    print(f"\n{'='*80}")
    print(f"Pre-computing Target Encoding Cache ({config.FINAL_N_SPLITS}-fold)")
    print(f"{'='*80}")
    fold_meta = precompute_te_cache(
        X, y, X_test, te_columns, ngram_columns, config.FINAL_N_SPLITS,
    )
    feature_names = load_fold(0, load_test=False)["X_tr"].columns.tolist()

    del train_df, test_df, orig_df, X, X_test
    import gc; gc.collect()

    # Final Training
    models, oof_preds, test_preds, fold_indices, fold_scores = train_final(
        y, fold_meta, best_params,
    )

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)
    save_models(models)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Done! Total time: {total_time/60:.1f} min")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
