"""
Original Data Reference Distribution + XGBoost
Experiment: 20260323_01_orig_data_reference

目的: 元データ（IBM Telco 7,043行）を参照分布として特徴量生成し、スコア向上を図る
アプローチ: 9つの特徴量グループ + Nested Target Encoding + XGBoost (5-Fold Stratified CV)
参考: https://www.kaggle.com/code/ozermehmet/original-data-fe-single-xgb-cv-0-919-lb-0-916
"""

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
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
import xgboost as xgb

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
    INNER_FOLDS = 5
    RANDOM_STATE = 42

    # Categorical / Numeric column definitions (competition format)
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

    # ORIG_proba: categories used for single and cross features
    ORIG_PROBA_CATS = [
        "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
        "TechSupport", "OnlineBackup", "DeviceProtection", "PaperlessBilling",
        "StreamingMovies", "StreamingTV", "Partner", "Dependents",
    ]
    ORIG_PROBA_CROSS_CATS = [
        "Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport",
    ]

    # N-gram top categories
    TOP_CATS_NGRAM = [
        "Contract", "InternetService", "PaymentMethod",
        "OnlineSecurity", "TechSupport", "PaperlessBilling",
    ]

    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }

    NUM_BOOST_ROUND = 5000
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 200


config = Config()

# ============================================================================
# Helpers
# ============================================================================
def pctrank_against(values, reference):
    """Percentile rank of values within reference distribution."""
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")


def zscore_against(values, reference):
    """Z-score of values relative to reference distribution."""
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
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    return train, test


def load_orig_data():
    """Load and preprocess original IBM Telco dataset."""
    print("Loading original data...")
    orig = pd.read_excel(config.ORIG_DATA)

    # Rename columns to match competition format
    col_map = {
        "Gender": "gender",
        "Senior Citizen": "SeniorCitizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "Tenure Months": "tenure",
        "Phone Service": "PhoneService",
        "Multiple Lines": "MultipleLines",
        "Internet Service": "InternetService",
        "Online Security": "OnlineSecurity",
        "Online Backup": "OnlineBackup",
        "Device Protection": "DeviceProtection",
        "Tech Support": "TechSupport",
        "Streaming TV": "StreamingTV",
        "Streaming Movies": "StreamingMovies",
        "Contract": "Contract",
        "Paperless Billing": "PaperlessBilling",
        "Payment Method": "PaymentMethod",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges",
        "Churn Label": "Churn",
    }
    orig = orig.rename(columns=col_map)

    # Convert SeniorCitizen: "Yes"/"No" -> 1/0
    if orig["SeniorCitizen"].dtype == object:
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)

    # Convert TotalCharges to numeric
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())

    # Binary target
    orig["Churn_binary"] = (orig["Churn"] == "Yes").astype(int)

    # Keep only relevant columns
    keep_cols = [config.ID_COL] if config.ID_COL in orig.columns else []
    keep_cols += config.CAT_COLS + config.NUM_COLS + ["SeniorCitizen", "Churn", "Churn_binary"]
    orig = orig[[c for c in keep_cols if c in orig.columns]]

    print(f"  Original: {orig.shape}")
    print(f"  Churn rate: {orig['Churn_binary'].mean():.4f}")
    return orig


# ============================================================================
# Feature Engineering (pre-CV: features that don't use target)
# ============================================================================
def add_pre_cv_features(train_df, test_df, orig_df):
    """
    Add features that do NOT depend on the target variable.
    These can be computed once before CV.
    9 feature groups from the reference notebook.
    """
    target_orig = orig_df["Churn_binary"].values
    dfs = [train_df, test_df]

    # ------------------------------------------------------------------
    # Group 1: Frequency Encoding (3 features)
    # ------------------------------------------------------------------
    print("  Group 1: Frequency Encoding...")
    for col in config.NUM_COLS:
        freq = pd.concat([train_df[col], orig_df[col], test_df[col]]).value_counts(normalize=True)
        for df in dfs + [orig_df]:
            df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")

    # ------------------------------------------------------------------
    # Group 2: Arithmetic Interactions (6 features)
    # ------------------------------------------------------------------
    print("  Group 2: Arithmetic Interactions...")
    for df in dfs + [orig_df]:
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
        df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
        df["is_first_month"] = (df["tenure"] == 1).astype("float32")
        df["dev_is_zero"] = (df["charges_deviation"] == 0).astype("float32")
        df["dev_sign"] = np.sign(df["charges_deviation"]).astype("float32")

    # ------------------------------------------------------------------
    # Group 3: Service Counts (3 features)
    # ------------------------------------------------------------------
    print("  Group 3: Service Counts...")
    for df in dfs + [orig_df]:
        df["service_count"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

    # ------------------------------------------------------------------
    # Group 4: ORIG_proba single (15 features)
    # ------------------------------------------------------------------
    print("  Group 4: ORIG_proba single...")
    # Categorical columns
    for col in config.ORIG_PROBA_CATS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    # Numeric columns
    for col in config.NUM_COLS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")

    # ------------------------------------------------------------------
    # Group 5: ORIG_proba cross (10 features)
    # ------------------------------------------------------------------
    print("  Group 5: ORIG_proba cross...")
    cross_pairs = list(combinations(config.ORIG_PROBA_CROSS_CATS, 2))
    for c1, c2 in cross_pairs:
        mapping = orig_df.groupby([c1, c2])["Churn_binary"].mean()
        name = f"ORIG_proba_{c1}_{c2}"
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[name] = idx.map(mapping).fillna(0.5).values.astype("float32")

    # ------------------------------------------------------------------
    # Group 6: Distribution Features (16 features)
    # ------------------------------------------------------------------
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
        tc = df["TotalCharges"].values
        mc = df["MonthlyCharges"].values
        t = df["tenure"].values

        # TotalCharges percentile ranks and z-scores
        df["pctrank_ch_TC"] = pctrank_against(tc, orig_ch_tc)
        df["pctrank_nc_TC"] = pctrank_against(tc, orig_nc_tc)
        df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
        df["pctrank_gap_TC"] = (pctrank_against(tc, orig_ch_tc) - pctrank_against(tc, orig_nc_tc)).astype("float32")
        df["zscore_ch_TC"] = zscore_against(tc, orig_ch_tc)
        df["zscore_nc_TC"] = zscore_against(tc, orig_nc_tc)
        df["zscore_gap_TC"] = (np.abs(zscore_against(tc, orig_ch_tc)) - np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")

        # MonthlyCharges percentile ranks
        df["pctrank_ch_MC"] = pctrank_against(mc, orig_ch_mc)
        df["pctrank_nc_MC"] = pctrank_against(mc, orig_nc_mc)
        df["pctrank_gap_MC"] = (pctrank_against(mc, orig_ch_mc) - pctrank_against(mc, orig_nc_mc)).astype("float32")

        # tenure percentile ranks
        df["pctrank_ch_T"] = pctrank_against(t, orig_ch_t)
        df["pctrank_nc_T"] = pctrank_against(t, orig_nc_t)
        df["pctrank_gap_T"] = (pctrank_against(t, orig_ch_t) - pctrank_against(t, orig_nc_t)).astype("float32")

        # Conditional percentile ranks
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

    # ------------------------------------------------------------------
    # Group 7: Quantile Distance Features (18 features)
    # ------------------------------------------------------------------
    print("  Group 7: Quantile Distance Features...")
    for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
        ch_q_tc = np.quantile(orig_ch_tc, q_val)
        nc_q_tc = np.quantile(orig_nc_tc, q_val)
        ch_q_t = np.quantile(orig_ch_t, q_val)
        nc_q_t = np.quantile(orig_nc_t, q_val)
        for df in dfs:
            df[f"dist_ch_TC_{q_label}"] = np.abs(df["TotalCharges"] - ch_q_tc).astype("float32")
            df[f"dist_nc_TC_{q_label}"] = np.abs(df["TotalCharges"] - nc_q_tc).astype("float32")
            df[f"qdist_gap_TC_{q_label}"] = (df[f"dist_nc_TC_{q_label}"] - df[f"dist_ch_TC_{q_label}"]).astype("float32")
            df[f"dist_ch_T_{q_label}"] = np.abs(df["tenure"] - ch_q_t).astype("float32")
            df[f"dist_nc_T_{q_label}"] = np.abs(df["tenure"] - nc_q_t).astype("float32")
            df[f"qdist_gap_T_{q_label}"] = (df[f"dist_nc_T_{q_label}"] - df[f"dist_ch_T_{q_label}"]).astype("float32")

    # ------------------------------------------------------------------
    # Group 8: Digit/Modular Features (17 features)
    # ------------------------------------------------------------------
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
    """
    Prepare base feature matrix: Label Encode categoricals.
    Returns X_train, X_test with base + engineered features (no target encoding yet).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Label encode categoricals
    label_encoders = {}
    for col in config.CAT_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le

    # Collect all feature columns (exclude target and id)
    exclude_cols = {config.TARGET_COL, config.ID_COL, "Churn_binary"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    print(f"  Base features: {X_train.shape[1]}")
    return X_train, X_test, label_encoders


# ============================================================================
# Group 9: N-gram features (created before CV, encoded within CV)
# ============================================================================
def create_ngram_columns(train_df, test_df):
    """Create bigram and trigram combination columns (string categories)."""
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

    # Num-as-cat columns
    num_as_cat_cols = []
    for col in config.NUM_COLS:
        name = f"CAT_{col}"
        for df in [train_df, test_df]:
            df[name] = df[col].astype(str)
        num_as_cat_cols.append(name)

    return bigram_cols, trigram_cols, num_as_cat_cols


# ============================================================================
# Target Encoding (within CV fold)
# ============================================================================
def apply_target_encoding(X_tr, y_tr, X_val, X_te,
                          te_columns, ngram_columns, inner_folds, seed):
    """
    Apply nested target encoding within a CV fold.
    Layer A: Inner KFold TE with std/min/max for te_columns
    Layer B: Inner KFold TE mean for ngram columns
    Layer C: sklearn TargetEncoder with CV smoothing for te_columns
    """
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    # ------------------------------------------------------------------
    # Layer A: TE with std/min/max statistics for te_columns
    # ------------------------------------------------------------------
    stats = ["std", "min", "max"]
    te1_cols = []
    for col in te_columns:
        for s in stats:
            te1_cols.append(f"TE1_{col}_{s}")

    # Initialize TE1 columns
    for c in te1_cols:
        X_tr[c] = np.float32(0)

    # Inner fold encoding for train
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr]
        y_tr2 = y_tr[in_tr]
        temp_df = X_tr2.copy()
        temp_df["__target__"] = y_tr2
        for col in te_columns:
            agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
            agg.columns = [f"TE1_{col}_{s}" for s in stats]
            for c in agg.columns:
                vals = pd.to_numeric(
                    X_tr.iloc[in_va][col].map(agg[c]), errors="coerce"
                ).fillna(0).astype("float32").values
                X_tr.iloc[in_va, X_tr.columns.get_loc(c)] = vals

    # Full fold encoding for val and test
    temp_df = X_tr.copy()
    temp_df["__target__"] = y_tr
    for col in te_columns:
        agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
        agg.columns = [f"TE1_{col}_{s}" for s in stats]
        for c in agg.columns:
            X_val[c] = pd.to_numeric(
                X_val[col].map(agg[c]), errors="coerce"
            ).fillna(0).astype("float32").values
            X_te[c] = pd.to_numeric(
                X_te[col].map(agg[c]), errors="coerce"
            ).fillna(0).astype("float32").values

    # ------------------------------------------------------------------
    # Layer B: TE mean for ngram columns
    # ------------------------------------------------------------------
    ng_te_cols = [f"TE_ng_{col}" for col in ngram_columns]
    for c in ng_te_cols:
        X_tr[c] = np.float32(0.5)

    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr]
        y_tr2 = y_tr[in_tr]
        temp_df = X_tr2.copy()
        temp_df["__target__"] = y_tr2
        for col in ngram_columns:
            ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
            te_col = f"TE_ng_{col}"
            vals = pd.to_numeric(
                X_tr.iloc[in_va][col].astype(str).map(ng_te), errors="coerce"
            ).fillna(0.5).astype("float32").values
            X_tr.iloc[in_va, X_tr.columns.get_loc(te_col)] = vals

    temp_df = X_tr.copy()
    temp_df["__target__"] = y_tr
    for col in ngram_columns:
        ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
        te_col = f"TE_ng_{col}"
        X_val[te_col] = pd.to_numeric(
            X_val[col].astype(str).map(ng_te), errors="coerce"
        ).fillna(0.5).astype("float32").values
        X_te[te_col] = pd.to_numeric(
            X_te[col].astype(str).map(ng_te), errors="coerce"
        ).fillna(0.5).astype("float32").values

    # ------------------------------------------------------------------
    # Layer C: sklearn TargetEncoder (mean with smoothing)
    # ------------------------------------------------------------------
    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te_encoder = TargetEncoder(
        cv=inner_folds, shuffle=True, smooth="auto",
        target_type="binary", random_state=seed,
    )
    X_tr[te_mean_cols] = te_encoder.fit_transform(X_tr[te_columns], y_tr)
    X_val[te_mean_cols] = te_encoder.transform(X_val[te_columns])
    X_te[te_mean_cols] = te_encoder.transform(X_te[te_columns])

    # ------------------------------------------------------------------
    # Drop raw ngram/num-as-cat string columns (not usable by XGBoost)
    # ------------------------------------------------------------------
    drop_cols = [c for c in ngram_columns if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_cols)
    X_val = X_val.drop(columns=drop_cols)
    X_te = X_te.drop(columns=drop_cols)

    return X_tr, X_val, X_te


# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y, X_test_base, te_columns, ngram_columns):
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation (XGBoost)")
    print(f"{'='*80}")

    oof_preds = np.zeros(len(X))
    test_preds_sum = None
    models = []
    fold_indices = {"train": [], "val": []}
    fold_scores = []

    skf = StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")

        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_val = y[val_idx]
        X_te = X_test_base.copy()

        print(f"  Train: {len(X_tr)}, Val: {len(X_val)}")

        # Apply target encoding within this fold
        print("  Applying target encoding...")
        X_tr, X_val, X_te = apply_target_encoding(
            X_tr, y_tr, X_val, X_te,
            te_columns, ngram_columns,
            config.INNER_FOLDS, config.RANDOM_STATE,
        )

        if fold == 0:
            print(f"  Features after TE: {X_tr.shape[1]}")
            # Initialize test predictions accumulator
            test_preds_sum = np.zeros(len(X_te))

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(
            config.XGB_PARAMS,
            dtrain,
            num_boost_round=config.NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            verbose_eval=config.VERBOSE_EVAL,
        )

        val_proba = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
        oof_preds[val_idx] = val_proba

        # Test prediction for this fold
        dtest = xgb.DMatrix(X_te)
        test_preds_sum += model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

        # Metrics
        fold_auc = roc_auc_score(y_val, val_proba)
        fold_logloss = log_loss(y_val, val_proba)
        fold_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))

        fold_scores.append({
            "fold": fold + 1,
            "auc": fold_auc,
            "logloss": fold_logloss,
            "accuracy": fold_acc,
            "best_iteration": model.best_iteration,
        })
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_logloss:.6f}, Accuracy: {fold_acc:.6f}, Best iter: {model.best_iteration}")

        models.append(model)
        fold_indices["train"].append(train_idx)
        fold_indices["val"].append(val_idx)

    # Test predictions: average over folds
    test_preds = test_preds_sum / config.N_SPLITS

    # Overall OOF
    oof_auc = roc_auc_score(y, oof_preds)
    oof_logloss = log_loss(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))

    print(f"\n{'='*80}")
    print("CV Results:")
    for s in fold_scores:
        print(f"  Fold {s['fold']}: AUC={s['auc']:.6f}, LogLoss={s['logloss']:.6f}, Acc={s['accuracy']:.6f}, Iter={s['best_iteration']}")
    print(f"\nOOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_logloss:.6f}")
    print(f"OOF Accuracy: {oof_acc:.6f}")
    print(f"{'='*80}")

    return models, oof_preds, test_preds, fold_indices, fold_scores


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


def plot_feature_importance(models, feature_names, save_path, top_n=30):
    importance = np.zeros(len(feature_names))
    for model in models:
        scores = model.get_score(importance_type="gain")
        for i, fname in enumerate(feature_names):
            importance[i] += scores.get(fname, 0)
    importance /= len(models)

    top_n = min(top_n, len(feature_names))
    idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    plt.barh(range(len(idx)), importance[idx], color="#2196F3")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Importance (Gain)")
    plt.title(f"Top {top_n} Feature Importance (XGBoost)")
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


# ============================================================================
# Save
# ============================================================================
def save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices):
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)

    # OOF
    oof_df = pd.DataFrame({
        config.ID_COL: train_ids,
        "prob": oof_preds,
        "predicted": (oof_preds > 0.5).astype(int),
        "true": y,
    })
    oof_df.to_csv(config.PRED_DIR / "oof.csv", index=False)

    # Test probabilities
    test_proba_df = pd.DataFrame({
        config.ID_COL: test_ids,
        "prob": test_preds,
    })
    test_proba_df.to_csv(config.PRED_DIR / "test_proba.csv", index=False)

    # Submission
    submit_df = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: test_preds,
    })
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False)

    # Fold indices
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Predictions saved to {config.PRED_DIR}")


def save_models(models):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fold, model in enumerate(models):
        model.save_model(str(config.MODEL_DIR / f"fold_{fold}.json"))
    print(f"Models saved to {config.MODEL_DIR}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("Original Data Reference Distribution + XGBoost")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()
    orig_df = load_orig_data()

    # Prepare target
    le = LabelEncoder()
    y = le.fit_transform(train_df[config.TARGET_COL])  # No=0, Yes=1
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print(f"\nTarget classes: {list(le.classes_)} -> [0, 1]")
    print(f"Churn rate: {y.mean():.4f}")

    # Feature engineering (pre-CV: Groups 1-8)
    print(f"\n{'='*80}")
    print("Feature Engineering (pre-CV)")
    print(f"{'='*80}")
    train_df, test_df = add_pre_cv_features(train_df, test_df, orig_df)

    # Group 9: Create n-gram columns (will be target-encoded in CV)
    print("  Group 9: N-gram columns...")
    bigram_cols, trigram_cols, num_as_cat_cols = create_ngram_columns(train_df, test_df)
    ngram_columns = num_as_cat_cols + bigram_cols + trigram_cols
    print(f"    Bigrams: {len(bigram_cols)}, Trigrams: {len(trigram_cols)}, Num-as-cat: {len(num_as_cat_cols)}")

    # Prepare base features (label encode categoricals)
    X, X_test, label_encoders = prepare_base_features(train_df, test_df)

    # Define TE columns (use label-encoded cat columns + num-as-cat)
    te_columns = config.CAT_COLS + num_as_cat_cols

    # Train with CV (target encoding applied within each fold)
    models, oof_preds, test_preds, fold_indices, fold_scores = train_with_cv(
        X, y, X_test, te_columns, ngram_columns,
    )

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)
    save_models(models)

    # Visualization
    oof_pred_labels = (oof_preds > 0.5).astype(int)
    plot_confusion_matrix(y, oof_pred_labels, config.BASE_DIR / "confusion_matrix.png")

    # Get feature names from first fold (after TE)
    # Re-run TE on full data just to get column names
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    first_train_idx, first_val_idx = next(iter(skf.split(X, y)))
    X_tr_tmp = X.iloc[first_train_idx].copy()
    X_val_tmp = X.iloc[first_val_idx].copy()
    X_te_tmp = X_test.copy()
    X_tr_tmp, X_val_tmp, X_te_tmp = apply_target_encoding(
        X_tr_tmp, y[first_train_idx], X_val_tmp, X_te_tmp,
        te_columns, ngram_columns, config.INNER_FOLDS, config.RANDOM_STATE,
    )
    feature_names = X_tr_tmp.columns.tolist()
    plot_feature_importance(models, feature_names, config.BASE_DIR / "feature_importance.png")
    plot_roc_curve(y, oof_preds, config.BASE_DIR / "roc_curve.png")

    # Classification report
    report = classification_report(y, oof_pred_labels, target_names=list(le.classes_))
    report_path = config.BASE_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
