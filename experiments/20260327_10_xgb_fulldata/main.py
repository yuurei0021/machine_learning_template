"""
XGBoost Full Data Training
Experiment: 20260327_10_xgb_fulldata

全データで学習し、テスト予測のみ生成（OOFなし）。
5-fold平均best_iteration(11570) x 1.25 = 14463をn_estimatorsとして使用。
FEパイプラインは20260323_02_xgb_optuna_tuningと同一。
LB提出で全データ学習の効果を検証する。
"""

import warnings
from pathlib import Path
from itertools import combinations
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.model_selection import StratifiedKFold
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

    RANDOM_STATE = 42
    INNER_FOLDS = 5

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

    # Optuna best params from 20260323_02
    XGB_PARAMS = {
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.7797316213800588,
        "colsample_bytree": 0.27036930195333186,
        "reg_alpha": 0.0023599517797442456,
        "reg_lambda": 9.90372658125681,
        "gamma": 1.1966979929346213,
    }

    # 5-fold mean best_iteration=11570, x1.25 for full data
    FULL_DATA_N_ESTIMATORS = 14463
    LEARNING_RATE = 0.004787126983706307


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
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test

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
    if not pd.api.types.is_numeric_dtype(orig["SeniorCitizen"]):
        orig["SeniorCitizen"] = (orig["SeniorCitizen"] == "Yes").astype(int)
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
    orig["TotalCharges"].fillna(orig["TotalCharges"].median(), inplace=True)
    orig["Churn_binary"] = (orig["Churn"] == "Yes").astype(int)
    keep_cols = config.CAT_COLS + config.NUM_COLS + ["SeniorCitizen", "Churn", "Churn_binary"]
    orig = orig[[c for c in keep_cols if c in orig.columns]]
    print(f"Orig: {orig.shape}")
    return orig


# ============================================================================
# Feature Engineering (same as 20260323_01/02)
# ============================================================================
def add_pre_cv_features(train_df, test_df, orig_df):
    dfs = [train_df, test_df]

    for col in config.NUM_COLS:
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
        df["service_count"] = (df[config.SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
        df["has_internet"] = (df["InternetService"] != "No").astype("float32")
        df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")

    for col in config.ORIG_PROBA_CATS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    for col in config.NUM_COLS:
        mapping = orig_df.groupby(col)["Churn_binary"].mean()
        for df in dfs:
            df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")

    cross_pairs = list(combinations(config.ORIG_PROBA_CROSS_CATS, 2))
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


def prepare_base_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in config.CAT_COLS:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    exclude_cols = {config.TARGET_COL, config.ID_COL, "Churn_binary"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    return train_df[feature_cols].copy(), test_df[feature_cols].copy()


def create_ngram_columns(train_df, test_df):
    bigram_cols = []
    for c1, c2 in combinations(config.TOP_CATS_NGRAM, 2):
        name = f"BG_{c1}_{c2}"
        for df in [train_df, test_df]:
            df[name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        bigram_cols.append(name)
    trigram_cols = []
    top4 = config.TOP_CATS_NGRAM[:4]
    for c1, c2, c3 in combinations(top4, 3):
        name = f"TG_{c1}_{c2}_{c3}"
        for df in [train_df, test_df]:
            df[name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        trigram_cols.append(name)
    num_as_cat_cols = []
    for col in config.NUM_COLS:
        name = f"CAT_{col}"
        for df in [train_df, test_df]:
            df[name] = df[col].astype(str)
        num_as_cat_cols.append(name)
    return bigram_cols, trigram_cols, num_as_cat_cols


def apply_fulldata_te(X_train, y_train, X_test, te_columns, ngram_columns, inner_folds, seed):
    """TE for full data training: inner K-fold on train, direct mapping on test."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    # Layer A: TE stats
    stats = ["std", "min", "max"]
    te1_cols = [f"TE1_{col}_{s}" for col in te_columns for s in stats]
    for c in te1_cols:
        X_train[c] = np.float32(0)
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_train, y_train)):
        X_tr2 = X_train.iloc[in_tr].copy()
        X_tr2["__target__"] = y_train[in_tr]
        for col in te_columns:
            agg = X_tr2.groupby(col, observed=False)["__target__"].agg(stats)
            agg.columns = [f"TE1_{col}_{s}" for s in stats]
            for c in agg.columns:
                vals = pd.to_numeric(X_train.iloc[in_va][col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values
                X_train.iloc[in_va, X_train.columns.get_loc(c)] = vals
    temp_df = X_train.copy()
    temp_df["__target__"] = y_train
    for col in te_columns:
        agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
        agg.columns = [f"TE1_{col}_{s}" for s in stats]
        for c in agg.columns:
            X_test[c] = pd.to_numeric(X_test[col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values

    # Layer B: N-gram TE
    ng_te_cols = [f"TE_ng_{col}" for col in ngram_columns]
    for c in ng_te_cols:
        X_train[c] = np.float32(0.5)
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_train, y_train)):
        X_tr2 = X_train.iloc[in_tr].copy()
        X_tr2["__target__"] = y_train[in_tr]
        for col in ngram_columns:
            ng_te = X_tr2.groupby(col, observed=False)["__target__"].mean()
            te_col = f"TE_ng_{col}"
            vals = pd.to_numeric(X_train.iloc[in_va][col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
            X_train.iloc[in_va, X_train.columns.get_loc(te_col)] = vals
    temp_df = X_train.copy()
    temp_df["__target__"] = y_train
    for col in ngram_columns:
        ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
        te_col = f"TE_ng_{col}"
        X_test[te_col] = pd.to_numeric(X_test[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")

    # Layer C: sklearn TargetEncoder
    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te_encoder = TargetEncoder(cv=inner_folds, shuffle=True, smooth="auto", target_type="binary", random_state=seed)
    X_train[te_mean_cols] = te_encoder.fit_transform(X_train[te_columns], y_train)
    X_test[te_mean_cols] = te_encoder.transform(X_test[te_columns])

    # Drop raw ngram columns
    drop_cols = [c for c in ngram_columns if c in X_train.columns]
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    return X_train, X_test


# ============================================================================
# Main
# ============================================================================
def main():
    config.PRED_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("XGBoost Full Data Training")
    print(f"  n_estimators: {config.FULL_DATA_N_ESTIMATORS}")
    print(f"  learning_rate: {config.LEARNING_RATE:.6f}")
    print("=" * 60)

    train_df, test_df = load_data()
    orig_df = load_orig_data()

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[config.TARGET_COL])
    test_ids = test_df[config.ID_COL].values

    print("\nFeature Engineering...")
    train_df, test_df = add_pre_cv_features(train_df, test_df, orig_df)

    bigram_cols, trigram_cols, num_as_cat_cols = create_ngram_columns(train_df, test_df)
    ngram_columns = num_as_cat_cols + bigram_cols + trigram_cols

    X, X_test = prepare_base_features(train_df, test_df)
    te_columns = config.CAT_COLS + num_as_cat_cols
    print(f"  Base features: {X.shape[1]}")

    print("\nTarget Encoding (full data)...")
    X, X_test = apply_fulldata_te(X, y, X_test, te_columns, ngram_columns, config.INNER_FOLDS, config.RANDOM_STATE)
    print(f"  Final features: {X.shape[1]}")

    print("\nTraining XGBoost on full data...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=config.FULL_DATA_N_ESTIMATORS,
        learning_rate=config.LEARNING_RATE,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **config.XGB_PARAMS,
    )
    model.fit(X, y, verbose=1000)

    print("\nPredicting...")
    test_preds = model.predict_proba(X_test)[:, 1]

    pd.DataFrame({"id": test_ids, "Churn": test_preds}).to_csv(config.PRED_DIR / "test.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": test_preds}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)

    print(f"\nTest pred range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"Test pred mean: {test_preds.mean():.4f}")
    print(f"Time: {(time.time()-t0)/60:.1f}min")
    print(f"Saved to {config.PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
