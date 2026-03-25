"""
MLP (Embedding + LayerNorm) + Original Data Reference Features
Experiment: 20260325_09_mlp_orig_data_reference

目的: PyTorch MLPでアンサンブル多様性を確保
アプローチ: 9つの特徴量グループ + Nested TE + Embedding MLP (5-Fold Stratified CV)
"""

import warnings
import argparse
from pathlib import Path
from itertools import combinations
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, log_loss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ============================================================================
# Config
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
    MODEL_DIR = BASE_DIR / "model"
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

    # MLP params
    HIDDEN_SIZES = [512, 256, 128]
    DROPOUT = 0.3
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 1024
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# ============================================================================
# Helpers (same as other experiments)
# ============================================================================
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
# Feature Engineering (same as other orig_data_reference experiments)
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
        for df in dfs:
            idx = df.set_index([c1, c2]).index
            df[f"ORIG_proba_{c1}_{c2}"] = idx.map(mapping).fillna(0.5).values.astype("float32")
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
    train_df, test_df = train_df.copy(), test_df.copy()
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
    for c1, c2, c3 in combinations(config.TOP_CATS_NGRAM[:4], 3):
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

def apply_target_encoding(X_tr, y_tr, X_val, X_te, te_columns, ngram_columns, inner_folds, seed):
    X_tr, X_val, X_te = X_tr.copy(), X_val.copy(), X_te.copy()
    skf_inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    stats = ["std", "min", "max"]
    te1_cols = [f"TE1_{col}_{s}" for col in te_columns for s in stats]
    for c in te1_cols:
        X_tr[c] = np.float32(0)
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        temp_df = X_tr.iloc[in_tr].copy(); temp_df["__target__"] = y_tr[in_tr]
        for col in te_columns:
            agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
            agg.columns = [f"TE1_{col}_{s}" for s in stats]
            for c in agg.columns:
                X_tr.iloc[in_va, X_tr.columns.get_loc(c)] = pd.to_numeric(X_tr.iloc[in_va][col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values
    temp_df = X_tr.copy(); temp_df["__target__"] = y_tr
    for col in te_columns:
        agg = temp_df.groupby(col, observed=False)["__target__"].agg(stats)
        agg.columns = [f"TE1_{col}_{s}" for s in stats]
        for c in agg.columns:
            X_val[c] = pd.to_numeric(X_val[col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values
            X_te[c] = pd.to_numeric(X_te[col].map(agg[c]), errors="coerce").fillna(0).astype("float32").values
    ng_te_cols = [f"TE_ng_{col}" for col in ngram_columns]
    for c in ng_te_cols:
        X_tr[c] = np.float32(0.5)
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        temp_df = X_tr.iloc[in_tr].copy(); temp_df["__target__"] = y_tr[in_tr]
        for col in ngram_columns:
            ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
            te_col = f"TE_ng_{col}"
            X_tr.iloc[in_va, X_tr.columns.get_loc(te_col)] = pd.to_numeric(X_tr.iloc[in_va][col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
    temp_df = X_tr.copy(); temp_df["__target__"] = y_tr
    for col in ngram_columns:
        ng_te = temp_df.groupby(col, observed=False)["__target__"].mean()
        te_col = f"TE_ng_{col}"
        X_val[te_col] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
        X_te[te_col] = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
    te_mean_cols = [f"TE_{col}" for col in te_columns]
    te_encoder = TargetEncoder(cv=inner_folds, shuffle=True, smooth="auto", target_type="binary", random_state=seed)
    X_tr[te_mean_cols] = te_encoder.fit_transform(X_tr[te_columns], y_tr)
    X_val[te_mean_cols] = te_encoder.transform(X_val[te_columns])
    X_te[te_mean_cols] = te_encoder.transform(X_te[te_columns])
    drop_cols = [c for c in ngram_columns if c in X_tr.columns]
    return X_tr.drop(columns=drop_cols), X_val.drop(columns=drop_cols), X_te.drop(columns=drop_cols)

# ============================================================================
# MLP Model
# ============================================================================
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X_tr, y_tr, X_va, y_va, X_te, epochs, device):
    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype("float32")
    X_va_s = scaler.transform(X_va).astype("float32")
    X_te_s = scaler.transform(X_te).astype("float32")

    # Tensors
    tr_ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr, dtype=torch.float32))
    tr_dl = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    X_va_t = torch.tensor(X_va_s).to(device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te_s).to(device)

    model = TabularMLP(X_tr_s.shape[1], config.HIDDEN_SIZES, config.DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        scheduler.step()
        train_loss /= len(tr_ds)

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t)
            va_proba = torch.sigmoid(va_logits).cpu().numpy()
        va_auc = roc_auc_score(y_va, va_proba)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs}: loss={train_loss:.5f}, val_AUC={va_auc:.6f}", flush=True)

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_proba = torch.sigmoid(model(X_va_t)).cpu().numpy()
        te_proba = torch.sigmoid(model(X_te_t)).cpu().numpy()

    return va_proba, te_proba, best_auc


# ============================================================================
# Main
# ============================================================================
def main(args):
    torch.manual_seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    epochs = args.epochs if args.epochs else config.EPOCHS
    device = config.DEVICE
    print("=" * 80)
    print(f"MLP + Original Data Reference Features (device={device}, epochs={epochs})")
    print("=" * 80)

    train_df = pd.read_csv(config.TRAIN_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
    orig_df = load_orig_data()
    print(f"Train: {train_df.shape}, Test: {test_df.shape}, Orig: {orig_df.shape}")

    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print("\nFeature Engineering...")
    train_df, test_df = add_pre_cv_features(train_df, test_df, orig_df)
    bigram_cols, trigram_cols, num_as_cat_cols = create_ngram_columns(train_df, test_df)
    ngram_columns = num_as_cat_cols + bigram_cols + trigram_cols
    X, X_test = prepare_base_features(train_df, test_df)
    te_columns = config.CAT_COLS + num_as_cat_cols
    print(f"  Base features: {X.shape[1]}")

    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{config.N_SPLITS} ---")
        X_tr, X_va, X_te = apply_target_encoding(
            X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], X_test,
            te_columns, ngram_columns, config.INNER_FOLDS, config.RANDOM_STATE,
        )
        if fold == 0:
            print(f"  Features after TE: {X_tr.shape[1]}")

        va_proba, te_proba, best_auc = train_one_fold(
            X_tr.values, y[tr_idx], X_va.values, y[va_idx], X_te.values,
            epochs, device,
        )
        oof_preds[va_idx] = va_proba
        test_preds += te_proba / config.N_SPLITS

        print(f"  Best AUC: {best_auc:.6f}")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\n{'='*80}")
    print(f"OOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_ll:.6f}")
    print(f"{'='*80}")

    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(config.PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({config.ID_COL: test_ids, "prob": test_preds}).to_csv(config.PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: test_preds}).to_csv(config.PRED_DIR / "test.csv", index=False)
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)
    print(f"Saved to {config.PRED_DIR}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default: use Config.EPOCHS)")
    main(parser.parse_args())
