"""
Stacking LGBM + Optuna
Experiment: 20260331_02_stacking_lgbm_optuna

Optuna-tuned LGBM with raw features + TabM/RealMLP/Ridge OOF predictions.
Phase 1: Optuna search (5-fold CV)
Phase 2: Final training with best params
"""

import warnings
import json
import time
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent.parent / "data" / "raw"
EXPERIMENTS_DIR = BASE_DIR.parent

TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

TARGET_COL = "Churn"
ID_COL = "id"
PRED_DIR = BASE_DIR / "predictions"

N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 50

STACK_MODELS = {
    "tabm_pred":    ("20260330_01_tabm_nb1feat_20fold", "oof.csv", "test_proba.csv"),
    "realmlp_pred": ("20260325_04_realmlp_orig_data_reference", "oof.csv", "test_proba.csv"),
    "ridge_pred":   ("20260328_04_ridge_201feat_oof", "oof.csv", "test_proba.csv"),
}


# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    print(f"Train: {train.shape}, Test: {test.shape}")
    return train, test


def load_stacking_features():
    print("Loading stacking features...")
    oof_features = {}
    test_features = {}
    for feat_name, (exp_name, oof_file, test_file) in STACK_MODELS.items():
        oof_df = pd.read_csv(EXPERIMENTS_DIR / exp_name / "predictions" / oof_file)
        test_df = pd.read_csv(EXPERIMENTS_DIR / exp_name / "predictions" / test_file)
        oof_features[feat_name] = oof_df["prob"].values
        test_features[feat_name] = test_df["prob"].values
        print(f"  {feat_name}: OOF {len(oof_df)}, Test {len(test_df)}")
    return oof_features, test_features


def extract_features(train_df, test_df):
    exclude_cols = [TARGET_COL, ID_COL]
    cat_cols = [c for c in train_df.columns
                if pd.api.types.is_string_dtype(train_df[c]) and c not in exclude_cols]
    num_cols = [c for c in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[c]) and c not in exclude_cols]

    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    train_df["Charge_Difference"] = train_df["TotalCharges"] - train_df["MonthlyCharges"] * train_df["tenure"]
    test_df["Charge_Difference"] = test_df["TotalCharges"] - test_df["MonthlyCharges"] * test_df["tenure"]

    feature_cols = num_cols + cat_cols + ["Charge_Difference"]
    return train_df[feature_cols], test_df[feature_cols], feature_cols


# ============================================================================
# Optuna
# ============================================================================
def create_objective(X_train, y, skf):

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.0125, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": RANDOM_STATE,
            "verbose": -1,
            "n_jobs": -1,
        }

        oof_preds = np.zeros(len(X_train))

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_val = y[tr_idx], y[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=5000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(0),
                ],
            )

            oof_preds[va_idx] = model.predict(X_val)

            trial.report(roc_auc_score(y_val, oof_preds[va_idx]), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        auc = roc_auc_score(y, oof_preds)
        print(f"  Trial {trial.number}: AUC={auc:.6f}", flush=True)
        return auc

    return objective


# ============================================================================
# Final Training
# ============================================================================
def train_final(X_train, y, X_test, best_params, skf):
    print(f"\nFinal Training with best params:")
    print(json.dumps(best_params, indent=2))

    final_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "subsample_freq": 1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
        **best_params,
    }

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_indices = {"train": [], "val": []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        fold_indices["train"].append(tr_idx)
        fold_indices["val"].append(va_idx)

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y[tr_idx], y[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            final_params, dtrain,
            num_boost_round=10000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(200),
            ],
        )

        val_proba = model.predict(X_val)
        oof_preds[va_idx] = val_proba
        test_preds += model.predict(X_test) / N_SPLITS

        fold_auc = roc_auc_score(y_val, val_proba)
        print(f"  AUC: {fold_auc:.6f}, Best iter: {model.best_iteration}")

    return oof_preds, test_preds, fold_indices


# ============================================================================
# Main
# ============================================================================
def main():
    PRED_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print(f"Stacking LGBM + Optuna ({N_TRIALS} trials)")
    print("=" * 60)

    train_df, test_df = load_data()
    le_target = LabelEncoder()
    y = le_target.fit_transform(train_df[TARGET_COL])
    train_ids = train_df[ID_COL].values
    test_ids = test_df[ID_COL].values

    X_train, X_test, feature_cols = extract_features(train_df, test_df)

    oof_features, test_features = load_stacking_features()
    stack_cols = []
    for feat_name, oof_vals in oof_features.items():
        X_train[feat_name] = oof_vals
        X_test[feat_name] = test_features[feat_name]
        stack_cols.append(feat_name)

    print(f"Total features: {len(feature_cols) + len(stack_cols)} (raw {len(feature_cols)} + stack {len(stack_cols)})")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Phase 1: Optuna
    print(f"\n{'='*60}")
    print(f"Phase 1: Optuna Search ({N_TRIALS} trials)")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = create_objective(X_train, y, skf)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    # Save Optuna results
    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials),
    }
    (BASE_DIR / "optuna_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Phase 2: Final training
    print(f"\n{'='*60}")
    print(f"Phase 2: Final Training")
    print(f"{'='*60}")

    oof_preds, test_preds, fold_indices = train_final(X_train, y, X_test, study.best_params, skf)

    oof_auc = roc_auc_score(y, oof_preds)
    oof_ll = log_loss(y, oof_preds)
    print(f"\n{'='*60}")
    print(f"OOF AUC: {oof_auc:.6f}, LogLoss: {oof_ll:.6f}")
    print(f"Total time: {(time.time()-t0)/60:.1f}min")
    print(f"{'='*60}")

    # Save
    pd.DataFrame({"id": train_ids, "prob": oof_preds, "predicted": (oof_preds > 0.5).astype(int), "true": y}).to_csv(PRED_DIR / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "prob": test_preds}).to_csv(PRED_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_preds}).to_csv(PRED_DIR / "test.csv", index=False)
    with open(PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Saved to {PRED_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
