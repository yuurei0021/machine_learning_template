"""
XGBoost with YDF-inspired Parameters
Experiment: 20260315_01_ydf_baseline

目的: YDF Discussion知見のパラメータをXGBoostで再現
アプローチ: grow_policy='lossguide' + max_depth=2 + Label Encoding + Charge_Difference
"""

import warnings
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_curve,
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

    TARGET_COL = "Churn"
    ID_COL = "id"

    PRED_DIR = BASE_DIR / "predictions"
    MODEL_DIR = BASE_DIR / "model"

    N_SPLITS = 5
    RANDOM_STATE = 42

    # YDF-inspired parameters:
    #   BEST_FIRST_GLOBAL -> grow_policy='lossguide'
    #   max_depth=2 (shallow trees, features are independent)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_depth": 2,
        "max_leaves": 0,  # unlimited leaves, constrained by max_depth
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }

    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

config = Config()

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

# ============================================================================
# Feature Engineering
# ============================================================================
def extract_features(train_df, test_df):
    """Label Encoding for categorical features + Charge_Difference."""
    exclude_cols = [config.TARGET_COL, config.ID_COL]

    cat_cols = [c for c in train_df.columns
                if pd.api.types.is_string_dtype(train_df[c]) and c not in exclude_cols]
    num_cols = [c for c in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[c]) and c not in exclude_cols]

    print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"  Numeric columns ({len(num_cols)}): {num_cols}")

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Label Encoding
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le

    # Charge_Difference
    train_df["Charge_Difference"] = train_df["TotalCharges"] - train_df["MonthlyCharges"] * train_df["tenure"]
    test_df["Charge_Difference"] = test_df["TotalCharges"] - test_df["MonthlyCharges"] * test_df["tenure"]

    feature_cols = num_cols + cat_cols + ["Charge_Difference"]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    print(f"  Total features: {X_train.shape[1]}")
    return X_train, X_test

# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y):
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation")
    print(f"  XGBoost with YDF-inspired params: grow_policy=lossguide, max_depth=2")
    print(f"{'='*80}")

    oof_preds = np.zeros(len(X))
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

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
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

    return models, oof_preds, fold_indices, fold_scores

def predict_test(models, X_test):
    print(f"\nPredicting on {len(X_test)} test samples...")
    dtest = xgb.DMatrix(X_test)
    test_preds = np.zeros(len(X_test))

    for model in models:
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    test_preds /= len(models)
    print("Test predictions complete.")
    return test_preds

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

def plot_feature_importance(models, feature_names, save_path, top_n=20):
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
    plt.title("Top Feature Importance (XGBoost lossguide, depth=2)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def plot_roc_curve(y_true, y_score, save_path):
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
    print("XGBoost with YDF-inspired Parameters (lossguide, depth=2)")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()

    # Prepare target
    le = LabelEncoder()
    y = le.fit_transform(train_df[config.TARGET_COL])
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print(f"\nTarget classes: {list(le.classes_)} -> [0, 1]")
    print(f"Churn rate: {y.mean():.4f}")

    # Feature engineering
    print(f"\n{'='*80}")
    print("Feature Engineering")
    print(f"{'='*80}")
    X, X_test = extract_features(train_df, test_df)

    # Train
    models, oof_preds, fold_indices, fold_scores = train_with_cv(X, y)

    # Predict test
    test_preds = predict_test(models, X_test)

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)
    save_models(models)

    # Visualization
    oof_pred_labels = (oof_preds > 0.5).astype(int)
    plot_confusion_matrix(y, oof_pred_labels, config.BASE_DIR / "confusion_matrix.png")
    plot_feature_importance(models, X.columns.tolist(), config.BASE_DIR / "feature_importance.png")
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
