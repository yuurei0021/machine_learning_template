"""
KNN Baseline
Experiment: 20260331_09_knn_baseline

目的: KNNによるベースライン構築（生値のみ）
アプローチ: One-Hot Encoding + StandardScaler + KNeighborsClassifier (5-Fold Stratified CV)
  - インスタンスベースの手法でツリー系・NN系と直交する誤差パターンを狙う
  - 合成データ（元データ7k行の約85コピー+ノイズ）とKNNの相性を検証
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_curve,
)

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

    # KNN params
    N_NEIGHBORS = 51
    WEIGHTS = "distance"  # distance weighting
    METRIC = "minkowski"
    P = 2  # Euclidean distance
    N_JOBS = -1

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
    """
    生値のみ: One-Hot Encoding for categoricals, StandardScaler for all.
    KNNは距離ベースのためOHE + スケーリングが必須。
    """
    exclude_cols = [config.TARGET_COL, config.ID_COL]

    cat_cols = [c for c in train_df.columns
                if pd.api.types.is_string_dtype(train_df[c]) and c not in exclude_cols]
    num_cols = [c for c in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[c]) and c not in exclude_cols]

    print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"  Numeric columns ({len(num_cols)}): {num_cols}")

    train_df = train_df.copy()
    test_df = test_df.copy()

    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
    train_cat = ohe.fit_transform(train_df[cat_cols])
    test_cat = ohe.transform(test_df[cat_cols])
    ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    # Numeric
    train_num = train_df[num_cols].values
    test_num = test_df[num_cols].values

    # Combine
    feature_names = num_cols + ohe_feature_names
    X_train = np.hstack([train_num, train_cat])
    X_test = np.hstack([test_num, test_cat])

    print(f"  Total features: {X_train.shape[1]}")
    return X_train, X_test, feature_names

# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y, feature_names):
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation (KNN, k={config.N_NEIGHBORS})")
    print(f"{'='*80}")

    oof_preds = np.zeros(len(X))
    fold_indices = {"train": [], "val": []}
    fold_scores = []
    scalers = []

    skf = StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

        # Scale per fold (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        scalers.append(scaler)

        # KNN
        knn = KNeighborsClassifier(
            n_neighbors=config.N_NEIGHBORS,
            weights=config.WEIGHTS,
            metric=config.METRIC,
            p=config.P,
            n_jobs=config.N_JOBS,
            algorithm="auto",
        )

        print("  Fitting KNN...")
        knn.fit(X_train_scaled, y_train)

        print("  Predicting validation set...")
        val_proba = knn.predict_proba(X_val_scaled)[:, 1]
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
        })
        print(f"  AUC: {fold_auc:.6f}, LogLoss: {fold_logloss:.6f}, Accuracy: {fold_acc:.6f}")

        fold_indices["train"].append(train_idx)
        fold_indices["val"].append(val_idx)

    # Overall OOF
    oof_auc = roc_auc_score(y, oof_preds)
    oof_logloss = log_loss(y, oof_preds)
    oof_acc = accuracy_score(y, (oof_preds > 0.5).astype(int))

    print(f"\n{'='*80}")
    print("CV Results:")
    for s in fold_scores:
        print(f"  Fold {s['fold']}: AUC={s['auc']:.6f}, LogLoss={s['logloss']:.6f}, Acc={s['accuracy']:.6f}")
    print(f"\nOOF AUC-ROC: {oof_auc:.6f}")
    print(f"OOF LogLoss: {oof_logloss:.6f}")
    print(f"OOF Accuracy: {oof_acc:.6f}")
    print(f"{'='*80}")

    return scalers, oof_preds, fold_indices, fold_scores

def predict_test(X, y, X_test):
    """Refit on full training data and predict test."""
    print(f"\nRefitting on full training data ({len(X)} samples)...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(
        n_neighbors=config.N_NEIGHBORS,
        weights=config.WEIGHTS,
        metric=config.METRIC,
        p=config.P,
        n_jobs=config.N_JOBS,
        algorithm="auto",
    )
    knn.fit(X_scaled, y)

    print(f"Predicting on {len(X_test)} test samples...")
    test_preds = knn.predict_proba(X_test_scaled)[:, 1]
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

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("KNN Baseline")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()

    # Prepare target
    le = LabelEncoder()
    y = le.fit_transform(train_df[config.TARGET_COL])  # No=0, Yes=1
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print(f"\nTarget classes: {list(le.classes_)} -> [0, 1]")
    print(f"Churn rate: {y.mean():.4f}")

    # Feature engineering
    print(f"\n{'='*80}")
    print("Feature Engineering")
    print(f"{'='*80}")
    X, X_test, feature_names = extract_features(train_df, test_df)

    # Train with CV
    scalers, oof_preds, fold_indices, fold_scores = train_with_cv(X, y, feature_names)

    # Predict test (refit on full data)
    test_preds = predict_test(X, y, X_test)

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)

    # Visualization
    oof_pred_labels = (oof_preds > 0.5).astype(int)
    plot_confusion_matrix(y, oof_pred_labels, config.BASE_DIR / "confusion_matrix.png")
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
