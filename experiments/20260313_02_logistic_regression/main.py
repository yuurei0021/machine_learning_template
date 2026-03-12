"""
Logistic Regression Baseline
Experiment: 20260313_02_logistic_regression

目的: ロジスティック回帰によるベースライン構築。特徴量が独立的であるため強いモデルが期待される（Discussion知見）
アプローチ: One-Hot Encoding + StandardScaler + LogisticRegression (5-Fold Stratified CV)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
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
    特徴量エンジニアリング:
    - カテゴリ変数: One-Hot Encoding
    - 数値変数: そのまま使用
    - 合成特徴量: Charge_Difference (TotalCharges - MonthlyCharges * tenure)
    """
    exclude_cols = [config.TARGET_COL, config.ID_COL]

    # カテゴリカラムと数値カラムの分離
    cat_cols = [c for c in train_df.columns
                if pd.api.types.is_string_dtype(train_df[c]) and c not in exclude_cols]
    num_cols = [c for c in train_df.columns
                if pd.api.types.is_numeric_dtype(train_df[c]) and c not in exclude_cols]

    print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"  Numeric columns ({len(num_cols)}): {num_cols}")

    # 合成特徴量
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["Charge_Difference"] = train_df["TotalCharges"] - train_df["MonthlyCharges"] * train_df["tenure"]
    test_df["Charge_Difference"] = test_df["TotalCharges"] - test_df["MonthlyCharges"] * test_df["tenure"]
    num_cols.append("Charge_Difference")

    # One-Hot Encoding
    train_cat = pd.get_dummies(train_df[cat_cols], drop_first=True)
    test_cat = pd.get_dummies(test_df[cat_cols], drop_first=True)

    # train/testのカラムを揃える
    missing_in_test = set(train_cat.columns) - set(test_cat.columns)
    missing_in_train = set(test_cat.columns) - set(train_cat.columns)
    for col in missing_in_test:
        test_cat[col] = 0
    for col in missing_in_train:
        train_cat[col] = 0
    test_cat = test_cat[train_cat.columns]

    # 数値 + One-Hot を結合
    X_train = pd.concat([train_df[num_cols].reset_index(drop=True),
                         train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[num_cols].reset_index(drop=True),
                        test_cat.reset_index(drop=True)], axis=1)

    # bool を int に変換
    for col in X_train.columns:
        if X_train[col].dtype == "bool":
            X_train[col] = X_train[col].astype(int)
            X_test[col] = X_test[col].astype(int)

    print(f"  Total features: {X_train.shape[1]}")
    return X_train, X_test

# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y):
    """5-Fold Stratified CV with Logistic Regression."""
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation (Logistic Regression)")
    print(f"{'='*80}")

    oof_preds = np.zeros(len(X))
    models = []
    scalers = []
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

        # StandardScaler (fold毎にfit)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Logistic Regression
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Predict probabilities (positive class = Churn=Yes)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
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

        models.append(model)
        scalers.append(scaler)
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

    return models, scalers, oof_preds, fold_indices, fold_scores

def predict_test(models, scalers, X_test):
    """Average predictions from all folds."""
    print(f"\nPredicting on {len(X_test)} test samples...")
    test_preds = np.zeros(len(X_test))

    for model, scaler in zip(models, scalers):
        X_test_scaled = scaler.transform(X_test)
        test_preds += model.predict_proba(X_test_scaled)[:, 1]

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

def plot_coefficient_importance(models, feature_names, save_path, top_n=30):
    """Plot average absolute coefficients across folds."""
    coefs = np.zeros(len(feature_names))
    for model in models:
        coefs += np.abs(model.coef_[0])
    coefs /= len(models)

    top_n = min(top_n, len(feature_names))
    idx = np.argsort(coefs)[-top_n:]

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.barh(range(len(idx)), coefs[idx], color="#2196F3")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Mean |Coefficient|")
    plt.title(f"Top {top_n} Feature Importance (Logistic Regression)")
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

    # Submission (id, Churn probability)
    submit_df = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: test_preds,
    })
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False)

    # Fold indices
    with open(config.PRED_DIR / "fold_indices.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    print(f"Predictions saved to {config.PRED_DIR}")

def save_models(models, scalers):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fold, (model, scaler) in enumerate(zip(models, scalers)):
        with open(config.MODEL_DIR / f"fold_{fold}.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Models saved to {config.MODEL_DIR}")

# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 80)
    print("Logistic Regression - Baseline")
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
    X, X_test = extract_features(train_df, test_df)

    # Train
    models, scalers, oof_preds, fold_indices, fold_scores = train_with_cv(X, y)

    # Predict test
    test_preds = predict_test(models, scalers, X_test)

    # Save
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices)
    save_models(models, scalers)

    # Visualization
    oof_pred_labels = (oof_preds > 0.5).astype(int)
    plot_confusion_matrix(y, oof_pred_labels, config.BASE_DIR / "confusion_matrix.png")
    plot_coefficient_importance(models, X.columns.tolist(), config.BASE_DIR / "feature_importance.png")
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
