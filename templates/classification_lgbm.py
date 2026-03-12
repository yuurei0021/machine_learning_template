"""
{実験名}
Experiment: {YYYYMMDD_NN_experiment_name}

目的: TODO
アプローチ: TODO
"""

# ============================================================================
# Import
# ============================================================================
import warnings
from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================================
# Config
# ============================================================================
class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent.parent / "data"
    INPUT_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"

    # --- TODO: Set your data files ---
    TRAIN_DATA = PROCESSED_DIR / "train.csv"
    TEST_DATA = PROCESSED_DIR / "test.csv"

    # --- TODO: Set column names ---
    TARGET_COL = "target"       # Target column name
    ID_COL = "id"               # ID column name

    # Output paths (auto-configured)
    PRED_DIR = BASE_DIR / "predictions"
    MODEL_DIR = BASE_DIR / "model"

    # CV settings
    N_SPLITS = 5
    RANDOM_STATE = 42

    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',      # auto-set: 'binary' or 'multiclass'
        'metric': 'multi_logloss',      # auto-set: 'binary_logloss' or 'multi_logloss'
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1,
    }

    NUM_BOOST_ROUND = 1000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

config = Config()

# ============================================================================
# Data Loading
# ============================================================================
def load_data():
    """Load training and test data."""
    print("Loading data...")
    train = pd.read_csv(config.TRAIN_DATA)
    test = pd.read_csv(config.TEST_DATA)
    print(f"  Train: {train.shape}")
    print(f"  Test: {test.shape}")
    return train, test

# ============================================================================
# Feature Engineering
# ============================================================================
def extract_features(train_df, test_df):
    """
    Extract/engineer features.

    TODO: Implement competition-specific feature engineering.
    Modify this function to create meaningful features from your data.
    """
    # TODO: Define which columns to exclude from features
    exclude_cols = [config.TARGET_COL, config.ID_COL]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # TODO: Add feature engineering here
    # Examples:
    #   X_train['new_feature'] = X_train['col_a'] * X_train['col_b']
    #   X_test['new_feature'] = X_test['col_a'] * X_test['col_b']
    #
    # For grouped/time-series data, iterate over groups:
    #   for group_id, group_df in raw_df.groupby('id'):
    #       features = compute_features(group_df)  # stats, FFT, etc.

    print(f"Features: {X_train.shape[1]} columns")
    return X_train, X_test

# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y, label_encoder):
    """Train with Stratified K-Fold CV."""
    n_classes = len(label_encoder.classes_)
    is_binary = (n_classes == 2)

    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation")
    print(f"  Classes: {n_classes} ({list(label_encoder.classes_)})")
    print(f"{'='*80}")

    # Auto-configure LightGBM params
    params = config.LGBM_PARAMS.copy()
    if is_binary:
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params.pop('num_class', None)
    else:
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'
        params['num_class'] = n_classes

    # Initialize OOF predictions
    if is_binary:
        oof_preds = np.zeros(len(X))
    else:
        oof_preds = np.zeros((len(X), n_classes))

    models = []
    fold_indices = {'train': [], 'val': []}
    fold_scores = []

    skf = StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{config.N_SPLITS} ---")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train: {len(X_train)}, Val: {len(X_val)}")

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=config.NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=config.VERBOSE_EVAL),
            ],
        )

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        # Fold metrics
        if is_binary:
            val_pred_labels = (val_preds > 0.5).astype(int)
            fold_logloss = log_loss(y_val, val_preds)
        else:
            val_pred_labels = np.argmax(val_preds, axis=1)
            fold_logloss = log_loss(y_val, val_preds)

        fold_f1 = f1_score(y_val, val_pred_labels, average='macro')
        fold_acc = accuracy_score(y_val, val_pred_labels)
        fold_scores.append({
            'fold': fold + 1,
            'f1_macro': fold_f1,
            'accuracy': fold_acc,
            'logloss': fold_logloss,
        })

        print(f"  F1 Macro: {fold_f1:.6f}, Accuracy: {fold_acc:.6f}, LogLoss: {fold_logloss:.6f}")

        models.append(model)
        fold_indices['train'].append(train_idx)
        fold_indices['val'].append(val_idx)

    # Overall OOF score
    if is_binary:
        oof_pred_labels = (oof_preds > 0.5).astype(int)
    else:
        oof_pred_labels = np.argmax(oof_preds, axis=1)

    oof_f1 = f1_score(y, oof_pred_labels, average='macro')
    oof_acc = accuracy_score(y, oof_pred_labels)

    print(f"\n{'='*80}")
    print(f"CV Results:")
    for s in fold_scores:
        print(f"  Fold {s['fold']}: F1={s['f1_macro']:.6f}, Acc={s['accuracy']:.6f}, Loss={s['logloss']:.6f}")
    print(f"\nOOF F1 Macro: {oof_f1:.6f}, Accuracy: {oof_acc:.6f}")
    print(f"{'='*80}")

    return models, oof_preds, fold_indices

def predict_test(models, X_test, is_binary=False):
    """Predict on test data (average of fold models)."""
    print(f"\nPredicting on {len(X_test)} test samples...")

    test_preds = None
    for fold, model in enumerate(models):
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        if test_preds is None:
            test_preds = np.zeros_like(preds)
        test_preds += preds

    test_preds /= len(models)
    print("Test predictions complete.")
    return test_preds

# ============================================================================
# Visualization
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (OOF)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_feature_importance(models, feature_names, save_path, top_n=30):
    """Plot average feature importance across folds."""
    importance = np.zeros(len(feature_names))
    for model in models:
        importance += model.feature_importance(importance_type='gain')
    importance /= len(models)

    top_n = min(top_n, len(feature_names))
    idx = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.barh(range(len(idx)), importance[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel('Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ============================================================================
# Save
# ============================================================================
def save_predictions(oof_preds, y, test_preds, train_ids, test_ids,
                     label_encoder, fold_indices):
    """Save OOF and test predictions."""
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)
    n_classes = len(label_encoder.classes_)
    is_binary = (n_classes == 2)
    class_names = list(label_encoder.classes_)

    # --- OOF predictions ---
    if is_binary:
        oof_df = pd.DataFrame({
            config.ID_COL: train_ids,
            'prob': oof_preds,
            'predicted': label_encoder.inverse_transform((oof_preds > 0.5).astype(int)),
            'true': label_encoder.inverse_transform(y),
        })
    else:
        oof_dict = {config.ID_COL: train_ids}
        for i, cls in enumerate(class_names):
            oof_dict[f'prob_{cls}'] = oof_preds[:, i]
        oof_dict['predicted'] = label_encoder.inverse_transform(np.argmax(oof_preds, axis=1))
        oof_dict['true'] = label_encoder.inverse_transform(y)
        oof_df = pd.DataFrame(oof_dict)

    oof_df.to_csv(config.PRED_DIR / "oof.csv", index=False)

    # --- Test predictions (probabilities) ---
    if is_binary:
        test_proba_df = pd.DataFrame({
            config.ID_COL: test_ids,
            'prob': test_preds,
        })
    else:
        test_dict = {config.ID_COL: test_ids}
        for i, cls in enumerate(class_names):
            test_dict[f'prob_{cls}'] = test_preds[:, i]
        test_proba_df = pd.DataFrame(test_dict)

    test_proba_df.to_csv(config.PRED_DIR / "test_proba.csv", index=False)

    # --- Test predictions (submission format) ---
    # TODO: Adjust to match competition submission format
    if is_binary:
        pred_labels = label_encoder.inverse_transform((test_preds > 0.5).astype(int))
    else:
        pred_labels = label_encoder.inverse_transform(np.argmax(test_preds, axis=1))

    submit_df = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: pred_labels,
    })
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False, header=False)

    # --- Fold indices ---
    with open(config.PRED_DIR / "fold_indices.pkl", 'wb') as f:
        pickle.dump(fold_indices, f)

    print(f"Predictions saved to {config.PRED_DIR}")

def save_models(models):
    """Save trained models."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fold, model in enumerate(models):
        model.save_model(str(config.MODEL_DIR / f"fold_{fold}.txt"))
    print(f"Models saved to {config.MODEL_DIR}")

# ============================================================================
# Main
# ============================================================================
def main(args):
    print("=" * 80)
    print("LightGBM Classification")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()

    # Prepare labels
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(train_df[config.TARGET_COL]))
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    n_classes = len(le.classes_)
    is_binary = (n_classes == 2)

    print(f"\nTarget: {n_classes} classes - {list(le.classes_)}")
    print(f"Distribution:\n{train_df[config.TARGET_COL].value_counts()}")

    # Feature engineering
    print(f"\n{'='*80}")
    print("Feature Engineering")
    print(f"{'='*80}")
    X, X_test = extract_features(train_df, test_df)

    # Train with CV
    models, oof_preds, fold_indices = train_with_cv(X, y, le)

    # Predict test
    test_preds = predict_test(models, X_test, is_binary)

    # Save predictions
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y.values, test_preds, train_ids, test_ids, le, fold_indices)

    if args.save_models:
        save_models(models)

    # Visualization
    if is_binary:
        oof_pred_labels = (oof_preds > 0.5).astype(int)
    else:
        oof_pred_labels = np.argmax(oof_preds, axis=1)

    plot_confusion_matrix(
        y.values, oof_pred_labels, list(le.classes_),
        config.BASE_DIR / "confusion_matrix.png",
    )
    plot_feature_importance(
        models, X.columns.tolist(),
        config.BASE_DIR / "feature_importance.png",
    )

    # Classification report
    report = classification_report(y, oof_pred_labels, target_names=list(le.classes_))
    report_path = config.BASE_DIR / "classification_report.txt"
    report_path.write_text(report, encoding='utf-8')
    print(f"Saved: {report_path}")

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM Classification")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models')
    main(parser.parse_args())
