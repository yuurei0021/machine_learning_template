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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        'objective': 'regression',
        'metric': 'rmse',
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

    print(f"Features: {X_train.shape[1]} columns")
    return X_train, X_test

# ============================================================================
# Training
# ============================================================================
def train_with_cv(X, y):
    """Train with K-Fold CV."""
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation")
    print(f"  Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"{'='*80}")

    params = config.LGBM_PARAMS.copy()

    oof_preds = np.zeros(len(X))
    models = []
    fold_indices = {'train': [], 'val': []}
    fold_scores = []

    kf = KFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
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
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        fold_mae = mean_absolute_error(y_val, val_preds)
        fold_r2 = r2_score(y_val, val_preds)
        fold_scores.append({
            'fold': fold + 1,
            'rmse': fold_rmse,
            'mae': fold_mae,
            'r2': fold_r2,
        })

        print(f"  RMSE: {fold_rmse:.6f}, MAE: {fold_mae:.6f}, R2: {fold_r2:.6f}")

        models.append(model)
        fold_indices['train'].append(train_idx)
        fold_indices['val'].append(val_idx)

    # Overall OOF score
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    oof_mae = mean_absolute_error(y, oof_preds)
    oof_r2 = r2_score(y, oof_preds)

    print(f"\n{'='*80}")
    print(f"CV Results:")
    for s in fold_scores:
        print(f"  Fold {s['fold']}: RMSE={s['rmse']:.6f}, MAE={s['mae']:.6f}, R2={s['r2']:.6f}")
    print(f"\nOOF RMSE: {oof_rmse:.6f}, MAE: {oof_mae:.6f}, R2: {oof_r2:.6f}")
    print(f"{'='*80}")

    return models, oof_preds, fold_indices

def predict_test(models, X_test):
    """Predict on test data (average of fold models)."""
    print(f"\nPredicting on {len(X_test)} test samples...")

    test_preds = np.zeros(len(X_test))
    for fold, model in enumerate(models):
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += preds

    test_preds /= len(models)
    print("Test predictions complete.")
    return test_preds

# ============================================================================
# Visualization
# ============================================================================
def plot_actual_vs_predicted(y_true, y_pred, save_path):
    """Plot actual vs predicted scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10)

    # Diagonal line
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    plt.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1)

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted (OOF)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_residuals(y_true, y_pred, save_path):
    """Plot residual distribution."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Residual')
    axes[0].set_title('Residuals vs Predicted')

    # Residual histogram
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual Distribution')

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
def save_predictions(oof_preds, y, test_preds, train_ids, test_ids, fold_indices):
    """Save OOF and test predictions."""
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)

    # OOF predictions
    oof_df = pd.DataFrame({
        config.ID_COL: train_ids,
        'predicted': oof_preds,
        'true': y,
    })
    oof_df.to_csv(config.PRED_DIR / "oof.csv", index=False)

    # Test predictions
    test_df = pd.DataFrame({
        config.ID_COL: test_ids,
        'predicted': test_preds,
    })
    test_df.to_csv(config.PRED_DIR / "test_proba.csv", index=False)

    # Submission format
    # TODO: Adjust to match competition submission format
    submit_df = pd.DataFrame({
        config.ID_COL: test_ids,
        config.TARGET_COL: test_preds,
    })
    submit_df.to_csv(config.PRED_DIR / "test.csv", index=False, header=False)

    # Fold indices
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
    print("LightGBM Regression")
    print("=" * 80)

    # Load data
    train_df, test_df = load_data()

    # Prepare target
    y = train_df[config.TARGET_COL].copy()
    train_ids = train_df[config.ID_COL].values
    test_ids = test_df[config.ID_COL].values

    print(f"\nTarget stats: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")

    # Feature engineering
    print(f"\n{'='*80}")
    print("Feature Engineering")
    print(f"{'='*80}")
    X, X_test = extract_features(train_df, test_df)

    # Train with CV
    models, oof_preds, fold_indices = train_with_cv(X, y)

    # Predict test
    test_preds = predict_test(models, X_test)

    # Save predictions
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")
    save_predictions(oof_preds, y.values, test_preds, train_ids, test_ids, fold_indices)

    if args.save_models:
        save_models(models)

    # Visualization
    plot_actual_vs_predicted(
        y.values, oof_preds,
        config.BASE_DIR / "actual_vs_predicted.png",
    )
    plot_residuals(
        y.values, oof_preds,
        config.BASE_DIR / "residuals.png",
    )
    plot_feature_importance(
        models, X.columns.tolist(),
        config.BASE_DIR / "feature_importance.png",
    )

    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM Regression")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models')
    main(parser.parse_args())
