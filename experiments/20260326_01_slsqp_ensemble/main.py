"""
SLSQP Weight-Optimized Ensemble
Experiment: 20260326_01_slsqp_ensemble

Optimize blending weights for all available OOF predictions using SLSQP,
then apply the same weights to test predictions for submission.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR.parent
PRED_DIR = BASE_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)

MODELS = {
    "logreg_baseline":          "20260313_02_logistic_regression",
    "xgb_baseline":             "20260313_03_xgboost_baseline",
    "xgb_depth1":               "20260313_04_xgboost_depth1",
    "lgbm_baseline":            "20260313_05_lightgbm_baseline",
    "bartz_baseline":           "20260313_06_bartz_baseline",
    "ydf_baseline":             "20260315_01_ydf_baseline",
    "xgb_orig_ref":             "20260323_01_orig_data_reference",
    "xgb_optuna":               "20260323_02_xgb_optuna_tuning",
    "lgbm_orig_ref":            "20260325_01_lgbm_orig_data_reference",
    "lgbm_optuna":              "20260325_02_lgbm_optuna_tuning",
    "catboost_orig_ref":        "20260325_03_catboost_orig_data_reference",
    "realmlp_orig_ref":         "20260325_04_realmlp_orig_data_reference",
    "logreg_baseline2":         "20260325_05_logreg_baseline",
    "logreg_orig_ref":          "20260325_06_logreg_orig_data_reference",
    "ridge_baseline":           "20260325_07_ridge_baseline",
    "ridge_orig_ref":           "20260325_08_ridge_orig_data_reference",
    "mlp_orig_ref":             "20260325_09_mlp_orig_data_reference",
    "tabm_orig_ref":            "20260325_10_tabm_orig_data_reference",
}


# ============================================================================
# Load OOF and test predictions
# ============================================================================
def load_predictions():
    oof_dict = {}
    test_dict = {}
    y_true = None
    oof_ids = None
    test_ids = None

    for short_name, exp_name in MODELS.items():
        oof_path = EXPERIMENTS_DIR / exp_name / "predictions" / "oof.csv"
        test_path = EXPERIMENTS_DIR / exp_name / "predictions" / "test_proba.csv"

        if not oof_path.exists():
            print(f"  SKIP {short_name}: no oof.csv")
            continue

        oof_df = pd.read_csv(oof_path)
        prob_col = "prob" if "prob" in oof_df.columns else "probability"
        true_col = "true" if "true" in oof_df.columns else "actual"

        oof_dict[short_name] = oof_df[prob_col].values

        if y_true is None:
            y_true = oof_df[true_col].values
            oof_ids = oof_df["id"].values if "id" in oof_df.columns else np.arange(len(oof_df))

        if test_path.exists():
            test_df = pd.read_csv(test_path)
            prob_col_t = next(c for c in ["prob", "probability", "Churn"] if c in test_df.columns)
            test_dict[short_name] = test_df[prob_col_t].values
            if test_ids is None:
                test_ids = test_df["id"].values
        else:
            print(f"  WARN {short_name}: no test_proba.csv")

    return oof_dict, test_dict, y_true, oof_ids, test_ids


# ============================================================================
# SLSQP optimization
# ============================================================================
def optimize_weights(oof_matrix, y_true):
    n_models = oof_matrix.shape[1]

    def neg_auc(weights):
        blend = oof_matrix @ weights
        return -roc_auc_score(y_true, blend)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.ones(n_models) / n_models

    result = minimize(neg_auc, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    return result.x, -result.fun


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("SLSQP Weight-Optimized Ensemble")
    print("=" * 70)

    print("\nLoading predictions...")
    oof_dict, test_dict, y_true, oof_ids, test_ids = load_predictions()

    common_models = sorted(set(oof_dict.keys()) & set(test_dict.keys()))
    print(f"\nModels with OOF + test: {len(common_models)}")

    print(f"\n{'Model':<25s} {'AUC':>10s}")
    print("-" * 37)
    for name in common_models:
        auc = roc_auc_score(y_true, oof_dict[name])
        print(f"{name:<25s} {auc:10.6f}")

    oof_matrix = np.column_stack([oof_dict[name] for name in common_models])
    test_matrix = np.column_stack([test_dict[name] for name in common_models])

    equal_blend = oof_matrix.mean(axis=1)
    equal_auc = roc_auc_score(y_true, equal_blend)
    print(f"\nEqual-weight AUC: {equal_auc:.6f}")

    print("\nRunning SLSQP optimization...")
    weights, opt_auc = optimize_weights(oof_matrix, y_true)

    print(f"\nOptimized AUC: {opt_auc:.6f} (improvement over equal: {opt_auc - equal_auc:+.6f})")

    print(f"\n{'Model':<25s} {'Weight':>8s} {'AUC':>10s}")
    print("-" * 45)
    for name, w in sorted(zip(common_models, weights), key=lambda x: -x[1]):
        auc = roc_auc_score(y_true, oof_dict[name])
        marker = " ***" if w > 0.01 else ""
        print(f"{name:<25s} {w:8.4f} {auc:10.6f}{marker}")

    oof_blend = oof_matrix @ weights
    test_blend = test_matrix @ weights

    pd.DataFrame({
        "id": oof_ids,
        "prob": oof_blend,
        "predicted": (oof_blend > 0.5).astype(int),
        "true": y_true,
    }).to_csv(PRED_DIR / "oof.csv", index=False)

    pd.DataFrame({
        "id": test_ids,
        "prob": test_blend,
    }).to_csv(PRED_DIR / "test_proba.csv", index=False)

    pd.DataFrame({
        "id": test_ids,
        "Churn": test_blend,
    }).to_csv(PRED_DIR / "test.csv", index=False)

    print(f"\nSaved to {PRED_DIR}")
    print(f"\nFinal OOF AUC: {opt_auc:.6f}")
    print("Done!")


if __name__ == "__main__":
    main()
