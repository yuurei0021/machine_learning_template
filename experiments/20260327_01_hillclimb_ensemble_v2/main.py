"""
Hill Climbing Ensemble v2
Experiment: 20260327_01_hillclimb_ensemble_v2

v1 (20260326_02) + Ridge→XGB models added (experiments 04/05 + Ridge単体OOF).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    # --- v1 models ---
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
    # --- v2: Ridge→XGB models ---
    "ridge_xgb_reproduce":      "20260326_04_ridge_xgb_reproduce",
    "ridge_xgb_adapted":        "20260326_05_ridge_xgb_adapted",
}

# Ridge単体OOF (separate file within Ridge→XGB experiments)
RIDGE_ONLY_MODELS = {
    "ridge_only_reproduce":     ("20260326_04_ridge_xgb_reproduce", "oof_ridge.csv", "test_proba_ridge.csv"),
    "ridge_only_adapted":       ("20260326_05_ridge_xgb_adapted", "oof_ridge.csv", "test_proba_ridge.csv"),
}


# ============================================================================
# Load predictions
# ============================================================================
def load_predictions():
    oof_dict = {}
    test_dict = {}
    y_true = None
    oof_ids = None
    test_ids = None

    # Standard models
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

    # Ridge-only models (separate OOF files)
    for short_name, (exp_name, oof_file, test_file) in RIDGE_ONLY_MODELS.items():
        oof_path = EXPERIMENTS_DIR / exp_name / "predictions" / oof_file
        test_path = EXPERIMENTS_DIR / exp_name / "predictions" / test_file

        if not oof_path.exists():
            print(f"  SKIP {short_name}: no {oof_file}")
            continue

        oof_df = pd.read_csv(oof_path)
        oof_dict[short_name] = oof_df["prob"].values

        if test_path.exists():
            test_df = pd.read_csv(test_path)
            test_dict[short_name] = test_df["prob"].values
        else:
            print(f"  WARN {short_name}: no {test_file}")

    return oof_dict, test_dict, y_true, oof_ids, test_ids


# ============================================================================
# Hill Climbing Ensemble
# ============================================================================
def hill_climbing_ensemble(oof_dict, y_true,
                           weight_step=0.01,
                           max_iterations=100,
                           min_improvement=0.000001):
    scores = {name: roc_auc_score(y_true, oof)
              for name, oof in oof_dict.items()}

    best_model = max(scores, key=scores.get)
    current_ensemble = oof_dict[best_model].copy()
    current_score = scores[best_model]

    used_models = {best_model: 1.0}
    remaining = [m for m in oof_dict.keys() if m != best_model]

    print(f"Start: {best_model} (score: {current_score:.6f})")
    print(f"Remaining models: {len(remaining)}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        best_improvement = 0
        best_candidate = None
        best_weight = None
        best_new_ensemble = None
        best_new_score = current_score

        for i, candidate in enumerate(remaining):
            candidate_oof = oof_dict[candidate]
            weights_to_try = np.arange(-0.5, 1.0, weight_step)

            for weight in weights_to_try:
                test_ensemble = (1 - weight) * current_ensemble + weight * candidate_oof
                test_score = roc_auc_score(y_true, test_ensemble)
                improvement = test_score - current_score

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = candidate
                    best_weight = weight
                    best_new_ensemble = test_ensemble.copy()
                    best_new_score = test_score

            if (i + 1) % 5 == 0:
                print(f"  Tested {i+1}/{len(remaining)} models...")

        if best_improvement <= min_improvement:
            print(f"\nNo significant improvement found.")
            print(f"  Best improvement was: {best_improvement:.8f}")
            print(f"  Threshold: {min_improvement:.8f}")
            print(f"Stopping at iteration {iteration-1}")
            break

        for model in used_models:
            used_models[model] *= (1 - best_weight)
        used_models[best_candidate] = best_weight

        current_ensemble = best_new_ensemble
        current_score = best_new_score
        remaining.remove(best_candidate)

        print(f"Added: {best_candidate}")
        print(f"  Weight: {best_weight:.2f}")
        print(f"  Score: {current_score:.6f} (+{best_improvement:.6f})")
        print(f"  Remaining models: {len(remaining)}")

        if len(remaining) == 0:
            print(f"\nAll models tested. Stopping.")
            break

    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE")
    print(f"{'='*50}")
    print(f"Models used: {len(used_models)}/{len(oof_dict)}")
    print(f"Final score: {current_score:.6f}")
    print(f"\nWeights:")
    for model, weight in sorted(used_models.items(),
                                key=lambda x: abs(x[1]),
                                reverse=True):
        print(f"  {model}: {weight:+.4f}")

    return current_ensemble, used_models, current_score


def apply_weights_to_test(test_dict, used_models):
    first_model = list(used_models.keys())[0]
    total_weight = sum(used_models.values())
    test_blend = np.zeros(len(test_dict[first_model]))
    for model, weight in used_models.items():
        test_blend += weight * test_dict[model]

    if abs(total_weight - 1.0) > 0.01:
        print(f"  Warning: weights sum to {total_weight:.4f}, normalizing")
        test_blend /= total_weight

    return test_blend


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Hill Climbing Ensemble v2 (+ Ridge->XGB models)")
    print("=" * 70)

    print("\nLoading predictions...")
    oof_dict, test_dict, y_true, oof_ids, test_ids = load_predictions()

    common_models = sorted(set(oof_dict.keys()) & set(test_dict.keys()))
    oof_dict = {k: oof_dict[k] for k in common_models}
    test_dict_filtered = {k: test_dict[k] for k in common_models}
    print(f"Models with OOF + test: {len(common_models)}")

    print(f"\n{'Model':<30s} {'AUC':>10s}")
    print("-" * 42)
    for name in common_models:
        auc = roc_auc_score(y_true, oof_dict[name])
        new = " [NEW]" if name in ["ridge_xgb_reproduce", "ridge_xgb_adapted", "ridge_only_reproduce", "ridge_only_adapted"] else ""
        print(f"{name:<30s} {auc:10.6f}{new}")

    print("\n" + "=" * 70)
    print("Running Hill Climbing...")
    print("=" * 70)
    oof_blend, used_models, final_score = hill_climbing_ensemble(
        oof_dict, y_true,
        weight_step=0.01,
        max_iterations=100,
        min_improvement=0.000001,
    )

    print("\nApplying weights to test predictions...")
    test_blend = apply_weights_to_test(test_dict_filtered, used_models)

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
    print(f"Final OOF AUC: {final_score:.6f}")
    print("Done!")


if __name__ == "__main__":
    main()
