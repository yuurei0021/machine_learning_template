"""
Hill Climbing Ensemble: v4+++ + TabTransformer (Kaggle GPU)
- v4+++ model set (17 models) + TabTransformer = 18 models
- Rank normalization (rank01)
- GPU parallel evaluation (CuPy)
- Re-selection of same model allowed
- GPU approximate AUC (search) + CPU exact AUC (final)
"""

import gc
import os
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================
DATA_DIR_1 = Path("/kaggle/input/datasets/nyhalcyon/hc-v4-oof-predictions")
DATA_DIR_2 = Path("/kaggle/input/datasets/nyhalcyon/tabtransformer-oof-predictions")
OUTPUT_DIR = Path("/kaggle/working")

# v4+++ models (17) + TabTransformer
MODELS = {
    "ridge_xgb_reproduce":      "20260326_04_ridge_xgb_reproduce",
    "xgb_optuna_20fold":        "20260327_02_xgb_optuna_20fold",
    "realmlp_orig_ref":         "20260325_04_realmlp_orig_data_reference",
    "tabm_orig_ref":            "20260325_10_tabm_orig_data_reference",
    "lgbm_optuna":              "20260325_02_lgbm_optuna_tuning",
    "catboost_orig_ref":        "20260325_03_catboost_orig_data_reference",
    "mlp_orig_ref":             "20260325_09_mlp_orig_data_reference",
    "logit3_te_pair":           "20260327_08_logit3_te_pair_logreg",
    "logreg_orig_ref":          "20260325_06_logreg_orig_data_reference",
    "xgb_baseline":             "20260313_03_xgboost_baseline",
    "xgb_depth1":               "20260313_04_xgboost_depth1",
    "bartz_baseline":           "20260313_06_bartz_baseline",
    "extratrees_orig_ref":      "20260327_07_extratrees_orig_data_reference",
    "ydf_orig_ref":             "20260327_11_ydf_orig_data_reference",
    "ridge_lgbm_reproduce":     "20260328_01_ridge_lgbm_reproduce",
    "tabm_nb1feat_10fold":      "20260330_01_tabm_nb1feat_20fold",
    "gnn_starter":              "20260330_02_gnn_starter",
    "tabtransformer":           "20260331_06_tabtransformer",
}

USE_NEGATIVE_WGT = True
MAX_MODELS       = 1000
TOL              = 1e-6

start_w = -0.50 if USE_NEGATIVE_WGT else 0.01
stop_w  =  1.00
step_w  =  0.01

FIRST = None

# ============================================================================
# Helpers
# ============================================================================
def rank01(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    r = rankdata(x, method="average")
    if len(r) == 1:
        return np.array([0.5], dtype=np.float64)
    return (r - 1.0) / (len(r) - 1.0)


def auc_cpu(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))


def multiple_roc_auc_scores(y_gpu, preds_gpu):
    n_pos = cp.sum(y_gpu)
    n_neg = y_gpu.size - n_pos
    ranks = cp.argsort(cp.argsort(preds_gpu, axis=0), axis=0) + 1
    aucs  = (
        cp.sum(ranks[y_gpu == 1, :], axis=0) - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)
    return aucs


# ============================================================================
# Load predictions (from multiple Kaggle datasets)
# ============================================================================
def find_file(filename):
    for d in [DATA_DIR_1, DATA_DIR_2]:
        p = d / filename
        if p.exists():
            return p
    return None


def load_predictions():
    oof_dict  = {}
    test_dict = {}
    y_true    = None
    oof_ids   = None
    test_ids  = None

    print(f"  DATA_DIR_1: {DATA_DIR_1} (exists={DATA_DIR_1.exists()})")
    print(f"  DATA_DIR_2: {DATA_DIR_2} (exists={DATA_DIR_2.exists()})")

    # Debug: list contents
    for dd in [DATA_DIR_1, DATA_DIR_2]:
        if dd.exists():
            files = sorted(os.listdir(dd))
            print(f"  {dd}: {len(files)} files")
        else:
            print(f"  {dd}: NOT FOUND")
            # Try to find it
            for d in os.listdir('/kaggle/input/'):
                full = f'/kaggle/input/{d}'
                if os.path.isdir(full):
                    for s in os.listdir(full):
                        sf = f'{full}/{s}'
                        if os.path.isdir(sf):
                            print(f"    {sf}/: {os.listdir(sf)[:3]}...")

    for short_name, exp_name in MODELS.items():
        oof_path  = find_file(f"oof_{exp_name}.csv")
        test_path = find_file(f"test_proba_{exp_name}.csv")

        if oof_path is None:
            print(f"  SKIP {short_name}: oof_{exp_name}.csv not found")
            continue

        oof_df   = pd.read_csv(oof_path)
        prob_col = "prob" if "prob" in oof_df.columns else "probability"
        true_col = "true" if "true" in oof_df.columns else "actual"

        oof_dict[short_name] = rank01(oof_df[prob_col].values)

        if y_true is None:
            y_true  = oof_df[true_col].values.astype(np.int32)
            oof_ids = oof_df["id"].values if "id" in oof_df.columns else np.arange(len(oof_df))

        if test_path is not None:
            test_df    = pd.read_csv(test_path)
            prob_col_t = next(c for c in ["prob", "probability", "Churn"] if c in test_df.columns)
            test_dict[short_name] = rank01(test_df[prob_col_t].values)
            if test_ids is None:
                test_ids = test_df["id"].values
        else:
            print(f"  WARN {short_name}: test_proba_{exp_name}.csv not found")

    return oof_dict, test_dict, y_true, oof_ids, test_ids


# ============================================================================
# Hill Climbing
# ============================================================================
def hill_climbing_ensemble(oof_dict, y_true):
    names   = list(oof_dict.keys())
    K       = len(names)
    x_train = np.column_stack([oof_dict[n] for n in names])

    print(f"\n{'Model':<30s} {'AUC':>10s}")
    print("-" * 42)
    single_aucs = []
    for k, name in enumerate(names):
        s = auc_cpu(y_true, x_train[:, k])
        single_aucs.append(s)
        print(f"{name:<30s} {s:10.6f}")

    best_single_idx = int(np.argmax(single_aucs))
    print(f"\nBest single: {names[best_single_idx]} (AUC={single_aucs[best_single_idx]:.6f})")

    if FIRST is None:
        start_idx = best_single_idx
    else:
        start_idx = names.index(FIRST)

    print(f"Starting from: {names[start_idx]} (AUC={single_aucs[start_idx]:.6f})")

    x_gpu = cp.asarray(x_train, dtype=cp.float32)
    y_gpu = cp.asarray(y_true,  dtype=np.int8)
    ww    = cp.arange(start_w, stop_w + 1e-9, step_w, dtype=cp.float32)

    best_ensemble = x_gpu[:, start_idx].copy()
    best_score    = float(single_aucs[start_idx])

    selected_indices = [start_idx]
    weights_step     = []

    print("\nStarting hill climb...\n")

    for it in range(1_000_000):
        if len(set(selected_indices)) >= MAX_MODELS:
            print(f"Reached MAX_MODELS={MAX_MODELS}")
            break

        base = best_ensemble[:, None] * (1.0 - ww)[None, :]

        best_it_score = -1.0
        best_it_idx   = -1
        best_it_w     = None
        best_it_ens   = None

        for k in range(K):
            cand = base + x_gpu[:, k][:, None] * ww[None, :]
            aucs = multiple_roc_auc_scores(y_gpu, cand)
            j = int(cp.argmax(aucs).item())
            s = float(aucs[j].item())
            if s > best_it_score:
                best_it_score = s
                best_it_idx   = k
                best_it_w     = float(ww[j].item())
                best_it_ens   = cand[:, j]
            del cand, aucs

        improve = best_it_score - best_score
        if improve < TOL:
            print(f"Stopped: improvement {improve:.8f} < TOL={TOL}")
            break

        print(f"Iter {it:3d}  AUC {best_it_score:.6f}"
              f" | add {names[best_it_idx]:<30s} | w = {best_it_w:+.3f}")

        selected_indices.append(best_it_idx)
        weights_step.append(best_it_w)
        best_ensemble = best_it_ens
        best_score    = best_it_score

        del base
        gc.collect()
        cp._default_memory_pool.free_all_blocks()

    final_oof = best_ensemble.get()
    final_auc = auc_cpu(y_true, final_oof)
    print(f"\nFinal OOF AUC (exact CPU): {final_auc:.6f}")

    wgt = np.array([1.0], dtype=np.float64)
    for w in weights_step:
        wgt = wgt * (1.0 - w)
        wgt = np.append(wgt, w)

    dfw = (
        pd.DataFrame({"model": [names[i] for i in selected_indices], "weight": wgt})
        .groupby("model", as_index=False)["weight"].sum()
        .sort_values("weight", ascending=False, key=abs)
        .reset_index(drop=True)
    )

    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE")
    print(f"{'='*50}")
    print(f"Models used: {len(dfw)}/{K}")
    print(f"Final score: {final_auc:.6f}")
    print(f"\nWeights:\n{dfw.to_string(index=False)}")
    print(f"\nWeight sum: {dfw['weight'].sum():.6f}")

    return final_oof, dfw, final_auc


def apply_weights_to_test(test_dict, names, dfw):
    K_test = len(next(iter(test_dict.values())))
    x_test = np.column_stack([test_dict[n] for n in names])
    blend  = np.zeros(K_test, dtype=np.float64)
    for _, row in dfw.iterrows():
        k = names.index(row["model"])
        blend += x_test[:, k] * float(row["weight"])
    return blend


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Hill Climbing: v4+++ + TabTransformer (18 models)")
    print("=" * 70)

    print("\nLoading predictions...")
    oof_dict, test_dict, y_true, oof_ids, test_ids = load_predictions()

    common_models = sorted(set(oof_dict.keys()) & set(test_dict.keys()))
    oof_dict  = {k: oof_dict[k]  for k in common_models}
    test_dict = {k: test_dict[k] for k in common_models}
    print(f"Models with OOF + test: {len(common_models)}")

    print("\n" + "=" * 70)
    print("Running Hill Climbing...")
    print("=" * 70)

    oof_blend, dfw, final_auc = hill_climbing_ensemble(oof_dict, y_true)

    print("\nApplying weights to test predictions...")
    names      = list(oof_dict.keys())
    test_blend = apply_weights_to_test(test_dict, names, dfw)

    pd.DataFrame({
        "id": oof_ids, "prob": oof_blend,
        "predicted": (oof_blend > 0.5).astype(int), "true": y_true,
    }).to_csv(OUTPUT_DIR / "oof.csv", index=False)

    pd.DataFrame({"id": test_ids, "prob": test_blend}).to_csv(OUTPUT_DIR / "test_proba.csv", index=False)
    pd.DataFrame({"id": test_ids, "Churn": test_blend}).to_csv(OUTPUT_DIR / "test.csv", index=False)
    dfw.to_csv(OUTPUT_DIR / "ensemble_weights.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}")
    print(f"Final OOF AUC: {final_auc:.6f}")
    print("Done!")


if __name__ == "__main__":
    main()
