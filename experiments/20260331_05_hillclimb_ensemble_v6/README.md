# 20260331_05_hillclimb_ensemble_v6

## 目的
HC v5から `lgbm_optuna` (20260325_02) を除外してヒルクライミングを再実行。
lgbm_optunaはOptuna 50trials/5-foldでチューニングされており、fold過学習が疑われる。

## アプローチ
- HC v5と同一設定（rank01, GPU CuPy, 再選択許可, 負の重み許可）
- 候補モデル: 19モデル（v5の20モデルから lgbm_optuna を除外）
- データセット: `nyhalcyon/hc-v4-oof-predictions`（既存）

## 除外モデル
- `lgbm_optuna` (20260325_02_lgbm_optuna_tuning): OOF 0.91880, LB 0.91659
  - Optuna 50trials/5-fold CVでチューニング → fold過学習の疑い

## 実行環境
- Kaggle GPU (T4)
- カーネルID: nyhalcyon/hill-climbing-ensemble-v6

## 結果
- OOF AUC: **0.919622**
- LB: **0.91712**
- 選択モデル数: 12/19

### 重み構成
| モデル | 重み |
|---|---|
| stacking_lgbm_optuna | +0.4714 |
| ridge_lgbm_reproduce | +0.2870 |
| stacking_xgb | -0.2576 |
| xgb_optuna_20fold | +0.2150 |
| realmlp_orig_ref | +0.2070 |
| tabm_nb1feat_10fold | +0.1185 |
| xgb_depth1 | -0.0630 |
| mlp_orig_ref | -0.0612 |
| gnn_starter | +0.0425 |
| xgb_baseline | +0.0399 |
| ydf_orig_ref | +0.0200 |
| logreg_orig_ref | -0.0195 |

### 比較
| バージョン | OOF AUC | LB |
|---|---|---|
| HC v4+++ | 0.91962 | **0.91715** |
| HC v5 | 0.91963 | 0.91713 |
| HC v6 | 0.91962 | 0.91712 |

## 考察
- lgbm_optuna除外によるLB改善は見られなかった（0.91713→0.91712）
- stackingモデル自体のOOFがヒルクライミングのOOF過適合を招いている可能性が高い
- HC v4+++（stackingモデルなし）がLBベスト (0.91715) を維持
