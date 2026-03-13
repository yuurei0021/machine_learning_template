# 20260313_06_bartz_baseline

## 目的
Bartz（Bayesian Additive Regression Trees）によるベースライン構築。GBDTとは異なるMCMCベースの手法で多様性のあるアンサンブル候補を作成。

## アプローチ
- Target Encoding（CV-basedスムージング、リーク防止）
- Pairwise interaction features（16カテゴリ変数の全組み合わせ → 120交互作用特徴量）
- 合計140特徴量
- Bartz params: `maxdepth=6, ntree=400, k=5, sigdf=3, sigquant=0.9`
- MCMC: NDPOST=1000, NSKIP=200, KEEPEVERY=2（合計2200イテレーション/fold）
- 5-Fold Stratified CV
- 予測値はMCMC後方サンプルの平均 → clip(1e-7, 1-1e-7)

## 結果

### OOF Metrics
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9158** |
| LogLoss | 0.3051 |
| Accuracy | 0.8615 |

### Fold-wise AUC
| Fold | AUC | LogLoss | Accuracy |
|------|-----|---------|----------|
| 1 | 0.9154 | 0.3048 | 0.8610 |
| 2 | 0.9164 | 0.3037 | 0.8621 |
| 3 | 0.9156 | 0.3054 | 0.8616 |
| 4 | 0.9171 | 0.3031 | 0.8626 |
| 5 | 0.9144 | 0.3083 | 0.8601 |

### 備考
- 初回実行時、MCMC予測値が[0,1]範囲外（max=1.154）となりlog_lossでエラー → clipで対処
- 実行時間: 約2時間（CPU、5 fold × ~20分/fold）
- GBDTモデル（XGBoost/LightGBM）とほぼ同等のAUC

## 全モデル比較
| Model | OOF AUC-ROC | Public LB AUC |
|-------|-------------|---------------|
| Logistic Regression | 0.9079 | 0.90504 |
| XGBoost (depth=1) | 0.9135 | 0.91039 |
| **Bartz (this)** | **0.9158** | TBD |
| LightGBM | 0.9163 | 0.91378 |
| XGBoost (depth=6) | 0.9164 | 0.91391 |

## 次のステップ
- アンサンブル（LR + XGB + LGB + Bartz）
- CatBoost実験
