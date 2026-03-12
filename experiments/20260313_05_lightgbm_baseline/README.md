# 20260313_05_lightgbm_baseline

## 目的
LightGBMによるベースライン構築（通常パラメータ）。XGBoostとの比較。

## アプローチ
- Label Encoding + Charge_Difference = 20特徴量
- categorical_featureをLightGBMに指定（ネイティブカテゴリ処理）
- LightGBM (gbdt, num_leaves=31, lr=0.05, subsample=0.8, colsample=0.8)
- 5-Fold Stratified CV, early_stopping_rounds=50, max 2000 rounds

## 結果

### OOF Metrics
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9163** |
| LogLoss | 0.2978 |
| Accuracy | 0.8615 |

### Fold-wise AUC
| Fold | AUC | LogLoss | Accuracy | Best Iter |
|------|-----|---------|----------|-----------|
| 1 | 0.9161 | 0.2984 | 0.8610 | 724 |
| 2 | 0.9170 | 0.2965 | 0.8616 | 648 |
| 3 | 0.9163 | 0.2979 | 0.8616 | 575 |
| 4 | 0.9176 | 0.2957 | 0.8626 | 713 |
| 5 | 0.9147 | 0.3003 | 0.8608 | 594 |

## 全モデル比較
| Model | OOF AUC-ROC | Public LB AUC |
|-------|-------------|---------------|
| Logistic Regression | 0.9079 | 0.90504 |
| XGBoost (depth=1) | 0.9135 | 0.91039 |
| XGBoost (depth=6) | 0.9164 | 0.91391 |
| **LightGBM (this)** | **0.9163** | TBD |

## 次のステップ
- アンサンブル（LR + XGB + LGB）
- CatBoost実験
