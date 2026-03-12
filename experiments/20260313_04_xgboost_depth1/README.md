# 20260313_04_xgboost_depth1

## 目的
max_depth=1でのXGBoost実験。Discussion知見（Chris Deotte）: 元データの特徴量が独立的なためdepth=1が最適。合成データではdepth=3-4が有効とされるが、depth=1の効果を確認する。

## アプローチ
- 03_xgboost_baselineと同一の特徴量（Label Encoding + Charge_Difference = 20特徴量）
- XGBoost (hist, **max_depth=1**, lr=0.05, subsample=0.8, colsample=0.8)
- 5-Fold Stratified CV, early_stopping_rounds=50, max 2000 rounds

## 結果

### OOF Metrics
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9135** |
| LogLoss | 0.3025 |
| Accuracy | 0.8588 |

### Fold-wise AUC
| Fold | AUC | LogLoss | Accuracy | Best Iter |
|------|-----|---------|----------|-----------|
| 1 | 0.9130 | 0.3034 | 0.8576 | 1999 |
| 2 | 0.9142 | 0.3013 | 0.8593 | 1999 |
| 3 | 0.9137 | 0.3023 | 0.8589 | 1999 |
| 4 | 0.9145 | 0.3007 | 0.8598 | 1999 |
| 5 | 0.9119 | 0.3050 | 0.8584 | 1999 |

**注**: 全foldで2000ラウンド上限に到達。early stoppingが効かず、まだ学習が進む余地あり。

## 比較
| Model | max_depth | OOF AUC-ROC |
|-------|-----------|-------------|
| Logistic Regression | - | 0.9079 |
| XGBoost | 6 | **0.9164** |
| XGBoost | 1 | 0.9135 |

## 考察
- max_depth=1はmax_depth=6より**AUC -0.003**低い
- Discussion知見では「元データではdepth=1が最適」だが、合成データにはfake signalが含まれており、depth>1でそれを捉えられる
- 全foldで2000ラウンド上限到達 → ラウンド数を増やせばさらに改善する可能性

## 次のステップ
- LightGBMベースライン
- max_depth=3-4の実験（合成データでの最適値とされる）
- アンサンブル
