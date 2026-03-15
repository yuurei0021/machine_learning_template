# 20260315_01_ydf_baseline

## 目的
YDF Discussion知見のパラメータ（grow_policy='lossguide', max_depth=2）をXGBoostで再現。
YDFはWindows非対応のため、対応するXGBoostパラメータで同等の実験を実施。

## アプローチ
- Label Encoding + Charge_Difference = 20特徴量（実験03/04と同一）
- XGBoost with YDF-inspired params:
  - `grow_policy='lossguide'`（YDFの`BEST_FIRST_GLOBAL`に対応）
  - `max_depth=2`（浅い木、特徴量の独立性に基づく）
  - `max_leaves=0`（制限なし、max_depthで制約）
  - 他パラメータは実験03と同一（lr=0.05, subsample=0.8等）
- 5-Fold Stratified CV, early_stopping_rounds=50, max 2000 rounds

## 結果

### OOF Metrics
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9159** |
| LogLoss | 0.2985 |
| Accuracy | 0.8611 |

### Fold-wise AUC
| Fold | AUC | LogLoss | Accuracy | Best Iter |
|------|-----|---------|----------|-----------|
| 1 | 0.9155 | 0.2993 | 0.8604 | 1995 |
| 2 | 0.9165 | 0.2973 | 0.8609 | 1999 |
| 3 | 0.9161 | 0.2984 | 0.8614 | 1998 |
| 4 | 0.9170 | 0.2967 | 0.8623 | 1999 |
| 5 | 0.9144 | 0.3009 | 0.8606 | 1999 |

### 備考
- 全foldで2000ラウンド上限到達（early stopping未発動）
- max_depth=2で学習が遅く、ラウンド数増加で改善余地あり
- 実験04（depthwise, depth=1: 0.9135）より良好、実験03（depthwise, depth=6: 0.9164）よりやや低い

## 全モデル比較
| Model | OOF AUC-ROC | Public LB AUC |
|-------|-------------|---------------|
| Logistic Regression | 0.9079 | 0.90504 |
| XGBoost (depth=1) | 0.9135 | 0.91039 |
| Bartz | 0.9158 | 0.91405 |
| **XGB lossguide depth=2 (this)** | **0.9159** | **0.91311** |
| LightGBM | 0.9163 | 0.91378 |
| XGBoost (depth=6) | 0.9164 | 0.91391 |

## 次のステップ
- ラウンド数増加（4000〜）で収束確認
- アンサンブル候補として活用
- ハイパーパラメータチューニング（将来タスク）
