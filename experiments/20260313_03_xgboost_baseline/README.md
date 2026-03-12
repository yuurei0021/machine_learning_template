# 20260313_03_xgboost_baseline

## 目的
XGBoostによるベースライン構築（通常パラメータ、max_depth=6）

## アプローチ
- Label Encoding（カテゴリ15列） + 数値4列 + Charge_Difference = 20特徴量
- XGBoost (tree_method=hist, max_depth=6, lr=0.05, subsample=0.8, colsample=0.8)
- 5-Fold Stratified CV, early_stopping_rounds=50, max 2000 rounds

## 結果

### OOF Metrics
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9164** |
| LogLoss | 0.2977 |
| Accuracy | 0.8615 |

### Fold-wise AUC
| Fold | AUC | LogLoss | Accuracy | Best Iter |
|------|-----|---------|----------|-----------|
| 1 | 0.9160 | 0.2984 | 0.8610 | 636 |
| 2 | 0.9170 | 0.2965 | 0.8610 | 599 |
| 3 | 0.9165 | 0.2976 | 0.8621 | 648 |
| 4 | 0.9176 | 0.2957 | 0.8630 | 587 |
| 5 | 0.9148 | 0.3001 | 0.8604 | 650 |

### 重要な特徴量（Gain順）
1. Contract (1014)
2. OnlineSecurity (312)
3. TechSupport (161)
4. InternetService (113)
5. tenure (57)

## 比較
| Model | OOF AUC-ROC |
|-------|-------------|
| Logistic Regression | 0.9079 |
| **XGBoost (this)** | **0.9164** |

## 次のステップ
- LightGBMベースラインとの比較
- max_depth=1実験（Discussion知見: 特徴量独立性を活かす）
- アンサンブル
