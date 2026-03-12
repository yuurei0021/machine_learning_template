# 20260313_02_logistic_regression

## 目的
ロジスティック回帰によるベースライン構築。Discussion知見によると特徴量間の相互作用がほぼないため、線形モデルが強いと期待される。

## アプローチ
- One-Hot Encoding（15カテゴリ → 27ダミー変数 + 4数値 = 31特徴量）
- 合成特徴量: `Charge_Difference = TotalCharges - MonthlyCharges * tenure`
- StandardScaler（fold毎にfit）
- LogisticRegression (C=1.0, solver=lbfgs, max_iter=1000)
- 5-Fold Stratified CV

## 結果

### OOF Metrics
| Metric | Score |
|--------|-------|
| **AUC-ROC** | **0.9079** |
| LogLoss | 0.3124 |
| Accuracy | 0.8545 |

### Fold-wise AUC
| Fold | AUC | LogLoss | Accuracy |
|------|-----|---------|----------|
| 1 | 0.9075 | 0.3129 | 0.8530 |
| 2 | 0.9089 | 0.3112 | 0.8551 |
| 3 | 0.9081 | 0.3121 | 0.8549 |
| 4 | 0.9091 | 0.3106 | 0.8557 |
| 5 | 0.9061 | 0.3151 | 0.8539 |

### 重要な特徴量（|係数|順）
1. tenure (1.81)
2. TotalCharges (0.81)
3. Contract_Two year (0.68)
4. InternetService_Fiber optic (0.45)
5. PaymentMethod_Electronic check (0.30)
6. Contract_One year (0.27)

### Confusion Matrix
- True Negative: 420,013 / True Positive: 87,729
- False Negative: 46,088 / False Positive: 40,364

## 次のステップ
- LightGBMベースラインとの比較
- アンサンブル候補としてOOF予測を保存済み
