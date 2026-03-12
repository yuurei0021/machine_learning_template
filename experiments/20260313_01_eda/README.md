# EDA Report: Predict Customer Churn

## 1. データ概要
- Train: 594,194 rows x 22 cols
- Test: 254,655 rows x 21 cols
- Missing values: train=0, test=0
- Churn rate: 22.52% (Yes=133,817, No=460,377)

## 2. 数値特徴量とChurnの相関 (Point-Biserial)
- tenure: r = -0.4185
- MonthlyCharges: r = +0.2730
- TotalCharges: r = -0.2184
- SeniorCitizen: r = +0.2364
- Charge_Difference: r = +0.0369

## 3. カテゴリ特徴量 - Churn率の差が大きい順
- **PaymentMethod** (spread=0.420): highest=Electronic check(0.489), lowest=Credit card (automatic)(0.069)
- **Contract** (spread=0.411): highest=Month-to-month(0.421), lowest=Two year(0.010)
- **InternetService** (spread=0.401): highest=Fiber optic(0.415), lowest=No(0.014)
- **OnlineSecurity** (spread=0.392): highest=No(0.406), lowest=No internet service(0.014)
- **TechSupport** (spread=0.387): highest=No(0.402), lowest=No internet service(0.014)
- **OnlineBackup** (spread=0.377): highest=No(0.391), lowest=No internet service(0.014)
- **DeviceProtection** (spread=0.366): highest=No(0.381), lowest=No internet service(0.014)
- **StreamingMovies** (spread=0.285): highest=No(0.299), lowest=No internet service(0.014)
- **StreamingTV** (spread=0.283): highest=No(0.297), lowest=No internet service(0.014)
- **PaperlessBilling** (spread=0.245): highest=Yes(0.319), lowest=No(0.075)
- **Dependents** (spread=0.219): highest=No(0.291), lowest=Yes(0.073)
- **Partner** (spread=0.191): highest=No(0.325), lowest=Yes(0.134)
- **MultipleLines** (spread=0.109): highest=Yes(0.277), lowest=No phone service(0.168)
- **PhoneService** (spread=0.061): highest=Yes(0.229), lowest=No(0.168)
- **gender** (spread=0.006): highest=Female(0.228), lowest=Male(0.222)

## 4. Charge_Difference アーティファクト
- Train: mean=-11.41, std=298.75
- Test: mean=-10.74, std=303.93
- Churnとの相関: +0.0369

## 5. Key Findings
1. **Contract** が最強のChurn予測因子（Month-to-month >> One year >> Two year）
2. **InternetService**: Fiber opticのChurn率がDSLより高い
3. **tenure**: 長期契約者ほどChurnしにくい（強い負の相関）
4. **MonthlyCharges**: 高額ほどChurnしやすい（正の相関）
5. **Charge_Difference**: 合成アーティファクトを捉え、Churnと相関あり
6. 欠損値なし（train/test共に）
7. train/testの分布は非常に類似（分布シフトなし）
8. **セキュリティ/サポート系サービス未加入者**のChurn率が高い

## 6. 生成された図
| # | ファイル名 | 内容 |
|---|----------|------|
| 01 | 01_target_distribution.png | ターゲット変数の分布 |
| 02 | 02_numeric_distributions.png | 数値特徴量の分布（train vs test） |
| 03 | 03_numeric_by_churn.png | 数値特徴量 x Churn |
| 04 | 04_categorical_churn_rates.png | カテゴリ特徴量ごとのChurn率 |
| 05 | 05_charge_difference_artifact.png | Charge_Differenceアーティファクト分析 |
| 06 | 06_correlation_matrix.png | 相関行列ヒートマップ |
| 07 | 07_train_vs_test.png | train vs test 分布比較 |
| 08 | 08_contract_tenure_churn.png | Contract x tenure x Churn クロス分析 |
| 09 | 09_internet_charges_churn.png | InternetService x MonthlyCharges x Churn |