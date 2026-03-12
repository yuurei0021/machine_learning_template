# EDA & the Synthetic Artifact in TotalCharges
- **Author**: DaylightH
- **Votes**: 13
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679407

## 内容

### Adversarial Validation 結果
- train vs test: **分布シフトなし** (AUC: 0.5112) → CVを信頼できる
- train vs 元データセット: **有意なドリフト** (AUC: 0.6628) → 合成データにアーティファクトあり

### TotalCharges のアーティファクト
- 元データでは `TotalCharges ≈ MonthlyCharges × tenure` の数学的関係が厳密に成立
- 合成データではこの関係が**崩れている**（合成生成器がこの制約を再現できなかった）
- AV モデルで最も重要な特徴量: `TotalCharges`, `MonthlyCharges`, `tenure`

### 推奨される特徴量エンジニアリング
```python
Charge_Difference = TotalCharges - (MonthlyCharges * tenure)
```
- この特徴量はモデルの Feature Importance で上位に表示される

### 元データの使用に関する注意
- 合成データと元データにはドリフトがあるため、**単純に結合して訓練するのは危険**
- 注意深い取り扱いが必要

### 参考ノートブック
- https://www.kaggle.com/code/daylighth/e6s3-eda-baseline
