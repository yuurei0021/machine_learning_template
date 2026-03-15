# Breaking 0.914+: Hunting the "Fake Signal" with Brute-Force Combinatorics & Ridge Stacking
- **Author**: Siddhu Jaykay (646th)
- **Votes**: 0
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681625

## 結果
- Local CV 0.9165+, Public LB 0.91407

## 4段階パイプライン

### 1. Brute-Force Feature Factory（25→500+列）
- 数値ビニング: pd.qcut（10分位数）
- 全2-way + 主要3-wayカテゴリカル交差
- GroupBy matrix: 各カテゴリグループごとのcharges mean/std
- 行方向統計: Total_Services_Subscribed等
- K-Means 5クラスタ + PCA 2成分

### 2. メモリ最適化
- float64/int64 → float32/int8でメモリ60%削減

### 3. GPU Optuna Tuning (TPE)
- 80/20固定split（K-Foldの代わり、60秒/trial）
- colsample_bytree上限0.5（ノイズ回避）
- max_depth最大10（3-way交差用）

### 4. 10-Fold Stack
- XGBoost + LightGBM + CatBoost → Ridge Regression meta
- XGBoostがアンサンブルの約50%

## Chris Deotteのコメント（重要）
> "The best way to extract the fake signal is to recognize that the synthetic data generation process makes multiple copies of the original dataset rows and adds noise. There are 7k original rows and train.csv is 600k rows, so each row has about 85 copies. Therefore you can try to find the original dataset rows to predict a given row, or look for similar copies of a given row when predicting."

## 実験への示唆
- Fake signalの本質: 元データの各行が約85コピー + ノイズ
- 元データの「類似行」を探すアプローチが有効
- colsample_bytree=0.2〜0.5が多特徴量時に有効
