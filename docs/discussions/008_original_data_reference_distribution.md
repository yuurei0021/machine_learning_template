# Using Original Data as a Reference Distribution + Single XGB
- **Author**: Mehmet Özer (147th)
- **Votes**: 1
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680877
- **Notebook**: https://www.kaggle.com/code/ozermehmet/original-data-fe-single-xgb-cv-0-919-lb-0-916

## 結果
- **CV 0.91902 | LB 0.91671**（単一XGBoostモデル）

## 核心アイデア
元データ（IBM Telco 7,043行）を訓練データに結合するのではなく、**参照分布として特徴量生成に活用**

## 9つの特徴量グループ（129基本特徴量 → 186モデル特徴量）

### 1. Frequency Encoding (3特徴量)
- tenure, MonthlyCharges, TotalChargesの値頻度（train+orig+test全体）

### 2. Arithmetic Interactions (6特徴量)
- charges_deviation = TotalCharges - tenure * MonthlyCharges
- monthly_to_total_ratio, avg_monthly_charges
- is_first_month, dev_is_zero, dev_sign
- **重要発見**: deviation=0のChurn率は65.9%（trainの4.5%、99.4%がtenure=1）

### 3. Service Counts (3特徴量)
- 契約サービス数合計、has_internet、has_phone

### 4. ORIG_proba single (15特徴量)
- 各カテゴリ/数値カラムの値を元データのChurn率にマッピング
- 元データの7k行を使うため**リーク無し**

### 5. ORIG_proba cross (10特徴量)
- 上位5カテゴリの2変数交互作用のChurn率（元データから算出）
- **Contract × InternetService の単一特徴量AUC: 0.859**

### 6. Distribution Features (16特徴量)
- 元データのChurner/Non-churner分布に対するpercentile rank、z-score
- 条件付きpercentile rank（例: 同じInternetService内でのTotalChargesランク）

### 7. Quantile Distance Features (18特徴量)
- Q25/Q50/Q75への距離（Churner分布 vs Non-churner分布）
- Gap特徴量: Non-churner距離 - Churner距離

### 8. Digit/Modular Features (17特徴量)
- tenure_mod12（**mod12=1のChurn率41% vs 他9%**）
- tenure_mod10, tenure_years, tenure_months_in_year
- mc_fractional, mc_rounded_10, tc_fractional等

### 9. Num-as-cat + N-grams (19特徴量)
- 数値カラムをカテゴリ文字列として扱いTarget Encoding
- 上位6カテゴリのBigram(15)、上位4カテゴリのTrigram(4)

## Target Encoding戦略
- Nested CV: 外側10-fold、内側5-fold
- sklearn TargetEncoderでmean、追加でstd/min/max
- 4エンコーディング × 19カテゴリ = 76追加特徴量

## XGBoostパラメータ（Optuna最適化済み）
- n_estimators=50000, lr=0.0063, max_depth=5
- subsample=0.81, colsample_bytree=0.32, min_child_weight=6
- reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790
- early_stopping_rounds=500, 10-fold CV

## 効果があったもの
- ORIG_proba特徴量（交互作用で単一特徴量AUC 0.859）
- Distribution特徴量（pctrank_ch_T単独AUC 0.794）
- Digit artifacts（tenure_mod12）
- Nested Target Encoding (std/min/max/mean)
- 10-fold CV（5-foldより低分散）

## 効果がなかったもの
- 元データを訓練データに結合（-0.00078 OOF）
- 追加のratio/zscore特徴量（高相関、新情報なし）

## 実験への示唆
- **元データを参照分布として使う**ことで大幅なスコア向上が可能
- tenure_mod12のデジタルアーティファクトは強力な特徴量
- 元データ単純結合は逆効果、参照マッピングが正解
