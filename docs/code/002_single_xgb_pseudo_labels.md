# 002: Single XGB, cuDF + Pseudo Labels | CV 0.91789

- **著者**: Traiko Dinev (include4eto)
- **URL**: https://www.kaggle.com/code/include4eto/single-xgb-cudf-pseudo-labels-cv-0-91789
- **CV**: 0.91789（単体XGBoost）
- **Votes**: 59

## アーキテクチャ: Single XGBoost + Pseudo Labels

1. 標準FE + Nested TE（5-fold outer, 5-fold inner）
2. XGBで初回学習、テスト予測を取得
3. 高確信度テスト予測（>0.999 or <0.001）をPseudo Labelsとして訓練データに追加
4. XGB再学習
5. **条件付き採用**: fold AUCが改善した場合のみmodel2を採用

## 当方パイプラインにない要素

### ORIG_std特徴量
```python
orig.groupby(col)['Churn'].agg(['mean', 'std'])
```
元データの各カテゴリ内Churn率の**標準偏差**を特徴量化。当方はmeanのみ（ORIG_proba）。stdはカテゴリ内のChurn率のばらつきを捉える。

### NUM_AS_CAT + TE
```python
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    df[f'CAT_{col}'] = df[col].astype(str)
```
数値列を文字列カテゴリに変換し、Target Encodingを適用。合成データの exact-value アーティファクト（同一値の行がChurn率を共有）を捉える。

### Pseudo Labels（条件付き）
- 閾値: 予測確率 > 0.999 または < 0.001
- 594k train + 254k test → 高確信度サンプル数千行が追加される可能性
- **fold AUCが改善した場合のみ採用** → ダウンサイドなし

### 全ペアワイズ組み合わせ
- 15カテゴリから C(15,2) = 105 ペアの組み合わせ特徴量
- 当方はORIG_PROBA_CROSS_CATSの上位5カテゴリ（10ペア）のみ

### TE_std（TargetEncodingの標準偏差）
- Inner fold TEでstdのみ計算（min/maxは不使用）
- sklearn TargetEncoder（smooth='auto'）と併用する2層TE構造

## XGBoostハイパーパラメータ

| パラメータ | NB3 | 当方Optuna |
|---|---|---|
| max_depth | 6 | 5 |
| learning_rate | 0.05 | 0.0048 |
| subsample | 0.8 | 0.95 |
| colsample_bytree | 0.8 | 0.27 |
| early_stopping | **1000** | 100 |

標準的なパラメータで特筆すべき点は少ない。

## 当方CVとの比較

| モデル | NB3 | 当方 | 差分 |
|---|---|---|---|
| XGBoost単体 | 0.91789 | **0.91893** | **+0.00104** |

当方が大幅に上回っている。FE量と Optuna チューニングの差。

## 実装への示唆

1. **ORIG_std**: 低コストで追加可能。mean + stdで情報量増加
2. **NUM_AS_CAT + TE**: 合成データ特有のアーティファクト活用。低コスト
3. **Pseudo Labels**: 条件付き採用で安全。XGB再学習のみで実装可能
4. **全ペアワイズ組み合わせ**: 105ペアは特徴量爆発のリスクあり。上位ペアに絞るのが安全
