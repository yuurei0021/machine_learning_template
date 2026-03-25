# 013: How Blending 5 Diverse Models Pushed My AUC Further Than Any Single Model!

- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679947
- **Author**: Sohail Khan (449th)
- **Upvotes**: 13
- **Notebook**: https://www.kaggle.com/code/sohailkhanlml/s6e3-xgb-xgb-opt-te-pairs-nn-lgbm (CV AUC: 0.918117)

## 概要

5つの構造的に異なるモデルを独立に学習し、4つのアンサンブル手法を比較した実験。

## 5つのモデル

1. **TE-Pair Logistic Regression**: 全C(n,2)ペアワイズ特徴量をTarget Encoding → logit^3変換 → LogReg
2. **XGBoost v1**: 浅い木 (depth=3)、ネイティブカテゴリカル
3. **EmbMLP Neural Net**: カテゴリカル埋め込み + LayerNorm MLP、cosine LR、smooth BCE
4. **LightGBM**: GPU高速化、leaf-wise
5. **XGBoost v2**: 深い木 (depth=5)、低LR、L1+L2正則化（v1と意図的に異なる設定）

全モデル同一の特徴量セット（charge ratios, service counts, contract risk flags, tenure bins）を使用。

## 4つのアンサンブル戦略

1. **Simple Mean**: 単純平均
2. **Rank Average**: 各モデルの予測をパーセンタイルランクに変換後平均。外れ値やキャリブレーション差に頑健
3. **Optimized Weights**: `scipy.optimize.minimize` (Nelder-Mead) でOOF予測のAUCを最大化する重み探索。データリークなし
4. **Level-2 Stacking**: OOF予測のlogit変換をLogRegメタモデルに入力（独自5-fold CV）

## 核心的知見

### アンサンブル貢献度 ≠ 単体AUC
- TE-pair LogRegはXGBoostより単体AUCが低いにも関わらず、最適化重みが**意外に高い**
- 理由: ツリーモデルとは**直交する**誤差パターンを持つため、相関が低い予測が全体を改善
- **多様性が個体性能より重要**

### なぜブレンディングが勝つか
- 構造的に異なるモデルは異なるミスをする
- 平均化で誤差が部分的に相殺される
- モデル誤差間の相関が1未満（常に成立）なら、アンサンブルの分散は必ず個体より低い

### Rank Averagingが最も安全なデフォルト
- 確率のキャリブレーション差に頑健
- Simple Meanと同等かそれ以上の性能

## 有効な特徴量

- `charges_per_tenure = MonthlyCharges / (tenure + 1)`
- `total_vs_expected = TotalCharges / (MonthlyCharges * tenure + 1)`
- `n_services = sum(全アクティブサービスフラグ)`
- `is_monthly = Contract == "Month-to-month"`
- `is_autopay = PaymentMethod contains "automatic"`

## 主要コメント

### Tilii (10th)
- 記述は妥当だがモデルの多様性はさらに増やせる
- 2つのツリーモデル(XGB v1, v2)は本質的に類似

### Sohail Khan (著者)
- 多様性の主張はやや誇張だったと認める
- XGB 2つはハイパーパラメータが違うだけで本質的に似ている

## 実験設計への示唆

1. **TE-Pair LogReg**はアンサンブルの多様性候補として有力（既存知見007と整合）
2. **Rank Averaging**をアンサンブルのデフォルト手法として使用すべき
3. **Optimized Weights** (Nelder-Mead on OOF) はHill Climbingの代替アプローチ
4. **多様なモデルタイプ**（ツリー系、NN系、線形モデル系）の組み合わせが鍵
5. 単体AUCが低くても誤差が直交していればアンサンブルへの貢献は大きい
