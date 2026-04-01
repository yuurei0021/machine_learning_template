# 20260331_06_tabtransformer

## 目的
Transformer系モデルによるアンサンブル多様性の確保。
ツリー系・MLP系とは異なるSelf-Attention構造でアンサンブル貢献を狙う。

## アプローチ
- **モデル**: TabTransformer (Keras/TensorFlow)
  - カテゴリカル埋め込み -> Multi-Head Self-Attention (2 blocks)
  - 数値特徴量 + TE-Pair Logit3特徴量 + Ridge予測値を結合
  - MLP Head (128-64) -> Sigmoid
- **特徴量**: 算術比率3 + Digit抽出1 + 量子化ビニング4 + 3-way交互作用1 + 数値カテゴリ化4 + ペアワイズTE Logit3
- **Ridge stacking**: 各foldで数値+TE特徴量からRidge予測値を生成し、TabTransformerの入力に追加
- **元データ**: 訓練foldに元データ(7,043行)を追加
- **CV**: 5-fold StratifiedKFold
- **EPOCHS**: 5 (EarlyStopping patience=15, ReduceLROnPlateau)

## 参照
- https://www.kaggle.com/code/include4eto/tabtransfomer-chatgpt-vibe-coding

## 実行環境
- Kaggle GPU (T4)
- カーネルID: nyhalcyon/tabtransformer-te-pair-logit3
- 依存: TensorFlow/Keras, cudf, cupy, cuml (RAPIDS)

## 結果
- OOF AUC: **0.916576**
- Fold別: 0.9180, 0.9169, 0.9163, 0.9176, 0.9170
- Ridge (中間) AUC: ~0.9137
- TEペア数: 300 (25 cat cols C(2))
- 実行時間: 約22分
- LB: (submit後記入)

## 考察
- OOF 0.9166はRealMLP (0.9190) やTabM (0.9190) より低い
- EPOCHS=5は少ない可能性あり（EarlyStopping patience=15だが5epoch固定）
- Transformer系アーキテクチャとしてアンサンブル多様性に貢献する可能性がある
