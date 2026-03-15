# GNN Starter - Graph Neural Network - CV 0.9155
- **Author**: Chris Deotte (1st)
- **Votes**: 26
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681020

## 内容

### GNN（Graph Neural Network）の概要
- CNN: 隣接列の情報を集約 → GNN: 「隣接行」の情報を集約
- ノード = 行（顧客）、エッジ = 類似度ベース
- エッジ定義: Nvidia cuML KNNによる行類似度

### 結果
- CV 0.9155
- Hill climbing でXGB + LR + MLP + GNNのアンサンブルに選択される → **多様性が高い**

### エッジ定義のバリエーション
- KNN類似度（デフォルト）
- 元データの最近傍レコードをアンカーとして接続（Don Maniの提案）
- 異なるエッジ定義で複数GNNを訓練可能

### 実験への示唆
- GNNはアンサンブルの多様性に貢献（GBDTとは全く異なるアプローチ）
- OOF予測を他モデルの入力特徴量として使用可能
- GPU（cuML）が必要
