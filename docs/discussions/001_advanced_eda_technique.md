# Advanced EDA Technique
- **Author**: Chris Deotte (1st place)
- **Votes**: 56
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680290

## 内容

### 発見: max_depth=1 が元データで最適
- 元データセットでXGBを様々な `max_depth` で訓練した結果、`max_depth=1` が最も良いパフォーマンス
- これは通常ではなく、**特徴量同士がほぼ独立**していることを示す

### 2つのシグナル
Playground コンペの合成データには2種類のシグナルが存在：
1. **Real signal**: 元データに由来する本物のパターン → `max_depth=1` を好む
2. **Fake signal**: 合成データ生成プロセスで生じるアーティファクト → より大きな `max_depth` を好む

### 実験への示唆
- 特徴量間の相互作用がほとんどないため、**ロジスティック回帰が強い**
- 特徴量の組み合わせ（interaction）を探るより、他のテクニックに注力すべき
- コメントでは `max_depth=3 or 4` が合成データでは最適との報告

### 参考ノートブック
- Basic EDA: https://www.kaggle.com/code/cdeotte/basic-eda-customer-churn/
- Logistic Regression + MLP starter: https://www.kaggle.com/code/cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178
