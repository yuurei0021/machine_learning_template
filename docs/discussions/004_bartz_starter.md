# Bartz Starter - CV 0.916401
- **Author**: Chris Deotte (1st place)
- **Votes**: 7
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680976

## 内容

### Bartz (Bayesian Additive Regression Trees)
- XGB/Cat/LGBMとは異なるアプローチ: 勾配ブースティングの代わりにベイズ事後サンプリング木を使用
- 多様なモデルとしてアンサンブルに貢献

### Chris Deotte のアンサンブル構成
1. XGBoost
2. Logistic Regression with Target Aggregation
3. PyTorch MLP
4. GNN (Graph Neural Network) - CV 0.9155
5. Bartz - CV 0.916401

これらを組み合わせてアンサンブルを構築し、1位を達成

### 参考ノートブック
- Bartz starter: https://www.kaggle.com/code/cdeotte/bartz-starter-with-hill-climbing-demo/notebook
- XGB + LR + MLP: https://www.kaggle.com/code/cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178
- GNN: https://www.kaggle.com/code/cdeotte/gnn-starter-cv-0-9155-with-hill-climbing-demo
