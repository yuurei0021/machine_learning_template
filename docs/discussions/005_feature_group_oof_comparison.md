# Feature Group OOF Comparison
- **Author**: Kawthar ELTARR
- **Votes**: 10
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680713

## 内容

### アプローチ
- 異なる特徴量グループごとにCatBoostモデルを訓練し、OOFスコアを比較
- S6E2の1位ソリューション（masayakawamata）の特徴量エンジニアリングを参考

### 実験への示唆
- 特徴量グループごとのablation studyはFEの方向性を決める上で有効
- CatBoostはカテゴリ変数の扱いに優れており、このデータセットに適している

### 参考ノートブック
- https://www.kaggle.com/code/sakuno/customer-churn-fe-methods-comparison/notebook
- S6E2 1st place: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t
