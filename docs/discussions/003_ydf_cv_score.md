# YDF gives pretty good CV score off default parameters
- **Author**: broccoli beef
- **Votes**: 27
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/679983

## 内容

### YDF (Yggdrasil Decision Forests)
- デフォルトに近いパラメータで **CV 0.91800 ± 0.00058** を達成
- 重要なパラメータ: `max_depth=2`, `growing_strategy='BEST_FIRST_GLOBAL'`, `categorical_algorithm='RANDOM'`
- 低い max_depth が効くのは Chris Deotte の発見（特徴量が独立）と一致

### パラメータ対応表（YDF → XGB/LGB）
- `shrinkage` → `learning_rate`
- `early_stopping_num_trees_look_ahead` → `early_stopping_rounds`
- `num_trees` → `n_estimators`
- `BEST_FIRST_GLOBAL` → `lossguide`

### 実験への示唆
- 多様なモデルの候補として YDF も検討する価値あり
- `max_depth=2` でも高いスコアが出る → 浅い木が有効
