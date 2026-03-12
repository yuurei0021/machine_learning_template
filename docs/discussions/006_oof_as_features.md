# OOF predictions from another model as features
- **Author**: W-Bruno
- **Votes**: 12
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680573

## 内容

### テクニック
- あるモデルのOOF予測値を、別のモデルの特徴量として追加する（スタッキングの一種）
- スコア向上に貢献

### 実験への示唆
- スタッキング/メタラーニングは有効な手法
- まずベースモデルを複数作成し、そのOOF予測を特徴量として次段モデルに投入
