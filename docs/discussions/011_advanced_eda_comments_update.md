# Advanced EDA Technique - コメント追加知見
- **Author**: Chris Deotte (1st)
- **Votes**: 69（更新時点）
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/680290
- **既存ファイル**: 001_advanced_eda_technique.md（メインポストの内容）

## コメントからの追加知見

### Chris Deotte: Stacking vs Hill Climbing
- **Stacking**: 特徴量間の交互作用が多い場合に最も効果的
- **Hill Climbing**: 交互作用が少ない場合に最も効果的
- このデータセットは交互作用が少ない → **Hill Climbing推奨**

### Chris Deotte: 決定木の限界（重要）
- 決定木はGreedyアルゴリズム → A×B交互作用を見つけるには、まずAだけでパターンを見つける必要がある
- A単独にもB単独にもパターンがない場合、木はその交互作用を発見できない
- **FE（A*B, A+B等の明示的作成）が必要な理由**
- NNは非Greedyなので異なるパターンを発見可能 → モデル多様性の根拠

### Chris Deotte: ハイパラチューニング哲学
- GBDTのハイパラ探索に最も少ない時間を使う
- max_depthとcolsample_by_treeを早期に設定、ほぼ再調整しない
- 特徴量追加時: max_depth ±1〜2調整、colsample_by_tree を下げる
- **FEとモデル多様性にほとんどの時間を投資**

### Chris Deotte: 数値特徴量のビニング手法
1. **ドメイン知識**: 自然な区分（例: 年齢→新生児/幼児/子供等）
2. **EDAベース**: 特徴量(x軸) vs ターゲット平均(y軸)をプロット、挙動が変わる点(不連続点)でビン作成
3. **試行錯誤**: 様々なビンサイズでCVスコアを比較

### Traiko Dinev (4th): max_depth=1 + max_leaves=1024
- 深さ1でmax_leavesを増やす実験 → ただしdepth=1では2葉しかないため効果なし

### Alexander Biller: Optuna 200-trial結果
- 合成データでは一貫してmax_depth=6-7が最適
- Chris回答: 元データではdepth=1最適（real signalのみ）、合成データではdepth高め（real+fake signal）

## 実験への示唆
- **Hill Climbing**アンサンブルを優先（stacking より有効な可能性）
- **明示的FE**（カテゴリカル組み合わせ、数値ビニング）が木モデルでは重要
- 数値特徴量のEDAベースビニングを検討（tenure, MonthlyCharges等）
- 多特徴量時はcolsample_by_treeを下げる（0.2〜0.5）
