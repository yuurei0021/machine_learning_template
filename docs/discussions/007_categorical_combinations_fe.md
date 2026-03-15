# I suggest categorical rather than mathematical combinations for FE
- **Author**: Tilii (45th)
- **Votes**: 12
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/681085

## 内容

### 遺伝的プログラミング(GP)による特徴量探索
- 元データで10,000世代GPを実行 → 発見された特徴量はほぼ既存特徴量のコピーか単純な変換
- 合成データでも1,000世代で同様の結果
- **数学的な特徴量組み合わせ（ratio, product等）はほぼ無意味**

### 推奨: カテゴリカル特徴量の組み合わせ
- 数学的FEよりも**カテゴリカル変数の組み合わせ**（ペアワイズ等）を推奨
- NNの場合: Embedding
- GBMの場合: Target Encoding

### 実験への示唆
- 特徴量の独立性を再確認（GP でも交互作用が見つからない）
- カテゴリカル組み合わせ + Target Encoding が有効な方向性
- ellynパッケージ: `g=2000以上, selection='lexicase', islands=True, num_islands=20`
