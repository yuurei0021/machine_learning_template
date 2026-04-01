# 20260331_10_knn_orig_data_reference

## 目的
201特徴量（元データ参照FE + Nested TE）でのKNN実装。生値版（実験09: AUC 0.900）からの改善幅を確認。

## アプローチ
- **特徴量**: 実験01(orig_data_reference)と同一の9グループFE + Nested TE
  - Groups 1-8: Freq Encoding, Arithmetic, Service Counts, ORIG_proba single/cross, Distribution, Quantile Distance, Digit/Modular
  - Group 9: N-gram (bigram/trigram/num-as-cat) + Nested Target Encoding (3 layers)
  - カテゴリカルはOHE（距離ベースモデル用、Label Encodingから変換）
  - 最終特徴量数: 212（OHE展開のため201より増加）
- **前処理**: StandardScaler（fold毎にfit）
- **モデル**: KNeighborsClassifier (k=51, weights="distance", Euclidean)
- **CV**: 5-Fold Stratified CV
- **テスト予測**: 5-fold平均（TE含むためfold毎にrefit）

## 結果

| Fold | AUC | LogLoss | Accuracy |
|------|------|---------|----------|
| 1 | 0.9095 | 0.3667 | 0.8571 |
| 2 | 0.9115 | 0.3570 | 0.8585 |
| 3 | 0.9099 | 0.3701 | 0.8567 |
| 4 | 0.9113 | 0.3650 | 0.8584 |
| 5 | 0.9081 | 0.3702 | 0.8570 |
| **OOF** | **0.9100** | **0.3658** | **0.8576** |

## 考察
- 生値版 (0.900) から **+0.010** の改善。201特徴量FEの効果を確認
- ただしツリー系での同等改善 (+0.002) より大きい → KNNはFE品質に敏感
- GNN (0.915) にはまだ及ばないが、インスタンスベースの独自の誤差パターンでアンサンブル貢献が期待できる

## 次のステップ
- Hill Climbingアンサンブルに追加して貢献度を検証
- k値のチューニング（現在k=51は暫定値）
