# 20260331_12_hc_v4ttt_knn_svm

## 目的
v4+++TT (18モデル) にKNN (2モデル) + SVM (1モデル) を追加し、アンサンブル多様性の改善を検証。

## アプローチ
- **ベース**: v4+++TT (18モデル) と同一のHill Climbingコード
- **追加モデル**:
  - knn_baseline (OOF AUC 0.900, 生値30特徴量)
  - knn_orig_ref (OOF AUC 0.910, 201特徴量)
  - svm_orig_ref (OOF AUC 0.916, Nystroem RBF + SGD)
- **合計**: 21モデル
- **実行**: Kaggle GPU (T4), rank01正規化, 負の重み許可

## 結果

- **OOF AUC: 0.919623** (v4+++TTの0.91962とほぼ同等)
- **選択モデル: 11/21** (KNN/SVMは未選択)

| モデル | 重み | 単体OOF AUC | タイプ |
|---|---|---|---|
| ridge_xgb_reproduce | +0.3336 | 0.9192 | Ridge→XGB |
| realmlp_orig_ref | +0.3265 | 0.9189 | NN |
| tabm_nb1feat_10fold | +0.2193 | 0.9190 | NN |
| ridge_lgbm_reproduce | +0.1020 | 0.9191 | Ridge→LGB |
| mlp_orig_ref | -0.0906 | 0.9172 | NN |
| xgb_depth1 | -0.0649 | 0.9135 | XGB |
| tabtransformer | +0.0578 | 0.9166 | Transformer |
| catboost_orig_ref | +0.0475 | 0.9185 | CatBoost |
| ydf_orig_ref | +0.0294 | 0.9174 | YDF |
| gnn_starter | +0.0200 | 0.9154 | GNN |
| xgb_baseline | +0.0194 | 0.9164 | XGB |

## 考察
- KNN (AUC 0.900, 0.910) と SVM (AUC 0.916) はいずれもHill Climbingで選択されなかった
- v4+++TT構成と完全に同一の11モデルが選択 → KNN/SVMの誤差パターンは既存モデルの組み合わせでカバーされている
- KNNは単体AUCが低すぎ (0.900-0.910) でアンサンブル候補に入れなかった可能性
- SVMはAUC 0.916だが、Nystroem近似のためカーネルSVMとしての多様性が不十分だった可能性

## 次のステップ
- KNN/SVMの改善（特徴量追加、ハイパラチューニング）よりも、他の多様なモデル（FT-Transformer, TabNet等）を検討すべき
