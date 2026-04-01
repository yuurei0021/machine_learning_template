# 20260401_01_hc_v4ttt_dart

## 目的
v4+++TT (18モデル) にRidge→LGB DART (1モデル) を追加し、アンサンブル多様性の改善を検証。

## アプローチ
- **ベース**: v4+++TT (18モデル) と同一のHill Climbingコード
- **追加モデル**: ridge_lgbm_dart (OOF AUC 0.91728, boosting_type=dart)
- **合計**: 19モデル
- **実行**: Kaggle GPU (T4), rank01正規化, 負の重み許可

## 結果

- **OOF AUC: 0.919640** (v4+++TTの0.91962から **+0.00002** 改善)
- **選択モデル: 12/19** (DARTが負の重みで選択！)

| モデル | 重み | 単体OOF AUC | タイプ |
|---|---|---|---|
| realmlp_orig_ref | +0.3278 | 0.9189 | NN |
| ridge_xgb_reproduce | +0.2982 | 0.9192 | Ridge→XGB |
| tabm_nb1feat_10fold | +0.2201 | 0.9190 | NN |
| **ridge_lgbm_dart** | **-0.1722** | **0.9173** | **DART (NEW)** |
| ridge_lgbm_reproduce | +0.1658 | 0.9191 | Ridge→LGB |
| tabtransformer | +0.0661 | 0.9166 | Transformer |
| xgb_optuna_20fold | +0.0618 | 0.9191 | XGB |
| catboost_orig_ref | +0.0420 | 0.9185 | CatBoost |
| mlp_orig_ref | -0.0300 | 0.9172 | NN |
| extratrees_orig_ref | +0.0207 | 0.9148 | ExtraTrees |
| xgb_depth1 | -0.0203 | 0.9135 | XGB |
| gnn_starter | +0.0202 | 0.9154 | GNN |

## 考察
- DARTがw=-0.172で4位の重みで選択。gbdt版 (ridge_lgbm_reproduce) とは反対の符号（+0.166 vs -0.172）
- gbdt版との差分が多様性として機能: gbdtが過学習するパターンをDARTのDropout正則化が補正
- extratrees_orig_ref (+0.021) が新たに選択 → DART追加により他モデルの選択も変化
- v4+++TTでは11モデルだったが12モデルに増加 → 多様性が向上

## 次のステップ
- LBスコアを確認してsubmit検討
