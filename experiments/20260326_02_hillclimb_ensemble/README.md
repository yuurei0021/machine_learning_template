# 20260326_02_hillclimb_ensemble

## 目的
全18モデルのOOF予測を用いたヒルクライミングアンサンブルで、LBスコアの最大化を図る。
Discussion知見（Chris Deotte）で「特徴量交互作用が少ないデータではHill Climbing > Stacking」とされており、SLSQPアンサンブル（実験01: LB 0.91630）からの改善を狙う。

## アプローチ
- **手法**: Greedy Hill Climbing（負の重みも許可）
- **初期モデル**: 単体AUC最高のRealMLP (0.918949) から開始
- **探索**: weight_step=0.01、範囲[-0.5, 1.0)で全候補モデル×全重みを探索
- **停止条件**: 改善量 < 0.000001
- **候補モデル**: 18モデル（ツリー系7、NN系4、線形系4、Bartz1、YDF1、depth1系1）

## 結果

### スコア
| 指標 | 値 |
|---|---|
| OOF AUC-ROC | **0.919391** |
| LB AUC-ROC | **0.91703** |

### 選択されたモデルと重み（8/18モデル）

| モデル | 重み | 単体AUC | タイプ |
|---|---|---|---|
| realmlp_orig_ref | +0.4624 | 0.918949 | NN |
| xgb_optuna | +0.4443 | 0.918931 | ツリー |
| tabm_orig_ref | +0.1600 | 0.918544 | NN |
| catboost_orig_ref | +0.0535 | 0.918530 | ツリー |
| xgb_baseline | +0.0315 | 0.916397 | ツリー |
| mlp_orig_ref | -0.0509 | 0.917203 | NN |
| xgb_depth1 | -0.0508 | 0.913454 | ツリー |
| xgb_orig_ref | -0.0500 | 0.918528 | ツリー |

### 反復履歴

| Iteration | 追加モデル | 重み | スコア | 改善量 |
|---|---|---|---|---|
| 0 | realmlp_orig_ref | 1.00 | 0.918949 | (初期) |
| 1 | xgb_optuna | 0.49 | 0.919345 | +0.000396 |
| 2 | tabm_orig_ref | 0.15 | 0.919369 | +0.000024 |
| 3 | xgb_depth1 | -0.05 | 0.919380 | +0.000011 |
| 4 | catboost_orig_ref | 0.05 | 0.919383 | +0.000003 |
| 5 | mlp_orig_ref | -0.05 | 0.919388 | +0.000005 |
| 6 | xgb_baseline | 0.03 | 0.919390 | +0.000002 |
| 7 | xgb_orig_ref | -0.05 | 0.919391 | +0.000002 |

### 他手法との比較

| 手法 | OOF AUC | LB |
|---|---|---|
| **Hill Climbing (本実験)** | **0.91939** | **0.91703** |
| TabM単体 | 0.91854 | 0.91657 |
| XGBoost orig_ref | 0.91853 | 0.91644 |
| SLSQP ensemble (18モデル) | 0.91883 | 0.91630 |

### 考察
- NN系（RealMLP + TabM）とツリー系（XGBoost Optuna）が主要な構成要素で、約46:44:16の配分
- 負の重み（mlp, xgb_depth1, xgb_orig_ref）が微小改善に寄与 — 誤差パターンの直交性を活用
- LGBM系、LogReg系、Ridge系、Bartz、YDFは選択されず — ツリー系の中ではXGBoostが多様性に優れる
- SLSQPは全18モデルに分散させた結果OOFに過適合し、LBで劣化。Hill Climbingの貪欲選択のほうがこのデータに適合

## 次のステップ
- Rank Averagingとの比較
- 上位モデルのハイパーパラメータチューニングによる単体性能向上 → 再アンサンブル
- 10-fold CVでのTabM/RealMLPの安定性向上
