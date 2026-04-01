# 20260327_09_hillclimb_ensemble_v3

## 目的
v2 (LB 0.91712) に新モデル（Logit3 TE-Pair LogReg, ExtraTrees, XGB Optuna 20-fold）を追加しつつ、低精度・重複モデルを除外してHill Climbingを効率化する。

## アプローチ
- **手法**: Greedy Hill Climbing（v1/v2と同一パラメータ）
- **候補モデル**: 22→13に絞り込み
  - **追加(3)**: logit3_te_pair, extratrees_orig_ref, xgb_optuna_20fold
  - **除外(12)**: 同系統の下位互換（xgb_orig_ref→xgb_optuna_20fold等）、低AUC（GNB 0.854/0.887）、アンサンブルOOF、重複（logreg_baseline×2）

### 除外判断基準
1. 同一モデル系統で上位版が存在 → 下位を除外
2. OOF AUC < 0.91 → 除外（GNB, ridge_baseline）
3. 完全重複 → 1つに集約

## 結果

### スコア
| 指標 | v3 | v2 | 差分 |
|---|---|---|---|
| OOF AUC-ROC | 0.91955 | 0.91955 | 0.00000 |
| LB AUC-ROC | **0.91711** | **0.91712** | **-0.00001** |
| モデル数 | 6/13 | 7/22 | |

### 選択されたモデルと重み（6/13モデル）

| モデル | 重み | 単体AUC | v2比較 |
|---|---|---|---|
| ridge_xgb_reproduce | +0.5283 | 0.91922 | v2: +0.58 |
| realmlp_orig_ref | +0.3825 | 0.91895 | v2: +0.42 |
| tabm_orig_ref | +0.1042 | 0.91854 | v2: +0.12 |
| xgb_optuna_20fold | +0.0800 | 0.91913 | **NEW** |
| mlp_orig_ref | -0.0675 | 0.91720 | v2: -0.07 |
| xgb_depth1 | -0.0276 | 0.91345 | v2: -0.03 |

### 選択されなかったモデル（7/13）
| モデル | 単体AUC | 不選択理由の推定 |
|---|---|---|
| logit3_te_pair | 0.91595 | 既存LogReg系の誤差と類似 |
| extratrees_orig_ref | 0.91484 | ツリー系の情報が主軸XGBで既にカバー |
| logreg_orig_ref | 0.91579 | v2のcatboostの代替にならず |
| lgbm_optuna | 0.91880 | XGB系と誤差パターンが類似 |
| catboost_orig_ref | 0.91853 | xgb_optuna_20foldに置換 |
| bartz_baseline | 0.91578 | 改善量が閾値未満 |
| xgb_baseline | 0.91640 | 上位XGB 2つで情報カバー |

### 考察
- **v2とほぼ同一のスコア**: 候補を絞っても結果は変わらず、コアメンバー（Ridge→XGB, RealMLP, TabM）の安定性を確認
- **新モデルの貢献は限定的**: Logit3 (0.916), ExtraTrees (0.915) は主軸モデル群 (0.919+) と十分な誤差直交性を持たなかった
- **xgb_optuna_20foldがcatboost_orig_refに代わって選択**: 20-fold OOFの品質がcatboostより有効
- **アンサンブル改善の天井**: 現在のモデル多様性ではOOF 0.9196付近が限界。さらなる改善には質的に異なるアプローチ（GNN、Pseudo Labels等）が必要

## 次のステップ
- GNNの実装（Chris Deotte手法、グラフベースで質的に異なる）
- 主軸モデル（Ridge→XGB, RealMLP）の20-fold化による個別精度向上
- Pseudo Labels（高確信度テスト予測を訓練に追加）
