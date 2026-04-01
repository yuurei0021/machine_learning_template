# 20260327_03_model_comparison_summary

## 目的
全実験の精度を一覧化し、モデル・アンサンブル性能の全体像を把握する。

---

## 単体モデル一覧

### ツリー系モデル

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| 03 | XGBoost baseline | 基本20特徴量 | 5 | 0.91640 | 0.91391 | max_depth=6 |
| 04 | XGBoost depth1 | 基本20特徴量 | 5 | 0.91345 | 0.91039 | max_depth=1 |
| 05 | LightGBM baseline | 基本20特徴量 | 5 | 0.91633 | 0.91378 | |
| 06 | Bartz baseline | TE+pairwise 140特徴量 | 5 | 0.91578 | 0.91405 | MCMC |
| ydf_01 | XGBoost lossguide | 基本20特徴量 | 5 | 0.91589 | 0.91311 | depth=2 |
| 01 | XGBoost orig_ref | 201特徴量 | 5 | 0.91853 | 0.91644 | 元データ参照FE |
| 02 | XGBoost Optuna | 201特徴量 | 5 | 0.91893 | 0.91656 | Optuna 50trials |
| lgbm_01 | LightGBM orig_ref | 201特徴量 | 5 | 0.91844 | 0.91631 | |
| lgbm_02 | LightGBM Optuna | 201特徴量 | 5 | 0.91880 | 0.91659 | Optuna 50trials |
| cb_03 | CatBoost orig_ref | 201特徴量 | 5 | 0.91853 | 0.91640 | |
| 04_reproduce | Ridge→XGB (NB1再現) | NB1準拠 | 20 | 0.91922 | 0.91685 | |
| 05_adapted | Ridge→XGB (適応) | 201特徴量+cross | 5 | 0.91888 | 0.91665 | |
| 02_20fold | XGBoost Optuna 20-fold | 201特徴量 | 20 | 0.91913 | 未submit | fold数検証 |
| et_06 | ExtraTrees baseline | ~40特徴量 | 5 | 0.91229 | - | ランダム分割 |
| et_07 | ExtraTrees orig_ref | 186特徴量(TE後) | 5 | 0.91484 | - | |
| ydf_11 | YDF orig_ref | 107特徴量 | 5 | 0.91744 | - | Kaggle CPU, max_depth=2 |
| 10_fulldata | XGB全データ学習 | 201特徴量 | - | - | 0.91662 | n_est=14463, OOFなし |
| 28_01 | Ridge→LGB | NB1準拠 | 20 | 0.91914 | 0.91680 | LGBM Optuna params |
| 28_03 | Ridge→XGB (201特徴量) | 201特徴量+ridge | 20 | 0.91902 | 未submit | ridge追加で悪化 |
| 28_06 | Ridge→XGB (NB1+ridge) | NB1+ridge | 20 | 0.91917 | 未submit | Kaggle CPU |

### NN系モデル

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| realmlp_04 | RealMLP | 42特徴量 | 5 | 0.91895 | 0.91655 | pytabkit, n_ens=8 |
| mlp_09 | MLP (PyTorch) | 201特徴量 | 5 | 0.91720 | 0.91483 | 512-256-128 |
| tabm_10 | TabM (当方FE) | 112+15特徴量 | 5 | 0.91854 | 0.91657 | Kaggle T4 GPU |
| **30_01** | **TabM (NB1 FE)** | NB1準拠 | **10** | **0.91898** | 未submit | **参照NB完全再現** |

### グラフ系モデル

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| 30_02 | GNN (GraphSAGE) | 25特徴量 | 5 | 0.91536 | 0.91370 | Chris Deotte手法 |

### 線形モデル

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| 02_lr | LogReg baseline | OHE 31特徴量 | 5 | 0.90794 | 0.90504 | |
| logreg_05 | LogReg baseline2 | 201特徴量 | 5 | 0.90794 | - | 重複 |
| logreg_06 | LogReg orig_ref | 201特徴量 | 5 | 0.91579 | 0.91285 | |
| ridge_07 | Ridge baseline | 201特徴量 | 5 | 0.89925 | - | |
| ridge_08 | Ridge orig_ref | 201特徴量 | 5 | 0.91084 | 0.90789 | |
| 28_04 | Ridge 201feat OOF | 201特徴量 | 20 | 0.91060 | - | stacking用 |
| 28_05 | Ridge NB1feat OOF | NB1準拠 | 20 | 0.91021 | - | stacking用 |
| logit3_08 | Logit3 TE-Pair LogReg | 513特徴量(171pair×3) | 5 | 0.91595 | 0.91348 | Chris Deotte手法 |

### ベイズ系モデル

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| gnb_04 | GNB baseline | 基本20特徴量 | 5 | 0.88729 | - | 特徴量独立性の仮定に適合 |
| gnb_05 | GNB orig_ref | 186特徴量(TE後) | 5 | 0.85389 | - | 相関特徴量で劣化、LogLoss 6.97 |

### Stackingモデル

| 実験 | モデル | 特徴量 | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| **31_01** | **Stacking LGBM** | 生20 + stack3 | 5 | **0.91926** | **0.91707** | **単体LB最高** |
| 31_02 | Stacking LGBM Optuna | 生20 + stack3 | 5 | 0.91934 | 0.91705 | OOF改善もLB微低下 |

---

## アンサンブル一覧

| 実験 | 手法 | モデル数 | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|
| ens_01 | SLSQP重み最適化 | 18/18 | 0.91883 | 0.91630 | 全モデルに分散、OOF過適合 |
| ens_02 | Hill Climbing v1 | 8/18 | 0.91939 | 0.91703 | realmlp+xgb_optuna主軸 |
| ens_03 | Stacking LR | 18/18 | 0.91679 | - | 単体以下、過学習 |
| ens_v2 | Hill Climbing v2 | 7/22 | 0.91955 | 0.91712 | ridge_xgb+realmlp主軸 |
| ens_v3 | Hill Climbing v3 | 6/13 | 0.91955 | 0.91711 | v2と同等、候補絞込み |
| ens_v4 | HC v4 (GPU) | 8/13 | 0.91956 | 0.91712 | rank01+再選択+8倍高速 |
| ens_v4+ | HC v4+ | 8/15 | 0.91957 | 0.91712 | +ridge_lgbm, ydf |
| ens_v4++ | HC v4++ | 8/16 | 0.91960 | 0.91715 | +tabm_nb1feat_10fold |
| **ens_v4+++** | **HC v4+++** | **11/17** | **0.91962** | **0.91715** | **+GNN、全提出最高** |

### HC v4+++ 構成（最新）

| モデル | 重み | 単体OOF AUC | タイプ |
|---|---|---|---|
| realmlp_orig_ref | +0.3551 | 0.91895 | NN |
| tabm_nb1feat_10fold | +0.2385 | 0.91898 | NN |
| ridge_xgb_reproduce | +0.2199 | 0.91922 | Ridge→XGB |
| ridge_lgbm_reproduce | +0.1592 | 0.91914 | Ridge→LGB |
| xgb_optuna_20fold | +0.0834 | 0.91913 | XGB |
| catboost_orig_ref | +0.0303 | 0.91853 | CatBoost |
| gnn_starter | +0.0296 | 0.91536 | GNN |
| ydf_orig_ref | +0.0210 | 0.91744 | YDF |
| xgb_baseline | +0.0210 | 0.91640 | XGB |
| mlp_orig_ref | -0.0985 | 0.91720 | NN |
| xgb_depth1 | -0.0592 | 0.91345 | XGB |

---

## 全提出LBスコア推移

| 日付 | モデル | LB | 備考 |
|---|---|---|---|
| 03/12 | LogReg baseline | 0.90504 | |
| 03/12 | XGBoost depth1 | 0.91039 | |
| 03/12 | XGBoost baseline | 0.91391 | |
| 03/12 | LightGBM baseline | 0.91378 | |
| 03/13 | Bartz baseline | 0.91405 | |
| 03/15 | XGBoost lossguide | 0.91311 | |
| 03/23 | XGBoost orig_ref | 0.91644 | 元データ参照FE導入 |
| 03/24 | XGBoost Optuna | 0.91656 | |
| 03/24 | LightGBM orig_ref | 0.91631 | |
| 03/25 | LightGBM Optuna | 0.91659 | |
| 03/25 | CatBoost orig_ref | 0.91640 | |
| 03/25 | RealMLP | 0.91655 | |
| 03/25 | MLP | 0.91483 | |
| 03/25 | LogReg orig_ref | 0.91285 | |
| 03/25 | Ridge orig_ref | 0.90789 | |
| 03/25 | TabM (当方FE) | 0.91657 | |
| 03/26 | SLSQP ensemble | 0.91630 | |
| 03/26 | Hill Climbing v1 | 0.91703 | |
| 03/26 | Ridge→XGB 20-fold | 0.91685 | |
| 03/26 | Ridge→XGB 5-fold | 0.91665 | |
| 03/26 | Hill Climbing v2 | 0.91712 | |
| 03/27 | Hill Climbing v3 | 0.91711 | |
| 03/27 | Logit3 TE-Pair LogReg | 0.91348 | |
| 03/27 | XGB全データ学習 | 0.91662 | |
| 03/28 | Ridge→LGB 20-fold | 0.91680 | |
| 03/28 | HC v4 (GPU) | 0.91712 | |
| 03/28 | HC v4+ | 0.91712 | |
| 03/28 | HC v4++ | 0.91715 | |
| 03/30 | GNN GraphSAGE | 0.91370 | |
| 03/30 | HC v4+++ | 0.91715 | |
| 03/31 | **Stacking LGBM** | **0.91707** | **単体LB最高** |
| 03/31 | Stacking LGBM Optuna | 0.91705 | OOF過適合 |

---

## 知見まとめ

### 単体モデル
- **元データ参照FE (201特徴量)** が全モデルで+0.002〜0.003の改善（最大の貢献）
- **20-fold CV** はパラメータ変更なしで+0.0002の改善（TE品質向上、OOF分散低減）
- **Ridge→XGBの2段階構造** の効果は限定的（20-fold同士で差0.00009）。fold数の効果が支配的
- **NN系 (RealMLP, TabM)** はツリー系と同等のCV（0.919前後）だがアーキテクチャが異なりアンサンブル多様性に貢献
- **TabM NB1 FE 10-fold** で参照ノートブック (0.91898) を完全再現。差分はDigit特徴量+18とSeniorCitizen categorical化+fold数
- **Stacking LGBM** が単体LB最高 (0.91707)。TabM/RealMLP/Ridgeの予測値がFeature Importanceの上位を独占

### アンサンブル
- **Hill Climbing > SLSQP > Stacking LR**: Discussion知見通り、特徴量交互作用が少ないデータではHill Climbingが最適
- **NN + ツリーの相補性** が鍵: realmlp (0.36) + tabm (0.24) + ridge_xgb (0.22) が主軸
- **GNN (0.030)** は単体AUC 0.915と低いがグラフベースの誤差パターンでアンサンブル貢献
- **11モデル選択**: 17モデル中11モデルが正または負の重みで選択。多様性の飽和が近い
- **LB 0.91715が天井に近い**: v4++以降、モデル追加によるLB改善が停滞

### その他
- **全データ学習** は5-fold平均と比べLBで+0.00006のみ。fold平均のほうがロバスト
- **Optuna過適合**: Stacking LGBM Optunaは OOF改善 (+0.00008) したがLBは微低下 (-0.00002)。少ないfold数でのチューニングは注意
- **ExtraTrees / GNB** は単体精度が低く、アンサンブルでも選択されない
- **Ridge→XGBに201特徴量のridge_predを追加すると悪化** (0.91913→0.91902)。201特徴量のTE群がRidgeの線形パターンを既にカバー
