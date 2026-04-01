# Model Comparison Report

全実験の精度一覧とアンサンブル結果、知見をまとめたレポート。

---

## 単体モデル

### ツリー系

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| 04_reproduce | Ridge→XGB (NB1再現) | NB1準拠 | 20 | 0.91922 | 0.91685 | |
| 02_20fold | XGBoost Optuna 20-fold | 201特徴量 | 20 | 0.91913 | 未submit | fold数検証 |
| 28_01 | Ridge→LGB | NB1準拠 | 20 | 0.91914 | 0.91680 | LGBM Optuna params |
| 31_13 | Ridge→LGB DART | NB1準拠 | 20 | 0.91728 | - | dart, 800rounds固定 |
| 28_06 | Ridge→XGB (NB1+ridge) | NB1+ridge | 20 | 0.91917 | 未submit | Kaggle CPU |
| 28_03 | Ridge→XGB (201特徴量) | 201特徴量+ridge | 20 | 0.91902 | 未submit | ridge追加で悪化 |
| 02 | XGBoost Optuna | 201特徴量 | 5 | 0.91893 | 0.91656 | Optuna 50trials |
| 05_adapted | Ridge→XGB (適応) | 201特徴量+cross | 5 | 0.91888 | 0.91665 | |
| lgbm_02 | LightGBM Optuna | 201特徴量 | 5 | 0.91880 | 0.91659 | Optuna 50trials |
| 01 | XGBoost orig_ref | 201特徴量 | 5 | 0.91853 | 0.91644 | 元データ参照FE |
| cb_03 | CatBoost orig_ref | 201特徴量 | 5 | 0.91853 | 0.91640 | |
| lgbm_01 | LightGBM orig_ref | 201特徴量 | 5 | 0.91844 | 0.91631 | |
| ydf_11 | YDF orig_ref | 107特徴量 | 5 | 0.91744 | - | Kaggle CPU, max_depth=2 |
| 03 | XGBoost baseline | 基本20特徴量 | 5 | 0.91640 | 0.91391 | max_depth=6 |
| 05 | LightGBM baseline | 基本20特徴量 | 5 | 0.91633 | 0.91378 | |
| ydf_01 | XGBoost lossguide | 基本20特徴量 | 5 | 0.91589 | 0.91311 | depth=2 |
| 06 | Bartz baseline | TE+pairwise 140特徴量 | 5 | 0.91578 | 0.91405 | MCMC |
| et_07 | ExtraTrees orig_ref | 186特徴量(TE後) | 5 | 0.91484 | - | |
| 04 | XGBoost depth1 | 基本20特徴量 | 5 | 0.91345 | 0.91039 | max_depth=1 |
| et_06 | ExtraTrees baseline | ~40特徴量 | 5 | 0.91229 | - | ランダム分割 |
| 10_fulldata | XGB全データ学習 | 201特徴量 | - | - | 0.91662 | n_est=14463, OOFなし |

### NN系

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| **30_01** | **TabM (NB1 FE)** | NB1準拠 | **10** | **0.91898** | 未submit | **参照NB完全再現** |
| realmlp_04 | RealMLP | 42特徴量 | 5 | 0.91895 | 0.91655 | pytabkit, n_ens=8 |
| tabm_10 | TabM (当方FE) | 112+15特徴量 | 5 | 0.91854 | 0.91657 | Kaggle T4 GPU |
| mlp_09 | MLP (PyTorch) | 201特徴量 | 5 | 0.91720 | 0.91483 | 512-256-128 |

### Transformer系

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| 31_06 | TabTransformer | TE-Pair Logit3 + Ridge | 5 | 0.91658 | 未submit | Keras, EPOCHS=5 |

### グラフ系

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| 30_02 | GNN (GraphSAGE) | 25特徴量 | 5 | 0.91536 | 0.91370 | Chris Deotte手法 |

### 線形モデル

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| logit3_08 | Logit3 TE-Pair LogReg | 513特徴量(171pair×3) | 5 | 0.91595 | 0.91348 | Chris Deotte手法 |
| logreg_06 | LogReg orig_ref | 201特徴量 | 5 | 0.91579 | 0.91285 | |
| ridge_08 | Ridge orig_ref | 201特徴量 | 5 | 0.91084 | 0.90789 | |
| 28_04 | Ridge 201feat OOF | 201特徴量 | 20 | 0.91060 | - | stacking用 |
| 28_05 | Ridge NB1feat OOF | NB1準拠 | 20 | 0.91021 | - | stacking用 |
| 02_lr | LogReg baseline | OHE 31特徴量 | 5 | 0.90794 | 0.90504 | |
| ridge_07 | Ridge baseline | 201特徴量 | 5 | 0.89925 | - | |

### SVM (カーネル近似)

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| svm_11 | Nystroem RBF + SGD | 201特徴量+OHE=212 | 5 | 0.91570 | - | n_comp=500, balanced |

### インスタンスベース

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| knn_10 | KNN (k=51) | 201特徴量+OHE=212 | 5 | 0.91004 | - | orig_ref FE + TE |
| knn_09 | KNN (k=51) | 生値OHE 30特徴量 | 5 | 0.90011 | - | distance weighting |

### ベイズ系

| 実験 | モデル | FE | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| gnb_04 | GNB baseline | 基本20特徴量 | 5 | 0.88729 | - | 特徴量独立性の仮定に適合 |
| gnb_05 | GNB orig_ref | 186特徴量(TE後) | 5 | 0.85389 | - | 相関特徴量で劣化 |

### Stacking

| 実験 | モデル | 特徴量 | Folds | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|---|
| **31_01** | **Stacking LGBM** | 生20 + stack3 | 5 | **0.91926** | **0.91707** | **単体LB最高** |
| 31_02 | Stacking LGBM Optuna | 生20 + stack3 | 5 | 0.91934 | 0.91705 | OOF改善もLB微低下 |
| 31_03 | Stacking XGB | 生20 + stack3 | 5 | 0.91909 | 0.91702 | LGBMが優位 |

---

## アンサンブル

| 実験 | 手法 | モデル数 | OOF AUC | LB | 備考 |
|---|---|---|---|---|---|
| **ens_v4TT+DART** | **HC v4TT+DART** | **12/19** | **0.91964** | **0.91720** | **DART w=-0.172、LB最高** |
| ens_v4TT+KS | HC v4TT+KNN+SVM | 11/21 | 0.91962 | 未submit | KNN/SVM未選択、v4TTと同一構成 |
| ens_v5+TT | HC v5+TT | 13/21 | 0.91964 | 未submit | +TabTransformer、OOF最高 |
| ens_v4+++TT | HC v4+++TT | 11/18 | 0.91962 | 未submit | +TabTransformer |
| ens_v6 | HC v6 | 12/19 | 0.91962 | 0.91712 | v5-lgbm_optuna |
| ens_v5 | HC v5 | 14/20 | 0.91963 | 0.91713 | +stacking3モデル |
| **ens_v4+++** | **HC v4+++** | **11/17** | **0.91962** | **0.91715** | **LB最高** |
| ens_v4++ | HC v4++ | 8/16 | 0.91960 | 0.91715 | +tabm_nb1feat_10fold |
| ens_v4+ | HC v4+ | 8/15 | 0.91957 | 0.91712 | +ridge_lgbm, ydf |
| ens_v4 | HC v4 (GPU) | 8/13 | 0.91956 | 0.91712 | rank01+再選択+8倍高速 |
| ens_v3 | Hill Climbing v3 | 6/13 | 0.91955 | 0.91711 | 候補絞込み |
| ens_v2 | Hill Climbing v2 | 7/22 | 0.91955 | 0.91712 | ridge_xgb+realmlp主軸 |
| ens_02 | Hill Climbing v1 | 8/18 | 0.91939 | 0.91703 | realmlp+xgb_optuna主軸 |
| ens_01 | SLSQP重み最適化 | 18/18 | 0.91883 | 0.91630 | 全モデルに分散、OOF過適合 |
| ens_03 | Stacking LR | 18/18 | 0.91679 | - | 単体以下、過学習 |

### HC v4TT+DART 構成（OOF 0.91964, LB最高 0.91720）

| モデル | 重み | 単体OOF AUC | タイプ |
|---|---|---|---|
| realmlp_orig_ref | +0.3278 | 0.91895 | NN |
| ridge_xgb_reproduce | +0.2982 | 0.91922 | Ridge→XGB |
| tabm_nb1feat_10fold | +0.2201 | 0.91898 | NN |
| **ridge_lgbm_dart** | **-0.1722** | **0.91728** | **DART** |
| ridge_lgbm_reproduce | +0.1658 | 0.91914 | Ridge→LGB |
| tabtransformer | +0.0661 | 0.91658 | Transformer |
| xgb_optuna_20fold | +0.0618 | 0.91913 | XGB |
| catboost_orig_ref | +0.0420 | 0.91853 | CatBoost |
| mlp_orig_ref | -0.0300 | 0.91720 | NN |
| extratrees_orig_ref | +0.0207 | 0.91484 | ExtraTrees |
| xgb_depth1 | -0.0203 | 0.91345 | XGB |
| gnn_starter | +0.0202 | 0.91536 | GNN |

---

## LBスコア推移

| 日付 | モデル | LB | 備考 |
|---|---|---|---|
| 03/12 | LogReg baseline | 0.90504 | |
| 03/12 | XGBoost depth1 | 0.91039 | |
| 03/12 | LightGBM baseline | 0.91378 | |
| 03/12 | XGBoost baseline | 0.91391 | |
| 03/13 | Bartz baseline | 0.91405 | |
| 03/15 | XGBoost lossguide | 0.91311 | |
| 03/23 | XGBoost orig_ref | 0.91644 | 元データ参照FE導入 |
| 03/24 | XGBoost Optuna | 0.91656 | |
| 03/24 | LightGBM orig_ref | 0.91631 | |
| 03/25 | LightGBM Optuna | 0.91659 | |
| 03/25 | CatBoost orig_ref | 0.91640 | |
| 03/25 | RealMLP | 0.91655 | |
| 03/25 | TabM (当方FE) | 0.91657 | |
| 03/25 | MLP | 0.91483 | |
| 03/25 | LogReg orig_ref | 0.91285 | |
| 03/25 | Ridge orig_ref | 0.90789 | |
| 03/26 | SLSQP ensemble | 0.91630 | |
| 03/26 | Ridge→XGB 20-fold | 0.91685 | |
| 03/26 | Ridge→XGB 5-fold | 0.91665 | |
| 03/26 | Hill Climbing v1 | 0.91703 | |
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
| 03/31 | Stacking XGB | 0.91702 | LGBMが優位 |
| 03/31 | HC v5 | 0.91713 | +stackingモデル、LB微減 |
| 03/31 | HC v6 | 0.91712 | v5-lgbm_optuna |
| 04/01 | **HC v4TT+DART** | **0.91720** | **+DART、LB最高** |

---

## 知見

### 単体モデル
- **元データ参照FE (201特徴量)** が全モデルで+0.002〜0.003の改善（最大の貢献）
- **20-fold CV** はパラメータ変更なしで+0.0002の改善（TE品質向上、OOF分散低減）
- **Ridge→XGBの2段階構造** の効果は限定的（20-fold同士で差0.00009）。fold数の効果が支配的
- **NN系 (RealMLP, TabM)** はツリー系と同等のCV（0.919前後）だがアーキテクチャが異なりアンサンブル多様性に貢献
- **TabM NB1 FE 10-fold** で参照ノートブック (0.91898) を完全再現。差分はDigit特徴量+18とSeniorCitizen categorical化+fold数
- **Stacking LGBM** が単体LB最高 (0.91707)。TabM/RealMLP/Ridgeの予測値がFeature Importanceの上位を独占
- **Ridge→XGBに201特徴量のridge_predを追加すると悪化** (0.91913→0.91902)。201特徴量のTE群がRidgeの線形パターンを既にカバー

### アンサンブル
- **Hill Climbing > SLSQP > Stacking LR**: Discussion知見通り、特徴量交互作用が少ないデータではHill Climbingが最適
- **NN + ツリーの相補性** が鍵: realmlp (0.36) + tabm (0.24) + ridge_xgb (0.22) が主軸
- **GNN (0.030)** は単体AUC 0.915と低いがグラフベースの誤差パターンでアンサンブル貢献
- **11モデル選択**: 17モデル中11モデルが正または負の重みで選択。多様性の飽和が近い
- **DART boosting** が多様性に大きく貢献: Ridge→LGB DARTはw=-0.172で4位の重み。gbdt版 (w=+0.166) と反対符号で、gbdtの過学習パターンをDropout正則化が補正
- **KNN / SVM はアンサンブルに貢献せず**: 単体AUC 0.900-0.916だがHCで未選択。異なるパラダイムでも誤差の直交性が不十分
- **TabTransformer (OOF 0.917)** は単体精度が低いがSelf-Attention構造の多様性でHCに貢献: v4+++TTでw=+0.058 (Iter3) で選択。GNNと同様のパターン
- **HC v4TT+DARTでLB 0.91720を達成**: v4+++ (0.91715) から+0.00005。DART追加でextratrees_orig_refも新たに選択（12モデル）

### その他
- **全データ学習** は5-fold平均と比べLBで+0.00006のみ。fold平均のほうがロバスト
- **Optuna過適合**: Stacking LGBM Optunaは OOF改善 (+0.00008) したがLBは微低下 (-0.00002)。少ないfold数でのチューニングは注意
- **ExtraTrees / GNB** は単体精度が低く、アンサンブルでも選択されない
- **YDF orig_ref** (0.917) はBartz baseline (0.916) を上回るが、XGB/LGBMには及ばず
- **Ridge→LGB** はRidge→XGBと同等精度を7倍速で達成
- **Stacking LGBM > Stacking XGB**: 同一特徴量でLGBMがXGBを+0.00017上回る (OOF)。LBでも+0.00005
- **HC v5でstackingモデルが主軸化**: stacking_lgbm_optunaが重み0.48で1位。ただしOOF最高でもLBはv4+++ (0.91715) を下回る
- **HC v6 (lgbm_optuna除外)**: LB 0.91712でv5 (0.91713) と変わらず。lgbm_optunaのfold過学習は主要因ではない
- **同一パイプライン・異なるboostingが有効**: DART (w=-0.172) > TabTransformer (w=+0.066) > GNN (w=+0.020) の順にアンサンブル貢献。パラダイムの違いより、同一FEでの正則化差異が有効
