# 20260327_01_hillclimb_ensemble_v2

## 目的
v1（20260326_02, LB 0.91703）にRidge→XGBモデル（実験04/05）とRidge単体OOFを追加し、Hill Climbingアンサンブルのスコア向上を図る。

## アプローチ
- **手法**: Greedy Hill Climbing（v1と同一パラメータ: weight_step=0.01, 範囲[-0.5, 1.0), 停止閾値0.000001）
- **候補モデル**: 22モデル（v1の18モデル + 新規4モデル）
  - **NEW**: ridge_xgb_reproduce（20-fold, OOF 0.91922）
  - **NEW**: ridge_xgb_adapted（5-fold, OOF 0.91888）
  - **NEW**: ridge_only_reproduce（Ridge単体, OOF 0.91021）
  - **NEW**: ridge_only_adapted（Ridge単体, OOF 0.91055）

## 結果

### スコア
| 指標 | v2 | v1 | 差分 |
|---|---|---|---|
| OOF AUC-ROC | **0.919553** | 0.919391 | **+0.000162** |
| LB AUC-ROC | **0.91712** | 0.91703 | **+0.00009** |

### 選択されたモデルと重み（7/22モデル）

| モデル | 重み | 単体AUC | タイプ | v1比較 |
|---|---|---|---|---|
| ridge_xgb_reproduce | +0.5837 | 0.91922 | Ridge→XGB | **NEW（主軸）** |
| realmlp_orig_ref | +0.4227 | 0.91895 | NN | v1: +0.46 |
| tabm_orig_ref | +0.1152 | 0.91854 | NN | v1: +0.16 |
| catboost_orig_ref | +0.0535 | 0.91853 | ツリー | v1: +0.05 |
| mlp_orig_ref | -0.0745 | 0.91720 | NN | v1: -0.05 |
| ridge_xgb_adapted | -0.0700 | 0.91888 | Ridge→XGB | **NEW** |
| xgb_depth1 | -0.0305 | 0.91345 | ツリー | v1: -0.05 |

### 反復履歴

| Iteration | 追加モデル | 重み | スコア | 改善量 |
|---|---|---|---|---|
| 0 | ridge_xgb_reproduce | 1.00 | 0.919215 | (初期) |
| 1 | realmlp_orig_ref | 0.44 | 0.919524 | +0.000309 |
| 2 | tabm_orig_ref | 0.12 | 0.919540 | +0.000016 |
| 3 | xgb_depth1 | -0.03 | 0.919544 | +0.000004 |
| 4 | mlp_orig_ref | -0.07 | 0.919547 | +0.000003 |
| 5 | catboost_orig_ref | 0.05 | 0.919550 | +0.000003 |
| 6 | ridge_xgb_adapted | -0.07 | 0.919553 | +0.000003 |

### v1→v2の変化
- **主軸がxgb_optuna → ridge_xgb_reproduceに交代**: Ridge→XGB 20-foldが単体最高スコア（0.91922）のため
- **xgb_optunaが選択されなくなった**: ridge_xgb_reproduceがXGBの情報を既に含んでおり冗長
- **ridge_xgb_adaptedが負の重みで選択**: 5-fold版の誤差パターンが20-fold版と微妙に異なり、補正に寄与
- **モデル数が8→7に減少**: より少ないモデルで高いスコアを達成

### 全提出スコア比較

| 手法 | OOF AUC | LB |
|---|---|---|
| **HC v2 (本実験)** | **0.91955** | **0.91712** |
| HC v1 | 0.91939 | 0.91703 |
| Ridge→XGB 20-fold | 0.91922 | 0.91685 |
| Ridge→XGB 5-fold | 0.91888 | 0.91665 |
| TabM単体 | 0.91854 | 0.91657 |
| XGBoost orig_ref | 0.91853 | 0.91644 |

## 次のステップ
- Ridge→XGBの20-fold CVを当方FEパイプラインでも実施（実験05を20-foldに）
- Logit3 TE-Pair LogReg（Chris Deotte手法）でアンサンブル多様性をさらに向上
- Pseudo Labelsの試行
