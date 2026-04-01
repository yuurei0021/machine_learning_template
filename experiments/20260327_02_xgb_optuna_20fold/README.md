# 20260327_02_xgb_optuna_20fold

## 目的
実験02（20260323_02_xgb_optuna_tuning）のOptuna最適パラメータを20-fold CVで実行し、fold数増加の効果を検証する。
Ridge→XGB (NB1, 0.91922) の優位性がRidge 2段階構造によるものか、20-fold CVによるものかを切り分ける。

## アプローチ
- **モデル**: XGBoost（Optuna最適パラメータをそのまま使用）
- **CV**: **20-fold** StratifiedKFold（元実験は5-fold）
- **FE**: 元実験と完全に同一（201特徴量、9グループ）
- **TE**: Inner 5-fold Target Encoding（パイプラインは元実験と同一）
- **Optuna探索なし**: 最適パラメータを直接使用

### XGBoostハイパーパラメータ（元実験Optuna結果）
| パラメータ | 値 |
|---|---|
| learning_rate | 0.00479 |
| max_depth | 5 |
| min_child_weight | 3 |
| subsample | 0.780 |
| colsample_bytree | 0.270 |
| reg_alpha | 0.00236 |
| reg_lambda | 9.904 |
| gamma | 1.197 |
| n_estimators | 50000 |
| early_stopping_rounds | 500 |

## 結果

| 指標 | 値 |
|---|---|
| **OOF AUC-ROC** | **0.919134** |
| OOF LogLoss | 0.293322 |
| 実行時間 | 133min |
| 平均best_iteration | ~11,800 |

### fold数による比較（同一パラメータ・同一FE）

| Folds | OOF AUC | 差分 |
|---|---|---|
| 5-fold（元実験02） | 0.91893 | - |
| **20-fold（本実験）** | **0.91913** | **+0.00020** |

### Ridge→XGBとの比較（20-fold同士）

| モデル | OOF AUC | Ridge有無 |
|---|---|---|
| NB1 Ridge→XGB | 0.91922 | あり |
| **本実験 XGB単体** | **0.91913** | **なし** |
| 差分 | -0.00009 | |

### 考察
- **20-fold化で+0.00020の改善**を確認。パラメータ変更なし、純粋にfold数の効果
- NB1 Ridge→XGB (0.91922) との差はわずか0.00009。Ridge 2段階構造の寄与は限定的で、**fold数増加が主因**という仮説を裏付ける
- 20-foldの改善メカニズム:
  - 各foldの訓練データ量: 475k (5-fold) → 564k (20-fold)、+19%
  - TEの推定品質向上（より多くの訓練データで計算）
  - OOFの分散低減（各fold予測が小さいval setに対して行われる）

## 次のステップ
- Hill Climbingアンサンブルへの組み込み
- 他モデル（LGBM, CatBoost等）でも20-fold化の効果を検証
