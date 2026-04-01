# 20260331_01_stacking_lgbm

## 目的
生特徴量にTabM/RealMLP/Ridgeの予測値を追加特徴量としてLGBMでstackingし、単体モデルの精度上限を引き上げる。

## アプローチ
- **モデル**: LightGBM（デフォルトパラメータ）
- **CV**: 5-fold StratifiedKFold
- **生特徴量 (20)**: Label Encoded categoricals (15) + numericals (4) + Charge_Difference (1)
- **Stacking特徴量 (3)**:
  - `tabm_pred`: TabM NB1feat 10-fold OOF (実験20260330_01, OOF AUC 0.91898)
  - `realmlp_pred`: RealMLP orig_ref OOF (実験20260325_04, OOF AUC 0.91895)
  - `ridge_pred`: Ridge 201feat 20-fold OOF (実験20260328_04, OOF AUC 0.91060)
- **合計特徴量**: 23

### Stacking特徴量の選定理由
- TabM/RealMLP: NN系の予測値。ツリーモデルでは捉えにくい非線形パターンを含む
- Ridge: 線形モデルの予測値。NNやツリーとは異なる線形的なシグナルを含む
- OOFを使用するためリーク無し（各サンプルは自身のfold外で予測された値を使用）

## 結果

| 指標 | 値 |
|---|---|
| **OOF AUC-ROC** | **0.919255** |
| OOF LogLoss | 0.293122 |
| **LB AUC-ROC** | **0.91707** |
| Best iteration (平均) | ~132 |

### Feature Importance (Fold 1, Gain)

| 特徴量 | Gain | タイプ |
|---|---|---|
| realmlp_pred | 1,140,366 | **STACK** |
| tabm_pred | 629,096 | **STACK** |
| ridge_pred | 115,255 | **STACK** |
| Charge_Difference | 2,864 | 生特徴量 |
| MonthlyCharges | 2,754 | 生特徴量 |
| TotalCharges | 2,021 | 生特徴量 |
| tenure | 1,219 | 生特徴量 |
| Contract | 372 | 生特徴量 |
| PaymentMethod | 286 | 生特徴量 |
| OnlineSecurity | 165 | 生特徴量 |

Stack特徴量のGainが生特徴量の40〜400倍。LGBMはStack特徴量を主に使い、生特徴量で微調整する構造。

### 他モデルとの比較

| モデル | OOF AUC | LB | タイプ |
|---|---|---|---|
| **Stacking LGBM** | **0.91926** | **0.91707** | Stacking |
| Ridge→XGB 20f (NB1) | 0.91922 | 0.91685 | 2段階 |
| XGB Optuna 20-fold | 0.91913 | - | 単体 |
| HC v4+++ | 0.91962 | 0.91715 | アンサンブル |

### 考察
- **単体モデルとしてLB最高 (0.91707)**: Ridge→XGB (0.91685) を+0.00022上回る
- **NN + 線形 + ツリーの3種stacking**: 異なるアーキテクチャの予測を統合することで、いずれか1つでは到達できない精度
- **生特徴量の役割は補助的**: Stack特徴量が支配的だが、生特徴量なしでは0.919+に届かない。残差の微調整に寄与
- **early stoppingが早い (iter ~130)**: Stack特徴量の情報量が大きいため少ないイテレーションで収束

## 次のステップ
- Optunaチューニング（実験02）で+0.001の改善を狙う
- Stack特徴量の追加（GNN、XGB Optuna等）
- 20-fold CV化
