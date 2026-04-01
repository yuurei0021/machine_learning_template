# 20260326_05_ridge_xgb_adapted

## 目的
BlamerXのRidge→XGB Two-Stageアプローチを当方のFEパイプラインに適応し、5-fold CVで効果を検証する。

## アプローチ
- **Two-Stage学習**: 実験04と同じRidge→XGBアーキテクチャ
- **5-fold CV**（Inner 5-fold for TE）— 高速検証用
- **当方FEパイプライン**: 実験04のFEに加え以下を追加
  - ORIG_proba cross（5カテゴリのペアワイズ、10特徴量）
  - Distribution拡張（MC系・T系のpctrank/zscore）
  - Arithmetic追加（is_first_month, dev_is_zero, dev_sign）
- **XGBパラメータ**: NB1のOptunaパラメータを流用

### 実験04との差分
| 項目 | 実験04（再現） | 実験05（適応） |
|---|---|---|
| CV | 20-fold | 5-fold |
| ORIG_proba cross | なし | 10ペア |
| Distribution MC/T系 | なし | あり |
| Arithmetic追加 | なし | 3特徴量 |

## 結果

| 指標 | 値 |
|---|---|
| Ridge OOF AUC | 0.910550 |
| **XGB OOF AUC** | **0.918883** |
| **LB AUC** | **0.91665** |
| 実行時間 | 48min（5-fold） |

### 他モデルとの比較

| モデル | OOF AUC | LB | Folds |
|---|---|---|---|
| 実験04 Ridge→XGB（再現） | 0.91922 | 0.91685 | 20 |
| **実験05 Ridge→XGB（適応）** | **0.91888** | **0.91665** | **5** |
| XGB Optuna（既存最高） | 0.91893 | - | 5 |
| TabM | 0.91854 | 0.91657 | 5 |

### 考察
- 5-foldでは既存XGB Optuna (0.91893) と同等水準（0.91888）
- ORIG_proba cross等の特徴量追加はRidge→XGBフレームワークでは大きな改善に繋がらず
- 実験04 vs 05の差（0.00034）は主に**20-fold vs 5-foldの差**に起因
  - 20-foldはTE品質向上・学習データ増加・OOF分散低減が寄与
- Ridge OOF AUC自体は実験05 (0.9106) > 実験04 (0.9102) であり、当方FE追加はRidgeには有効

## 次のステップ
- 実験04/05のOOFをHill Climbingアンサンブルに組み込み
- 当方パイプラインでも20-fold CVを試行
