# 20260328_03_ridge_xgb_201feat

## 目的
当方の201特徴量パイプラインにRidge予測値を追加し、XGBoostの精度向上効果を検証する。

### 背景
- NB1再現 (Ridge→XGB, NB1 FE, 20-fold): OOF 0.91922
- 当方 (XGB Optuna, 201特徴量, 20-fold): OOF 0.91913
- 差は0.00009。特徴量の違い（Digit追加20個 vs ORIG_proba cross等）とRidge予測値の有無が要因

本実験で201特徴量 + ridge_predの組み合わせにより、NB1を上回れるか検証する。

## アプローチ
- **Stage 1**: Ridge回帰（201特徴量をStandardScaler + OHEカテゴリで学習）
- **Stage 2**: XGBoost（201特徴量 + ridge_pred + Nested TE）
- **CV**: 20-fold（NB1再現と同条件）
- **XGBパラメータ**: 当方Optuna最適パラメータ
- **FE**: 20260323_01と同一の201特徴量体系（ORIG_proba cross含む）
- **ベース実験**: 20260326_05_ridge_xgb_adapted（5-fold版、OOF 0.91888）を20-foldに拡張

## 比較対象

| 実験 | FE | Ridge | Folds | OOF AUC |
|---|---|---|---|---|
| NB1再現 (04) | NB1準拠 (179 XGB feat) | あり | 20 | 0.91922 |
| XGB Optuna 20fold (02) | 201特徴量 | なし | 20 | 0.91913 |
| **本実験** | **201特徴量** | **あり** | **20** | **?** |

## 期待される結果
- Ridge予測値が201特徴量のORIG_proba cross等と相補的であれば、0.9192+
- 201特徴量がXGBにとって既に十分な情報を含んでいれば、Ridge追加の効果は限定的

## 結果
TODO

## 次のステップ
TODO
