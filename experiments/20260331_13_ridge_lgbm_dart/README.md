# 20260331_13_ridge_lgbm_dart

## 目的
ridge_lgbm_reproduce (28_01) のboosting_typeをgbdt→dartに変更。Dropout正則化によりgbdtと異なる誤差パターンを生成し、アンサンブル多様性に貢献する。

## アプローチ
- **ベース**: 28_01 (Ridge→LightGBM, NB1準拠FE, 20-fold) と同一
- **変更点**:
  - `boosting_type`: `gbdt` → `dart`
  - `drop_rate`: 0.1 (各ラウンドで10%のツリーをドロップ)
  - `skip_drop`: 0.5 (50%の確率でドロップをスキップ)
  - Early stopping無効化 → 固定800ラウンド (gbdt版best_iter 600-900)
- **その他パラメータ・FE**: 28_01と完全同一

## 結果

- **Ridge OOF AUC: 0.91021** (28_01と同一、同じRidgeモデル)
- **DART OOF AUC: 0.91728** (gbdt版 0.91914 から **-0.0019**)
- 実行時間: 28.2min (gbdt版とほぼ同等)

## 考察
- gbdt版 (0.91914) より-0.0019低下。DARTのDropout正則化により予測力がやや犠牲になっている
- ただしアンサンブルでの貢献はgbdtと異なる誤差パターンにより期待できる
- LogLossは800ラウンドで収束傾向あり (600→800でわずかに改善)。ラウンド数増加の余地あり

## 次のステップ
- Hill Climbingアンサンブルに追加して貢献度を検証
- ラウンド数を1200まで増加して精度改善の余地を確認
