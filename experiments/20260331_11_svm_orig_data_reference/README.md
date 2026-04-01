# 20260331_11_svm_orig_data_reference

## 目的
201特徴量でのSVM実装。マージンベースの非線形決定境界でアンサンブル多様性を確保する。

## アプローチ
- **特徴量**: 実験01(orig_data_reference)と同一の9グループFE + Nested TE + OHE = 212特徴量
- **前処理**: StandardScaler → Nystroem kernel approximation (RBF, n_components=500)
  - 594K行で標準SVC(RBF)はO(n^2)で非現実的 → Nystroem近似で線形時間に
  - gamma = 1/(n_features * X.var()) ≈ 0.00556 (sklearn 'scale' equivalent)
- **モデル**: SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-5, class_weight='balanced')
  - Nystroem空間でのロジスティック回帰 = 近似RBFカーネルロジスティック回帰
  - log_lossでcalibratedな確率を出力
- **CV**: 5-Fold Stratified CV、テスト予測は5-fold平均

## 結果

| Fold | AUC | LogLoss | Accuracy |
|------|------|---------|----------|
| 1 | 0.9156 | 0.3813 | 0.8150 |
| 2 | 0.9165 | 0.3782 | 0.8157 |
| 3 | 0.9160 | 0.3775 | 0.8157 |
| 4 | 0.9168 | 0.3988 | 0.8071 |
| 5 | 0.9142 | 0.3866 | 0.8107 |
| **OOF** | **0.9157** | **0.3845** | **0.8128** |

## 考察
- OOF AUC 0.9157 はGNN (0.9154) を上回り、Bartz (0.9158) と同等
- Accuracyが0.813と低い（他モデルは0.86前後）のはclass_weight='balanced'の影響（少数クラスのrecallを重視）
- LogLossも0.384と高めだが、AUC-ROCが評価指標なのでrank順序の品質が重要
- カーネルベースの非線形決定境界はツリー系・NN系・線形系と全く異なるパラダイム
- n_components=500は控えめな設定。増加で精度向上の余地あり

## 次のステップ
- Hill Climbingアンサンブルに追加してアンサンブル貢献度を検証
- n_components増加（1000, 2000）での精度改善
- class_weight='balanced'除外版の比較
