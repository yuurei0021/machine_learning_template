# 20260326_04_ridge_xgb_reproduce

## 目的
BlamerXのRidge→XGB Two-Stage ノートブック（CV 0.91927）をローカル環境で忠実に再現し、スコアの再現性を確認する。

参照: https://www.kaggle.com/code/blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv

## アプローチ
- **Two-Stage学習**: Stage1でRidge回帰、Stage2でXGBoostにRidge予測値を特徴量として追加
- **20-fold CV**（Inner 5-fold for TE）
- **FE**: Freq Encoding、Arithmetic、Service Counts、ORIG_proba single、Distribution、Quantile Distance、Digit Features（35個）、Bigram TE（15個）+ Trigram TE（4個）、NUM_AS_CAT、TE std/min/max
- **Ridge**: alpha=10.0、StandardScaler + OHE categoricals（sparse）
- **XGB**: NB1のOptunaパラメータ（lr=0.0063, max_depth=5, reg_alpha=3.50, early_stopping=500）

### 元ノートブックからの変更点
- ローカルパス（Kaggleパスから変更）
- 元データ: xlsx読み込み + カラム名マッピング
- device=cpu（ローカルGPUなし）
- 可視化コード削除

## 結果

| 指標 | 値 |
|---|---|
| Ridge OOF AUC | 0.910208 |
| **XGB OOF AUC** | **0.919215** |
| **LB AUC** | **0.91685** |
| 実行時間 | 139min（20-fold） |

### 元ノートブックとの比較

| | 元NB | 本実験 | 差分 |
|---|---|---|---|
| XGB OOF AUC | 0.91927 | 0.91922 | -0.00005 |

ほぼ完全に再現。微差は元データソースの違い（Kaggle CSV vs ローカルxlsx）に起因する可能性。

## 次のステップ
- 当方FEパイプラインへの適応（実験05）
- Ridge→XGB OOFをHill Climbingアンサンブルに組み込み
