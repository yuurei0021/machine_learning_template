# 001: S6E3 | Ridge → XGB + N-gram | CV 0.91927

- **著者**: BlamerX
- **URL**: https://www.kaggle.com/code/blamerx/s6e3-ridge-xgb-n-gram-0-91927-cv
- **CV**: 0.91927（単体パイプライン）
- **Votes**: 99

## アーキテクチャ: Two-Stage Ridge → XGBoost

1. **Stage 1**: Ridge回帰を全特徴量（数値 + OHE categoricals + TE）で学習（alpha=10.0）
2. **Stage 2**: XGBoostが全特徴量 + `ridge_pred`（Ridge予測値）を特徴量として学習

Ridgeが線形パターンを捉え、XGBがそれを活用しつつ非線形補正を行う。`ridge_pred`はXGB importance上位にランクイン。

## 特徴量エンジニアリング

### 当方パイプラインと共通
- Frequency Encoding（数値列）
- Arithmetic Features（charges_deviation, monthly_to_total_ratio, avg_monthly_charges）
- Service Counts（service_count, has_internet, has_phone）
- ORIG_proba single（CAT + NUM）
- Distribution Features（pctrank, zscore, cond_pctrank, resid_IS_MC）
- Quantile Distance Features
- Digit Features
- Bigram TE（Top6カテゴリのペアワイズ）

### 当方パイプラインにない要素
- **Tri-gram TE**: Top4カテゴリの3-way組み合わせ（C(4,3)=4個）をTE
- **TE with std/min/max stats**: カテゴリカルのTE時にmeanだけでなくstd, min, maxも計算（※当方のLGBM実験01/02には存在するが、XGB実験やNN実験には未実装）
- **Frequency Encoding**: train + orig のみ使用（testを含めない）

## CV戦略

- **20-fold outer CV**（当方は5-fold）
  - 594k行で20-fold → 各fold 564k train / 30k val
  - OOF/TEの品質向上、学習データ増加
- Inner 5-fold for Target Encoding（当方と同じ）

## XGBoostハイパーパラメータ

| パラメータ | NB1 | 当方Optuna | 差異 |
|---|---|---|---|
| learning_rate | **0.0063** | 0.0048 | 近い |
| max_depth | 5 | 5 | 同じ |
| colsample_bytree | **0.32** | 0.27 | 近い |
| reg_alpha | **3.50** | 0.002 | **大幅に異なる** |
| reg_lambda | 1.29 | 9.90 | 逆方向 |
| gamma | 0.79 | 1.20 | |
| min_child_weight | 6 | 3 | |
| early_stopping | **500** | 100 | **5倍** |
| n_estimators上限 | 5000 | 5000 | 同じ |

注目点: L1正則化（reg_alpha）が当方の1750倍。L2（reg_lambda）は逆に低い。

## 当方CVとの比較

| モデル | NB1 | 当方 | 差分 |
|---|---|---|---|
| XGBoost単体 | **0.91927** | 0.91893 | **-0.00034** |

## 実装への示唆

1. **Ridge→XGB 2段階パイプライン**: 最も効果が高い。Ridge予測値を特徴量に追加するだけで実装可能
2. **20-fold CV**: OOF品質向上。計算時間は4倍だが精度改善に寄与
3. **reg_alpha の大幅引き上げ**: Optuna探索範囲を見直す価値あり
4. **early_stopping=500**: 早期打ち切りを緩くすることで微小改善を逃さない
