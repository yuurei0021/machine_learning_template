# 20260323_01_orig_data_reference

## 目的
元データ（IBM Telco 7,043行）を参照分布として特徴量生成し、XGBoostのスコア向上を図る。
参考: https://www.kaggle.com/code/ozermehmet/original-data-fe-single-xgb-cv-0-919-lb-0-916

## アプローチ
実験03（XGBoost baseline）をベースに、Notebookの9つの特徴量グループ + Nested Target Encodingを追加。

### 特徴量グループ（129基本特徴量 → 201モデル特徴量）
1. **Frequency Encoding** (3): tenure, MonthlyCharges, TotalChargesの値頻度
2. **Arithmetic Interactions** (6): charges_deviation, ratio, avg, is_first_month, dev_is_zero, dev_sign
3. **Service Counts** (3): service_count, has_internet, has_phone
4. **ORIG_proba single** (15): 各カラムの値→元データChurn率マッピング
5. **ORIG_proba cross** (10): 上位5カテゴリの2変数交互作用Churn率
6. **Distribution Features** (16): percentile rank, z-score (Churner/Non-churner分布)
7. **Quantile Distance Features** (18): Q25/Q50/Q75への距離（Churner vs Non-churner）
8. **Digit/Modular Features** (17): tenure_mod12, mc_fractional等の合成アーティファクト
9. **Num-as-cat + N-grams** (19→TE): 数値カテゴリ化 + Bigram/Trigram組み合わせ

### Target Encoding戦略（CV内部で適用、72追加特徴量）
- **Layer A**: Inner 5-fold TE (std/min/max) for 18 columns = 54特徴量
- **Layer B**: Inner 5-fold TE mean for 22 n-gram columns = 22特徴量
- **Layer C**: sklearn TargetEncoder (mean with smoothing) for 18 columns = 18特徴量
- Raw n-gram列はドロップ → 最終201特徴量

### XGBoostパラメータ
- 実験03と同一（max_depth=6, lr=0.05）
- NUM_BOOST_ROUND=5000, EARLY_STOPPING_ROUNDS=100に拡大

## 結果

| Metric | Value |
|--------|-------|
| OOF AUC-ROC | **0.9185** |
| OOF LogLoss | 0.2943 |
| OOF Accuracy | 0.8635 |

### Fold別結果
| Fold | AUC | LogLoss | Accuracy | Best Iter |
|------|-----|---------|----------|-----------|
| 1 | 0.9184 | 0.2946 | 0.8636 | 576 |
| 2 | 0.9191 | 0.2932 | 0.8633 | 613 |
| 3 | 0.9185 | 0.2946 | 0.8635 | 487 |
| 4 | 0.9198 | 0.2922 | 0.8648 | 523 |
| 5 | 0.9169 | 0.2969 | 0.8623 | 494 |

### モデル比較
| Experiment | OOF AUC | LB AUC | 差分 |
|-----------|---------|--------|------|
| 03_xgboost_baseline (depth=6) | 0.9164 | 0.91391 | baseline |
| **本実験 (orig_data_reference)** | **0.9185** | **0.91644** | **+0.0021** |
| Notebook参考値 (10-fold, Optuna) | 0.9190 | 0.91671 | +0.0026 |

## 次のステップ
- Kaggle提出してLBスコアを確認
- ハイパーパラメータチューニング（Optunaで最適化）
- 10-fold CVでの再実験（Notebookと同条件）
- アンサンブル（本実験のOOF + 他モデルのOOF）
