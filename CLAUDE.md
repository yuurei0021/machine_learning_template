# CLAUDE.md

## プロジェクト概要
このリポジトリはKaggleコンペ「Predict Customer Churn」（Playground Series - Season 6 Episode 3）のデータ分析およびモデル作成を行う

### 目的
顧客の離脱（Churn）の確率を予測する二値分類タスク

### 評価指標
- **AUC-ROC（Area Under the ROC Curve）**: コンペティションの公式評価指標
- 予測確率と実際のターゲットの間のROC曲線下面積で評価
- モデル開発・評価時は必ずAUC-ROCを主要指標として使用すること

## 環境

- **Python**: 3.11以上
- **パッケージマネージャ**: uv
- **依存関係**: `pyproject.toml` に定義（numpy, pandas, scikit-learn, xgboost, lightgbm, matplotlib, seaborn）
- **実行**: `uv run python experiments/YYYYMMDD_NN_実験名/main.py`
- **実験作成**: `uv run python scripts/create_experiment.py <実験名> [--template classification|regression]`

## Kaggle GPU実行

ローカルGPUのVRAMが不足するモデル（TabM等）はKaggle GPU (T4 x2, 30GB VRAM) で実行する。

### ファイル構成

```
experiments/YYYYMMDD_NN_実験名/
├── main.py                    # 正のコード（ローカル再現用、全ロジック含む）
├── kaggle_kernel/             # Kaggle実行用
│   ├── kernel-metadata.json   # Kaggle API設定（GPU、データソース等）
│   └── tabm_kernel.py         # main.pyのKaggleパス調整版（薄いラッパー）
├── predictions/               # Kaggle出力をダウンロードして配置
└── README.md                  # Kaggle実行であること、カーネルID等を記録
```

### 実行手順

```bash
# 1. カーネルをpush（Kaggle GPU上で自動実行開始）
uv run kaggle kernels push -p experiments/YYYYMMDD_NN_実験名/kaggle_kernel/

# 2. 状態確認（complete になるまで待機）
uv run kaggle kernels status ユーザー名/カーネル名

# 3. 結果ダウンロード
uv run kaggle kernels output ユーザー名/カーネル名 -p experiments/YYYYMMDD_NN_実験名/predictions/
```

### 原則

- **main.py が正**: kaggle_kernel/ のスクリプトはパス調整のみの薄いコピー。ロジックはmain.pyと同一であること
- **結果はリポジトリに格納**: ダウンロードしたOOF/テスト予測を `predictions/` に配置し、他の実験と同じ構造にする
- **README.mdに実行条件を記録**: カーネルID、GPU種別、実行時間、pytabkitバージョン等
- **元データDataset**: `sohailkhanlml/customer-churn-prediction-original-s6e3` を使用

## ファイル構成

```
project_root/
├── data/
│   ├── raw/                # 元データ（Kaggleからダウンロード）
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/          # 前処理済みデータ
├── experiments/            # 実験用フォルダ
│   └── YYYYMMDD_NN_実験名/
│       ├── main.py         # 実験用スクリプト（自己完結）
│       ├── predictions/    # 予測結果
│       │   ├── oof.csv            # OOF予測（ID + 確率）
│       │   ├── test_proba.csv     # テスト予測確率
│       │   ├── test.csv           # テスト予測（提出用）
│       │   └── fold_indices.pkl   # CV fold情報（オプション）
│       ├── model/          # 学習済みモデル
│       └── README.md       # 実験の詳細記録
├── templates/              # 実験テンプレート（コピー元）
│   ├── classification_lgbm.py  # LightGBM分類テンプレート
│   └── regression_lgbm.py      # LightGBM回帰テンプレート
├── scripts/                # ユーティリティスクリプト
│   └── create_experiment.py    # 実験フォルダ自動生成CLI
├── .venv/                  # 仮想環境（gitignore対象）
├── pyproject.toml          # プロジェクト設定・依存関係
├── .python-version         # Python バージョン指定
├── .gitignore              # Git除外設定
└── CLAUDE.md               # このファイル
```

## データ形式

- データソース: IBM顧客離脱予測データセットからDeep Learningモデルで生成された合成データ
- 元データセット: https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm

### 訓練データ
- **ファイル**: `data/raw/train.csv`
- **行数**: 594,194件（21カラム）
- **ターゲット**: `Churn`（二値: Yes/No、Yes=22.5%, No=77.5%）
- **ID**: `id`（int64）
- **数値カラム**: `SeniorCitizen`(int), `tenure`(int), `MonthlyCharges`(float), `TotalCharges`(float)
- **カテゴリカラム**: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

### テストデータ
- **ファイル**: `data/raw/test.csv`
- **行数**: 254,655件（20カラム、`Churn`列を除く）

### 提出ファイル
```csv
id,Churn
594194,0.1
594195,0.3
594196,0.2
```
- `id`: テストデータのID
- `Churn`: 離脱確率（0〜1の確率値）

## コンペ知見（Discussion要点）

詳細は `docs/discussions/` を参照。以下は実験設計に影響する重要な知見：

- **特徴量間の相互作用がほぼない**: 元データで `max_depth=1` が最適。ロジスティック回帰が強い。合成データでは `max_depth=2〜4` が有効（Chris Deotte, 1st place）
- **合成データに2種類のシグナル**: (1) Real signal（元データ由来）と (2) Fake signal（合成アーティファクト）。両方を捉える必要がある
- **Fake signalの本質**: 元データ7k行 → train 600k行（各行約85コピー+ノイズ）。類似コピーを探すアプローチが有効（Chris Deotte）
- **TotalCharges のアーティファクト**: 元データでは `TotalCharges ≈ MonthlyCharges × tenure` が厳密に成立するが、合成データでは崩れている。`Charge_Difference = TotalCharges - MonthlyCharges * tenure` が有効な特徴量
- **train vs test に分布シフトなし** (AV AUC=0.51): CVスコアを信頼できる
- **train vs 元データにドリフトあり** (AV AUC=0.66): 元データを単純に結合するのは危険（-0.00078 OOF）
- **元データを参照分布として活用**: 結合ではなく、元データのChurn率を特徴量としてマッピング → 単一XGBで CV 0.919 / LB 0.917（008参照）
- **有効なモデル**: XGBoost, LightGBM, CatBoost, YDF, Logistic Regression, RealMLP, GNN, Bartz。多様性のあるアンサンブルが鍵
- **RealMLP (PyTabKit)**: 単体でCV 0.919+, LB 0.916+。n_ens=32が最も効果的。簡素なFE(42特徴量)でもツリーモデルと同等（012参照）
- **Hill Climbing > Stacking**: 特徴量交互作用が少ないためHill Climbingが有効。Stackingは交互作用が多い場合に有効（Chris Deotte）
- **Rank Averagingが安全なデフォルト**: 確率のキャリブレーション差に頑健。アンサンブル貢献度≠単体AUC、誤差が直交するモデルが重要（013参照）
- **TE-Pair LogRegがアンサンブルで高貢献**: 単体AUCが低くてもツリーモデルと直交する誤差パターンで高い重みを得る（013参照）
- **スコア目安**: 合成データのみで CV 0.9148〜0.9150 が天井。元データ活用やアンサンブルで 0.916+ が可能

### 特徴量エンジニアリング知見
- **数学的FEは無効**: GP(遺伝的プログラミング)で10,000世代探索しても既存特徴量のコピーしか見つからない（007参照）
- **カテゴリカル組み合わせが有効**: ペアワイズ組み合わせ + Target Encoding が推奨（007参照）
- **元データ参照特徴量が強力**: ORIG_proba（元データのChurn率マッピング）、Contract×InternetService単一特徴量AUC 0.859（008参照）
- **Digit/Modularアーティファクト**: tenure_mod12=1のChurn率41% vs 他9%。合成データ生成プロセスの痕跡（008参照）
- **Service Counts**: 契約サービス数合計、has_internet、has_phoneフラグ（008参照）
- **数値ビニング**: EDAベース（特徴量 vs ターゲット平均プロットで不連続点を発見）が推奨（011参照）
- **決定木の限界**: Greedyアルゴリズムのため、A単独・B単独にパターンがない交互作用は発見不可 → 明示的FE（A*B等）が必要（011参照）

### ツール
- **動的Webページの取得**: `playwright-cli` スキルを使用すること

### ファイルアクセス制限
- **リポジトリ外のファイル読み書き禁止**: プロジェクトディレクトリ外への Read/Write/Edit/Glob/Grep は PreToolUse hook でブロックされる
- **許可パス**: プロジェクトディレクトリ、`~/.claude/`、`~/.kaggle/` のみ

## 実験管理ルール

### 基本原則
- 各実験は独立したフォルダで管理し、再現性と追跡可能性を確保する
- **各実験の main.py は自己完結的に動作すること**（外部モジュールに依存しない）
- templates/ はコピー元のテンプレートであり、実験スクリプトから import しない
- **CLAUDE.md更新ルール**: リポジトリに変更を加えた際、その情報をAI Agentが今後の作業で考慮する必要がある場合は、本ファイル（CLAUDE.md）を更新すること

### 実験の作成方法
```bash
# 分類タスク（デフォルト）
uv run python scripts/create_experiment.py baseline_lgbm

# 回帰タスク
uv run python scripts/create_experiment.py baseline_lgbm --template regression
```
- テンプレートから新しい実験フォルダを自動生成
- 日付・連番は自動で設定される

### フォルダ命名規則
- **形式**: `YYYYMMDD_NN_実験名`（NNは連番: 01, 02, 03...）
- **例**: `20260313_01_data_validation`, `20260313_02_baseline_lgbm`
- **場所**: `experiments/` ディレクトリ配下
- **同日複数実験**: 連番により実行順序を明確化

### 実験フォルダ構成
各実験フォルダには以下を含める：

1. **main.py** (必須)
   - データ読み込みから予測まで一連の処理を実装
   - 自己完結的に動作するPythonスクリプト
   - コマンドライン引数でパラメータ調整可能にする

2. **predictions/** (必須)
   - `oof.csv`: OOF予測結果（ID, 確率, 正解）
   - `test_proba.csv`: テスト予測確率（ID, 確率）
   - `test.csv`: テスト予測（提出用: `id,Churn`）
   - `fold_indices.pkl`: CV fold情報（オプション、再現性確保のため）

3. **model/** (必要に応じて)
   - 学習済みモデルファイル
   - CV使用時は各foldのモデルを保存（例: `fold_0.txt`, `fold_1.txt`, ...）

4. **README.md** (必須)
   - 実験の詳細な記録（目的、アプローチ、結果、次のステップ）

### 実験履歴の記録
- 各実験完了後、本ファイル（CLAUDE.md）の「実験履歴」セクションに要約を追加
- 見出しは実験フォルダ名と一致させる（例: `### 20260313_01_data_validation`）
- 目的、アプローチ、結果、次のステップを記録

### コミットルール
- 実験フォルダ作成時にコミット
- 実験完了・中断時にコミット
- コミットメッセージ例: `Add experiment: 20260313_01_data_validation`
- pushはユーザーから明示的に指示されたときのみ実行すること

## 実験履歴

### 20260313_01_eda
**目的**: データの全体像を把握し、各特徴量の分布・Churnとの関係・合成アーティファクトを分析
**アプローチ**: 基本統計量、数値/カテゴリ特徴量の分布可視化、Churn率分析、Charge_Differenceアーティファクト分析、相関行列、train vs test分布比較、クロス分析（9枚の図を生成）
**結果**:
- Churn率: 22.52%（No=77.5%, Yes=22.5%）、欠損値なし
- 最強のChurn予測因子: Contract（Month-to-month: 42.1% vs Two year: 1.0%）、PaymentMethod（Electronic check: 48.9%）
- 数値特徴量の相関: tenure(r=-0.418) > MonthlyCharges(r=+0.273) > SeniorCitizen(r=+0.236) > TotalCharges(r=-0.218)
- Charge_Difference: Churnとの相関は弱い(r=+0.037)が、合成アーティファクトとしてモデルが活用可能
- train/test間の分布シフトなし → CVスコアを信頼できる
- セキュリティ/サポート系サービス未加入者のChurn率が顕著に高い（40%超）
**次のステップ**: ベースラインモデル構築（LightGBM）

---

### 20260313_02_logistic_regression
**目的**: ロジスティック回帰によるベースライン構築。特徴量が独立的であるため強いモデルが期待される（Discussion知見）
**アプローチ**: One-Hot Encoding（31特徴量） + StandardScaler + LogisticRegression、Charge_Difference特徴量追加、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9079**（各Fold: 0.9075, 0.9089, 0.9081, 0.9091, 0.9061）
- OOF LogLoss: 0.3124, Accuracy: 0.8545
- 重要な特徴量（係数の絶対値順）: tenure, TotalCharges, Contract_Two year, InternetService_Fiber optic, PaymentMethod_Electronic check
- Confusion Matrix: FN=46,088（Churn=Yesを見逃し）、FP=40,364
**次のステップ**: LightGBMベースラインとの比較、アンサンブル候補としてOOF予測を保存済み

---

### 20260313_03_xgboost_baseline
**目的**: XGBoostによるベースライン構築（通常パラメータ、max_depth=6）
**アプローチ**: Label Encoding + Charge_Difference（20特徴量）、XGBoost (hist, max_depth=6, lr=0.05)、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9164**（各Fold: 0.9160, 0.9170, 0.9165, 0.9176, 0.9148）
- OOF LogLoss: 0.2977, Accuracy: 0.8615
- 重要な特徴量（Gain順）: Contract, OnlineSecurity, TechSupport, InternetService, tenure
- LB AUC-ROC: **0.91391**
**次のステップ**: LightGBMベースライン、max_depth=1実験、アンサンブル

---

### 20260313_04_xgboost_depth1
**目的**: max_depth=1でのXGBoost実験（Discussion知見: 元データの特徴量が独立的でdepth=1が最適）
**アプローチ**: 03と同一特徴量、XGBoost (hist, max_depth=1, lr=0.05)、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9135**（max_depth=6の0.9164から-0.003）
- 全foldで2000ラウンド上限到達（early stopping未発動）
- 合成データのfake signalはdepth>1で捉えられるため、depth=1では若干劣る結果
**次のステップ**: LightGBMベースライン、アンサンブル

---

### 20260313_05_lightgbm_baseline
**目的**: LightGBMによるベースライン構築（通常パラメータ）
**アプローチ**: Label Encoding + Charge_Difference（20特徴量）、LightGBM (gbdt, num_leaves=31, lr=0.05)、categorical_feature指定、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9163**（XGBoost depth=6の0.9164とほぼ同等）
- OOF LogLoss: 0.2978, Accuracy: 0.8615
- Best iteration: 575〜724（XGBoostより少ないラウンドで収束）
**次のステップ**: アンサンブル、CatBoost実験

---

### 20260313_06_bartz_baseline
**目的**: Bartz（Bayesian Additive Regression Trees）によるベースライン構築。MCMCベースの手法でアンサンブル多様性を確保
**アプローチ**: Target Encoding（CV-based） + Pairwise interaction features（120個） = 140特徴量、Bartz (maxdepth=6, ntree=400, k=5)、MCMC (NDPOST=1000, NSKIP=200, KEEPEVERY=2)、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9158**（各Fold: 0.9154, 0.9164, 0.9156, 0.9171, 0.9144）
- OOF LogLoss: 0.3051, Accuracy: 0.8615
- MCMC予測値が[0,1]範囲外になる問題をclipで対処
- 実行時間: 約2時間（CPU）
- LB AUC-ROC: **0.91405**（全モデル中最高）
**次のステップ**: アンサンブル（LR + XGB + LGB + Bartz）、CatBoost実験

---

### 20260315_01_ydf_baseline
**目的**: YDF Discussion知見のパラメータ（grow_policy='lossguide', max_depth=2）をXGBoostで再現
**アプローチ**: Label Encoding + Charge_Difference（20特徴量）、XGBoost (hist, grow_policy=lossguide, max_depth=2, lr=0.05)、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9159**（各Fold: 0.9155, 0.9165, 0.9161, 0.9170, 0.9144）
- OOF LogLoss: 0.2985, Accuracy: 0.8611
- 全foldで2000ラウンド上限到達（early stopping未発動、ラウンド数増加で改善余地あり）
- LB AUC-ROC: **0.91311**
**次のステップ**: ラウンド数増加実験、アンサンブル、ハイパーパラメータチューニング

---

### 20260323_01_orig_data_reference
**目的**: 元データ（IBM Telco 7,043行）を参照分布として特徴量生成し、XGBoostのスコア向上を図る
**アプローチ**: 実験03ベース + 9つの特徴量グループ（Freq Encoding, Arithmetic, Service Counts, ORIG_proba single/cross, Distribution, Quantile Distance, Digit/Modular, N-gram） + Nested Target Encoding（Inner 5-fold）。129基本特徴量 → 201モデル特徴量。XGBoost (hist, max_depth=6, lr=0.05)、5-Fold Stratified CV
**結果**:
- OOF AUC-ROC: **0.9185**（各Fold: 0.9184, 0.9191, 0.9185, 0.9198, 0.9169）
- OOF LogLoss: 0.2943, Accuracy: 0.8635
- ベースライン（実験03: 0.9164）から **+0.0021** の改善
- 参考Notebook（10-fold, Optuna最適化）のCV 0.9190にほぼ匹敵
- LB AUC-ROC: **0.91644**（全提出中最高、前最高Bartz 0.91405から+0.00239）
**次のステップ**: Optunaチューニング、10-fold CV実験、アンサンブル
