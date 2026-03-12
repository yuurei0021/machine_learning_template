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
- **TotalCharges のアーティファクト**: 元データでは `TotalCharges ≈ MonthlyCharges × tenure` が厳密に成立するが、合成データでは崩れている。`Charge_Difference = TotalCharges - MonthlyCharges * tenure` が有効な特徴量
- **train vs test に分布シフトなし** (AV AUC=0.51): CVスコアを信頼できる
- **train vs 元データにドリフトあり** (AV AUC=0.66): 元データを単純に結合するのは危険
- **有効なモデル**: XGBoost, LightGBM, CatBoost, YDF, Logistic Regression, MLP, GNN, Bartz。多様性のあるアンサンブルが鍵
- **スタッキング**: OOF予測を別モデルの特徴量に追加するテクニックが有効
- **スコア目安**: 合成データのみで CV 0.9148〜0.9150 が天井。元データ活用やアンサンブルで 0.916+ が可能

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
