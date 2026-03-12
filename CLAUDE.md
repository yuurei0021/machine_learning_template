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
- 全体サイズ: 116.57 MB、43カラム
- カラム型: String 22列、Boolean 9列、Integer 5列、Other 7列
- 元データセット: https://www.kaggle.com/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm

### 訓練データ
- **ファイル**: `data/raw/train.csv`
- **ターゲット**: `Churn`（二値）
- **ID**: `id`

### テストデータ
- **ファイル**: `data/raw/test.csv`
- **カラム**: 訓練データと同じ（`Churn`列を除く）

### 提出ファイル
```csv
id,Churn
594194,0.1
594195,0.3
594196,0.2
```
- `id`: テストデータのID
- `Churn`: 離脱確率（0〜1の確率値）

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

<!--
### YYYYMMDD_NN_実験名
**目的**:
**アプローチ**:
**結果**:
**次のステップ**:
-->
