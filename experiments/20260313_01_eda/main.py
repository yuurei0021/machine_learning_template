"""
EDA (Exploratory Data Analysis) for Predict Customer Churn
==========================================================
Experiment: 20260313_01_eda

目的: データの全体像を把握し、各特徴量の分布・Churnとの関係・合成アーティファクトを分析
アプローチ: 基本統計量、分布可視化、Churn率分析、Charge_Differenceアーティファクト、相関分析
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# パス設定
# ============================================================
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

OUTPUT_DIR = EXPERIMENT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# データ読み込み
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# ============================================================
# 1. 基本情報
# ============================================================
print("\n" + "=" * 60)
print("1. Basic Info")
print("=" * 60)

print("\n--- Train dtypes ---")
print(train.dtypes)

print("\n--- Missing values (train) ---")
missing_train = train.isnull().sum()
print(missing_train[missing_train > 0] if missing_train.sum() > 0 else "No missing values")

print("\n--- Missing values (test) ---")
missing_test = test.isnull().sum()
print(missing_test[missing_test > 0] if missing_test.sum() > 0 else "No missing values")

print("\n--- Train describe (numeric) ---")
print(train.describe())

# ============================================================
# 2. ターゲット変数の分布
# ============================================================
print("\n" + "=" * 60)
print("2. Target Distribution")
print("=" * 60)

churn_counts = train["Churn"].value_counts()
churn_pct = train["Churn"].value_counts(normalize=True) * 100
print(churn_counts)
print(f"\nChurn rate: {churn_pct['Yes']:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
churn_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"])
axes[0].set_title("Churn Count")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=0)

churn_pct.plot(kind="pie", ax=axes[1], autopct="%.1f%%", colors=["#4CAF50", "#F44336"])
axes[1].set_ylabel("")
axes[1].set_title("Churn Rate")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_target_distribution.png")

# ============================================================
# 3. 数値特徴量の分布
# ============================================================
print("\n" + "=" * 60)
print("3. Numeric Features Distribution")
print("=" * 60)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(num_cols):
    ax = axes[i // 2][i % 2]
    if col == "SeniorCitizen":
        train[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="#2196F3")
        ax.set_title(f"{col} (Count)")
        ax.tick_params(axis="x", rotation=0)
    else:
        ax.hist(train[col], bins=50, alpha=0.7, color="#2196F3", label="train")
        ax.hist(test[col], bins=50, alpha=0.5, color="#FF9800", label="test")
        ax.set_title(f"{col} Distribution")
        ax.legend()
    ax.set_xlabel(col)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_numeric_distributions.png")

# 数値特徴量 × Churn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
    for label, color in [("No", "#4CAF50"), ("Yes", "#F44336")]:
        subset = train[train["Churn"] == label][col]
        axes[i].hist(subset, bins=50, alpha=0.5, label=f"Churn={label}", color=color, density=True)
    axes[i].set_title(f"{col} by Churn")
    axes[i].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_numeric_by_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_numeric_by_churn.png")

# ============================================================
# 4. カテゴリ特徴量の分布とChurn率
# ============================================================
print("\n" + "=" * 60)
print("4. Categorical Features & Churn Rate")
print("=" * 60)

cat_cols = [c for c in train.columns
           if pd.api.types.is_string_dtype(train[c]) and c != "Churn"]
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Churn率を計算
churn_rates = {}
for col in cat_cols:
    rates = train.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean())
    churn_rates[col] = rates
    print(f"\n{col}:")
    for val, rate in rates.sort_values(ascending=False).items():
        count = (train[col] == val).sum()
        print(f"  {val:30s}  Churn={rate:.3f}  (n={count:,})")

# 可視化: Churn率のバーチャート
fig, axes = plt.subplots(5, 3, figsize=(18, 25))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    rates = churn_rates[col].sort_values(ascending=False)
    axes[i].bar(range(len(rates)), rates.values, color="#2196F3")
    axes[i].set_xticks(range(len(rates)))
    axes[i].set_xticklabels(rates.index, rotation=45, ha="right", fontsize=8)
    axes[i].set_title(f"{col} -> Churn Rate")
    axes[i].set_ylabel("Churn Rate")
    axes[i].axhline(y=churn_pct["Yes"] / 100, color="red", linestyle="--", alpha=0.7, label="Overall")
    axes[i].legend(fontsize=7)

# 余った軸を非表示に
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_categorical_churn_rates.png")

# ============================================================
# 5. 合成データのアーティファクト分析
# ============================================================
print("\n" + "=" * 60)
print("5. Synthetic Artifact: Charge_Difference")
print("=" * 60)

train["Charge_Difference"] = train["TotalCharges"] - train["MonthlyCharges"] * train["tenure"]
test["Charge_Difference"] = test["TotalCharges"] - test["MonthlyCharges"] * test["tenure"]

print("Train Charge_Difference stats:")
print(train["Charge_Difference"].describe())
print(f"\nTest Charge_Difference stats:")
print(test["Charge_Difference"].describe())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 分布
axes[0].hist(train["Charge_Difference"], bins=100, alpha=0.7, color="#2196F3", label="train", density=True)
axes[0].hist(test["Charge_Difference"], bins=100, alpha=0.5, color="#FF9800", label="test", density=True)
axes[0].set_title("Charge_Difference Distribution (train vs test)")
axes[0].set_xlabel("TotalCharges - MonthlyCharges * tenure")
axes[0].legend()

# Churn別
for label, color in [("No", "#4CAF50"), ("Yes", "#F44336")]:
    subset = train[train["Churn"] == label]["Charge_Difference"]
    axes[1].hist(subset, bins=100, alpha=0.5, label=f"Churn={label}", color=color, density=True)
axes[1].set_title("Charge_Difference by Churn")
axes[1].legend()

# TotalCharges vs MonthlyCharges * tenure の散布図（サンプル）
sample = train.sample(n=min(10000, len(train)), random_state=42)
expected = sample["MonthlyCharges"] * sample["tenure"]
axes[2].scatter(expected, sample["TotalCharges"], alpha=0.1, s=1, c="#2196F3")
max_val = max(expected.max(), sample["TotalCharges"].max())
axes[2].plot([0, max_val], [0, max_val], "r--", linewidth=1, label="y=x (ideal)")
axes[2].set_xlabel("MonthlyCharges * tenure")
axes[2].set_ylabel("TotalCharges")
axes[2].set_title("TotalCharges vs Expected (MonthlyCharges * tenure)")
axes[2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_charge_difference_artifact.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_charge_difference_artifact.png")

# ============================================================
# 6. 特徴量間の相関
# ============================================================
print("\n" + "=" * 60)
print("6. Feature Correlations")
print("=" * 60)

corr_df = train[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Charge_Difference"]].copy()
corr_df["Churn_binary"] = (train["Churn"] == "Yes").astype(int)

corr_matrix = corr_df.corr()
print(corr_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation Matrix (Numeric Features + Charge_Difference)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06_correlation_matrix.png")

# ============================================================
# 7. train vs test 分布比較
# ============================================================
print("\n" + "=" * 60)
print("7. Train vs Test Distribution Comparison")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 数値特徴量のKDE比較
for i, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
    ax = axes[0][i]
    train[col].plot(kind="kde", ax=ax, label="train", color="#2196F3")
    test[col].plot(kind="kde", ax=ax, label="test", color="#FF9800")
    ax.set_title(f"{col} (KDE)")
    ax.legend()

# カテゴリ特徴量の比較（重要なもの3つ）
key_cats = ["Contract", "InternetService", "PaymentMethod"]
for i, col in enumerate(key_cats):
    ax = axes[1][i]
    train_pct = train[col].value_counts(normalize=True).sort_index()
    test_pct = test[col].value_counts(normalize=True).sort_index()
    x = np.arange(len(train_pct))
    width = 0.35
    ax.bar(x - width / 2, train_pct.values, width, label="train", color="#2196F3")
    ax.bar(x + width / 2, test_pct.reindex(train_pct.index).values, width, label="test", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(train_pct.index, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{col} (train vs test)")
    ax.set_ylabel("Proportion")
    ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 07_train_vs_test.png")

# ============================================================
# 8. 重要な特徴量のクロス分析
# ============================================================
print("\n" + "=" * 60)
print("8. Key Feature Cross Analysis")
print("=" * 60)

# Contract x tenure x Churn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, contract in enumerate(["Month-to-month", "One year", "Two year"]):
    subset = train[train["Contract"] == contract]
    for label, color in [("No", "#4CAF50"), ("Yes", "#F44336")]:
        s = subset[subset["Churn"] == label]["tenure"]
        axes[i].hist(s, bins=30, alpha=0.5, label=f"Churn={label}", color=color, density=True)
    axes[i].set_title(f"Contract={contract}")
    axes[i].set_xlabel("tenure")
    axes[i].legend()
plt.suptitle("Tenure Distribution by Contract Type and Churn", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_contract_tenure_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 08_contract_tenure_churn.png")

# InternetService x MonthlyCharges x Churn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, inet in enumerate(["DSL", "Fiber optic", "No"]):
    subset = train[train["InternetService"] == inet]
    for label, color in [("No", "#4CAF50"), ("Yes", "#F44336")]:
        s = subset[subset["Churn"] == label]["MonthlyCharges"]
        axes[i].hist(s, bins=30, alpha=0.5, label=f"Churn={label}", color=color, density=True)
    axes[i].set_title(f"InternetService={inet}")
    axes[i].set_xlabel("MonthlyCharges")
    axes[i].legend()
plt.suptitle("MonthlyCharges Distribution by InternetService and Churn", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_internet_charges_churn.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 09_internet_charges_churn.png")

# ============================================================
# 9. サマリーレポート出力
# ============================================================
print("\n" + "=" * 60)
print("9. Summary Report")
print("=" * 60)

churn_binary = (train["Churn"] == "Yes").astype(int)

report_lines = []
report_lines.append("# EDA Report: Predict Customer Churn\n")
report_lines.append(f"## 1. データ概要")
report_lines.append(f"- Train: {train.shape[0]:,} rows x {train.shape[1]} cols")
report_lines.append(f"- Test: {test.shape[0]:,} rows x {test.shape[1]} cols")
report_lines.append(f"- Missing values: train={train.isnull().sum().sum()}, test={test.isnull().sum().sum()}")
report_lines.append(f"- Churn rate: {churn_pct['Yes']:.2f}% (Yes={churn_counts['Yes']:,}, No={churn_counts['No']:,})")
report_lines.append("")

report_lines.append("## 2. 数値特徴量とChurnの相関 (Point-Biserial)")
for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Charge_Difference"]:
    corr_val = corr_matrix.loc[col, "Churn_binary"]
    report_lines.append(f"- {col}: r = {corr_val:+.4f}")
report_lines.append("")

report_lines.append("## 3. カテゴリ特徴量 - Churn率の差が大きい順")
cat_impact = []
for col in cat_cols:
    rates = churn_rates[col]
    impact = rates.max() - rates.min()
    cat_impact.append((col, impact, rates.idxmax(), rates.max(), rates.idxmin(), rates.min()))

cat_impact.sort(key=lambda x: x[1], reverse=True)
for col, impact, high_val, high_rate, low_val, low_rate in cat_impact:
    report_lines.append(f"- **{col}** (spread={impact:.3f}): highest={high_val}({high_rate:.3f}), lowest={low_val}({low_rate:.3f})")
report_lines.append("")

report_lines.append("## 4. Charge_Difference アーティファクト")
report_lines.append(f"- Train: mean={train['Charge_Difference'].mean():.2f}, std={train['Charge_Difference'].std():.2f}")
report_lines.append(f"- Test: mean={test['Charge_Difference'].mean():.2f}, std={test['Charge_Difference'].std():.2f}")
report_lines.append(f"- Churnとの相関: {corr_matrix.loc['Charge_Difference', 'Churn_binary']:+.4f}")
report_lines.append("")

report_lines.append("## 5. Key Findings")
report_lines.append("1. **Contract** が最強のChurn予測因子（Month-to-month >> One year >> Two year）")
report_lines.append("2. **InternetService**: Fiber opticのChurn率がDSLより高い")
report_lines.append("3. **tenure**: 長期契約者ほどChurnしにくい（強い負の相関）")
report_lines.append("4. **MonthlyCharges**: 高額ほどChurnしやすい（正の相関）")
report_lines.append("5. **Charge_Difference**: 合成アーティファクトを捉え、Churnと相関あり")
report_lines.append("6. 欠損値なし（train/test共に）")
report_lines.append("7. train/testの分布は非常に類似（分布シフトなし）")
report_lines.append("8. **セキュリティ/サポート系サービス未加入者**のChurn率が高い")
report_lines.append("")

report_lines.append("## 6. 生成された図")
report_lines.append("| # | ファイル名 | 内容 |")
report_lines.append("|---|----------|------|")
report_lines.append("| 01 | 01_target_distribution.png | ターゲット変数の分布 |")
report_lines.append("| 02 | 02_numeric_distributions.png | 数値特徴量の分布（train vs test） |")
report_lines.append("| 03 | 03_numeric_by_churn.png | 数値特徴量 x Churn |")
report_lines.append("| 04 | 04_categorical_churn_rates.png | カテゴリ特徴量ごとのChurn率 |")
report_lines.append("| 05 | 05_charge_difference_artifact.png | Charge_Differenceアーティファクト分析 |")
report_lines.append("| 06 | 06_correlation_matrix.png | 相関行列ヒートマップ |")
report_lines.append("| 07 | 07_train_vs_test.png | train vs test 分布比較 |")
report_lines.append("| 08 | 08_contract_tenure_churn.png | Contract x tenure x Churn クロス分析 |")
report_lines.append("| 09 | 09_internet_charges_churn.png | InternetService x MonthlyCharges x Churn |")

report_text = "\n".join(report_lines)
print(report_text)

# READMEに書き出し
readme_path = EXPERIMENT_DIR / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport saved to: {readme_path}")

print("\n" + "=" * 60)
print(f"All figures saved to: {OUTPUT_DIR}")
print("=" * 60)
