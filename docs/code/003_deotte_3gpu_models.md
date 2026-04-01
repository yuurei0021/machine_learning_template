# 003: ChatGPT Vibe Coding - 3 GPU Models | CV 0.9178

- **著者**: Chris Deotte (1st place)
- **URL**: https://www.kaggle.com/code/cdeotte/chatgpt-vibe-coding-3xgpu-models-cv-0-9178
- **CV**: 0.9178（3モデルアンサンブル）、個別: LogReg 0.9160, XGB 0.9166, MLP 0.9164
- **Votes**: 94

## アーキテクチャ: 3つの多様なGPUモデル

### Model 1: Logit3 TE-Pair LogReg（CV 0.9160）

質的に異なるLogistic Regression:

1. 全ペアワイズ特徴量組み合わせ（C(19,2) = 171ペア）を生成
2. cuML TargetEncoderで各ペアをTE → 171個のTE値
3. Logit変換: `z = logit(TE_value)`
4. 多項式展開: `[z, z^2, z^3]` → 513特徴量
5. StandardScale → cuML LogisticRegression（L2, C=0.5）

**ポイント**: z^2, z^3 が TE値の非線形パターンを線形モデルで捉えられるようにする。OHEベースの従来LogRegとは全く異なるアプローチ。アンサンブルでツリーモデルと直交する誤差パターンを生成。

### Model 2: XGB（CV 0.9166）

**特徴量エンジニアリングなし**。生の特徴量 + `enable_categorical=True` のみ。

| パラメータ | 値 |
|---|---|
| max_depth | 3 |
| learning_rate | 0.1 |
| min_child_weight | 5 |
| subsample | 0.85 |
| colsample_bytree | 0.85 |

**示唆**: 重いFEを積み上げたXGBとFEなしXGBのCV差は小さい（0.9166 vs 当方0.9164 baseline）。FEの限界収益逓減を示す。ただしアンサンブル時の多様性確保には有効。

### Model 3: PyTorch MLP（CV 0.9164）

数値→カテゴリ変換（snapping）手法:

1. train fold内の各数値列の value_counts を計算
2. 出現回数 ≥ 25 の値を「frequent values」として特定
3. 希少値は最近傍の frequent value に snap
4. snap後の離散値に embedding を割り当て

| パラメータ | 値 |
|---|---|
| Embedding dim | `1.8 * cardinality^0.25`（4〜64） |
| Hidden layers | 512, 256 |
| Dropout | 0.30 |
| Embedding dropout | 0.10 |
| Learning rate | 2.5e-5 |
| Weight decay | 3e-4 |
| Epochs | 10 |
| Batch size | 256 |
| Scheduler | Cosine with warmup |

### cuML TargetEncoder: smooth=0

スムージングなし（raw mean encoding）。sklearn の `smooth='auto'`（ベイズ平滑化）とは異なる。合成データのアーティファクトをより強く捉える。

## 当方CVとの比較

| モデル | NB4 | 当方 | 差分 |
|---|---|---|---|
| LogReg | 0.9160 (Logit3) | 0.91579 | **-0.00021** |
| XGB（FEなし） | 0.9166 | 0.91640 (baseline) | -0.00020 |
| MLP | 0.9164 | **0.91720** | **+0.00080** |
| アンサンブル | 0.9178 | **0.91939** (HC) | **+0.00159** |

当方MLPとアンサンブルが上回るが、Logit3 LogRegは未実装で多様性に貢献する可能性あり。

## 実装への示唆

1. **Logit3 TE-Pair LogReg**: アンサンブル多様性の大幅向上。ツリー/NNと直交する誤差パターン。実装コストは高いがHill Climbingでの貢献度が高い可能性
2. **smooth=0のTE**: 合成データ向け。簡単に試せる
3. **数値→カテゴリsnapping**: TabM/RealMLPに応用可能だが、pytabkitの内部処理と重複する可能性
4. **FEなしXGB**: 多様性確保用。当方baseline（0.9164）と同等スコアで既に保有
