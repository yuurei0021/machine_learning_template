# 012: RealMLP Parameter Adjustment Experiment

- **URL**: https://www.kaggle.com/competitions/playground-series-s6e3/discussion/683316
- **Author**: YUNSUXIAOZI (27th place)
- **Upvotes**: 9

## 概要

RealMLP（PyTabKit）のパラメータ調整実験。参照Notebook「PS|S6|E3: RealMLP - PyTabKit」(by yekenot) をベースに各パラメータの影響を検証。

## パラメータ調整結果

| Method | valid(20%) 1-AUC | Improvement |
|---|---|---|
| origin parameter | 0.082573 | -- |
| without tfms | 0.082590 | No |
| add_front_scale=True | 0.082763 | No |
| without bias_init_mode | 0.082538 | Yes |
| n_ens=16 | 0.082451 | Yes |
| ls_eps=0.02 | 0.082541 | Yes |
| plr_lr_factor=0.2 | 0.082600 | No |
| embedding_size=8 | 0.082487 | Yes |
| n_ens=32 | 0.082419 | Yes |
| without TE | 0.082486 | Yes |

**注**: 1-AUCが低いほど良い。n_ens=32が最も効果的（AUC ~0.9176）。

## 参照Notebookスコア比較

| Notebook | Model | CV | LB |
|---|---|---|---|
| RealMLP - PyTabKit | RealMLP | 0.91912 | 0.91658 |
| TabM + Advanced Features | TabM | 0.91898 | 0.91682 |
| Ridge XGB N-gram | XGBoost | 0.91927 | 0.91684 |

## RealMLPベースパラメータ

```python
params = {
    'random_state': 42,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',
    'n_ens': 8,
    'n_epochs': 3,
    'batch_size': 256,
    'use_early_stopping': True,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,
    'lr': 0.075,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',
    'first_layer_lr_factor': 0.25,
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',
    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'flat_cos',
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    'ls_eps': 0.01,
    'ls_eps_sched': 'cos',
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}
```

## 主要コメント

### Vladimir Demidov (30th, 8 upvotes)
- RealMLPは「マスターピース」: 深層学習の本質はモデルの深さではなく、最適化の豊かさと前処理の巧みさ
- レイヤーごと・特徴量ごとのLRスケーリングと動的正則化による自己制御
- 特徴量選択が重要: AI生成特徴量を盲目的に追加しない
- パラメータ詳細は `class RealMLPConstructorMixin` と `def get_schedule()` を参照

### Tilii (10th, 4 upvotes)
- 5つの算術特徴量 + 3つのカテゴリカル特徴量を追加（合計42特徴量）
- RealMLPのパラメータは変更せず → **CV: 0.91927, LB: 0.91674**
- エポック数が少ないとLRスケジューラーのペースが変わる

### Dhruv Pai Dukle (178th)
- RealMLP + LightGBM + CatBoost のアンサンブルで **LB 0.91663**

## 実験設計への示唆

1. **RealMLPは強力なNNモデル**: 単体でCV 0.919+, LB 0.916+（ツリーモデルと同等）
2. **n_ensの増加が最も効果的**: n_ens=32で最良結果
3. **アンサンブル多様性**: ツリーモデルと異なるアルゴリズムなので、アンサンブルで有効
4. **ライブラリ**: `pytabkit` パッケージの `RealMLP_TD_Classifier`
5. **簡素なFEでも高性能**: 42特徴量でCV 0.919+を達成
