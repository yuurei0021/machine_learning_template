"""
TabTransformer + TE-Pair Logit3 + Ridge stacking
Based on: https://www.kaggle.com/code/include4eto/tabtransfomer-chatgpt-vibe-coding
Experiment: 20260331_06_tabtransformer
"""

import os
import gc
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import Ridge

import cudf
import cupy as cp
from cuml.preprocessing import TargetEncoder

warnings.filterwarnings("ignore")

# ============================================================================
# Paths (Kaggle)
# ============================================================================
COMP_DIR = Path("/kaggle/input/competitions/playground-series-s6e3")
ORIG_DIR = Path("/kaggle/input/telco-customer-churn")
OUTPUT_DIR = Path("/kaggle/working")

TRAIN_PATH = COMP_DIR / "train.csv"
TEST_PATH = COMP_DIR / "test.csv"
ORIG_PATH = ORIG_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ============================================================================
# Seed
# ============================================================================
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

seed_everything(seed=42)

# ============================================================================
# Load Data
# ============================================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Find original dataset with fallback glob search
import glob as _glob
orig_path = str(ORIG_PATH)
if not os.path.exists(orig_path):
    print(f"  Primary path not found: {orig_path}")
    print(f"  /kaggle/input/ contents: {os.listdir('/kaggle/input/')}")
    for d in os.listdir("/kaggle/input/"):
        full = f"/kaggle/input/{d}"
        if os.path.isdir(full):
            sub = os.listdir(full)
            print(f"    {full}/: {sub[:5]}")
            for s in sub:
                sf = f"{full}/{s}"
                if os.path.isdir(sf):
                    print(f"      {sf}/: {os.listdir(sf)[:5]}")
    candidates = _glob.glob("/kaggle/input/**/*.csv", recursive=True)
    print(f"  Scanning {len(candidates)} CSV files...")
    for c in candidates:
        basename = os.path.basename(c).lower()
        if ("churn" in basename or "telco" in basename) \
           and "train" not in basename and "test" not in basename \
           and "sample" not in basename and "submission" not in basename:
            orig_path = c
            print(f"  Using orig: {orig_path}")
            break
orig = pd.read_csv(orig_path)

_tc = pd.to_numeric(orig.TotalCharges, errors="coerce")
_tc[_tc.isnull()] = orig[_tc.isnull()].MonthlyCharges * orig[_tc.isnull()].tenure
orig.TotalCharges = _tc
print("Train shape:", train.shape)
print("Test shape :", test.shape)

# ============================================================================
# Feature Engineering
# ============================================================================
ID = "id"
TARGET = "Churn"
train[TARGET] = train[TARGET].map({"Yes": 1, "No": 0})
orig[TARGET] = orig[TARGET].map({"Yes": 1, "No": 0})

X = train.drop([ID, TARGET], axis=1)
train_id = train[ID]
y = train[TARGET]

X_test = test.drop([ID], axis=1)
test_id = test[ID]
X_orig = orig.drop(["customerID", TARGET], axis=1)
y_orig = orig[TARGET]

print("X      init shape:", X.shape)
print("X_test init shape:", X_test.shape)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()
print("init len(cat_cols):", len(cat_cols))
print("init len(num_cols):", len(num_cols))

category_map = {}

def feature_engineering(df, fit=False):
    # Arithmetic interaction
    df["_MonthlyCharges_DIV_TotalCharges"] = (
        df["MonthlyCharges"] / (df["TotalCharges"] + 1e-6)
    ).astype("float32")
    df["_TotalCharges_DIV_tenure"] = (
        df["TotalCharges"] / (df["tenure"] + 1e-6)
    ).astype("float32")
    df["_TotalCharges_DIV_MonthlyCharges"] = (
        df["TotalCharges"] / (df["MonthlyCharges"] + 1e-6)
    ).astype("float32")

    # Digit extraction
    for col in ["TotalCharges"]:
        k = -3
        digit_name = f"{col}_d{k}_"
        df[digit_name] = ((df[col] * 10**k) % 10).astype("int8")
        df[digit_name] = df[digit_name].astype("category")

    # Discretize numericals
    bin_config = {"TotalCharges": [4000, 500], "MonthlyCharges": [200, 100]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_bin_"
            if fit:
                kb = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode="ordinal",
                    strategy="quantile",
                    subsample=None,
                )
                binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
                category_map[bin_name] = kb
            else:
                kb = category_map[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype("int32")
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype("category")

    # Create interaction categories
    important_combos = [
        ("Contract", "InternetService", "PaymentMethod"),
    ]
    combo_names = []
    for col1, col2, col3 in important_combos:
        combo_name = f"{col1}_{col2}_{col3}_"
        combo_names.append(combo_name)
        combo_series = (
            df[col1].astype(str) + "_" + df[col2].astype(str) + "_" + df[col3].astype(str)
        )
        if fit:
            codes, uniques = combo_series.factorize()
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype("category")

    # Categorize numericals
    for col in num_cols:
        cat_name = f"{col}_cat_"
        round_level = 0
        if fit:
            round_flag = col == "TotalCharges"
            series = df[col].round(round_level) if round_flag else df[col]
            codes, uniques = series.factorize()
            category_map[col] = {"uniques": uniques, "round_flag": round_flag}
        else:
            round_flag = category_map[col]["round_flag"]
            uniques = category_map[col]["uniques"]
            series = df[col].round(round_level) if round_flag else df[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = series.map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")

    new_cat_cols = [col for col in df.columns if col.endswith("_")]
    new_num_cols = [col for col in df.columns if col.startswith("_")]
    return df, new_cat_cols, new_num_cols


df_all = pd.concat([X, X_test, X_orig], axis=0).reset_index(drop=True)
print("Total rows for FE:", len(df_all))
df_tf, new_cat_cols, new_num_cols = feature_engineering(df_all, fit=True)

X = df_tf.iloc[: len(train)]
X_test = df_tf.iloc[len(train) : len(train) + len(test)]
X_orig = df_tf.iloc[len(train) + len(test) :]

cat_cols += new_cat_cols
num_cols += new_num_cols
print("len(new_cat_cols):", len(new_cat_cols))
print("len(new_num_cols):", len(new_num_cols))
print("prep len(cat_cols):", len(cat_cols))
print("prep len(num_cols):", len(num_cols))
print("X      prep shape:", X.shape)
print("X_test prep shape:", X_test.shape)

# Cast categories to string
for c in cat_cols:
    X[c] = X[c].astype(str).astype("category")
    X_test[c] = X_test[c].astype(str).astype("category")
    X_orig[c] = X_orig[c].astype(str).astype("category")

# ============================================================================
# CONFIG
# ============================================================================
SEED = 0
N_SPLITS = 5
TE_INNER_SPLITS = 5
BATCH_SIZE = 4096
EPOCHS = 5

LEARNING_RATE = 1e-3
USE_X_ORIG = True
USE_LOGIT3 = True
MAX_PAIRS = None
TE_SMOOTH = 0
EMBED_DIM = 16
NUM_HEADS = 4
FF_DIM = 64
NUM_TRANSFORMER_BLOCKS = 2
MLP_HIDDEN = (128, 64)
DROPOUT = 0.2

keras.utils.set_random_seed(SEED)
outer_kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ============================================================================
# BASIC COLUMN SETUP
# ============================================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X = X.copy()
X_test = X_test.copy()
X_orig = X_orig.copy()

for c in cat_cols:
    X[c] = X[c].astype(str).fillna("__MISSING__")
    X_test[c] = X_test[c].astype(str).fillna("__MISSING__")
    X_orig[c] = X_orig[c].astype(str).fillna("__MISSING__")

for c in num_cols:
    X[c] = X[c].astype("float32")
    X_test[c] = X_test[c].astype("float32")
    X_orig[c] = X_orig[c].astype("float32")

# ============================================================================
# CATEGORICAL VOCAB FOR KERAS STRINGLOOKUP
# ============================================================================
cat_vocab = {}
for c in cat_cols:
    vals = pd.concat([X[c], X_test[c], X_orig[c]], axis=0).astype(str).unique().tolist()
    cat_vocab[c] = sorted(vals)

# ============================================================================
# INTEGER-ENCODED MIRROR DATA FOR GPU TARGET ENCODING
# ============================================================================
def label_encode_frames(train_df, test_df, orig_df, cols):
    train_out = train_df.copy()
    test_out = test_df.copy()
    orig_out = orig_df.copy()
    mappings = {}
    for c in cols:
        all_vals = pd.concat(
            [
                train_out[c].astype("string").fillna("__MISSING__"),
                test_out[c].astype("string").fillna("__MISSING__"),
                orig_out[c].astype("string").fillna("__MISSING__"),
            ],
            axis=0,
        )
        uniq = pd.Index(all_vals.unique())
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        mappings[c] = mapping
        train_out[c] = train_out[c].astype("string").fillna("__MISSING__").map(mapping).astype(np.int32)
        test_out[c] = test_out[c].astype("string").fillna("__MISSING__").map(mapping).astype(np.int32)
        orig_out[c] = orig_out[c].astype("string").fillna("__MISSING__").map(mapping).astype(np.int32)
    return train_out, test_out, orig_out, mappings


X_teenc, X_test_teenc, X_orig_teenc, _ = label_encode_frames(X, X_test, X_orig, cat_cols)

# ============================================================================
# PAIR LIST
# ============================================================================
pair_cols = list(combinations(cat_cols, 2))
if MAX_PAIRS is not None:
    pair_cols = pair_cols[:MAX_PAIRS]

print(f"num cat cols: {len(cat_cols)}")
print(f"num num cols: {len(num_cols)}")
print(f"num TE pairs: {len(pair_cols)}")

# ============================================================================
# GPU HELPERS
# ============================================================================
def to_cudf(df, cols):
    out = {}
    for c in cols:
        out[c] = cudf.Series(df[c].values).astype("int32")
    return cudf.DataFrame(out)


def make_logit3_features(tr_m, va_m, te_m, eps=1e-5):
    tr_m = np.clip(tr_m, eps, 1.0 - eps)
    va_m = np.clip(va_m, eps, 1.0 - eps)
    te_m = np.clip(te_m, eps, 1.0 - eps)

    z_tr = np.log(tr_m / (1.0 - tr_m)).astype(np.float32)
    z_va = np.log(va_m / (1.0 - va_m)).astype(np.float32)
    z_te = np.log(te_m / (1.0 - te_m)).astype(np.float32)

    X_tr = np.hstack([z_tr, z_tr**2, z_tr**3]).astype(np.float32)
    X_va = np.hstack([z_va, z_va**2, z_va**3]).astype(np.float32)
    X_te = np.hstack([z_te, z_te**2, z_te**3]).astype(np.float32)
    return X_tr, X_va, X_te


def build_pair_te_features_gpu(
    Xtr_int, ytr, Xva_int, Xte_int,
    pair_cols,
    inner_splits=5,
    seed=0,
    smooth=0,
    use_logit3=True,
):
    n_tr = len(Xtr_int)
    n_va = len(Xva_int)
    n_te = len(Xte_int)
    n_pair = len(pair_cols)

    tr_pair = np.zeros((n_tr, n_pair), dtype=np.float32)
    va_pair = np.zeros((n_va, n_pair), dtype=np.float32)
    te_pair = np.zeros((n_te, n_pair), dtype=np.float32)

    Xtr_g = to_cudf(Xtr_int, cat_cols)
    Xva_g = to_cudf(Xva_int, cat_cols)
    Xte_g = to_cudf(Xte_int, cat_cols)
    ytr_g = cudf.Series(ytr.astype(np.int32))

    te = TargetEncoder(
        n_folds=inner_splits,
        smooth=smooth,
        seed=seed,
        split_method="random",
        stat="mean",
        output_type="cupy",
    )

    for j, (f1, f2) in enumerate(pair_cols):
        if j == 0 or (j + 1) % 20 == 0 or (j + 1) == n_pair:
            print(f"  pair {j+1:>4d}/{n_pair}: {f1} x {f2}")

        Xtr_ij = Xtr_g[[f1, f2]]
        Xva_ij = Xva_g[[f1, f2]]
        Xte_ij = Xte_g[[f1, f2]]

        # OOF encodings for train
        tr_oof_cp = te.fit_transform(Xtr_ij, ytr_g)
        tr_pair[:, j] = cp.asnumpy(tr_oof_cp).ravel().astype(np.float32)

        # full-train encoding for valid/test
        te.fit(Xtr_ij, ytr_g)
        va_pair[:, j] = cp.asnumpy(te.transform(Xva_ij)).ravel().astype(np.float32)
        te_pair[:, j] = cp.asnumpy(te.transform(Xte_ij)).ravel().astype(np.float32)

    if use_logit3:
        tr_pair, va_pair, te_pair = make_logit3_features(tr_pair, va_pair, te_pair)

    return tr_pair, va_pair, te_pair


# ============================================================================
# MODEL INPUT HELPERS
# ============================================================================
def make_inputs(df, te_features):
    d = {}
    for c in cat_cols:
        d[c] = df[c].astype(str).values
    for c in num_cols:
        d[c] = df[c].astype("float32").values.reshape(-1, 1)
    d["te_feats"] = te_features.astype(np.float32)
    return d


# ============================================================================
# TABTRANSFORMER + TE MODEL
# ============================================================================
def build_tabtransformer_te_model(
    cat_cols,
    num_cols,
    cat_vocab,
    te_dim,
    embed_dim=16,
    num_heads=4,
    ff_dim=64,
    num_transformer_blocks=2,
    mlp_hidden=(128, 64),
    dropout=0.2,
    lr=1e-3,
):
    inputs = {}
    cat_embeds = []

    for c in cat_cols:
        inp = keras.Input(shape=(1,), name=c, dtype="string")
        inputs[c] = inp

        lookup = keras.layers.StringLookup(
            vocabulary=cat_vocab[c],
            mask_token=None,
            num_oov_indices=1,
            output_mode="int",
        )

        x = lookup(inp)
        x = keras.layers.Embedding(
            input_dim=lookup.vocabulary_size(),
            output_dim=embed_dim,
        )(x)
        x = keras.layers.Reshape((embed_dim,))(x)
        cat_embeds.append(x)

    if len(cat_embeds) == 0:
        raise ValueError("TabTransformer needs at least one categorical column.")

    cat_tokens = keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(cat_embeds)

    x_cat = cat_tokens
    for _ in range(num_transformer_blocks):
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout,
        )(x_cat, x_cat)

        x_cat = keras.layers.Add()([x_cat, attn_out])
        x_cat = keras.layers.LayerNormalization(epsilon=1e-6)(x_cat)

        ff = keras.layers.Dense(ff_dim, activation="gelu")(x_cat)
        ff = keras.layers.Dropout(dropout)(ff)
        ff = keras.layers.Dense(embed_dim)(ff)

        x_cat = keras.layers.Add()([x_cat, ff])
        x_cat = keras.layers.LayerNormalization(epsilon=1e-6)(x_cat)

    x_cat = keras.layers.Flatten()(x_cat)

    num_features = []
    for c in num_cols:
        inp = keras.Input(shape=(1,), name=c, dtype="float32")
        inputs[c] = inp
        num_features.append(inp)

    te_inp = keras.Input(shape=(te_dim,), name="te_feats", dtype="float32")
    inputs["te_feats"] = te_inp

    te_x = keras.layers.BatchNormalization()(te_inp)
    te_x = keras.layers.Dense(min(256, max(32, te_dim // 2)))(te_x)
    te_x = keras.layers.Activation("relu")(te_x)
    te_x = keras.layers.Dropout(dropout)(te_x)

    feats = [x_cat, te_x]
    if num_features:
        nums = keras.layers.Concatenate()(num_features)
        feats.append(nums)

    if len(feats) == 1:
        x = feats[0]
    else:
        x = keras.layers.Concatenate()(feats)

    for h in mlp_hidden:
        x = keras.layers.Dense(h)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(dropout)(x)

    out = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


# ============================================================================
# TRAIN
# ============================================================================
oof = np.zeros(len(X), dtype=np.float32)
test_preds = np.zeros(len(X_test), dtype=np.float32)

for fold, (t, v) in enumerate(outer_kfold.split(X, y), 1):
    print("\n" + "=" * 80)
    print(f"fold {fold}")
    print("=" * 80)

    Xt_raw, yt = X.iloc[t].copy(), y.iloc[t]
    Xv_raw, yv = X.iloc[v].copy(), y.iloc[v]

    Xt_int = X_teenc.iloc[t].copy()
    Xv_int = X_teenc.iloc[v].copy()

    if USE_X_ORIG:
        Xt_raw = pd.concat([Xt_raw, X_orig], axis=0).reset_index(drop=True)
        Xt_int = pd.concat([Xt_int, X_orig_teenc], axis=0).reset_index(drop=True)
        yt = np.concatenate([yt, y_orig], axis=0)

    X_test_raw = X_test.copy()
    X_test_int = X_test_teenc.copy()

    if num_cols:
        scaler_num = StandardScaler()
        Xt_raw[num_cols] = scaler_num.fit_transform(Xt_raw[num_cols])
        Xv_raw[num_cols] = scaler_num.transform(Xv_raw[num_cols])
        X_test_raw[num_cols] = scaler_num.transform(X_test_raw[num_cols])

    tr_te, va_te, te_te = build_pair_te_features_gpu(
        Xt_int, yt,
        Xv_int, X_test_int,
        pair_cols=pair_cols,
        inner_splits=TE_INNER_SPLITS,
        seed=SEED + fold,
        smooth=TE_SMOOTH,
        use_logit3=USE_LOGIT3,
    )

    scaler_te = StandardScaler()
    tr_te = scaler_te.fit_transform(tr_te).astype(np.float32)
    va_te = scaler_te.transform(va_te).astype(np.float32)
    te_te = scaler_te.transform(te_te).astype(np.float32)

    # RIDGE
    Xt_ridge = np.hstack([Xt_raw[num_cols].values, tr_te])
    Xv_ridge = np.hstack([Xv_raw[num_cols].values, va_te])
    X_test_ridge = np.hstack([X_test_raw[num_cols].values, te_te])

    ridge = Ridge(alpha=10, random_state=42)
    ridge.fit(Xt_ridge, yt)

    ridge_tr_pred = np.clip(ridge.predict(Xt_ridge), 0, 1)
    ridge_val_pred = np.clip(ridge.predict(Xv_ridge), 0, 1)
    ridge_te_pred = np.clip(ridge.predict(X_test_ridge), 0, 1)

    print(f"fold {fold} ridge auc: {roc_auc_score(yv, ridge_val_pred):.6f}")

    num_cols.append("ridge_pred")
    Xt_raw["ridge_pred"] = ridge_tr_pred
    Xv_raw["ridge_pred"] = ridge_val_pred
    X_test_raw["ridge_pred"] = ridge_te_pred

    model = build_tabtransformer_te_model(
        cat_cols=cat_cols,
        num_cols=num_cols,
        cat_vocab=cat_vocab,
        te_dim=tr_te.shape[1],
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_hidden=MLP_HIDDEN,
        dropout=DROPOUT,
        lr=LEARNING_RATE,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    model.fit(
        make_inputs(Xt_raw, tr_te),
        yt,
        validation_data=(make_inputs(Xv_raw, va_te), yv),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    oof[v] = model.predict(
        make_inputs(Xv_raw, va_te),
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()

    test_preds += model.predict(
        make_inputs(X_test_raw, te_te),
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()

    num_cols.remove("ridge_pred")
    print(f"fold {fold} auc: {roc_auc_score(yv, oof[v]):.6f}")

    del model, tr_te, va_te, te_te
    gc.collect()

test_preds /= N_SPLITS

oof_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {oof_auc:.6f}")

# ============================================================================
# Save predictions
# ============================================================================
pd.DataFrame({
    "id": train_id.values,
    "prob": oof,
    "predicted": (oof > 0.5).astype(int),
    "true": y.values,
}).to_csv(OUTPUT_DIR / "oof.csv", index=False)

pd.DataFrame({
    "id": test_id.values,
    "prob": test_preds,
}).to_csv(OUTPUT_DIR / "test_proba.csv", index=False)

pd.DataFrame({
    "id": test_id.values,
    "Churn": test_preds,
}).to_csv(OUTPUT_DIR / "test.csv", index=False)

print(f"Saved to {OUTPUT_DIR}")
print(f"Final OOF AUC: {oof_auc:.6f}")
print("Done!")
