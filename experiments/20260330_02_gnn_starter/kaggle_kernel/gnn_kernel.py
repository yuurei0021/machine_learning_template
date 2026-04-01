import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch-geometric"])

# ============================================================
# TELCO CHURN - GNN SIMPLE 25 FEATURES (GraphSAGE, FAST BATCHED)
#
# Node features (same simple 25 features):
#   - 16 original categorical columns
#   - 3 snapped numeric->categorical proxy columns
#   - 3 numeric rare-flag categorical columns
#   - 3 numeric columns as direct inputs
#
# Graph construction:
#   - OHE of 16 original categorical columns
#   - PLUS 3 numeric columns
#   - numeric columns are StandardScaler normalized
#   - then multiplied by GRAPH_NUMERIC_MULTIPLIER
#
# Saves:
#   - oof_gnn_v{VER}.npy
#   - pred_gnn_v{VER}.npy
# ============================================================

import numpy as np
import pandas as pd
import gc
import time
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cuml.neighbors import NearestNeighbors

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = "Churn"
DROP_COLS = ["customerID", "id"]

VER = 1909

SEED = 42
N_FOLDS = 5
RARE_MIN_COUNT = 25

# Faster graph/training settings
K = 8
EPOCHS = 5
PATIENCE = 8
VAL_EVERY = 1
MIN_EPOCHS = 6

GRAPH_NUMERIC_MULTIPLIER = 3.0

# Manual mini-batch settings
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 16384
FANOUTS = [6, 4]   # 2 hops for 2 GraphSAGE layers

USE_AMP = True

BASE_NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
BASE_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

CAT_PROXY = [f"{c}__cat" for c in BASE_NUMS]
CAT_RARE = [f"{c}__is_rare" for c in BASE_NUMS]

CAT_COLS = BASE_CATS + CAT_PROXY + CAT_RARE   # 22 categorical node features
NUM_COLS = BASE_NUMS[:]                       # 3 numeric node features

GRAPH_CAT_COLS = BASE_CATS[:]                 # 16 categoricals for graph
GRAPH_NUM_COLS = BASE_NUMS[:]                 # 3 numerics for graph

# -----------------------------
# REPRO
# -----------------------------
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------------
# LOAD
# -----------------------------
train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/train.csv")
test = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/test.csv")
print("train:", train.shape, "test:", test.shape)

# Save IDs before dropping
train_ids = train["id"].values.copy() if "id" in train.columns else np.arange(len(train))
test_ids = test["id"].values.copy() if "id" in test.columns else np.arange(len(test))

for c in DROP_COLS:
    if c in train.columns:
        train.drop(columns=[c], inplace=True)
    if c in test.columns:
        test.drop(columns=[c], inplace=True)

if TARGET not in train.columns:
    raise KeyError(f"TARGET column '{TARGET}' not found in train.csv")

y = train[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0}).values.astype(np.float32)

# -----------------------------
# CLEAN BASE DATA
# -----------------------------
def clean_totalcharges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

train = clean_totalcharges(train)
test = clean_totalcharges(test)

def preprocess_base(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df.copy()
    te = test_df.copy()

    # numeric cleanup
    for df in (tr, te):
        for c in BASE_NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # categorical cleanup
    for df in (tr, te):
        for c in BASE_CATS:
            df[c] = df[c].astype(str).fillna("missing").str.strip()

    # fill numeric NaNs using TRAIN medians only
    for c in BASE_NUMS:
        med = float(np.nanmedian(tr[c].to_numpy(np.float32)))
        if not np.isfinite(med):
            med = 0.0
        tr[c] = tr[c].fillna(med).astype(np.float32)
        te[c] = te[c].fillna(med).astype(np.float32)

    return tr, te

train_base, test_base = preprocess_base(train, test)

# -----------------------------
# SIMPLE 25 FEATURE ENGINEERING
# -----------------------------
def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    """
    From TRAIN data only:
      - frequent values = count >= rare_min_count
      - rare values snapped to nearest frequent value
      - rare flag returned too
    """
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)

    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)

    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)

    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series):
        x = pd.to_numeric(series_any, errors="coerce").astype(np.float32).values
        is_nan = np.isnan(x)

        is_rare = np.ones_like(x, dtype=np.int32)
        for i, v in enumerate(x):
            if np.isnan(v):
                is_rare[i] = 1
            else:
                is_rare[i] = 0 if float(v) in frequent_set else 1

        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.searchsorted(frequent_vals, v)
                pos = np.clip(pos, 0, len(frequent_vals) - 1)

                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                right = pos

                left_vals = frequent_vals[left]
                right_vals = frequent_vals[right]

                choose_right = (np.abs(v - right_vals) <= np.abs(v - left_vals))
                nearest = np.where(choose_right, right_vals, left_vals)
                x_snapped[idx_snap] = nearest.astype(np.float32)

        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform

def make_simple25_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df.copy()
    te = test_df.copy()

    for col in BASE_NUMS:
        snapper = build_numeric_snapper(tr[col], rare_min_count=RARE_MIN_COUNT)

        tr_snap, tr_israre = snapper(tr[col])
        te_snap, te_israre = snapper(te[col])

        tr[f"{col}__cat"] = pd.Series(tr_snap).astype(str).values
        te[f"{col}__cat"] = pd.Series(te_snap).astype(str).values

        tr[f"{col}__is_rare"] = pd.Series(tr_israre).astype(str).values
        te[f"{col}__is_rare"] = pd.Series(te_israre).astype(str).values

    for df in (tr, te):
        for c in CAT_COLS:
            df[c] = df[c].astype(str).fillna("missing")

    return tr, te

train_fe, test_fe = make_simple25_features(train_base, test_base)

# -----------------------------
# ENCODE NODE CATEGORICALS
# -----------------------------
def encode_cats(train_df, test_df, cat_cols):
    tr_codes = []
    te_codes = []
    cardinals = []

    for c in cat_cols:
        all_vals = pd.concat(
            [train_df[c].astype(str), test_df[c].astype(str)],
            axis=0,
            ignore_index=True
        )
        uniq = all_vals.unique().tolist()
        mapping = {v: i for i, v in enumerate(uniq)}

        tr_c = train_df[c].astype(str).map(mapping).fillna(0).astype(np.int64).values
        te_c = test_df[c].astype(str).map(mapping).fillna(0).astype(np.int64).values

        tr_codes.append(tr_c)
        te_codes.append(te_c)
        cardinals.append(len(mapping))

    Xc_tr = np.stack(tr_codes, axis=1) if len(tr_codes) else np.zeros((len(train_df), 0), dtype=np.int64)
    Xc_te = np.stack(te_codes, axis=1) if len(te_codes) else np.zeros((len(test_df), 0), dtype=np.int64)

    return Xc_tr, Xc_te, cardinals

Xc_train, Xc_test, cat_cardinals = encode_cats(train_fe, test_fe, CAT_COLS)

# -----------------------------
# NODE NUMERIC MATRIX (same simple 25 features)
# -----------------------------
for c in NUM_COLS:
    train_fe[c] = pd.to_numeric(train_fe[c], errors="coerce").fillna(0).astype(np.float32)
    test_fe[c]  = pd.to_numeric(test_fe[c], errors="coerce").fillna(0).astype(np.float32)

Xn_train = train_fe[NUM_COLS].values.astype(np.float32)
Xn_test  = test_fe[NUM_COLS].values.astype(np.float32)

node_scaler = StandardScaler()
Xn_train = node_scaler.fit_transform(Xn_train).astype(np.float32)
Xn_test  = node_scaler.transform(Xn_test).astype(np.float32)

Xn_all = np.vstack([Xn_train, Xn_test])
Xc_all = np.vstack([Xc_train, Xc_test])

n_train = len(train_fe)
n_test = len(test_fe)
n_all = n_train + n_test

print("Using same simple 25 node features:")
print("NUM_COLS:", NUM_COLS)
print("CAT_COLS:", CAT_COLS)
print("Xn_all:", Xn_all.shape, "Xc_all:", Xc_all.shape)

# ============================================================
# BUILD KNN GRAPH ON:
#   OHE(16 base cats) + standardized(3 numerics) * multiplier
# ============================================================
graph_train_cat = train_fe[GRAPH_CAT_COLS].astype(str).copy()
graph_test_cat  = test_fe[GRAPH_CAT_COLS].astype(str).copy()
graph_all_cat = pd.concat([graph_train_cat, graph_test_cat], axis=0, ignore_index=True)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

X_graph_cat = ohe.fit_transform(graph_all_cat).astype(np.float32)

graph_train_num = train_fe[GRAPH_NUM_COLS].copy()
graph_test_num  = test_fe[GRAPH_NUM_COLS].copy()

for c in GRAPH_NUM_COLS:
    graph_train_num[c] = pd.to_numeric(graph_train_num[c], errors="coerce").fillna(0).astype(np.float32)
    graph_test_num[c]  = pd.to_numeric(graph_test_num[c], errors="coerce").fillna(0).astype(np.float32)

graph_scaler = StandardScaler()
X_graph_num_train = graph_scaler.fit_transform(graph_train_num.values.astype(np.float32)).astype(np.float32)
X_graph_num_test  = graph_scaler.transform(graph_test_num.values.astype(np.float32)).astype(np.float32)

X_graph_num = np.vstack([X_graph_num_train, X_graph_num_test]).astype(np.float32)
X_graph_num *= GRAPH_NUMERIC_MULTIPLIER

X_graph = np.concatenate([X_graph_cat, X_graph_num], axis=1).astype(np.float32)

print("Building cuML KNN edges on:")
print(" - OHE of 16 base categorical features")
print(" - plus standardized 3 numerics multiplied by", GRAPH_NUMERIC_MULTIPLIER)
print("Graph categorical matrix shape:", X_graph_cat.shape)
print("Graph numeric matrix shape    :", X_graph_num.shape)
print("Graph final matrix shape      :", X_graph.shape)

knn = NearestNeighbors(n_neighbors=K)
knn.fit(X_graph)
_, idx = knn.kneighbors(X_graph)

neighbors = idx.astype(np.int32)

print("neighbors:", neighbors.shape)

del X_graph, X_graph_cat, X_graph_num, idx, knn
gc.collect()

# ============================================================
# BASE DATA ON CPU
# ============================================================
x_num_cpu = torch.tensor(Xn_all, dtype=torch.float32).pin_memory()
x_cat_cpu = torch.tensor(Xc_all, dtype=torch.long).pin_memory()
y_all = np.concatenate([y, np.full(n_test, -1, np.float32)]).astype(np.float32)
y_cpu = torch.tensor(y_all, dtype=torch.float32).pin_memory()

print("CPU tensors ready:")
print("x_num:", tuple(x_num_cpu.shape), "x_cat:", tuple(x_cat_cpu.shape), "y:", tuple(y_cpu.shape))

# ============================================================
# MODEL: Cat Embeddings + GraphSAGE (FAST 2-LAYER VERSION)
# ============================================================
def emb_dim_from_card(card: int) -> int:
    d = int(round(1.8 * (card ** 0.25)))
    return int(np.clip(d, 4, 24))

class CatEmbed(nn.Module):
    def __init__(self, cardinals):
        super().__init__()
        self.embs = nn.ModuleList()
        out_dim = 0
        for card in cardinals:
            card = max(2, int(card))
            d = emb_dim_from_card(card)
            self.embs.append(nn.Embedding(card, d))
            out_dim += d
        self.out_dim = out_dim

        for e in self.embs:
            nn.init.normal_(e.weight, mean=0.0, std=0.02)

    def forward(self, x_cat):
        zs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embs)]
        return torch.cat(zs, dim=1)

class SAGEWithCats(nn.Module):
    def __init__(self, num_in, cat_cardinals, hidden=128, dropout=0.20):
        super().__init__()
        self.cat = CatEmbed(cat_cardinals)
        in_dim = num_in + self.cat.out_dim

        self.lin_in = nn.Linear(in_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.dropout = dropout

        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        zc = self.cat(data.x_cat)
        x = torch.cat([data.x_num, zc], dim=1)

        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x1 = F.relu(self.norm1(self.conv1(x, data.edge_index)))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x = x + 0.5 * x1

        x2 = F.relu(self.norm2(self.conv2(x, data.edge_index)))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = x + 0.5 * x2

        return self.head(x).squeeze(-1)

class SmoothBCE(nn.Module):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        t = targets * (1 - self.eps) + 0.5 * self.eps
        return F.binary_cross_entropy_with_logits(logits, t)

# ============================================================
# FAST SUBGRAPH SAMPLER
# ============================================================
global_pos = np.full(n_all, -1, dtype=np.int32)

def build_subgraph_from_seeds_fast(
    seed_nodes,
    neighbors,
    x_num_cpu,
    x_cat_cpu,
    y_cpu,
    fanouts,
    device,
    offset=0,
):
    seed_nodes = np.asarray(seed_nodes, dtype=np.int32)

    frontier = seed_nodes
    collected = [seed_nodes]

    for hop, fanout in enumerate(fanouts):
        nbr = neighbors[frontier]  # [num_frontier, K]
        start = (offset + hop) % nbr.shape[1]
        cols = (np.arange(fanout) + start) % nbr.shape[1]
        nbr = nbr[:, cols]

        frontier = np.unique(nbr.reshape(-1))
        collected.append(frontier)

    nodes = np.unique(np.concatenate(collected))
    m = len(nodes)

    global_pos[nodes] = np.arange(m, dtype=np.int32)
    seed_local = global_pos[seed_nodes]

    sub_nbr = neighbors[nodes]
    dst_local_all = global_pos[sub_nbr]
    mask = dst_local_all >= 0

    src_local = np.repeat(np.arange(m, dtype=np.int64), sub_nbr.shape[1])[mask.reshape(-1)]
    dst_local = dst_local_all[mask].astype(np.int64)

    edge_index = torch.tensor(
        np.vstack([src_local, dst_local]),
        dtype=torch.long,
        device=device
    )

    batch = Data(
        x_num=x_num_cpu[nodes].to(device, non_blocking=True),
        x_cat=x_cat_cpu[nodes].to(device, non_blocking=True),
        y=y_cpu[nodes].to(device, non_blocking=True),
        edge_index=edge_index,
    )
    batch.seed_local = torch.tensor(seed_local, dtype=torch.long, device=device)

    global_pos[nodes] = -1
    return batch

def iterate_seed_batches(seed_nodes, batch_size, shuffle):
    arr = np.asarray(seed_nodes, dtype=np.int32).copy()
    if shuffle:
        np.random.shuffle(arr)
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]

@torch.no_grad()
def predict_seed_nodes_fast(
    model,
    seed_nodes,
    neighbors,
    x_num_cpu,
    x_cat_cpu,
    y_cpu,
    fanouts,
    batch_size,
    device,
    offset=0,
):
    model.eval()
    out = np.zeros(len(seed_nodes), dtype=np.float32)
    pos = 0

    for batch_seed_nodes in iterate_seed_batches(seed_nodes, batch_size, shuffle=False):
        batch = build_subgraph_from_seeds_fast(
            seed_nodes=batch_seed_nodes,
            neighbors=neighbors,
            x_num_cpu=x_num_cpu,
            x_cat_cpu=x_cat_cpu,
            y_cpu=y_cpu,
            fanouts=fanouts,
            device=device,
            offset=offset,
        )

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and DEVICE == "cuda")):
            logits = model(batch)

        probs = torch.sigmoid(logits[batch.seed_local]).float().cpu().numpy()
        out[pos:pos + len(batch_seed_nodes)] = probs.astype(np.float32)
        pos += len(batch_seed_nodes)

        del batch, logits, probs

    return out

# ============================================================
# CV TRAIN
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(n_train, dtype=np.float32)
pred_test = np.zeros(n_test, dtype=np.float32)

loss_fn = SmoothBCE(eps=0.01)
test_nodes = np.arange(n_train, n_all, dtype=np.int32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y), 1):
    print(f"\n================ Fold {fold}/{N_FOLDS} ================")

    model = SAGEWithCats(
        num_in=Xn_all.shape[1],
        cat_cardinals=cat_cardinals,
        hidden=128,
        dropout=0.20,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE == "cuda"))

    best_auc = -1.0
    best_state = None
    bad = 0
    t_fold = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        offset = epoch % K

        for batch_seed_nodes in iterate_seed_batches(tr_idx, BATCH_SIZE, shuffle=True):
            batch = build_subgraph_from_seeds_fast(
                seed_nodes=batch_seed_nodes,
                neighbors=neighbors,
                x_num_cpu=x_num_cpu,
                x_cat_cpu=x_cat_cpu,
                y_cpu=y_cpu,
                fanouts=FANOUTS,
                device=DEVICE,
                offset=offset,
            )

            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(USE_AMP and DEVICE == "cuda")):
                logits = model(batch)
                loss = loss_fn(logits[batch.seed_local], batch.y[batch.seed_local])

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())
            del batch, logits, loss

        do_val = (epoch % VAL_EVERY == 0) or (epoch == 1)

        if do_val:
            val_pred = predict_seed_nodes_fast(
                model=model,
                seed_nodes=va_idx,
                neighbors=neighbors,
                x_num_cpu=x_num_cpu,
                x_cat_cpu=x_cat_cpu,
                y_cpu=y_cpu,
                fanouts=FANOUTS,
                batch_size=INFER_BATCH_SIZE,
                device=DEVICE,
                offset=offset,
            )
            auc = roc_auc_score(y[va_idx], val_pred)

            print(f"epoch {epoch:03d} | loss {np.mean(losses):.5f} | val_auc {auc:.6f}")

            if auc > best_auc + 1e-6:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if epoch >= MIN_EPOCHS and bad >= PATIENCE:
                    break

    model.load_state_dict(best_state)
    model.eval()

    val_pred = predict_seed_nodes_fast(
        model=model,
        seed_nodes=va_idx,
        neighbors=neighbors,
        x_num_cpu=x_num_cpu,
        x_cat_cpu=x_cat_cpu,
        y_cpu=y_cpu,
        fanouts=FANOUTS,
        batch_size=INFER_BATCH_SIZE,
        device=DEVICE,
        offset=0,
    )
    oof[va_idx] = val_pred.astype(np.float32)

    test_pred = predict_seed_nodes_fast(
        model=model,
        seed_nodes=test_nodes,
        neighbors=neighbors,
        x_num_cpu=x_num_cpu,
        x_cat_cpu=x_cat_cpu,
        y_cpu=y_cpu,
        fanouts=FANOUTS,
        batch_size=INFER_BATCH_SIZE,
        device=DEVICE,
        offset=0,
    )
    pred_test += test_pred.astype(np.float32) / N_FOLDS

    print(f"[Fold {fold}] best AUC: {best_auc:.6f} | fold time {time.time() - t_fold:.1f}s")

    del model, opt, scaler, best_state, val_pred, test_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

cv_auc = roc_auc_score(y, oof)
print("\n====================")
print("OOF CV AUC:", cv_auc)
print("====================")

np.save(f"oof_gnn_v{VER}.npy", oof.astype(np.float32))
np.save(f"pred_gnn_v{VER}.npy", pred_test.astype(np.float32))
print(f"Saved: oof_gnn_v{VER}.npy")
print(f"Saved: pred_gnn_v{VER}.npy")

# Save in project format (CSV)
OUTPUT_DIR = "/kaggle/working"
pd.DataFrame({"id": train_ids, "prob": oof, "predicted": (oof > 0.5).astype(int), "true": y.astype(int)}).to_csv(f"{OUTPUT_DIR}/oof.csv", index=False)
pd.DataFrame({"id": test_ids, "prob": pred_test}).to_csv(f"{OUTPUT_DIR}/test_proba.csv", index=False)
pd.DataFrame({"id": test_ids, "Churn": pred_test}).to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
print(f"Saved CSV to {OUTPUT_DIR}")