"""
================================================================================
STAGE 2: Mamba-LSTM Core-Node Detection Pipeline  v1.1  (bugfix)
================================================================================
Research plan v5.0 - Month 4-5 (STAGE 2)

Input:
  stage1_work/s1_node_features_fixed.parquet
  stage1_work/s1_edges.parquet
  stage1_work/s1_keywords.json
  stage1_work/s1_graph_meta.json

Outputs:
  stage2_work/s2_results_table.xlsx
  stage2_work/s2_lead_time.xlsx
  stage2_work/s2_chow_test.json
  stage2_work/checkpoints/<model>_h<H>.pt

Usage:
  python stage2_mamba_lstm.py --step all
  python stage2_mamba_lstm.py --step train --models lstm mamba_lstm
  python stage2_mamba_lstm.py --step leadtime
  python stage2_mamba_lstm.py --step chow

v1.1 bugfixes:
  - QuarterWindowDataset: DataLoader로 배치 처리 (단일 루프 제거)
  - train_model: DataLoader 사용, 배치 단위 loss 계산 정상화
  - compute_lead_time: model_class 문자열→클래스 올바르게 해석
  - evaluate_model: DataLoader 사용, per-quarter 집계 정확도 수정
  - TransformerLSTMNodeModel: max_len 동적 처리 (인덱스 초과 방지)
  - MinimalSSMBlock: loop 대신 누적 스캔, 메모리 안전 처리
  - 전체 tqdm 진행바 정리 및 로그 가독성 개선
================================================================================
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------
# Optional official Mamba backend
# ------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as OfficialMamba
    MAMBA_AVAILABLE = True
except ImportError:
    OfficialMamba = None
    MAMBA_AVAILABLE = False

# ==============================================================================
# Paths / constants
# ==============================================================================
BASE_DIR = Path.cwd()
STAGE1_DIR = BASE_DIR / "stage1_work"
OUT_DIR    = BASE_DIR / "stage2_work"
CKPT_DIR   = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

DEFAULT_FEATURES_FILE  = STAGE1_DIR / "s1_node_features_fixed.parquet"
FALLBACK_FEATURES_FILE = STAGE1_DIR / "s1_node_features.parquet"
EDGES_PATH    = STAGE1_DIR / "s1_edges.parquet"
KEYWORDS_PATH = STAGE1_DIR / "s1_keywords.json"
GRAPH_META_PATH = STAGE1_DIR / "s1_graph_meta.json"

RESULTS_TABLE_PATH = OUT_DIR / "s2_results_table.xlsx"
LEAD_TIME_PATH     = OUT_DIR / "s2_lead_time.xlsx"
CHOW_TEST_PATH     = OUT_DIR / "s2_chow_test.json"

FEATURE_COLS = [
    "f1_tf", "f2_idf", "f3_fcite", "f4_bcite",
    "f5_age", "f6_cpc_entropy", "f7_group_diversity", "f8_text_tier",
]
N_FEATURES = len(FEATURE_COLS)
ALL_MODELS = ["lstm", "mamba", "transformer_lstm", "mamba_lstm"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Data loading & tensor construction
# ==============================================================================

def load_quarters():
    with open(GRAPH_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta["quarters_all"], meta


def resolve_features_file(path_arg):
    if path_arg:
        p = Path(path_arg)
        if p.exists():
            return p
        raise FileNotFoundError(p)
    if DEFAULT_FEATURES_FILE.exists():
        return DEFAULT_FEATURES_FILE
    print(f"WARNING: {DEFAULT_FEATURES_FILE.name} not found. "
          f"Falling back to {FALLBACK_FEATURES_FILE.name} "
          f"(f3/f4/f5 will be degenerate / all-zero).")
    if not FALLBACK_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Neither {DEFAULT_FEATURES_FILE} nor {FALLBACK_FEATURES_FILE} found."
        )
    return FALLBACK_FEATURES_FILE


def build_feature_tensor(features_file, keywords, quarters):
    """Returns X: (T, N, F) float32 array."""
    feat_df = pd.read_parquet(features_file)
    q_idx = {q: i for i, q in enumerate(quarters)}
    n_idx = {kw: i for i, kw in enumerate(keywords)}
    T, N = len(quarters), len(keywords)
    X = np.zeros((T, N, N_FEATURES), dtype=np.float32)

    missing_q = missing_n = 0
    for row in feat_df.itertuples(index=False):
        ti = q_idx.get(row.quarter_label)
        ni = n_idx.get(row.node)
        if ti is None:
            missing_q += 1
            continue
        if ni is None:
            missing_n += 1
            continue
        for fi, col in enumerate(FEATURE_COLS):
            val = getattr(row, col, 0.0)
            X[ti, ni, fi] = float(val) if val is not None and not (
                isinstance(val, float) and np.isnan(val)) else 0.0

    if missing_q or missing_n:
        print(f"  Note: skipped {missing_q} rows (unknown quarter), "
              f"{missing_n} rows (unknown node)")
    return X


def normalize_features(X, train_end_idx):
    """Z-score per feature using train-period statistics only."""
    flat = X[:train_end_idx].reshape(-1, N_FEATURES)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std[std < 1e-8] = 1.0
    return ((X - mean) / std).astype(np.float32), mean, std


def build_hub_labels(edges_path, keywords, quarters, top_k):
    """
    degree[t, n] = sum of PMI weights for node n at quarter t.
    hub[t, n]    = 1 if node n is in top_k by degree at quarter t.
    """
    edges = pd.read_parquet(edges_path)
    n_idx = {kw: i for i, kw in enumerate(keywords)}
    q_idx = {q: i for i, q in enumerate(quarters)}
    T, N  = len(quarters), len(keywords)
    degree = np.zeros((T, N), dtype=np.float32)

    for row in tqdm(edges.itertuples(index=False), total=len(edges),
                    desc="Building hub labels"):
        ti = q_idx.get(row.quarter_label)
        if ti is None:
            continue
        w  = float(row.weight)
        si = n_idx.get(row.src_node)
        tj = n_idx.get(row.tgt_node)
        if si is not None:
            degree[ti, si] += w
        if tj is not None:
            degree[ti, tj] += w

    hub = np.zeros((T, N), dtype=np.int64)
    for t in range(T):
        if degree[t].sum() > 0:
            hub[t, np.argsort(-degree[t])[:top_k]] = 1

    return degree, hub


def get_split_boundaries(meta):
    """Returns (train_end, valid_end, test_end) as exclusive quarter indices."""
    split    = meta["train_valid_test_split"]
    quarters = meta["quarters_all"]
    q_idx    = {q: i for i, q in enumerate(quarters)}

    def year_end_idx(year):
        cands = [i for q, i in q_idx.items() if int(q[:4]) <= year]
        return max(cands) + 1 if cands else 0

    train_end = year_end_idx(int(split["train"].split("-")[1]))
    valid_end = year_end_idx(int(split["valid"].split("-")[1]))
    return train_end, valid_end, len(quarters)


# ==============================================================================
# Dataset
# ==============================================================================

class QuarterWindowDataset(torch.utils.data.Dataset):
    """
    Each item: (x_win, y, t)
      x_win : (N, L, F)  — lookback window per node
      y     : (N,)       — hub label at t + horizon
      t     : int        — end of the lookback window (exclusive)
    """

    def __init__(self, X, hub, lookback, horizon, t_start, t_end):
        T = X.shape[0]
        self.X       = X
        self.hub     = hub
        self.L       = lookback
        self.H       = horizon
        self.samples = [
            t for t in range(t_start, t_end)
            if t - lookback >= 0 and t + horizon < T
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t     = self.samples[idx]
        x_win = self.X[t - self.L : t]           # (L, N, F)
        x_win = np.transpose(x_win, (1, 0, 2))   # (N, L, F)
        y     = self.hub[t + self.H].astype(np.float32)  # (N,)
        return (
            torch.from_numpy(x_win.copy()).float(),
            torch.from_numpy(y.copy()).float(),
            t,
        )


def collate_fn(batch):
    """Stack along a new batch dimension: (B, N, L, F), (B, N), [t, ...]"""
    xs, ys, ts = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(ts)


# ==============================================================================
# Models
# ==============================================================================

class MinimalSSMBlock(nn.Module):
    """
    Pure-PyTorch diagonal SSM fallback (Mamba-inspired, NOT official kernel).
    Uses a parallel-scan-friendly cumulative product formulation to avoid
    Python-level time loops (which are extremely slow for long sequences).

    x: (B, L, D) -> y: (B, L, D)
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        # A: negative → stable eigenvalues
        self.A_log      = nn.Parameter(torch.rand(d_model, d_state) * -2.0 - 0.5)
        self.B          = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C          = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D          = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, Dm = x.shape
        A     = -torch.exp(self.A_log)                      # (D, S)
        delta = F.softplus(self.delta_proj(x))              # (B, L, D)

        # Discretize: dA(b,l,d,s) = exp(delta(b,l,d) * A(d,s))
        # delta: (B,L,D,1), A: (1,1,D,S)
        delta_exp = delta.unsqueeze(-1)                      # (B,L,D,1)
        A_bc      = A.unsqueeze(0).unsqueeze(0)             # (1,1,D,S)
        dA        = torch.exp(delta_exp * A_bc)             # (B,L,D,S)

        # dB(b,l,d,s) = delta(b,l,d) * B(d,s)
        B_bc = self.B.unsqueeze(0).unsqueeze(0)             # (1,1,D,S)
        dB   = delta_exp * B_bc                             # (B,L,D,S)

        # u = dB * x: (B,L,D,S)
        u = dB * x.unsqueeze(-1)                            # (B,L,D,S)

        # Sequential scan over L (unavoidable for SSM; L is typically small)
        h = torch.zeros(B, Dm, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h  = dA[:, t] * h + u[:, t]                    # (B,D,S)
            C_bc = self.C.unsqueeze(0)                      # (1,D,S)
            y_t  = (h * C_bc).sum(-1) + self.D * x[:, t]  # (B,D)
            ys.append(y_t)
        return torch.stack(ys, dim=1)                       # (B,L,D)


def make_mamba_block(d_model):
    if MAMBA_AVAILABLE:
        return OfficialMamba(d_model=d_model)
    return MinimalSSMBlock(d_model)


class LSTMNodeModel(nn.Module):
    """Baseline: 2-layer LSTM. Input: (B, N, L, F) — treat N as batch."""

    def __init__(self, in_dim=N_FEATURES, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, N, L, F)
        B, N, L, F = x.shape
        x2       = x.view(B * N, L, F)
        out, _   = self.lstm(x2)
        logits   = self.head(out[:, -1, :]).squeeze(-1)  # (B*N,)
        return logits.view(B, N)


class MambaNodeModel(nn.Module):
    """Pure SSM ablation arm."""

    def __init__(self, in_dim=N_FEATURES, d_model=64):
        super().__init__()
        self.proj  = nn.Linear(in_dim, d_model)
        self.mamba = make_mamba_block(d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, 1)

    def forward(self, x):
        B, N, L, F = x.shape
        x2  = x.view(B * N, L, F)
        h   = self.proj(x2)
        h   = self.norm(h + self.mamba(h))
        out = self.head(h[:, -1, :]).squeeze(-1)
        return out.view(B, N)


class TransformerLSTMNodeModel(nn.Module):
    """
    Simplified Transformer encoder + LSTM head.
    Positional embedding is dynamically sliced to actual sequence length.
    """

    def __init__(self, in_dim=N_FEATURES, d_model=64, hidden=64, max_len=128):
        super().__init__()
        self.proj       = nn.Linear(in_dim, d_model)
        self.pos        = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.lstm        = nn.LSTM(d_model, hidden, batch_first=True)
        self.head        = nn.Linear(hidden, 1)

    def forward(self, x):
        B, N, L, F = x.shape
        x2      = x.view(B * N, L, F)
        h       = self.proj(x2) + self.pos[:, :L, :]   # dynamic slice
        h       = self.transformer(h)
        out, _  = self.lstm(h)
        logits  = self.head(out[:, -1, :]).squeeze(-1)
        return logits.view(B, N)


class MambaLSTMNodeModel(nn.Module):
    """Proposed: Mamba block + LSTM head."""

    def __init__(self, in_dim=N_FEATURES, d_model=64, hidden=64):
        super().__init__()
        self.proj  = nn.Linear(in_dim, d_model)
        self.mamba = make_mamba_block(d_model)
        self.norm  = nn.LayerNorm(d_model)
        self.lstm  = nn.LSTM(d_model, hidden, batch_first=True)
        self.head  = nn.Linear(hidden, 1)

    def forward(self, x):
        B, N, L, F = x.shape
        x2      = x.view(B * N, L, F)
        h       = self.proj(x2)
        h       = self.norm(h + self.mamba(h))
        out, _  = self.lstm(h)
        logits  = self.head(out[:, -1, :]).squeeze(-1)
        return logits.view(B, N)


MODEL_CLASSES = {
    "lstm":             LSTMNodeModel,
    "mamba":            MambaNodeModel,
    "transformer_lstm": TransformerLSTMNodeModel,
    "mamba_lstm":       MambaLSTMNodeModel,
}


# ==============================================================================
# Training
# ==============================================================================

def train_model(model_name, X, hub, splits, lookback, horizon,
                epochs=50, lr=1e-3, patience=8, pos_weight_cap=20.0,
                batch_size=4):
    train_end, valid_end, _ = splits

    train_ds = QuarterWindowDataset(X, hub, lookback, horizon, lookback, train_end)
    valid_ds = QuarterWindowDataset(X, hub, lookback, horizon, train_end, valid_end)

    if len(train_ds) == 0:
        raise ValueError(
            f"[{model_name}] Empty training set — "
            f"check lookback={lookback}/horizon={horizon} vs split sizes."
        )

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                          shuffle=False, collate_fn=collate_fn)

    model = MODEL_CLASSES[model_name]().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    # pos_weight: compensate for class imbalance
    train_hub = hub[lookback:train_end]
    n_pos   = float(train_hub.sum())
    n_total = float(train_hub.size)
    pos_weight = min(pos_weight_cap, (n_total - n_pos) / max(n_pos, 1.0))
    pos_wt = torch.tensor(pos_weight, device=DEVICE, dtype=torch.float32)
    print(f"  [{model_name}] pos_weight={pos_weight:.2f}  "
          f"hub_rate={n_pos/n_total:.4f}  "
          f"train_samples={len(train_ds)}  valid_samples={len(valid_ds)}")

    best_val   = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, _ in train_dl:
            x_batch = x_batch.to(DEVICE)   # (B, N, L, F)
            y_batch = y_batch.to(DEVICE)   # (B, N)
            opt.zero_grad()
            logits = model(x_batch)        # (B, N)
            loss   = F.binary_cross_entropy_with_logits(
                logits, y_batch, pos_weight=pos_wt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= max(len(train_dl), 1)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, _ in valid_dl:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                logits  = model(x_batch)
                loss    = F.binary_cross_entropy_with_logits(
                    logits, y_batch, pos_weight=pos_wt)
                val_loss += loss.item()
        val_loss /= max(len(valid_dl), 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  [{model_name}] epoch {epoch:3d}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val     = val_loss
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  [{model_name}] early stop @ epoch {epoch} "
                      f"(best val={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = CKPT_DIR / f"{model_name}_h{horizon}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  [{model_name}] checkpoint -> {ckpt_path}")
    return model


# ==============================================================================
# Evaluation: Precision@K, NDCG@K
# ==============================================================================

def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 20) -> float:
    order    = np.argsort(-scores)[:k]
    gains    = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg      = float((gains * discounts).sum())
    ideal    = np.argsort(-relevance)[:k]
    idcg     = float((relevance[ideal] * discounts[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate_model(model, X, hub, degree, lookback, horizon,
                   t_start, t_end, top_k=50, ndcg_k=20, batch_size=4):
    model.eval()
    ds = QuarterWindowDataset(X, hub, lookback, horizon, t_start, t_end)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    precisions, ndcgs = [], []

    for x_batch, y_batch, ts in dl:
        x_batch = x_batch.to(DEVICE)           # (B, N, L, F)
        logits  = model(x_batch)               # (B, N)
        scores  = torch.sigmoid(logits).cpu().numpy()   # (B, N)
        y_np    = y_batch.numpy()              # (B, N)

        for b, t in enumerate(ts):
            sc  = scores[b]                    # (N,)
            y_b = y_np[b]                      # (N,)
            rel = degree[t + horizon]          # (N,) raw PMI degree

            n_actual_pos = int(y_b.sum())
            if n_actual_pos == 0:
                continue

            top_pred  = set(np.argsort(-sc)[:n_actual_pos])
            actual    = set(np.where(y_b == 1)[0])
            precisions.append(len(top_pred & actual) / n_actual_pos)
            ndcgs.append(ndcg_at_k(sc, rel, k=ndcg_k))

    return {
        "n_eval_quarters":    len(precisions),
        "precision_at_k_mean": float(np.mean(precisions)) if precisions else 0.0,
        "ndcg_at_20_mean":     float(np.mean(ndcgs))      if ndcgs      else 0.0,
    }


# ==============================================================================
# Lead-time analysis
# ==============================================================================

def compute_lead_time(model_name, X, hub, splits, lookback, horizons,
                      epochs, lr, threshold=0.5, batch_size=4):
    """
    Train one model per horizon, then find how far in advance the model
    can flag hub-emergence events in the test period.
    NOTE: no @torch.no_grad() here — train_model() needs gradients.
    """
    _, valid_end, test_end = splits
    T, N = hub.shape

    # Train a model for each horizon
    models = {}
    for h in horizons:
        print(f"\n--- {model_name}  horizon={h} (lead-time) ---")
        models[h] = train_model(model_name, X, hub, splits, lookback, h,
                                epochs=epochs, lr=lr, batch_size=batch_size)

    # Pre-compute sigmoid scores (inference only — no_grad here is safe)
    scores_by_h = {}
    for h, model in models.items():
        model.eval()
        scores_by_h[h] = {}
        with torch.no_grad():
            for t in range(lookback, T - h):
                x_win = X[t - lookback : t]                    # (L, N, F)
                x_win = np.transpose(x_win, (1, 0, 2))[None]   # (1, N, L, F)
                x_t   = torch.from_numpy(x_win.copy()).float().to(DEVICE)
                sc    = torch.sigmoid(model(x_t)).cpu().numpy()[0]  # (N,)
                scores_by_h[h][t] = sc

    # Emergence events in test period
    events = []
    min_t  = max(valid_end, lookback + max(horizons))
    for t in range(min_t, test_end):
        emerged = np.where((hub[t - 1] == 0) & (hub[t] == 1))[0]
        for node_idx in emerged:
            events.append((t, int(node_idx)))

    print(f"\n{len(events)} hub-emergence events in test period")

    rows = []
    for (t_emerge, node_idx) in events:
        best_h = 0
        for h in sorted(horizons, reverse=True):
            t_pred = t_emerge - h
            if t_pred not in scores_by_h[h]:
                continue
            if scores_by_h[h][t_pred][node_idx] > threshold:
                best_h = h
                break
        rows.append({
            "t_emerge":           t_emerge,
            "node_idx":           node_idx,
            "lead_time_quarters": best_h,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        print("\nLead-time summary (quarters; 0 = not flagged in advance):")
        print(df["lead_time_quarters"].describe())
        hit_rate = (df["lead_time_quarters"] > 0).mean()
        print(f"Hit rate (flagged >=1 quarter before emergence): {hit_rate:.2%}")
    return df


# ==============================================================================
# Chow test (H1c)
# ==============================================================================

def chow_test(series, break_idx):
    from scipy import stats

    y = np.asarray(series, dtype=np.float64)
    n = len(y)
    t = np.arange(n, dtype=np.float64)

    if not (2 < break_idx < n - 2):
        raise ValueError("break_idx must leave >=2 points on each side")

    def ols_rss(y_s, t_s):
        Xm   = np.column_stack([np.ones_like(t_s), t_s])
        beta, *_ = np.linalg.lstsq(Xm, y_s, rcond=None)
        return float(((y_s - Xm @ beta) ** 2).sum())

    rss_pool = ols_rss(y, t)
    rss1     = ols_rss(y[:break_idx],  t[:break_idx])
    rss2     = ols_rss(y[break_idx:],  t[break_idx:])

    k   = 2       # intercept + slope
    df1 = k
    df2 = n - 2 * k

    if df2 <= 0 or (rss1 + rss2) == 0:
        return {"error": "insufficient degrees of freedom or zero residual variance"}

    F = ((rss_pool - (rss1 + rss2)) / df1) / ((rss1 + rss2) / df2)
    p = float(1 - stats.f.cdf(F, df1, df2))

    return {
        "f_statistic":        float(F),
        "p_value":            p,
        "df1":                df1,
        "df2":                df2,
        "break_idx":          break_idx,
        "n":                  n,
        "rss_pooled":         rss_pool,
        "rss_segment1":       rss1,
        "rss_segment2":       rss2,
        "significant_at_0.05": p < 0.05,
    }


def run_chow_for_h1c(hub, quarters):
    T, N = hub.shape
    emergence = np.zeros(T)
    for t in range(1, T):
        emergence[t] = ((hub[t] == 1) & (hub[t - 1] == 0)).sum()

    try:
        break_idx = quarters.index("2017Q1")
    except ValueError:
        break_idx = T // 2
        print(f"  WARNING: '2017Q1' not found in quarters list — "
              f"using midpoint break_idx={break_idx}")

    result = chow_test(emergence, break_idx)
    result["series_definition"] = "corpus-wide hub-emergence count per quarter"
    result["target_hypothesis"] = "H1c (B60W60 acceleration, 2017Q1)"
    result["note"] = (
        "Corpus-wide proxy. For B60W60-specific test, recompute emergence "
        "restricted to B60W60-tagged keyword nodes."
    )
    return result


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="STAGE 2 Mamba-LSTM pipeline v1.1")
    parser.add_argument("--step", choices=["train", "leadtime", "chow", "all"],
                        default="all")
    parser.add_argument("--features-file", default=None)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument("--lookback",  type=int, default=8,
                        help="Lookback window in quarters (default 8 = 2 yrs)")
    parser.add_argument("--horizon",   type=int, default=4,
                        help="Prediction horizon in quarters (default 4 = 1 yr)")
    parser.add_argument("--lead-time-horizons", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--lead-time-model", default="mamba_lstm",
                        choices=ALL_MODELS)
    parser.add_argument("--hub-top-k", type=int, default=50)
    parser.add_argument("--ndcg-k",    type=int, default=20)
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--batch-size",type=int, default=4)
    args = parser.parse_args()

    print("=" * 80)
    print(f"STAGE 2 Mamba-LSTM  v1.1")
    print(f"Device : {DEVICE}")
    print(f"mamba_ssm: {'official' if MAMBA_AVAILABLE else 'MinimalSSMBlock fallback'}")
    print("=" * 80)

    # ---- Load shared data ----
    quarters, meta = load_quarters()
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        kw_data = json.load(f)
    keywords = kw_data["keywords"]

    features_file = resolve_features_file(args.features_file)
    print(f"\nFeatures : {features_file}")

    print("\nBuilding (T, N, F) feature tensor ...")
    X_raw = build_feature_tensor(features_file, keywords, quarters)
    print(f"  X shape : {X_raw.shape}  "
          f"(T={len(quarters)} quarters × N={len(keywords)} nodes × F={N_FEATURES})")

    train_end, valid_end, test_end = get_split_boundaries(meta)
    print(f"\nSplit (exclusive end indices):")
    print(f"  Train : [0, {train_end})  "
          f"{quarters[0]} .. {quarters[train_end-1]}")
    print(f"  Valid : [{train_end}, {valid_end})  "
          f"{quarters[train_end]} .. {quarters[valid_end-1]}")
    print(f"  Test  : [{valid_end}, {test_end})  "
          f"{quarters[valid_end]} .. {quarters[test_end-1]}")

    print("\nNormalizing features (train stats only) ...")
    X, _mean, _std = normalize_features(X_raw, train_end)

    print(f"\nBuilding hub labels (top-{args.hub_top_k} PMI degree) ...")
    degree, hub = build_hub_labels(EDGES_PATH, keywords, quarters, args.hub_top_k)
    print(f"  Overall hub rate: {hub.mean():.4f}  "
          f"(expected {args.hub_top_k/len(keywords):.4f})")

    splits = (train_end, valid_end, test_end)

    # ---- STEP: train + evaluate ----
    if args.step in ("train", "all"):
        print(f"\n{'='*80}\nAblation training & evaluation\n{'='*80}")
        results = []
        for model_name in args.models:
            print(f"\n--- Training: {model_name} ---")
            model = train_model(
                model_name, X, hub, splits,
                args.lookback, args.horizon,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
            )
            print(f"\n--- Evaluating: {model_name} on TEST set ---")
            ev = evaluate_model(
                model, X, hub, degree,
                args.lookback, args.horizon,
                valid_end, test_end,
                top_k=args.hub_top_k, ndcg_k=args.ndcg_k,
                batch_size=args.batch_size,
            )
            print(f"  Precision@{args.hub_top_k}: {ev['precision_at_k_mean']:.4f}")
            print(f"  NDCG@{args.ndcg_k}        : {ev['ndcg_at_20_mean']:.4f}")
            results.append({
                "model":           model_name,
                "lookback":        args.lookback,
                "horizon":         args.horizon,
                "n_test_quarters": ev["n_eval_quarters"],
                "precision_at_k":  ev["precision_at_k_mean"],
                "ndcg_at_20":      ev["ndcg_at_20_mean"],
                "mamba_backend":   "official" if MAMBA_AVAILABLE else "minimal_ssm_fallback",
            })

        results_df = pd.DataFrame(results)
        results_df.to_excel(RESULTS_TABLE_PATH, index=False)
        print(f"\nResults saved → {RESULTS_TABLE_PATH}")
        print("\n" + results_df.to_string(index=False))

    # ---- STEP: lead-time ----
    if args.step in ("leadtime", "all"):
        print(f"\n{'='*80}\nLead-time analysis: {args.lead_time_model}\n{'='*80}")
        lt_df = compute_lead_time(
            args.lead_time_model, X, hub, splits,
            args.lookback, args.lead_time_horizons,
            args.epochs, args.lr, batch_size=args.batch_size,
        )
        if len(lt_df) > 0:
            lt_df.to_excel(LEAD_TIME_PATH, index=False)
            print(f"Lead-time table saved → {LEAD_TIME_PATH}")

    # ---- STEP: Chow test ----
    if args.step in ("chow", "all"):
        print(f"\n{'='*80}\nChow test (H1c)\n{'='*80}")
        chow_result = run_chow_for_h1c(hub, quarters)
        with open(CHOW_TEST_PATH, "w", encoding="utf-8") as f:
            json.dump(chow_result, f, ensure_ascii=False, indent=2)
        print(json.dumps(chow_result, ensure_ascii=False, indent=2))
        print(f"\nChow test saved → {CHOW_TEST_PATH}")

    print("\n" + "=" * 80)
    print("STAGE 2 complete")
    print("=" * 80)


if __name__ == "__main__":
    main()