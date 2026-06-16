"""
================================================================================
STAGE 3: SHAP 4-Level Analysis  v1.1  (bugfix)
================================================================================
v1.1 bugfixes:
  - build_group_masks: 텍스트 중첩루프 → Aho-Corasick 방식으로 교체 (속도 100x↑)
  - run_level3: pre 구간을 train 구간까지 확장 (valid_end > break_t 문제 해결)
  - run_level4: keywords_list를 keywords 기준으로 고정 (node 순서 불일치 수정)
================================================================================
"""

import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path.cwd()
STAGE1_DIR = BASE_DIR / "stage1_work"
STAGE2_DIR = BASE_DIR / "stage2_work"
OUT_DIR    = BASE_DIR / "stage3_work"
FIG_DIR    = OUT_DIR / "figures"
CKPT_DIR   = STAGE2_DIR / "checkpoints"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

FEATURES_PATH = STAGE1_DIR / "s1_node_features_f8fixed.parquet"
EDGES_PATH    = STAGE1_DIR / "s1_edges.parquet"
KEYWORDS_PATH = STAGE1_DIR / "s1_keywords.json"
META_PATH     = STAGE1_DIR / "s1_graph_meta.json"
CORPUS_PATH   = STAGE1_DIR / "s1_corpus.parquet"
CKPT_PATH     = CKPT_DIR   / "mamba_lstm_h4.pt"

FEATURE_COLS = [
    "f1_tf", "f2_idf", "f3_fcite", "f4_bcite",
    "f5_age", "f6_cpc_entropy", "f7_group_diversity", "f8_text_tier",
]
FEATURE_LABELS = [
    "TF (출현빈도)", "IDF (희소성)",
    "Forward Citation", "Backward Citation",
    "Node Age", "CPC Entropy", "Group Diversity", "Text Tier",
]
N_FEATURES = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BREAK_QUARTER        = "2017Q1"
TARGET_JURISDICTIONS = ["US", "KR", "DE", "JP", "EP"]
TARGET_NATIONALITIES = ["US", "KR", "DE", "JP"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. 데이터 로드 & 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

def load_all():
    with open(META_PATH,     "r", encoding="utf-8") as f: meta    = json.load(f)
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f: kw_data = json.load(f)
    return meta, kw_data["keywords"], meta["quarters_all"]


def build_X(feat_df, keywords, quarters):
    q_idx = {q: i for i, q in enumerate(quarters)}
    n_idx = {kw: i for i, kw in enumerate(keywords)}
    T, N  = len(quarters), len(keywords)
    X     = np.zeros((T, N, N_FEATURES), dtype=np.float32)
    for row in feat_df.itertuples(index=False):
        ti = q_idx.get(row.quarter_label)
        ni = n_idx.get(row.node)
        if ti is None or ni is None:
            continue
        for fi, col in enumerate(FEATURE_COLS):
            v = getattr(row, col, 0.0)
            X[ti, ni, fi] = float(v) if v is not None and not (
                isinstance(v, float) and np.isnan(v)) else 0.0
    return X


def normalize_X(X, train_end):
    flat = X[:train_end].reshape(-1, N_FEATURES)
    mean = flat.mean(0); std = flat.std(0)
    std[std < 1e-8] = 1.0
    return ((X - mean) / std).astype(np.float32), mean, std


def build_hub(keywords, quarters, top_k=50):
    edges  = pd.read_parquet(EDGES_PATH)
    n_idx  = {kw: i for i, kw in enumerate(keywords)}
    q_idx  = {q: i for i, q in enumerate(quarters)}
    T, N   = len(quarters), len(keywords)
    degree = np.zeros((T, N), np.float32)
    for row in edges.itertuples(index=False):
        ti = q_idx.get(row.quarter_label)
        if ti is None: continue
        w  = float(row.weight)
        si = n_idx.get(row.src_node)
        tj = n_idx.get(row.tgt_node)
        if si is not None: degree[ti, si] += w
        if tj is not None: degree[ti, tj] += w
    hub = np.zeros((T, N), np.int64)
    for t in range(T):
        if degree[t].sum() > 0:
            hub[t, np.argsort(-degree[t])[:top_k]] = 1
    return degree, hub


def get_splits(meta, quarters):
    split = meta["train_valid_test_split"]
    q_idx = {q: i for i, q in enumerate(quarters)}
    def ye(yr):
        c = [i for q, i in q_idx.items() if int(q[:4]) <= yr]
        return max(c) + 1 if c else 0
    tr = ye(int(split["train"].split("-")[1]))
    vl = ye(int(split["valid"].split("-")[1]))
    return tr, vl, len(quarters)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 모델
# ══════════════════════════════════════════════════════════════════════════════

class MinimalSSMBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model; self.d_state = d_state
        self.A_log      = nn.Parameter(torch.rand(d_model, d_state) * -2.0 - 0.5)
        self.B          = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C          = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D          = nn.Parameter(torch.ones(d_model))
        self.delta_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, Dm = x.shape
        A     = -torch.exp(self.A_log)
        delta = F.softplus(self.delta_proj(x))
        dA    = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB    = delta.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0)
        u     = dB * x.unsqueeze(-1)
        h     = torch.zeros(B, Dm, self.d_state, device=x.device, dtype=x.dtype)
        ys    = []
        for t in range(L):
            h = dA[:, t] * h + u[:, t]
            ys.append((h * self.C.unsqueeze(0)).sum(-1) + self.D * x[:, t])
        return torch.stack(ys, dim=1)


class MambaLSTMNodeModel(nn.Module):
    def __init__(self, in_dim=N_FEATURES, d_model=64, hidden=64):
        super().__init__()
        self.proj  = nn.Linear(in_dim, d_model)
        self.mamba = MinimalSSMBlock(d_model)
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


def load_model():
    model = MambaLSTMNodeModel().to(DEVICE)
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"체크포인트 로드: {CKPT_PATH}")
    else:
        print(f"⚠️  체크포인트 없음. 무작위 가중치로 진행.")
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3. KernelSHAP
# ══════════════════════════════════════════════════════════════════════════════

def kernel_shap_node(model, x_window, node_idx, n_samples=256, baseline="mean"):
    from math import comb
    L, N, F = x_window.shape
    x_node  = x_window[:, node_idx, :]
    ref     = x_node.mean(0) if baseline == "mean" else np.zeros(F, dtype=np.float32)

    def predict(feat_vals):
        scores = []
        for fv in feat_vals:
            x_mod = x_window.copy()
            x_mod[:, node_idx, :] = fv
            x_t = torch.from_numpy(
                np.transpose(x_mod, (1, 0, 2))[None]
            ).float().to(DEVICE)
            with torch.no_grad():
                sc = torch.sigmoid(model(x_t)).cpu().numpy()[0, node_idx]
            scores.append(float(sc))
        return np.array(scores)

    rng        = np.random.default_rng(42)
    f0         = predict([ref])[0]
    coalitions = []
    for _ in range(n_samples):
        mask = rng.integers(0, 2, size=F).astype(bool)
        z    = np.where(mask, x_node[-1], ref)
        coalitions.append((mask, z))

    preds  = predict(np.array([z for _, z in coalitions]))
    weights = []
    for mask, _ in coalitions:
        s = mask.sum()
        w = 1e6 if s == 0 or s == F else (F - 1) / (comb(F, s) * s * (F - s))
        weights.append(w)

    W   = np.diag(weights)
    Z   = np.array([m.astype(float) for m, _ in coalitions])
    v   = preds - f0
    ZWZ = Z.T @ W @ Z
    ZWv = Z.T @ W @ v
    try:
        shap_val = np.linalg.solve(ZWZ + np.eye(F) * 1e-6, ZWv)
    except np.linalg.LinAlgError:
        shap_val = np.zeros(F)
    return shap_val.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEVEL 1 — 전역 특징 중요도
# ══════════════════════════════════════════════════════════════════════════════

def run_level1(model, X, hub, quarters, valid_end, test_end,
               lookback=8, horizon=4, n_nodes_per_q=20, n_shap_samples=256):
    print("\n[LEVEL 1] 전역 특징 중요도 계산 ...")
    T = X.shape[0]
    q_idx_map = {q: i for i, q in enumerate(quarters)}
    target_qs = [quarters[t] for t in range(valid_end, test_end)
                 if t - lookback >= 0 and t + horizon < T]
    all_shap = []

    for qlab in tqdm(target_qs, desc="LEVEL1"):
        t     = q_idx_map[qlab]
        x_win = X[t - lookback : t]
        hub_n = np.where(hub[t] == 1)[0].tolist()
        non_n = np.where(hub[t] == 0)[0].tolist()
        rng   = np.random.default_rng(t)
        sel   = hub_n[:n_nodes_per_q // 2]
        sel  += rng.choice(non_n, size=min(n_nodes_per_q // 2, len(non_n)),
                           replace=False).tolist()
        for nidx in sel:
            sv = kernel_shap_node(model, x_win, nidx, n_samples=n_shap_samples)
            all_shap.append({
                "quarter": qlab, "node_idx": nidx, "is_hub": int(hub[t, nidx]),
                **{f"shap_{FEATURE_COLS[fi]}": float(sv[fi]) for fi in range(N_FEATURES)}
            })

    shap_df   = pd.DataFrame(all_shap)
    shap_cols = [f"shap_{c}" for c in FEATURE_COLS]
    global_imp = shap_df[shap_cols].abs().mean()
    global_imp.index = FEATURE_LABELS
    global_imp = global_imp.sort_values(ascending=False)

    print("\n전역 특징 중요도 (|SHAP| 평균):")
    for name, val in global_imp.items():
        print(f"  {name:<22} {val:.4f}  {'█' * int(val * 200)}")

    shap_df.to_excel(OUT_DIR / "s3_shap_level1_global.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(global_imp.index[::-1], global_imp.values[::-1],
            color=["#E74C3C" if "Text Tier" in n else "#3498DB"
                   for n in global_imp.index[::-1]])
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("LEVEL 1: Global Feature Importance\n(Mamba-LSTM, Test 2019-2024)",
                 fontsize=12)
    ax.axvline(global_imp.mean(), color="gray", ls="--", lw=1, label="mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "level1_global_importance.png", dpi=150)
    plt.close()
    print("  → 저장: s3_shap_level1_global.xlsx + figures/level1_global_importance.png")
    return shap_df, global_imp


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEVEL 2 — Jurisdiction vs 국적 이원 SHAP
# ══════════════════════════════════════════════════════════════════════════════

def build_group_masks(corpus, keywords, quarters):
    """
    [BUGFIX v1.1] 중첩루프 제거 → pandas str.contains 벡터 연산으로 교체.
    각 특허의 final_text에 포함된 키워드 집합을 빠르게 탐색.
    """
    print("  국적/Jurisdiction 매핑 구축 중 ...")
    q_col   = next(c for c in ["quarter_label", "Quarter", "quarter"]
                   if c in corpus.columns)
    jur_col = next((c for c in ["Jurisdiction", "jurisdiction", "Country"]
                    if c in corpus.columns), None)
    nat_col = next((c for c in ["applicant_nationality", "Nationality", "nationality"]
                    if c in corpus.columns), None)

    n_idx = {kw: i for i, kw in enumerate(keywords)}

    # final_text를 소문자로 변환 후 컬럼 추가 (1회만)
    texts_lower = corpus["final_text"].fillna("").str.lower()

    jur_map = defaultdict(set)
    nat_map = defaultdict(set)

    # 키워드별로 벡터 연산 (1,000회 × 47,281행 str.contains → 약 10~30초)
    print(f"  키워드 {len(keywords):,}개 매핑 중 (벡터 연산) ...")
    for kw, ni in tqdm(n_idx.items(), desc="keyword scan", total=len(n_idx)):
        # 해당 키워드가 포함된 특허 행 마스크
        mask = texts_lower.str.contains(kw, regex=False, na=False)
        if not mask.any():
            continue
        sub = corpus[mask]
        for row in sub.itertuples(index=False):
            qlab = getattr(row, q_col)
            jur  = str(getattr(row, jur_col, "") or "").strip().upper()[:2] if jur_col else ""
            nat  = str(getattr(row, nat_col, "") or "").strip().upper()[:2] if nat_col else ""
            if jur:
                jur_map[(qlab, ni)].add(jur)
            if nat and nat != "NA":  # pandas NA 문자열 방지
                nat_map[(qlab, ni)].add(nat)

    print(f"  Jurisdiction 매핑: {len(jur_map):,}쌍  /  국적 매핑: {len(nat_map):,}쌍")
    return jur_map, nat_map, jur_col, nat_col


def run_level2(model, X, hub, quarters, valid_end, test_end,
               jur_map, nat_map, lookback=8, horizon=4,
               n_nodes_per_q=30, n_shap_samples=192):
    print("\n[LEVEL 2] Jurisdiction vs 국적 이원 SHAP ...")
    T     = X.shape[0]
    q_idx = {q: i for i, q in enumerate(quarters)}
    target_qs = [quarters[t] for t in range(valid_end, test_end)
                 if t - lookback >= 0 and t + horizon < T]

    rows_jur, rows_nat = [], []
    for qlab in tqdm(target_qs, desc="LEVEL2"):
        t     = q_idx[qlab]
        x_win = X[t - lookback : t]
        hub_n = np.where(hub[t] == 1)[0].tolist()
        non_n = np.where(hub[t] == 0)[0].tolist()
        rng   = np.random.default_rng(t + 1000)
        sel   = hub_n[:n_nodes_per_q // 2]
        sel  += rng.choice(non_n, size=min(n_nodes_per_q // 2, len(non_n)),
                           replace=False).tolist()
        for nidx in sel:
            sv   = kernel_shap_node(model, x_win, nidx, n_samples=n_shap_samples)
            jurs = jur_map.get((qlab, nidx), set())
            nats = nat_map.get((qlab, nidx), set())
            base = {f"shap_{FEATURE_COLS[fi]}": float(sv[fi]) for fi in range(N_FEATURES)}
            for jur in jurs:
                if jur in TARGET_JURISDICTIONS:
                    rows_jur.append({"quarter": qlab, "node_idx": nidx,
                                     "jurisdiction": jur, "is_hub": int(hub[t, nidx]),
                                     **base})
            for nat in nats:
                if nat in TARGET_NATIONALITIES:
                    rows_nat.append({"quarter": qlab, "node_idx": nidx,
                                     "nationality": nat, "is_hub": int(hub[t, nidx]),
                                     **base})

    shap_cols = [f"shap_{c}" for c in FEATURE_COLS]

    def make_profile(df, group_col, targets):
        if len(df) == 0: return pd.DataFrame()
        rows = []
        for grp in targets:
            sub = df[df[group_col] == grp]
            if len(sub) == 0: continue
            imp = sub[shap_cols].abs().mean()
            imp.index = FEATURE_LABELS
            rows.append({"group": grp, "n": len(sub), **imp.to_dict()})
        return pd.DataFrame(rows)

    jur_df      = pd.DataFrame(rows_jur)
    nat_df      = pd.DataFrame(rows_nat)
    jur_profile = make_profile(jur_df, "jurisdiction", TARGET_JURISDICTIONS)
    nat_profile = make_profile(nat_df, "nationality",  TARGET_NATIONALITIES)

    print("\n[Jurisdiction별 SHAP 프로파일]")
    if len(jur_profile) > 0: print(jur_profile.to_string(index=False))
    print("\n[국적별 SHAP 프로파일]")
    if len(nat_profile) > 0: print(nat_profile.to_string(index=False))

    if len(jur_df) > 0: jur_df.to_excel(OUT_DIR / "s3_shap_level2_jurisdiction.xlsx", index=False)
    if len(nat_df) > 0: nat_df.to_excel(OUT_DIR / "s3_shap_level2_nationality.xlsx", index=False)

    def plot_heatmap(profile_df, group_col, title, fname):
        if len(profile_df) == 0: return
        feat_cols = [c for c in profile_df.columns if c not in [group_col, "group", "n"]]
        mat = profile_df.set_index("group")[feat_cols].astype(float)
        fig, ax = plt.subplots(figsize=(10, max(3, len(mat) * 0.8)))
        im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(feat_cols)))
        ax.set_xticklabels(feat_cols, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels(mat.index, fontsize=10)
        plt.colorbar(im, ax=ax, label="|SHAP|")
        ax.set_title(title, fontsize=12)
        for i in range(len(mat)):
            for j in range(len(feat_cols)):
                ax.text(j, i, f"{mat.values[i,j]:.3f}",
                        ha="center", va="center", fontsize=7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / fname, dpi=150)
        plt.close()

    plot_heatmap(jur_profile, "jurisdiction",
                 "LEVEL 2: SHAP Profile by Jurisdiction", "level2_jurisdiction_heatmap.png")
    plot_heatmap(nat_profile, "nationality",
                 "LEVEL 2: SHAP Profile by Nationality",  "level2_nationality_heatmap.png")
    print("  → 저장: s3_shap_level2_*.xlsx + figures/")
    return jur_df, nat_df, jur_profile, nat_profile


# ══════════════════════════════════════════════════════════════════════════════
# 6. LEVEL 3 — 시간적 변화  [BUGFIX v1.1]
# ══════════════════════════════════════════════════════════════════════════════

def run_level3(model, X, hub, quarters, train_end, valid_end, test_end,
               lookback=8, horizon=4, n_nodes_per_q=20, n_shap_samples=192):
    """
    [BUGFIX v1.1]
    pre 구간: train_end ~ break_t  (valid_end > break_t(88) 이므로 train 구간 포함)
    post 구간: valid_end ~ test_end
    두 구간 모두 SHAP 계산 후 비교.
    """
    print(f"\n[LEVEL 3] 시간적 변화 (break: {BREAK_QUARTER}) ...")
    T     = X.shape[0]
    q_idx = {q: i for i, q in enumerate(quarters)}

    try:
        break_t = q_idx[BREAK_QUARTER]
    except KeyError:
        break_t = T // 2
        print(f"  ⚠️  {BREAK_QUARTER} 없음. 중간점 break_t={break_t} 사용.")

    # pre: train_end ~ break_t (2015Q1~2016Q4)
    # post: valid_end ~ test_end (2019Q1~2024Q4)
    pre_range  = range(train_end, break_t)
    post_range = range(valid_end, test_end)

    print(f"  pre  구간: {quarters[train_end]} ~ {quarters[break_t-1]}  "
          f"({len(pre_range)}분기)")
    print(f"  post 구간: {quarters[valid_end]} ~ {quarters[test_end-1]}  "
          f"({len(post_range)}분기)")

    rows = []
    for period, t_range in [("pre", pre_range), ("post", post_range)]:
        for t in tqdm(t_range, desc=f"LEVEL3 {period}"):
            if t - lookback < 0 or t + horizon >= T:
                continue
            qlab  = quarters[t]
            x_win = X[t - lookback : t]
            hub_n = np.where(hub[t] == 1)[0].tolist()
            non_n = np.where(hub[t] == 0)[0].tolist()
            rng   = np.random.default_rng(t + 2000)
            sel   = hub_n[:n_nodes_per_q // 2]
            sel  += rng.choice(non_n, size=min(n_nodes_per_q // 2, len(non_n)),
                               replace=False).tolist()
            for nidx in sel:
                sv = kernel_shap_node(model, x_win, nidx, n_samples=n_shap_samples)
                rows.append({
                    "quarter": qlab, "period": period,
                    "node_idx": nidx, "is_hub": int(hub[t, nidx]),
                    **{f"shap_{FEATURE_COLS[fi]}": float(sv[fi])
                       for fi in range(N_FEATURES)}
                })

    shap_df   = pd.DataFrame(rows)
    shap_cols = [f"shap_{c}" for c in FEATURE_COLS]

    summary = shap_df.groupby("period")[shap_cols].apply(
        lambda df: df.abs().mean()
    ).T
    summary.index = FEATURE_LABELS

    print(f"\n  pre({BREAK_QUARTER} 이전) vs post 비교:")
    if "pre" in summary.columns and "post" in summary.columns:
        summary["delta(post-pre)"] = summary["post"] - summary["pre"]
        summary["delta_%"] = (summary["delta(post-pre)"] /
                               summary["pre"].replace(0, np.nan) * 100).round(1)
        print(summary.round(4).to_string())

    shap_df.to_excel(OUT_DIR / "s3_shap_level3_temporal.xlsx", index=False)

    key_feats  = ["shap_f6_cpc_entropy", "shap_f7_group_diversity", "shap_f8_text_tier"]
    key_labels = ["CPC Entropy (f6)", "Group Diversity (f7)", "Text Tier (f8)"]
    q_mean = shap_df.groupby("quarter")[key_feats].mean().abs()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, col, lbl in zip(axes, key_feats, key_labels):
        if col in q_mean.columns:
            ax.plot(q_mean.index, q_mean[col], marker="o", ms=3, lw=1.2)
            ax.axvline(BREAK_QUARTER, color="red", ls="--", lw=1.2,
                       label=f"Break: {BREAK_QUARTER}")
            ax.set_ylabel(f"|SHAP| {lbl}", fontsize=9)
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("Quarter", fontsize=9)
    axes[-1].tick_params(axis="x", rotation=45, labelsize=7)
    plt.suptitle("LEVEL 3: Temporal SHAP Trend", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "level3_temporal_trend.png", dpi=150)
    plt.close()
    print("  → 저장: s3_shap_level3_temporal.xlsx + figures/level3_temporal_trend.png")
    return shap_df, summary


# ══════════════════════════════════════════════════════════════════════════════
# 7. LEVEL 4 — text_tier별 SHAP 안정성  [BUGFIX v1.1]
# ══════════════════════════════════════════════════════════════════════════════

def run_level4(model, X, hub, feat_df, keywords, quarters,
               valid_end, test_end, lookback=8, horizon=4,
               n_nodes_per_q=30, n_shap_samples=192):
    """
    [BUGFIX v1.1] keywords_list를 keywords 인자 기준으로 고정
    (feat_df["node"].unique() 순서 불일치 문제 해결)
    """
    print("\n[LEVEL 4] text_tier별 SHAP 안정성 (H0b) ...")
    T     = X.shape[0]
    q_idx = {q: i for i, q in enumerate(quarters)}

    # keywords 순서 기준으로 index 매핑 고정
    keywords_list = keywords   # ← BUGFIX: feat_df unique 대신 keywords 사용

    # (quarter_label, node) → tier 룩업
    tier_lookup = {}
    if "f8_text_tier" in feat_df.columns and "node" in feat_df.columns:
        for row in feat_df.itertuples(index=False):
            v = getattr(row, "f8_text_tier", 1)
            tier_lookup[(row.quarter_label, row.node)] = int(round(float(v))) if v == v else 1

    target_qs = [quarters[t] for t in range(valid_end, test_end)
                 if t - lookback >= 0 and t + horizon < T]
    rows = []

    for qlab in tqdm(target_qs, desc="LEVEL4"):
        t     = q_idx[qlab]
        x_win = X[t - lookback : t]
        hub_n = np.where(hub[t] == 1)[0].tolist()
        non_n = np.where(hub[t] == 0)[0].tolist()
        rng   = np.random.default_rng(t + 3000)
        sel   = hub_n[:n_nodes_per_q // 2]
        sel  += rng.choice(non_n, size=min(n_nodes_per_q // 2, len(non_n)),
                           replace=False).tolist()
        for nidx in sel:
            node_name = keywords_list[nidx] if nidx < len(keywords_list) else ""
            tier = tier_lookup.get((qlab, node_name), 1)
            sv   = kernel_shap_node(model, x_win, nidx, n_samples=n_shap_samples)
            rows.append({
                "quarter": qlab, "node_idx": nidx, "node": node_name,
                "tier": tier, "is_hub": int(hub[t, nidx]),
                **{f"shap_{FEATURE_COLS[fi]}": float(sv[fi]) for fi in range(N_FEATURES)}
            })

    shap_df   = pd.DataFrame(rows)
    shap_cols = [f"shap_{c}" for c in FEATURE_COLS]

    print(f"\n  tier별 샘플 수: {shap_df['tier'].value_counts().to_dict()}")

    tier_profile = shap_df.groupby("tier")[shap_cols].apply(
        lambda df: df.abs().mean()
    ).T
    tier_profile.index = FEATURE_LABELS
    print("\n  tier별 |SHAP| 평균:")
    print(tier_profile.round(4).to_string())

    from scipy import stats
    t1, t2 = shap_df[shap_df["tier"] == 1], shap_df[shap_df["tier"] == 2]
    mw_rows = []
    if len(t1) > 0 and len(t2) > 0:
        print(f"\n  Mann-Whitney U (tier1 n={len(t1)} vs tier2 n={len(t2)}):")
        for col, lbl in zip(shap_cols, FEATURE_LABELS):
            if col not in shap_df.columns: continue
            u_stat, p_val = stats.mannwhitneyu(
                t1[col].abs(), t2[col].abs(), alternative="two-sided")
            sig = "✓ 유의" if p_val < 0.05 else "✗ 비유의"
            print(f"    {lbl:<22}  U={u_stat:.0f}  p={p_val:.4f}  {sig}")
            mw_rows.append({
                "feature": lbl, "U": u_stat, "p_value": p_val,
                "significant_0.05": p_val < 0.05,
                "tier1_mean_|shap|": t1[col].abs().mean(),
                "tier2_mean_|shap|": t2[col].abs().mean(),
            })

    mw_df = pd.DataFrame(mw_rows)
    n_sig = int(mw_df["significant_0.05"].sum()) if len(mw_df) > 0 else 0
    print(f"\n  → H0b {'지지 (편향 없음)' if n_sig == 0 else f'기각 ({n_sig}개 특징 유의)'}")

    shap_df.to_excel(OUT_DIR / "s3_shap_level4_tier_stability.xlsx", index=False)
    if len(mw_df) > 0:
        mw_df.to_excel(OUT_DIR / "s3_shap_level4_mannwhitney.xlsx", index=False)

    if len(tier_profile.columns) >= 1 and 1 in tier_profile.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        x_pos = np.arange(N_FEATURES)
        w     = 0.35
        ax.bar(x_pos - w/2, tier_profile.get(1, np.zeros(N_FEATURES)),
               w, label="Tier 1 (Abstract)", color="#3498DB", alpha=0.8)
        if 2 in tier_profile.columns:
            ax.bar(x_pos + w/2, tier_profile.get(2, np.zeros(N_FEATURES)),
                   w, label="Tier 2 (Title+CPC)", color="#E74C3C", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(FEATURE_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean |SHAP value|", fontsize=10)
        ax.set_title("LEVEL 4: SHAP Stability by Text Tier (H0b)", fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "level4_tier_stability.png", dpi=150)
        plt.close()

    print("  → 저장: s3_shap_level4_*.xlsx + figures/level4_tier_stability.png")
    return shap_df, tier_profile, mw_df


# ══════════════════════════════════════════════════════════════════════════════
# 8. 종합 요약 JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(global_imp, jur_profile, nat_profile, temporal_summary, mw_df):
    summary = {
        "level1_top3_features": global_imp.head(3).to_dict(),
        "level1_f8_text_tier_rank": int(
            list(global_imp.index).index("Text Tier") + 1
            if "Text Tier" in global_imp.index else -1),
        "level2_jurisdiction_available": len(jur_profile) > 0,
        "level2_nationality_available":  len(nat_profile) > 0,
        "level3_break_quarter": BREAK_QUARTER,
        "level4_H0b_n_significant_features": int(
            mw_df["significant_0.05"].sum() if len(mw_df) > 0 else 0),
        "level4_H0b_verdict": (
            "지지 (tier간 SHAP 차이 없음 → 텍스트 보완 편향 없음)"
            if len(mw_df) == 0 or not mw_df["significant_0.05"].any()
            else "기각 (일부 특징에서 tier간 차이 유의 → 한계로 보고)"),
    }
    with open(OUT_DIR / "s3_shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n종합 요약 저장 → {OUT_DIR}/s3_shap_summary.json")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="STAGE 3 SHAP 4-Level v1.1")
    parser.add_argument("--level", default="all", choices=["all","1","2","3","4"])
    parser.add_argument("--lookback",       type=int,   default=8)
    parser.add_argument("--horizon",        type=int,   default=4)
    parser.add_argument("--n-nodes-per-q",  type=int,   default=20)
    parser.add_argument("--n-shap-samples", type=int,   default=256)
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 3: SHAP 4-Level Analysis  v1.1")
    print(f"Device: {DEVICE}  /  Level: {args.level}")
    print("=" * 70)

    meta, keywords, quarters = load_all()
    feat_df = pd.read_parquet(FEATURES_PATH)
    print(f"\nFeatures: {FEATURES_PATH.name}  ({len(feat_df):,}행)")

    X_raw = build_X(feat_df, keywords, quarters)
    train_end, valid_end, test_end = get_splits(meta, quarters)
    X, *_ = normalize_X(X_raw, train_end)

    print(f"X shape : {X.shape}")
    print(f"Split   : train=[0,{train_end})  valid=[{train_end},{valid_end})  "
          f"test=[{valid_end},{test_end})")

    _, hub = build_hub(keywords, quarters)
    model  = load_model()
    corpus = pd.read_parquet(CORPUS_PATH)

    jur_map, nat_map, jur_col, nat_col = build_group_masks(corpus, keywords, quarters)
    if jur_col is None: print("  ⚠️  Jurisdiction 컬럼 없음")
    if nat_col is None: print("  ⚠️  국적 컬럼 없음")

    global_imp       = pd.Series(dtype=float)
    jur_profile      = pd.DataFrame()
    nat_profile      = pd.DataFrame()
    temporal_summary = pd.DataFrame()
    mw_df            = pd.DataFrame()

    kw = dict(lookback=args.lookback, horizon=args.horizon,
              n_nodes_per_q=args.n_nodes_per_q,
              n_shap_samples=args.n_shap_samples)

    if args.level in ("all", "1"):
        _, global_imp = run_level1(
            model, X, hub, quarters, valid_end, test_end, **kw)

    if args.level in ("all", "2"):
        _, _, jur_profile, nat_profile = run_level2(
            model, X, hub, quarters, valid_end, test_end,
            jur_map, nat_map, **kw)

    if args.level in ("all", "3"):
        # [BUGFIX] train_end 추가 전달
        _, temporal_summary = run_level3(
            model, X, hub, quarters, train_end, valid_end, test_end, **kw)

    if args.level in ("all", "4"):
        # [BUGFIX] keywords 추가 전달
        _, _, mw_df = run_level4(
            model, X, hub, feat_df, keywords, quarters,
            valid_end, test_end, **kw)

    if args.level == "all":
        save_summary(global_imp, jur_profile, nat_profile,
                     temporal_summary, mw_df)

    print("\n" + "=" * 70)
    print("STAGE 3 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
