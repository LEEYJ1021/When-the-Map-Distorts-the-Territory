"""
================================================================================
STAGE 2 Chow Test — B60W60 특화 재실행  v1.0
================================================================================
목적:
  corpus-wide proxy 대신 B60W60(전기/하이브리드 파워트레인 제어) 관련
  키워드 노드만 필터링하여 Chow test (H1c) 를 재검정.

  두 가지 방법으로 B60W60 노드를 식별:
    [방법 A] CPC 코드 기반 — corpus에서 B60W60 태그 특허를 찾아
             해당 특허에 등장하는 키워드를 B60W60 노드로 간주
    [방법 B] 키워드 의미 기반 — 전기차/하이브리드/배터리 관련 용어를
             포함하는 키워드를 직접 필터링 (CPC 컬럼 없을 때 fallback)

  두 결과 모두 저장 후 비교.

입력:
  stage1_work/s1_node_features_fixed.parquet
  stage1_work/s1_edges.parquet
  stage1_work/s1_keywords.json
  stage1_work/s1_graph_meta.json
  stage1_work/s1_corpus.parquet            (방법 A에 필요)

출력:
  stage2_work/s2_chow_b60w60_methodA.json  (CPC 기반)
  stage2_work/s2_chow_b60w60_methodB.json  (키워드 의미 기반)
  stage2_work/s2_chow_b60w60_summary.xlsx  (두 방법 비교표)

실행:
  python stage2_chow_b60w60.py
  python stage2_chow_b60w60.py --break-quarter 2017Q1
  python stage2_chow_b60w60.py --break-quarter 2017Q1 --min-node-count 5
================================================================================
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# 경로
# ==============================================================================
BASE_DIR   = Path.cwd()
STAGE1_DIR = BASE_DIR / "stage1_work"
OUT_DIR    = BASE_DIR / "stage2_work"
OUT_DIR.mkdir(exist_ok=True)

CORPUS_PATH        = STAGE1_DIR / "s1_corpus.parquet"
KEYWORDS_PATH      = STAGE1_DIR / "s1_keywords.json"
GRAPH_META_PATH    = STAGE1_DIR / "s1_graph_meta.json"
EDGES_PATH         = STAGE1_DIR / "s1_edges.parquet"
NODE_FEATURES_PATH = STAGE1_DIR / "s1_node_features_fixed.parquet"

OUT_A       = OUT_DIR / "s2_chow_b60w60_methodA.json"
OUT_B       = OUT_DIR / "s2_chow_b60w60_methodB.json"
OUT_SUMMARY = OUT_DIR / "s2_chow_b60w60_summary.xlsx"

# B60W60 의미 기반 키워드 패턴 (방법 B)
B60W60_SEMANTIC_PATTERNS = [
    "electric vehicle", "battery", "energy management",
    "regenerative braking", "regenerative", "hybrid",
    "fuel cell", "range extender", "state of charge",
    "state-of-charge", "powertrain", "electric motor",
    "charging", "plug-in", "plug in",
    "hev", "phev", "bev", "erev",
    "traction motor", "inverter", "dc-dc",
    "energy storage", "battery management", "thermal management",
    "torque distribution", "power split", "series hybrid",
    "parallel hybrid", "mild hybrid",
]

# CPC 컬럼 후보
CPC_COL_CANDIDATES = [
    "CPC Classifications", "CPC", "IPC Classifications",
    "Classification", "Classifications",
]


# ==============================================================================
# 공통 유틸
# ==============================================================================

def load_meta():
    with open(GRAPH_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_keywords():
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["keywords"]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def chow_test(series, break_idx):
    """
    Chow test: y_t = a + b*t + e, 구조 전환점 break_idx.
    Returns dict with F, p, rss 등.
    """
    y = np.asarray(series, dtype=np.float64)
    n = len(y)
    t = np.arange(n, dtype=np.float64)

    if not (2 < break_idx < n - 2):
        return {
            "error": f"break_idx={break_idx} must leave >=2 points on each side (n={n})"
        }

    def ols_rss(y_s, t_s):
        Xm   = np.column_stack([np.ones_like(t_s), t_s])
        beta, *_ = np.linalg.lstsq(Xm, y_s, rcond=None)
        return float(((y_s - Xm @ beta) ** 2).sum()), beta.tolist()

    rss_pool, beta_pool = ols_rss(y, t)
    rss1,     beta1     = ols_rss(y[:break_idx], t[:break_idx])
    rss2,     beta2     = ols_rss(y[break_idx:], t[break_idx:])

    k   = 2
    df1 = k
    df2 = n - 2 * k

    if df2 <= 0 or (rss1 + rss2) == 0:
        return {"error": "insufficient degrees of freedom or zero residual variance"}

    F = ((rss_pool - (rss1 + rss2)) / df1) / ((rss1 + rss2) / df2)
    p = float(1 - stats.f.cdf(F, df1, df2))

    return {
        "f_statistic":         float(F),
        "p_value":             p,
        "df1":                 df1,
        "df2":                 df2,
        "break_idx":           break_idx,
        "n":                   n,
        "rss_pooled":          rss_pool,
        "rss_segment1":        rss1,
        "rss_segment2":        rss2,
        "beta_pooled":         beta_pool,     # [intercept, slope]
        "beta_pre_break":      beta1,
        "beta_post_break":     beta2,
        "slope_change":        float(beta2[1] - beta1[1]),
        "significant_at_0.05": p < 0.05,
        "significant_at_0.10": p < 0.10,
    }


def compute_emergence_series(hub, node_indices, quarters):
    """
    지정 node_indices 대상으로 분기별 hub-emergence 수 계산.
    emergence[t] = (hub[t-1]==0 & hub[t]==1) 노드 수 (subset 내)
    """
    T = hub.shape[0]
    series = np.zeros(T)
    idx_arr = np.array(list(node_indices), dtype=int)
    if len(idx_arr) == 0:
        return series
    for t in range(1, T):
        prev = hub[t - 1, idx_arr]
        curr = hub[t,     idx_arr]
        series[t] = ((prev == 0) & (curr == 1)).sum()
    return series


def build_hub_from_edges(keywords, quarters):
    """degree[t,n] 및 hub[t,n] 재계산 (edges 파일 기반)."""
    edges = pd.read_parquet(EDGES_PATH)
    n_idx = {kw: i for i, kw in enumerate(keywords)}
    q_idx = {q: i for i, q in enumerate(quarters)}
    T, N  = len(quarters), len(keywords)
    degree = np.zeros((T, N), dtype=np.float32)

    for row in edges.itertuples(index=False):
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
    top_k = 50
    for t in range(T):
        if degree[t].sum() > 0:
            hub[t, np.argsort(-degree[t])[:top_k]] = 1

    return degree, hub


# ==============================================================================
# 방법 A: CPC 코드 기반 B60W60 노드 식별
# ==============================================================================

def method_a_cpc(keywords, quarters, hub, break_idx):
    """
    corpus에서 CPC에 'B60W60'이 포함된 특허를 찾아,
    해당 특허의 final_text에 등장하는 키워드를 B60W60 노드로 간주.
    """
    print("\n[방법 A] CPC 코드 기반 B60W60 노드 식별")

    if not CORPUS_PATH.exists():
        return None, "corpus 파일 없음 (s1_corpus.parquet)"

    corpus = pd.read_parquet(CORPUS_PATH)
    cpc_col = find_col(corpus, CPC_COL_CANDIDATES)

    if cpc_col is None:
        return None, f"CPC 컬럼 없음. 가용 컬럼: {list(corpus.columns)}"

    print(f"  CPC 컬럼: '{cpc_col}'")

    # B60W60 태그 특허 필터
    mask_b60w60 = corpus[cpc_col].astype(str).str.contains(
        "B60W60", case=False, na=False
    )
    b60w60_patents = corpus[mask_b60w60]
    print(f"  B60W60 태그 특허: {len(b60w60_patents):,}건 / 전체 {len(corpus):,}건")

    if len(b60w60_patents) == 0:
        # B60W60이 없으면 B60W 전체로 확장
        mask_b60w = corpus[cpc_col].astype(str).str.contains(
            "B60W", case=False, na=False
        )
        b60w60_patents = corpus[mask_b60w]
        print(f"  → B60W60 미발견, B60W 전체로 확장: {len(b60w60_patents):,}건")

    if len(b60w60_patents) == 0:
        return None, "B60W60 / B60W 태그 특허 없음"

    # 해당 특허들의 텍스트에서 키워드 등장 여부
    kw_set = set(keywords)
    n_idx  = {kw: i for i, kw in enumerate(keywords)}
    b60w60_node_counts = defaultdict(int)

    texts = b60w60_patents["final_text"].dropna().tolist()
    for text in texts:
        tl = text.lower()
        for kw in kw_set:
            if kw in tl:
                b60w60_node_counts[kw] += 1

    # 최소 등장 횟수 이상인 키워드만 선택 (min_count=3)
    min_count = 3
    b60w60_nodes = {kw: cnt for kw, cnt in b60w60_node_counts.items()
                    if cnt >= min_count}
    b60w60_indices = {n_idx[kw] for kw in b60w60_nodes}

    print(f"  B60W60 관련 노드: {len(b60w60_indices):,}개 "
          f"(최소 {min_count}건 등장 기준)")
    print(f"  상위 20 키워드: "
          f"{sorted(b60w60_nodes, key=lambda x: -b60w60_node_counts[x])[:20]}")

    # Emergence series
    series = compute_emergence_series(hub, b60w60_indices, quarters)
    total_emergence = int(series.sum())
    print(f"  B60W60 노드 hub-emergence 총 건수: {total_emergence}")

    if total_emergence < 10:
        print(f"  WARNING: emergence 건수가 너무 적어 Chow test 신뢰도 낮음")

    result = chow_test(series, break_idx)
    result.update({
        "method":              "A_cpc_based",
        "cpc_column":          cpc_col,
        "n_b60w60_patents":    int(len(b60w60_patents)),
        "n_b60w60_nodes":      len(b60w60_indices),
        "b60w60_keywords_top30": sorted(
            b60w60_nodes, key=lambda x: -b60w60_node_counts[x])[:30],
        "series_total_emergence": total_emergence,
        "break_quarter":       quarters[break_idx] if break_idx < len(quarters) else "?",
        "series":              series.tolist(),
    })
    return result, None


# ==============================================================================
# 방법 B: 키워드 의미 기반 B60W60 노드 식별
# ==============================================================================

def method_b_semantic(keywords, quarters, hub, break_idx, min_node_count=5):
    """
    B60W60_SEMANTIC_PATTERNS 중 하나 이상을 포함하는 키워드를
    B60W60 관련 노드로 간주.
    """
    print("\n[방법 B] 키워드 의미 기반 B60W60 노드 식별")
    print(f"  패턴 목록 ({len(B60W60_SEMANTIC_PATTERNS)}개):")
    print(f"  {B60W60_SEMANTIC_PATTERNS}")

    n_idx = {kw: i for i, kw in enumerate(keywords)}
    matched = {}
    for kw in keywords:
        kl = kw.lower()
        for pat in B60W60_SEMANTIC_PATTERNS:
            if pat in kl:
                matched[kw] = pat
                break

    b60w60_indices = {n_idx[kw] for kw in matched}
    print(f"\n  매칭된 키워드 ({len(matched)}개):")
    for kw, pat in sorted(matched.items(), key=lambda x: x[0]):
        print(f"    '{kw}'  ← 패턴: '{pat}'")

    if len(b60w60_indices) == 0:
        return {
            "method": "B_semantic",
            "error":  "매칭된 키워드 없음 — 패턴 목록을 확장하세요",
        }, None

    # Emergence series
    series = compute_emergence_series(hub, b60w60_indices, quarters)
    total_emergence = int(series.sum())
    print(f"\n  B60W60 노드 hub-emergence 총 건수: {total_emergence}")

    # 분기별 series 출력 (비0 분기만)
    print("\n  분기별 emergence (비0만):")
    for i, (q, v) in enumerate(zip(quarters, series)):
        if v > 0:
            print(f"    {q}: {int(v)}")

    if total_emergence < 5:
        print(f"  WARNING: emergence 건수가 너무 적어 Chow test 신뢰도 낮음")

    result = chow_test(series, break_idx)
    result.update({
        "method":                "B_semantic",
        "patterns_used":         B60W60_SEMANTIC_PATTERNS,
        "n_b60w60_nodes":        len(b60w60_indices),
        "b60w60_keywords":       list(matched.keys()),
        "b60w60_keyword_patterns": matched,
        "series_total_emergence": total_emergence,
        "break_quarter":         quarters[break_idx] if break_idx < len(quarters) else "?",
        "series":                series.tolist(),
    })
    return result, None


# ==============================================================================
# 비교 요약 엑셀
# ==============================================================================

def save_summary(result_a, result_b, quarters):
    rows = []
    for res, label in [(result_a, "A: CPC 기반"), (result_b, "B: 의미 기반")]:
        if res is None or "error" in res:
            rows.append({
                "방법": label,
                "B60W60 노드 수": "-",
                "Emergence 총건수": "-",
                "F 통계량": "-",
                "p-value": "-",
                "유의(p<0.05)": "-",
                "유의(p<0.10)": "-",
                "기울기 변화(post-pre)": "-",
                "break quarter": "-",
                "비고": res.get("error", "None") if res else "실행 안됨",
            })
        else:
            rows.append({
                "방법": label,
                "B60W60 노드 수": res.get("n_b60w60_nodes", "-"),
                "Emergence 총건수": res.get("series_total_emergence", "-"),
                "F 통계량": round(res.get("f_statistic", 0), 4),
                "p-value": round(res.get("p_value", 1), 4),
                "유의(p<0.05)": "✓" if res.get("significant_at_0.05") else "✗",
                "유의(p<0.10)": "✓" if res.get("significant_at_0.10") else "✗",
                "기울기 변화(post-pre)": round(res.get("slope_change", 0), 4),
                "break quarter": res.get("break_quarter", "-"),
                "비고": "",
            })

    summary_df = pd.DataFrame(rows)

    # 분기별 series 비교
    series_rows = []
    for i, q in enumerate(quarters):
        row = {"quarter": q}
        if result_a and "series" in result_a:
            row["emergence_A_cpc"] = result_a["series"][i] if i < len(result_a["series"]) else 0
        if result_b and "series" in result_b:
            row["emergence_B_semantic"] = result_b["series"][i] if i < len(result_b["series"]) else 0
        series_rows.append(row)
    series_df = pd.DataFrame(series_rows)

    with pd.ExcelWriter(OUT_SUMMARY) as writer:
        summary_df.to_excel(writer, sheet_name="Chow_비교", index=False)
        series_df.to_excel(writer, sheet_name="분기별_emergence", index=False)
        if result_b and "b60w60_keywords" in result_b:
            kw_df = pd.DataFrame({
                "keyword": result_b["b60w60_keywords"],
                "pattern": [result_b["b60w60_keyword_patterns"][k]
                            for k in result_b["b60w60_keywords"]],
            })
            kw_df.to_excel(writer, sheet_name="B60W60_키워드", index=False)

    print(f"\n비교 요약 저장 → {OUT_SUMMARY}")
    print("\n" + summary_df.to_string(index=False))


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B60W60 특화 Chow test (H1c)"
    )
    parser.add_argument(
        "--break-quarter", default="2017Q1",
        help="구조 전환점 분기 (기본: 2017Q1)"
    )
    parser.add_argument(
        "--min-node-count", type=int, default=3,
        help="방법A: B60W60 특허 내 최소 키워드 등장 횟수 (기본: 3)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("B60W60 특화 Chow test  v1.0")
    print(f"Break quarter: {args.break_quarter}")
    print("=" * 70)

    # 공통 데이터 로드
    meta     = load_meta()
    quarters = meta["quarters_all"]
    keywords = load_keywords()

    try:
        break_idx = quarters.index(args.break_quarter)
    except ValueError:
        print(f"ERROR: '{args.break_quarter}' not in quarters list.")
        print(f"Available: {quarters[:5]} ... {quarters[-5:]}")
        return

    print(f"\nBreak index: {break_idx}  ({quarters[break_idx]})")
    print(f"Keywords: {len(keywords):,}")
    print(f"Quarters: {quarters[0]} .. {quarters[-1]}")

    # Hub 라벨 재계산
    print("\nHub 라벨 로드 중 (edges 기반) ...")
    _, hub = build_hub_from_edges(keywords, quarters)
    print(f"Hub shape: {hub.shape}, hub rate: {hub.mean():.4f}")

    # ---- 방법 A ----
    result_a, err_a = method_a_cpc(keywords, quarters, hub, break_idx)
    if err_a:
        print(f"\n  방법 A 실패: {err_a}")
        result_a = {"error": err_a}
    else:
        with open(OUT_A, "w", encoding="utf-8") as f:
            # series는 float list라 JSON 직렬화 가능하지만 길어서 요약본만 저장
            save_obj = {k: v for k, v in result_a.items() if k != "series"}
            json.dump(save_obj, f, ensure_ascii=False, indent=2)
        print(f"\n방법 A 결과:")
        print(f"  F={result_a.get('f_statistic', 'N/A'):.4f}  "
              f"p={result_a.get('p_value', 'N/A'):.4f}  "
              f"유의(p<0.05): {result_a.get('significant_at_0.05', '?')}")
        print(f"  기울기 변화: {result_a.get('slope_change', 0):.4f} "
              f"({'증가' if result_a.get('slope_change', 0) > 0 else '감소'})")
        print(f"  저장 → {OUT_A}")

    # ---- 방법 B ----
    result_b, err_b = method_b_semantic(
        keywords, quarters, hub, break_idx,
        min_node_count=args.min_node_count
    )
    if err_b:
        print(f"\n  방법 B 실패: {err_b}")
        result_b = {"error": err_b}
    else:
        with open(OUT_B, "w", encoding="utf-8") as f:
            save_obj = {k: v for k, v in result_b.items()
                        if k not in ("series",)}
            json.dump(save_obj, f, ensure_ascii=False, indent=2)
        print(f"\n방법 B 결과:")
        print(f"  F={result_b.get('f_statistic', 'N/A'):.4f}  "
              f"p={result_b.get('p_value', 'N/A'):.4f}  "
              f"유의(p<0.05): {result_b.get('significant_at_0.05', '?')}")
        print(f"  기울기 변화: {result_b.get('slope_change', 0):.4f} "
              f"({'증가' if result_b.get('slope_change', 0) > 0 else '감소'})")
        print(f"  저장 → {OUT_B}")

    # ---- 비교 요약 ----
    save_summary(result_a, result_b, quarters)

    # ---- 해석 가이드 ----
    print("\n" + "=" * 70)
    print("결과 해석 가이드")
    print("=" * 70)
    for res, label in [(result_a, "방법 A"), (result_b, "방법 B")]:
        if not res or "error" in res:
            continue
        sig  = res.get("significant_at_0.05", False)
        sig10 = res.get("significant_at_0.10", False)
        slope = res.get("slope_change", 0)
        print(f"\n[{label}]")
        if sig:
            print(f"  ✓ H1c 지지: p={res['p_value']:.4f} < 0.05 → "
                  f"2017Q1 전후 구조적 전환 유의")
        elif sig10:
            print(f"  △ H1c 약한 지지: p={res['p_value']:.4f} < 0.10 → "
                  f"10% 유의수준에서 전환 감지")
        else:
            print(f"  ✗ H1c 기각: p={res['p_value']:.4f} ≥ 0.05 → "
                  f"유의한 구조 전환 미감지")
        if slope > 0:
            print(f"  → 기울기 {slope:+.4f}: 2017Q1 이후 B60W60 hub-emergence 증가 추세")
        else:
            print(f"  → 기울기 {slope:+.4f}: 2017Q1 이후 B60W60 hub-emergence 감소/유지")

    print("\n완료.")


if __name__ == "__main__":
    main()
