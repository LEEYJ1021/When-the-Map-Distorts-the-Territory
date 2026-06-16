"""
================================================================================
STAGE 1 PATCH v1.0 — Fix f3 (forward citation), f4 (backward citation),
                      and f5 (node age)
================================================================================

DIAGNOSIS (from the STAGE1 run log)
--------------------------------------------------------------------------
1) f3_fcite and f4_bcite are 0.0 for ALL 64,236 rows.

   Root cause: step_F_node_features() looked for citation columns named
   "Forward Citations" / "Citations Received" / "Cited By Count" and
   "Backward Citations" / "References Count" / "Citations Made". None of
   these exist in the corpus. The actual columns (visible in the STAGE1
   log's full column list) are:

       "Cited by Patent Count"  -> forward citations (how many later
                                    patents cite this one)
       "Cites Patent Count"     -> backward citations (how many earlier
                                    patents this one cites)

   With fcite_col = bcite_col = None, the running sums never accumulated,
   so f3 = f4 = 0 for every (quarter, node) pair. This silently zeroes out
   one of the corpus's main selling points -- "53,199 citation-validated
   (>=1 forward citation) patents" -- inside the node feature matrix.

2) f5_age is 0.0 for ALL rows -- and this is a DESIGN bug, not a missing
   column.

       f5 = max(0, q_year - avg_application_year_of_patents_in_quarter_q)

   Because quarter_label is itself derived from Application Date, every
   patent counted in quarter q has application year == q_year. So
   avg_year == q_year and f5 is tautologically 0, regardless of data.
   It can never be anything else as originally specified.

   REDEFINITION: f5 is now "node age" -- the number of quarters elapsed
   since this keyword/node FIRST appeared (with patent_count > 0)
   anywhere in the 1995Q1-2024Q4 corpus:

       f5_age(node, q) = quarter_index(q) - quarter_index(first_appearance(node))

   This varies meaningfully across (node, quarter), is monotonically
   increasing within a node's lifetime, and is exactly the kind of
   "how long has this technology concept been around" signal that
   motivates a long-range sequence model (Mamba) over a short-memory
   baseline (LSTM) -- consistent with the H1a/H1b/H1c framing in v5.0.

WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------
- Recomputes f3_fcite / f4_bcite from s1_corpus.parquet using the correct
  citation columns.
- Recomputes f5_age from the existing s1_node_features.parquet using the
  node-age definition above.
- Leaves f1, f2, f6, f7, f8 untouched (these were computed correctly).
- Writes s1_node_features_fixed.parquet and a before/after summary.

This does NOT require re-running embeddings, keyword extraction, or edge
construction -- only s1_corpus.parquet, s1_keywords.json,
s1_node_features.parquet and s1_graph_meta.json are needed.
================================================================================
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path.cwd()
WORK_DIR = BASE_DIR / "stage1_work"

CORPUS_PATH = WORK_DIR / "s1_corpus.parquet"
KEYWORDS_PATH = WORK_DIR / "s1_keywords.json"
NODE_FEATURES_PATH = WORK_DIR / "s1_node_features.parquet"
GRAPH_META_PATH = WORK_DIR / "s1_graph_meta.json"

OUTPUT_PATH = WORK_DIR / "s1_node_features_fixed.parquet"
SUMMARY_PATH = WORK_DIR / "s1_feature_fix_summary.json"

# Correct citation columns (confirmed present in s1_corpus.parquet)
FCITE_COL_CANDIDATES = ["Cited by Patent Count", "Cited By Patent Count"]
BCITE_COL_CANDIDATES = ["Cites Patent Count"]


def safe_to_parquet(df: pd.DataFrame, path):
    df_save = df.copy()
    for col in df_save.columns:
        if df_save[col].dtype == object:
            sample = df_save[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                df_save[col] = df_save[col].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if x is not None and not (isinstance(x, float) and np.isnan(x))
                    else None
                )
            else:
                df_save[col] = df_save[col].where(df_save[col].notna(), None)
                df_save[col] = df_save[col].astype(str).where(df_save[col].notna(), None)
    df_save.to_parquet(path, index=False)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_kws(text, keywords):
    if not isinstance(text, str) or not text.strip():
        return set()
    tl = text.lower()
    return {kw for kw in keywords if kw in tl}


def recompute_f3_f4(df, keywords, quarters):
    """Recompute average forward/backward citation per (quarter, node)."""
    fcite_col = find_col(df, FCITE_COL_CANDIDATES)
    bcite_col = find_col(df, BCITE_COL_CANDIDATES)
    if fcite_col is None or bcite_col is None:
        raise ValueError(
            f"Citation columns not found. Available columns:\n{list(df.columns)}"
        )
    print(f"Using forward-citation column:  '{fcite_col}'")
    print(f"Using backward-citation column: '{bcite_col}'")

    rows = []
    for qlab in tqdm(quarters, desc="Recomputing f3/f4 by quarter"):
        subset = df[df["quarter_label"] == qlab]
        if len(subset) == 0:
            continue

        kw_fsum = defaultdict(float)
        kw_bsum = defaultdict(float)
        kw_cnt = defaultdict(int)

        for _, row in subset.iterrows():
            kws = extract_kws(str(row.get("final_text", "")), keywords)
            if not kws:
                continue

            fc = row.get(fcite_col)
            bc = row.get(bcite_col)
            fc = float(fc) if pd.notna(fc) else 0.0
            bc = float(bc) if pd.notna(bc) else 0.0

            for kw in kws:
                kw_fsum[kw] += fc
                kw_bsum[kw] += bc
                kw_cnt[kw] += 1

        for kw, cnt in kw_cnt.items():
            rows.append({
                "quarter_label": qlab,
                "node": kw,
                "f3_fcite_new": round(kw_fsum[kw] / cnt, 4),
                "f4_bcite_new": round(kw_bsum[kw] / cnt, 4),
            })

    return pd.DataFrame(rows)


def recompute_f5_node_age(feat_df, quarters):
    """
    Redefine f5 as node age: quarters elapsed since this node first appeared
    (patent_count > 0) anywhere in the corpus.
    """
    q_idx = {q: i for i, q in enumerate(quarters)}
    feat_df = feat_df.copy()
    feat_df["_q_idx"] = feat_df["quarter_label"].map(q_idx)

    # Drop rows whose quarter isn't in the recognized quarter list (shouldn't
    # normally happen, but guards against UNKNOWN labels slipping through)
    n_dropped = feat_df["_q_idx"].isna().sum()
    if n_dropped:
        print(f"  Note: dropping {n_dropped} rows with unrecognized quarter_label")
    feat_df = feat_df.dropna(subset=["_q_idx"])
    feat_df["_q_idx"] = feat_df["_q_idx"].astype(int)

    first_appear = feat_df.groupby("node")["_q_idx"].min().to_dict()
    feat_df["f5_age_new"] = feat_df.apply(
        lambda r: r["_q_idx"] - first_appear[r["node"]], axis=1
    )
    feat_df = feat_df.drop(columns=["_q_idx"])
    return feat_df


def main():
    print("=" * 80)
    print("STAGE 1 PATCH: fixing f3 (forward citation), f4 (backward citation), "
          "f5 (node age)")
    print("=" * 80)

    for p in (CORPUS_PATH, KEYWORDS_PATH, NODE_FEATURES_PATH, GRAPH_META_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    df = pd.read_parquet(CORPUS_PATH)
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        kw_data = json.load(f)
    keywords = kw_data["keywords"]

    feat_df = pd.read_parquet(NODE_FEATURES_PATH)

    with open(GRAPH_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    quarters = meta["quarters_all"]

    print(f"\nCorpus: {len(df):,} patents")
    print(f"Keywords (nodes): {len(keywords):,}")
    print(f"Existing node-feature rows: {len(feat_df):,}")
    print(f"Quarters: {quarters[0]} .. {quarters[-1]} ({len(quarters)} total)")

    # ---- Before snapshot ----
    before = {
        "f3_fcite": feat_df["f3_fcite"].describe().to_dict(),
        "f4_bcite": feat_df["f4_bcite"].describe().to_dict(),
        "f5_age": feat_df["f5_age"].describe().to_dict(),
    }

    # ---- f3 / f4 ----
    print("\n[1/2] Recomputing f3_fcite / f4_bcite ...")
    f34_df = recompute_f3_f4(df, keywords, quarters)
    feat_df = feat_df.merge(f34_df, on=["quarter_label", "node"], how="left")
    feat_df["f3_fcite"] = feat_df["f3_fcite_new"].fillna(0.0)
    feat_df["f4_bcite"] = feat_df["f4_bcite_new"].fillna(0.0)
    feat_df = feat_df.drop(columns=["f3_fcite_new", "f4_bcite_new"])

    # ---- f5 ----
    print("\n[2/2] Recomputing f5_age (node age) ...")
    feat_df = recompute_f5_node_age(feat_df, quarters)
    feat_df["f5_age"] = feat_df["f5_age_new"]
    feat_df = feat_df.drop(columns=["f5_age_new"])

    # ---- After snapshot ----
    after = {
        "f3_fcite": feat_df["f3_fcite"].describe().to_dict(),
        "f4_bcite": feat_df["f4_bcite"].describe().to_dict(),
        "f5_age": feat_df["f5_age"].describe().to_dict(),
    }

    print("\n" + "-" * 80)
    print("BEFORE -> AFTER (describe())")
    print("-" * 80)
    for feat in ("f3_fcite", "f4_bcite", "f5_age"):
        b, a = before[feat], after[feat]
        print(f"\n{feat}:")
        print(f"  mean : {b['mean']:.4f} -> {a['mean']:.4f}")
        print(f"  std  : {b['std']:.4f} -> {a['std']:.4f}")
        print(f"  min  : {b['min']:.4f} -> {a['min']:.4f}")
        print(f"  max  : {b['max']:.4f} -> {a['max']:.4f}")

    safe_to_parquet(feat_df, OUTPUT_PATH)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({"before": before, "after": after}, f, ensure_ascii=False, indent=2)

    print(f"\nFixed feature table saved to: {OUTPUT_PATH}")
    print(f"Before/after summary saved to: {SUMMARY_PATH}")
    print("\nNext step: point stage2_mamba_lstm.py at "
          f"'{OUTPUT_PATH.name}' (default) instead of "
          "'s1_node_features.parquet'.")


if __name__ == "__main__":
    main()
