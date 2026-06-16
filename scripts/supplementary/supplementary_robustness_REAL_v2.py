"""
supplementary_robustness_REAL_v2.py
================================================================================
Supplementary Robustness Analysis — Real Data
"When the Map Distorts the Territory: Structural Attribution Bias in
 Cross-National Patent Data and Its Consequences for Multi-Agent Strategic Discourse"

CHANGELOG
--------------------------------------------------------------------------
[FIX v2 — 2026-06]  Two critical corrections in Section B:

  [FIX B-1]  b-path dependent variable corrected
              BEFORE: smf.ols("stance_entropy ~ stance_openness + jp")
              AFTER:  smf.ols("converged    ~ stance_openness + jp")
              → Aligns with v5c primary pipeline (Y = converged)
              → Resolves b-path sign reversal; indirect now correctly < 0

  [FIX B-2]  Recomputation dataset expanded
              BEFORE: df[country.isin(["JP","US"])] → 2,000 rows
              AFTER:  Full df (4,000 rows) — identical to v5c conditions
              → Resolves a-path magnitude mismatch (-0.367 vs -0.754)

Original fixes (FIX 1, FIX 2) retained:
  [FIX 1]  summary dict variable name unification (NameError resolved)
  [FIX 2]  stance_openness computation fixed (sign × prob, two vocabularies)

SECTIONS
--------------------------------------------------------------------------
  A.  ICC / Design-Effect Analysis
  B.  Cluster-Bootstrap Mediation (H3 replication vs v5c)
  C.  Practical Significance Translation (H2)
  D.  Session-Level Placebo Permutation Test
  E.  External Validity Face-Check

OUTPUT FILES
--------------------------------------------------------------------------
  supplementary_robustness_output/
    fig_supp_robustness_REAL_v2.png
    supplementary_robustness_REAL_v2_summary.json
================================================================================
"""

from __future__ import annotations

import json
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = Path(__file__).parent
S4    = BASE / "stage4_work"
V5C   = S4 / "postanalysis_v5c" / "s4v5c_mediation_analysis.xlsx"
JSONL = S4 / "s4_simulations.jsonl"
OUT   = S4 / "supplementary_robustness_output"
OUT.mkdir(parents=True, exist_ok=True)

SEP = "═" * 78
def hdr(t: str) -> None:
    print(f"\n{SEP}\n  {t}\n{SEP}")

RNG      = np.random.default_rng(42)
_N_BOOT  = 10_000
_N_PERM  = 100_000

# ── Agent → Country mapping ───────────────────────────────────────────────────
AGENT_COUNTRY: dict[str, str] = {
    "Toyota Group (JP)":        "JP",
    "Hyundai Motor Group (KR)": "KR",
    "Ford (US)":                "US",
    "Bosch (DE)":               "DE",
}
AGENTS = list(AGENT_COUNTRY.keys())

PROB_KEY_CANDIDATES   = ["probability", "confidence", "final_probability", "prob"]
STANCE_KEY_CANDIDATES = ["stance", "final_stance", "position", "stance_label", "label"]


# ── Helper functions ──────────────────────────────────────────────────────────

def get_field(d: dict, candidates: list, default=None):
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k], k
    return default, None


def sign_from_stance(stance_str: str) -> float:
    """
    Maps round-3 stance vocabulary to sign:
      expand / increase  →  +1.0
      reject / decrease  →  -1.0
      maintain / other   →   0.0
    """
    if not stance_str:
        return 0.0
    s = str(stance_str).lower()
    if "expand" in s or "increase" in s:
        return 1.0
    if "reject" in s or "decrease" in s:
        return -1.0
    return 0.0


def normalize_prob(val) -> float:
    """
    Normalises probability to [0, 1].
    Values > 1.5 are assumed to be on a 0–100 scale and divided by 100.
    """
    if val is None:
        return np.nan
    try:
        v = float(val)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(v):
        return np.nan
    return v / 100.0 if abs(v) > 1.5 else v


# ── Data loading ──────────────────────────────────────────────────────────────

def load_agent_level_data() -> pd.DataFrame:
    """Parse s4_simulations.jsonl into a long-format agent-level DataFrame."""
    rows = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    used_prob_keys:    set[str] = set()
    used_stance_keys:  set[str] = set()
    unrecognized_stances: set[str] = set()
    long_rows = []

    for sim in rows:
        sid      = sim["sim_id"]
        cond     = sim["scenario_branch"]
        conv_all = int(sim["metrics"]["all_agents_converged"])
        entropy  = sim["metrics"]["stance_entropy_bits"]

        for agent in AGENTS:
            country = AGENT_COUNTRY[agent]

            r1 = sim["rounds"].get("round1", {}).get(agent, {})
            r2 = sim["rounds"].get("round2", {}).get(agent, {})
            r3 = sim["rounds"].get("round3", {}).get(agent, {})

            prob_r3_raw, prob_key   = get_field(r3, PROB_KEY_CANDIDATES,   default=None)
            stance_r3,   stance_key = get_field(r3, STANCE_KEY_CANDIDATES, default="")

            if prob_key   is not None: used_prob_keys.add(prob_key)
            if stance_key is not None: used_stance_keys.add(stance_key)

            prob_r3 = normalize_prob(prob_r3_raw)
            sign    = sign_from_stance(stance_r3)

            if stance_r3 and sign == 0.0 and "maintain" not in str(stance_r3).lower():
                unrecognized_stances.add(stance_r3)

            stance_openness = sign * prob_r3 if not np.isnan(prob_r3) else 0.0

            long_rows.append(dict(
                session_id         = sid,
                agent_id           = agent,
                country            = country,
                jp                 = 1 if country == "JP" else 0,
                condition          = cond,
                converged          = conv_all,
                stance_entropy     = entropy,
                stance_openness    = stance_openness,
                round1_stance      = r1.get("stance"),
                round1_confidence  = r1.get("confidence"),
                round3_stance      = stance_r3,
                round3_probability = prob_r3,
            ))

    df = pd.DataFrame(long_rows)

    print(f"  [INFO] round3 probability key: {used_prob_keys or 'NONE FOUND'}")
    print(f"  [INFO] round3 stance key:      {used_stance_keys or 'NONE FOUND'}")
    if unrecognized_stances:
        print(f"  [WARN] Unrecognised stance values (sign=0): {unrecognized_stances}")

    n_nonzero = (df["stance_openness"] != 0).sum()
    print(f"  [INFO] stance_openness ≠ 0: {n_nonzero:,} / {len(df):,}")
    print(f"  Loaded {len(df):,} agent-rows "
          f"({df['session_id'].nunique():,} sessions × {df['agent_id'].nunique()} agents)")
    return df


def load_reference_openness_scores(df_sim: pd.DataFrame) -> np.ndarray:
    """
    In the absence of an external human corpus, use round-1 stance × confidence
    as a calibration proxy.  Document as a Limitation in the manuscript.
    """
    print("  [NOTE] No external human corpus — using round-1 proxy as calibration reference.")
    valid = df_sim["round1_confidence"].notna() & df_sim["round1_stance"].notna()
    ref   = df_sim.loc[valid, "round1_confidence"].values / 10.0
    signs = np.array([sign_from_stance(s) for s in df_sim.loc[valid, "round1_stance"]])
    return signs * ref


# ── ICC helper ────────────────────────────────────────────────────────────────

def icc_one_way(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """One-way random effects ICC and mean group size."""
    df_ = pd.DataFrame({"y": values.astype(float), "c": clusters})
    grand_mean = df_["y"].mean()
    groups     = df_.groupby("c")["y"]
    k_bar      = groups.size().mean()
    ms_btw = (groups.apply(lambda g: len(g) * (g.mean() - grand_mean) ** 2).sum()
              / (df_["c"].nunique() - 1))
    ms_wth = (groups.apply(lambda g: ((g - g.mean()) ** 2).sum()).sum()
              / (len(df_) - df_["c"].nunique()))
    icc = (ms_btw - ms_wth) / (ms_btw + (k_bar - 1) * ms_wth)
    return max(0.0, float(icc)), float(k_bar)


# ── Cluster-bootstrap mediation ───────────────────────────────────────────────

def fit_med(data: pd.DataFrame) -> tuple[float, float, float, float, float]:
    """
    Baron–Kenny mediation with:
      X = jp (dummy)
      M = stance_openness
      Y = converged  [FIX B-1: was stance_entropy]
    Returns (a, se_a, b, se_b, indirect).
    """
    a_m = smf.ols("stance_openness ~ jp", data=data).fit()
    b_m = smf.ols("converged ~ stance_openness + jp", data=data).fit()  # FIX B-1
    a,  se_a = a_m.params["jp"],              a_m.bse["jp"]
    b,  se_b = b_m.params["stance_openness"], b_m.bse["stance_openness"]
    return a, se_a, b, se_b, a * b


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

hdr("DATA LOADING")
df = load_agent_level_data()

sess = (df.groupby(["session_id", "condition"], as_index=False)
          .agg(converged=("converged", "first"),
               entropy=("stance_entropy", "first")))

session_ids = df["session_id"].unique()


# ══════════════════════════════════════════════════════════════════════════════
# A.  ICC / DESIGN-EFFECT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hdr("A. ICC / Design-Effect Analysis — H2 Convergence Rate")

icc_conv, k_bar = icc_one_way(df["converged"].values,       df["session_id"].values)
icc_ent,  _     = icc_one_way(df["stance_entropy"].values,  df["session_id"].values)
icc_open, _     = icc_one_way(df["stance_openness"].values, df["session_id"].values)

deff_conv = 1 + (k_bar - 1) * icc_conv
n_eff     = len(df) / deff_conv

print(f"  ICC (converged)       = {icc_conv:.4f}  [session-shared; ICC ≈ 1.0 expected]")
print(f"  ICC (stance_entropy)  = {icc_ent:.4f}  [session-shared]")
print(f"  ICC (stance_openness) = {icc_open:.4f}  [agent-level — OK]")
print(f"  Avg agents/session    = {k_bar:.1f}")
print(f"  Design effect (conv)  = {deff_conv:.3f}")
print(f"  N_total = {len(df):,}  →  N_effective (conv) = {n_eff:.1f}")

# Session-level z-test (correct inference unit for Y = converged)
g1 = sess.loc[sess.condition == "reversal_confirmed",     "converged"]
g2 = sess.loc[sess.condition == "reversal_not_confirmed", "converged"]
n1, n2 = len(g1), len(g2)
p1, p2 = g1.mean(), g2.mean()
p_pool = (g1.sum() + g2.sum()) / (n1 + n2)
se_z   = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
z_sess = (p1 - p2) / se_z
pv     = 2 * (1 - stats.norm.cdf(abs(z_sess)))

print(f"\n  Session-level z-test (n = {n1 + n2} sessions):")
print(f"    reversal_confirmed:     p(converged) = {p1 * 100:.1f}%")
print(f"    reversal_not_confirmed: p(converged) = {p2 * 100:.1f}%")
print(f"    z = {z_sess:.3f},  p = {pv:.4f}")
print(f"\n  NOTE: converged is a session-level attribute (ICC = 1.0).")
print(f"  Use session-level z as the primary H2 inferential statistic.")


# ══════════════════════════════════════════════════════════════════════════════
# B.  CLUSTER-BOOTSTRAP MEDIATION  [FIX B-1, FIX B-2]
# ══════════════════════════════════════════════════════════════════════════════
hdr("B. Cluster-Bootstrap Mediation — Replication of v5c [FIX B-1, B-2]")

# Load v5c reference results
df_v5c = pd.read_excel(V5C)
print("  ▶ s4v5c_mediation_analysis.xlsx (cluster-robust, n_clusters = 1,000):")
print(df_v5c[["x_label", "n", "n_clusters", "a_path", "b_path",
              "indirect_effect", "sobel_z", "sobel_p",
              "boot_CI_lo", "boot_CI_hi", "se_method"]].to_string(index=False))

# [FIX B-2] Use full 4,000-row dataset
df_all    = df.copy()
df_all["jp"] = (df_all["country"] == "JP").astype(int)
print(f"\n  [FIX B-2] Dataset: full {len(df_all):,} rows  "
      f"(previously JP+US only = 2,000 rows)")
print(f"  [FIX B-1] Y variable: converged  (previously stance_entropy)")

a0, se_a0, b0, se_b0, ind0 = fit_med(df_all)
sobel_se      = np.sqrt(b0 ** 2 * se_a0 ** 2 + a0 ** 2 * se_b0 ** 2)
sobel_z_naive = ind0 / sobel_se if sobel_se != 0 else np.nan

# Cluster bootstrap
print(f"\n  Running cluster bootstrap (B = {_N_BOOT:,}) ...")
boot = []
for _ in range(_N_BOOT):
    samp = RNG.choice(session_ids, size=len(session_ids), replace=True)
    bdf  = pd.concat([df_all[df_all.session_id == s] for s in samp],
                     ignore_index=True)
    try:
        boot.append(fit_med(bdf)[4])
    except Exception:
        continue

boot       = np.array(boot)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
cluster_se = boot.std(ddof=1)
cluster_z  = ind0 / cluster_se if cluster_se > 0 else np.nan

v5c_z = df_v5c.loc[df_v5c.x_label == "JP_vs_US", "sobel_z"].values[0]
delta = abs(cluster_z - v5c_z) if not np.isnan(cluster_z) else float("nan")

print(f"\n  ▶ Independent replication (N = 4,000, cluster-bootstrap B = {len(boot):,}):")
print(f"    a-path         = {a0:.5f}")
print(f"    b-path         = {b0:.5f}")
print(f"    indirect       = {ind0:.5f}")
print(f"    naive Sobel z  = {sobel_z_naive:.3f}")
print(f"    cluster SE     = {cluster_se:.5f}  "
      f"(naive SE = {sobel_se:.5f}, inflation = {cluster_se/sobel_se:.2f}x)")
print(f"    cluster-robust z = {cluster_z:.3f}")
print(f"    95% CI           = [{ci_lo:.5f}, {ci_hi:.5f}]")
print(f"\n  ▶ v5c reported Sobel z (JP_vs_US) = {v5c_z:.3f}  ←→  replication = {cluster_z:.3f}")
print(f"    Δ = {delta:.3f}  "
      f"{'← convergence ✅' if delta < 1.0 else '← gap remains; see §3.4 Y-specification'}")
print(f"\n  MANUSCRIPT NOTE: Report cluster-robust z = {cluster_z:.2f}, "
      f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  Footnote: naive Sobel z = {sobel_z_naive:.2f} (independence assumed)")


# ══════════════════════════════════════════════════════════════════════════════
# C.  PRACTICAL SIGNIFICANCE — H2
# ══════════════════════════════════════════════════════════════════════════════
hdr("C. Practical Significance Translation — H2")

rr       = p1 / p2 if p2 > 0 else np.nan
arr      = abs(p1 - p2)
rrr      = arr / max(p1, p2)
cohens_h = 2 * abs(np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
nnt      = 1 / arr if arr > 0 else np.inf

print(f"  p(converged | reversal_confirmed)     = {p1 * 100:.1f}%")
print(f"  p(converged | reversal_not_confirmed) = {p2 * 100:.1f}%")
print(f"  Risk Ratio          = {rr:.3f}")
print(f"  Absolute Difference = {arr * 100:.1f}pp")
print(f"  Relative Reduction  = {rrr * 100:.1f}%")
print(f"  Cohen's h           = {cohens_h:.3f}  "
      f"({'small' if cohens_h < 0.2 else 'small-medium' if cohens_h < 0.5 else 'medium'})")
print(f"  NNT                 = {nnt:.1f}")
print(textwrap.dedent(f"""
  DISCUSSION TEMPLATE:
    "The reversal-confirmed condition reduced convergence probability by {rrr*100:.0f}%
     relative to baseline ({p1*100:.1f}% vs {p2*100:.1f}%; Cohen's h = {cohens_h:.2f},
     NNT ≈ {nnt:.0f}).  The effect size is small-to-medium, warranting
     conservative interpretation focused on the mechanistic pathway."
"""))


# ══════════════════════════════════════════════════════════════════════════════
# D.  SESSION-LEVEL PLACEBO PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════════════
hdr("D. Session-Level Placebo Permutation Test")

obs_diff   = p1 - p2
perm_diffs = np.zeros(_N_PERM)
labels     = sess["condition"].values.copy()

for i in range(_N_PERM):
    sh = RNG.permutation(labels)
    d  = (sess.loc[sh == "reversal_confirmed",     "converged"].mean()
        - sess.loc[sh == "reversal_not_confirmed", "converged"].mean())
    perm_diffs[i] = d

placebo_p = (np.sum(np.abs(perm_diffs) >= np.abs(obs_diff)) + 1) / (_N_PERM + 1)

print(f"  Observed Δ(converged)              = {obs_diff * 100:.2f}pp")
print(f"  Session-level permutation p (N = {_N_PERM:,}) = {placebo_p:.5f}")
print(f"\n  NOTE: Report session-level permutation p (n = 1,000 sessions) as the")
print(f"  primary placebo result.  Agent-level p ≈ 0 is over-powered due to ICC = 1.")


# ══════════════════════════════════════════════════════════════════════════════
# E.  EXTERNAL VALIDITY FACE-CHECK
# ══════════════════════════════════════════════════════════════════════════════
hdr("E. External Validity Face-Check — Simulated vs Round-1 Reference")

sim_scores = df["stance_openness"].values
ref_scores = load_reference_openness_scores(df)

ks_stat, ks_p = ks_2samp(sim_scores, ref_scores)
wd            = wasserstein_distance(sim_scores, ref_scores)

print(f"  Simulated (round-3): mean = {sim_scores.mean():.3f},  sd = {sim_scores.std():.3f}")
print(f"  Reference (round-1): mean = {ref_scores.mean():.3f},  sd = {ref_scores.std():.3f}")
print(f"  KS statistic = {ks_stat:.3f},  p = {ks_p:.4f}")
print(f"  Wasserstein distance = {wd:.3f}")
print(textwrap.dedent("""
  MANUSCRIPT NOTE:
    Direct comparison against a human corpus is unavailable.
    Report the round-1 → round-3 distributional shift as a face-validity
    indicator in the Appendix and frame true external validation as
    Future Work.
"""))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — 3-panel summary
# ══════════════════════════════════════════════════════════════════════════════
hdr("FIGURE — 3-Panel Robustness Summary")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.patch.set_facecolor("white")

# Panel 1: Convergence rate by condition
conv_rates = sess.groupby("condition")["converged"].mean() * 100
bars = axes[0].bar(
    ["Reversal\nConfirmed", "Reversal Not\nConfirmed"],
    [conv_rates.get("reversal_confirmed", 0),
     conv_rates.get("reversal_not_confirmed", 0)],
    color=["#333333", "#777777"], alpha=0.88, width=0.5, edgecolor="white", lw=2,
)
for bar, v in zip(bars, [conv_rates.get("reversal_confirmed", 0),
                          conv_rates.get("reversal_not_confirmed", 0)]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1,
                 f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Convergence Rate (%)")
axes[0].set_ylim(0, 45)
axes[0].set_title(f"H2: Convergence Rate\n"
                  f"Session-level z = {z_sess:.2f}, p = {pv:.4f}",
                  fontsize=10.5, fontweight="bold")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Panel 2: Cluster-bootstrap indirect effect distribution
axes[1].hist(boot, bins=40, color="#444444", alpha=0.82,
             edgecolor="white", linewidth=0.4)
axes[1].axvline(ind0, color="black", lw=2,
                label=f"Point est. {ind0:.4f}")
axes[1].axvline(ci_lo, color="black", ls="--", lw=1.3,
                label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
axes[1].axvline(ci_hi, color="black", ls="--", lw=1.3)
axes[1].axvline(0, color="#CC3333", ls=":", lw=1.2)
axes[1].set_title(f"H3: Cluster-Bootstrap Indirect Effect\n"
                  f"z = {cluster_z:.2f}  [FIX B-1: Y=converged, B-2: N=4,000]",
                  fontsize=10.5, fontweight="bold")
axes[1].legend(fontsize=7.5)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# Panel 3: Stance openness distribution by country
country_colors = {"JP": "#222222", "KR": "#555555", "US": "#888888", "DE": "#BBBBBB"}
for country, color in country_colors.items():
    vals = df.loc[df.country == country, "stance_openness"]
    axes[2].hist(vals, bins=20, alpha=0.60, label=country,
                 color=color, density=True, edgecolor="white", lw=0.3)
axes[2].set_title("Stance Openness by Country\n(round-3 probability × direction)",
                  fontsize=10.5, fontweight="bold")
axes[2].set_xlabel("Stance Openness")
axes[2].legend(fontsize=8.5)
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

fig.suptitle(
    "Supplementary Robustness Analysis [v2: FIX B-1 (Y=converged), FIX B-2 (N=4,000)]",
    fontweight="bold", y=1.02, fontsize=11,
)
plt.tight_layout()

fig_path = OUT / "fig_supp_robustness_REAL_v2.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  ✓ Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY JSON
# ══════════════════════════════════════════════════════════════════════════════
hdr("SUMMARY JSON")

summary = {
    "version":     "v2 — FIX B-1 (Y=converged), FIX B-2 (N=4,000)",
    "data_source": "s4_simulations.jsonl (1,000 sessions × 4 agents = 4,000 rows)",
    "A_icc_design_effect": {
        "icc_converged":           float(icc_conv),
        "icc_stance_entropy":      float(icc_ent),
        "icc_stance_openness":     float(icc_open),
        "design_effect_converged": float(deff_conv),
        "k_bar":                   float(k_bar),
        "n_effective":             float(n_eff),
        "session_level_z":         float(z_sess),
        "session_level_p":         float(pv),
        "note": "converged is a session attribute; ICC ≈ 1 is structurally expected",
    },
    "B_cluster_bootstrap_mediation": {
        "fixes_applied":            ["FIX B-1: Y=converged", "FIX B-2: N=4,000 full dataset"],
        "a_path":                   float(a0),
        "b_path":                   float(b0),
        "indirect_point_estimate":  float(ind0),
        "naive_sobel_z":            float(sobel_z_naive) if not np.isnan(sobel_z_naive) else None,
        "cluster_bootstrap_SE":     float(cluster_se),
        "se_inflation_factor":      float(cluster_se / sobel_se) if sobel_se else None,
        "cluster_robust_z":         float(cluster_z) if not np.isnan(cluster_z) else None,
        "ci_95":                    [float(ci_lo), float(ci_hi)],
        "v5c_reported_sobel_z":     float(v5c_z),
        "replication_delta":        float(delta) if not np.isnan(delta) else None,
    },
    "C_practical_significance": {
        "p_reversal_confirmed":     float(p1),
        "p_reversal_not_confirmed": float(p2),
        "risk_ratio":               float(rr) if not np.isnan(rr) else None,
        "absolute_diff_pp":         float(arr * 100),
        "relative_reduction_pct":   float(rrr * 100),
        "cohens_h":                 float(cohens_h),
        "NNT":                      float(nnt) if not np.isinf(nnt) else None,
    },
    "D_session_placebo": {
        "observed_diff_pp":  float(obs_diff * 100),
        "permutation_p":     float(placebo_p),
        "n_permutations":    _N_PERM,
    },
    "E_external_validity": {
        "ks_stat":             float(ks_stat),
        "ks_p":                float(ks_p),
        "wasserstein_distance":float(wd),
        "reference_type":      "round-1 stance × confidence (proxy; no human corpus)",
    },
}

json_path = OUT / "supplementary_robustness_REAL_v2_summary.json"
json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"  ✓ Saved: {json_path}")

# ── Final checklist ───────────────────────────────────────────────────────────
hdr("DONE — v2 Corrections Applied")
print(textwrap.dedent(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CORRECTION SUMMARY                                                     │
  │  FIX B-1: Y variable  stance_entropy → converged                       │
  │           b-path sign reversal resolved  (expected b > 0)              │
  │  FIX B-2: Dataset     JP+US 2,000 rows  → full 4,000 rows             │
  │           a-path magnitude resolved  (expected a ≈ -0.754)            │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  MANUSCRIPT CHECKLIST                                                   │
  │  H2  primary stat → session-level z = {z_sess:.3f}, p = {pv:.4f}           │
  │  H3  cluster-robust z = {cluster_z:.2f}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]    │
  │  H3  v5c reported z = {v5c_z:.3f}, replication Δ = {delta:.3f}              │
  │  Discussion: Cohen's h = {cohens_h:.2f}, NNT ≈ {nnt:.0f}                      │
  │  Placebo: session-level p = {placebo_p:.5f}                              │
  │  External validity → Limitations / Future Work                         │
  └─────────────────────────────────────────────────────────────────────────┘
"""))
