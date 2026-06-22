# When the Map Distorts the Territory

### Structural Attribution Bias in Cross-National Patent Data and Its Consequences for Multi-Agent Strategic Discourse

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![R 4.3+](https://img.shields.io/badge/R-4.3%2B-276DC3.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v7.0-green.svg)]()

> **Before running any script:** place `b60w_data.xlsx` at the repository root.  
> This file is not tracked in git. See [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md) for schema and loading instructions.

---

## The Story in One Paragraph

The race to build self-driving cars is one of the most patent-intensive technology competitions in history. Governments, investors, and corporate strategists track this race through patent databases — treating them as an objective scoreboard of who is innovating, and where. But what if the scoreboard's nationality labels are themselves an artifact of corporate filing strategy rather than a neutral record of where innovation occurs?

This study begins with that observation. In the CPC B60W corpus (N = 53,199 citation-validated granted patents), Japanese-origin patents are filed in their home jurisdiction at only **6.2%**, compared to **91.1%** for US-origin patents (χ²(21) = 18,336.51, p < .001, Cramér's V = 0.44). This is not random error. It is a direct consequence of Japanese automakers' global IP filing strategy: Toyota, Honda, and their peers route the vast majority of their patents through US jurisdictions. The scoreboard doesn't just miscount Japan — it structurally transfers Japan's innovation record to the US column.

We then ask a downstream question: if AI agents are briefed with this distorted picture of the autonomous driving patent landscape, does it change how they negotiate competitive strategy? The answer is yes. Agents exposed to the attribution-distortion framing **converged on a shared strategy 10.3 percentage points less often** (17.6% vs. 27.9%; z = −3.877, OR = 0.552, NNT = 9.7) than those in the baseline condition. The mechanism, traced through cross-level mediation analysis, is that the Japanese agent adopts a **closed, defensive posture** — a withdrawal that propagates through the conversation, blocking agreement among all four parties. This stance closure accounts for **35.2% of the pathway at the agent level and 60.3% at the session level** — the level matching convergence's native unit of variation.

A broken database, in other words, doesn't just distort statistics. It distorts judgment.

---

## Domain Context — CPC B60W

**B60W** is the Cooperative Patent Classification (CPC) subclass for *conjoint control of vehicle sub-units of different type or different function*, covering the core technologies of autonomous and intelligent vehicle systems: adaptive cruise control, lane-keeping, collision avoidance, path planning, sensor fusion, and Level 2–5 autonomy stacks.

This subclass draws filings from OEMs (Toyota, Hyundai, Ford, GM), Tier-1 suppliers (Bosch, Continental, Denso), and technology entrants (Waymo, Mobileye, Baidu). The B60W60 subgroup (autonomous driving specifically) grew from near-zero filings in 2012 to a sustained peak after 2018.

When corporate strategists, policy analysts, or AI agents reason about who leads in AV technology, they draw on patent data. A systematic bias that undercounts Japan's contribution does not merely affect bibliometric rankings — it shapes strategic inference about competitive threats, collaboration targets, and investment priorities.

---

## Corpus Construction — N = 53,199 and N = 47,281

Two corpus sizes appear throughout this project. **Both are correct; they describe different pipeline stages.**

```
B60W* (all filings, all years)             559,974
       │
       │  filter: granted patents only
       ▼
Granted patents                            195,531   (grant rate: 34.9%)
       │
       │  filter: forward citation ≥ 1
       ▼
Citation-validated corpus  [Stage 0/3]      53,199   ← H1 attribution analysis
       │
       │  filter: application year 1995–2024
       ▼
Time-series analysis corpus [Stage 1–3]     47,281   ← H2/H3, SHAP, graph analyses
       (excluded 5,918: pre-1995 = 5,756 | post-2024 = 34 | date missing = 128)
```

**N = 53,199** is the full citation-validated corpus used for the H1 nationality-attribution analysis and for building the group-to-nationality crosswalk (Appendix Table A1). **N = 47,281** is the 1995–2024 subset used in all time-series, graph-based, and simulation stages. Any paper section or figure using one of these numbers should identify which stage it belongs to.

---

## The Three-Hypothesis Chain

### H1 — Structural MNAR Bias in Patent-Nationality Attribution

> *Patent-nationality attribution in the CPC B60W corpus is Missing Not at Random (MNAR): the rate at which a patent's filing jurisdiction matches the applicant's true national origin depends systematically on that origin's global IP filing strategy.*

**Method.** Applicant records were first resolved to corporate group labels via an LLM-assisted classification pipeline (`stage0_pipeline.py`), mapping 33,896 of 53,199 patents (63.7%) to one of 925 distinct groups. We then built an explicit, auditable group-to-nationality crosswalk based on each group's publicly known corporate headquarters location (full crosswalk in Appendix Table A1 below and in `scripts/supplementary/extract_crosswalk_A1.py`). Home-jurisdiction filing rate = fraction of a nationality group's patents filed in their own home jurisdiction.

| Nationality | Home-filing rate | n (patents) |
|---|---|---|
| US | **91.1%** | 9,021 |
| KR | 49.6% | 3,949 |
| DE | 24.6% | 5,581 |
| JP | **6.2%** | 13,094 |

**Key statistics:** χ²(21) = 18,336.51, p < .001, Cramér's V = 0.44 (large effect).  
**Toyota Group alone:** 6,016 patents; JP-jurisdiction filings = 269 (4.5%); US-jurisdiction filings = 4,863 (80.8%).

Because the probability that a patent's filing jurisdiction correctly reflects the applicant's nationality depends on a strategic, nationality-correlated choice (where to file for IP protection), missingness is informative and not ignorable (Little & Rubin, 2002) — a MNAR pattern.

> ⚠️ **Crosswalk transparency note.** The nationality crosswalk is researcher-constructed from publicly available headquarters information, not from a verified legal-nationality registry. The complete mapping for the top 60 groups (covering > 85% of mapped patents) is reproduced in Appendix Table A1 of this README and in the supplementary extraction script. Results should be interpreted as a transparent, reproducible approximation.

### H2 — Attribution Distortion Suppresses AI Strategic Convergence

> *When AI agents are made aware that nationality attribution in the underlying patent data systematically understates a given nation's innovation record, multi-agent convergence on a shared strategy is suppressed relative to a baseline condition.*

**Design.** 1,000 simulated three-round negotiation sessions; four LLM-powered agents (Toyota/JP, Hyundai/KR, Ford/US, Bosch/DE). Treatment condition (`reversal_confirmed`, n ≈ 488): briefing explicitly flags the JP attribution distortion. Control condition (`reversal_not_confirmed`, n ≈ 512): same data, no distortion framing. Convergence = `all_agents_converged = TRUE` at end of round 3.

| Condition | Convergence rate | n sessions |
|---|---|---|
| Baseline (`reversal_not_confirmed`) | 27.9% | ≈ 512 |
| Distortion salient (`reversal_confirmed`) | 17.6% | ≈ 488 |
| **Δ** | **−10.3 pp** | — |

**Key statistics:** z = −3.877, p = .0001, OR = 0.552, NNT = 9.7.  
**Placebo permutation test** (N = 100,000 session-label shuffles): p = 0.00016.

A NNT of 9.7 means that, on average, one additional session fails to reach consensus for every 10 sessions in which the distortion-salience framing is introduced.

### H3 — JP Stance Closure Partially Mediates Convergence Failure

> *The Japanese agent's progressive stance closure under distortion salience partially mediates the pathway from distortion-salience exposure to convergence failure, with the proportion mediated depending on whether the outcome is modeled at the agent level or the session level.*

**Cross-level design.** Predictor (JP nationality) and mediator (stance) are agent-level variables (N = 4,000 agent-rounds across 1,000 sessions). Convergence is a session-level outcome shared identically by all four agents within a session — confirmed: within-session variance = 0, ICC(converged) = 1.000. This is a 2-1-2 cross-level mediation design (Preacher, Zyphur & Zhang, 2010).

| Specification | Y | M | indirect | z | 95% CI | c (total) | PM |
|---|---|---|---|---|---|---|---|
| **Agent-level (primary)** | probability | stance_direction | −0.02888 | **−15.686** | [−0.033, −0.025] | −0.082 | **35.2%** |
| **Session-level (robustness)** | converged | mean_openness | −0.06213 | **−4.899** | [−0.088, −0.039] | −0.103 | **60.3%** |
| Agent robustness | converged | stance_openness | −0.09042 | −9.033 | [−0.110, −0.071] | — | — |
| GLMM (random intercept) | converged | stance_* | ≈0 | ≈0 | — | — | ⚠ degenerate |

**PM reporting convention.** The two PM values (35.2% and 60.3%) are not alternative estimates of the same quantity — they answer related but distinct questions at different levels of the cross-level design. Both are reported; the range 35–60% is used as the substantive estimate. Neither value is reported alone without its level label (agent-level) or (session-level).

**GLMM note.** The mixed-effects model with a random session intercept returns fixed effects indistinguishable from zero. This is not evidence against the mediation hypothesis. Because convergence has zero within-session variance by construction, the random intercept absorbs all outcome variance, leaving no within-cluster signal for agent-level fixed effects — a known degenerate case for multilevel models applied to ICC = 1 outcomes (Hox, Moerbeek & van de Schoot, 2017).

**ICC diagnostics:**

| Variable | ICC | N_eff | Interpretation |
|---|---|---|---|
| `converged` | 1.000 | 1,000 | Session-shared; agent-level model degenerate |
| `stance_entropy` | 1.000 | 1,000 | Session-shared |
| `probability` | 0.103 | 3,057 | Agent-level variation exists |
| `stance_openness` | 0.090 | 3,148 | Agent-level variation exists |

---

## Results at a Glance

| Hypothesis | Claim | Verdict | Primary Statistic |
|---|---|---|---|
| **H1** | B60W nationality attribution is MNAR | ✔ Supported | χ²(21) = 18,336.51, p < .001, V = 0.44 |
| **H2** | Attribution distortion suppresses AI convergence | ✔ Strongly supported | Δ = 10.3 pp, z = −3.877, OR = 0.552, NNT = 9.7 |
| **H3** | JP stance closure partially mediates failure | ✔ Strongly supported | PM = 35.2% (agent) / 60.3% (session); z = −15.686 / −4.899 |

All primary tests survive both Benjamini–Hochberg (FDR) and Bonferroni correction.

---

## Causal Architecture

```
[H1: Structural MNAR]  ──distortion salience──►  [H2: Convergence Suppression]
  JP home-filing: 6.2%                              Δ = −10.3 pp (17.6% vs 27.9%)
  US home-filing: 91.1%                             z = −3.877, OR = 0.552, NNT = 9.7
  χ²(21) = 18,336.51, V = 0.44
       │
       │  Salience → H3 mediation
       ▼
  [M: JP Stance Closure]  ──b = +0.038***──►  [Outcome: Discourse Failure]
   a = −0.754***                                 17.6% convergence vs 27.9% baseline
       │
       └── indirect = −0.029*** (agent-level PM = 35.2%)
           indirect = −0.062*** (session-level PM = 60.3%)
           ICC(converged) = 1.000 → 2-1-2 cross-level design
```

---

## Figures

All paper figures (Fig. 1, 2, 4, 5, A, B) are generated by `scripts/Figures_all.R`. Fig. 3 (H2 convergence panels) is generated by `scripts/stage4_postanalysis_v5c.py`.

| Figure | File | Generated by | What it shows |
|---|---|---|---|
| **Fig. 1** | `outputs/fig1_theory_v6.png` | `Figures_all.R` | Causal architecture: H1 → H2 → H3 path diagram with confirmed effect sizes |
| **Fig. 2** | `outputs/fig2_H1_mnar_v6.png` | `Figures_all.R` | H1 MNAR evidence: home-filing rates by nationality; Toyota Group filing destinations; home-filing vs. missingness scatter |
| **Fig. 3** | `outputs/paper_v4/figures/fig3_H2_convergence_v4.png` | `stage4_postanalysis_v5c.py` | H2 evidence: convergence rate by condition; stance entropy; JP closure Cohen's d; effect size summary |
| **Fig. 4** | `outputs/fig4_H3_mediation_v6.png` | `Figures_all.R` | H3 evidence: mediation path diagram (agent-level); indirect effects by specification; ICC diagnostics |
| **Fig. 5** | `outputs/fig5_integrated_dashboard_v6.png` | `Figures_all.R` | Integrated 2×6 evidence dashboard across H1, H2, H3 |
| **Fig. A** | `outputs/figA_crosslevel_architecture_v6.png` | `Figures_all.R` | Cross-level (2-1-2) mediation architecture; ICC diagnostics panel |
| **Fig. B** | `outputs/figB_PM_comparison_v6.png` | `Figures_all.R` | PM specification comparison: PM by level; indirect effects with 95% CI; Y×M 2×2 decomposition |

> **Fig. 1 and Fig. A caption note.** The session-level PM shown inside these figures is 61.2% (as computed in the R script). The Methods text and all statistical tables use **60.3%** (from `fix_v3_session_level_mediation.csv`, indirect = −0.06213 / c = −0.103). The 0.9 pp difference reflects rounding in intermediate outputs. Captions should note: *"Session-level PM shown as 61.2% in figure; text reports 60.3% from primary CSV output (difference < 1 pp, attributable to rounding)."*

### Running the R Figures

```r
# Requires: ggplot2, gridExtra, grid, scales
# Output directory: ./outputs/

source("scripts/Figures_all.R")
# Generates: fig1_theory_v6.png, fig2_H1_mnar_v6.png,
#            fig4_H3_mediation_v6.png, fig5_integrated_dashboard_v6.png,
#            figA_crosslevel_architecture_v6.png, figB_PM_comparison_v6.png
```

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── b60w_data.xlsx                  # ★ PRIMARY SOURCE — NOT tracked in git (43.4 MB)
│                                   #   Place at repo root before running any stage.
│
├── data/
│   └── DATA_DESCRIPTION.md         # Full schema, column reference, loading instructions
│
├── config/
│   └── agents_config.json          # Agent personas (JP/KR/US/DE) for simulation
│
├── scripts/
│   ├── Figures_all.R                   # ★ R script: generates Fig 1,2,4,5,A,B (v6.0 numbers)
│   ├── stage0_pipeline.py              # Stage 0: Raw B60W ingestion & LLM applicant grouping
│   ├── stage1_graph_pipeline.py        # Stage 1: Keyword extraction, citation graph, embeddings
│   ├── stage1-2_patch_features.py      # Stage 1 patch: citation columns, node-age fix
│   ├── stage2_mamba_lstm.py            # Stage 2: Mamba vs LSTM forecasting
│   ├── stage2-2_chow_b60w60.py         # Stage 2: Chow structural-break test (B60W60)
│   ├── stage3_shap.py                  # Stage 3: TF-SHAP attribution (nationality profile)
│   ├── stage4_sllm_multiagent.py       # Stage 4: Multi-agent simulation engine (1,000 sessions)
│   ├── stage4_postanalysis_v5c.py      # Stage 4: H2 convergence test + H3 mediation + Fig 3
│   │
│   └── supplementary/
│       ├── extract_crosswalk_A1.py              # ★ Extracts Appendix Table A1 from parquet
│       ├── supplementary_fix_v3.py              # 2×2 Y×M decomposition; ICC-corrected mediation
│       ├── supplementary_robustness_REAL_v2.py  # Independent replication + session-level placebo
│       └── diagnose_sobel_z_discrepancy.py      # Root-cause diagnosis of z-statistic difference
│
└── outputs/
    ├── fig1_theory_v6.png                  # ★ R-generated
    ├── fig2_H1_mnar_v6.png                 # ★ R-generated
    ├── fig4_H3_mediation_v6.png            # ★ R-generated
    ├── fig5_integrated_dashboard_v6.png    # ★ R-generated
    ├── figA_crosslevel_architecture_v6.png # ★ R-generated
    ├── figB_PM_comparison_v6.png           # ★ R-generated
    │
    ├── paper_v4/
    │   └── figures/
    │       ├── fig3_H2_convergence_v4.png  # Python-generated (stage4_postanalysis_v5c.py)
    │       └── appA_composition_v4.png     # Exploratory (not in primary analysis)
    │
    ├── stage_work/
    │   ├── stage0/   step3_applicant_groups.parquet              (58.4 MB)
    │   ├── stage1/   s1_node_features_f8fixed.parquet            (804 KB)
    │   ├── stage2/   s2_top20_v5.xlsx, s2_chow_b60w60_summary.xlsx
    │   ├── stage3/   s3_shap_level2_nationality.xlsx, s3_shap_level2_jurisdiction.xlsx
    │   └── stage4/
    │       ├── s4_simulations.jsonl                              (6.8 MB)
    │       └── postanalysis_v5c/
    │           └── s4v5c_mediation_analysis.xlsx
    │
    └── supplementary/
        ├── fix_v3/
        │   ├── fix_v3_unified_table3.csv       # All Y×M specs — Table 4 source
        │   ├── fix_v3_Y_decomposition_table.csv
        │   ├── fix_v3_ICC_diagnostics.csv      # ICC table source
        │   ├── fix_v3_session_level_mediation.csv  # Session-level PM = 60.3% source
        │   ├── fix_v3_GEE_results.csv
        │   ├── fix_v3_GLMM_results.csv
        │   └── fig_fix_v3_panel.png
        └── sobel_diagnosis/
            └── sobel_diagnosis_summary.json
```

---

## Data

### Primary Source — `b60w_data.xlsx`

| File | Size | In git | Description |
|---|---|---|---|
| `b60w_data.xlsx` | 43.4 MB | **No** | Citation-validated B60W patent corpus — place at repo root |

**Corpus construction funnel:**

| Stage | Count | Notes |
|---|---|---|
| B60W* all filings | 559,974 | All CPC B60W applications, all years |
| Granted patents | 195,531 | Grant rate: 34.9% |
| **Citation-validated (Stage 0 corpus)** | **53,199** | **≥ 1 forward citation; used for H1** |
| **Time-series corpus (Stage 1+)** | **47,281** | 1995–2024 window; used for H2/H3/SHAP |

**Quick facts (N = 53,199):**

| Property | Value |
|---|---|
| Forward citations | 100% have ≥ 1 (filter criterion) |
| Application date range | 1897–2025; analysis window 1995Q1–2024Q4 |
| Jurisdictions | 28 (US 71.6%, KR 8.7%, DE 5.7%, EP 4.8%, JP 4.1%) |
| Top applicant | Toyota Group — 6,016 patents (crosswalk-matched) |
| LLM-mapped patents | 33,896 (63.7%); 925 distinct corporate groups |
| B60W60 subgroup | 2,210 patents (first observed 2012Q1, peak 2020) |

### Intermediate Files

| Description | Path | Size |
|---|---|---|
| Applicant groups (Stage 0) | `outputs/stage_work/stage0/step3_applicant_groups.parquet` | 58.4 MB |
| Node features, patched (Stage 1) | `outputs/stage_work/stage1/s1_node_features_f8fixed.parquet` | 804 KB |
| Forecasting results (Stage 2) | `outputs/stage_work/stage2/s2_top20_v5.xlsx` | 109 KB |
| SHAP nationality profile (Stage 3) | `outputs/stage_work/stage3/s3_shap_level2_nationality.xlsx` | 64 KB |
| Raw simulation logs (Stage 4) | `outputs/stage_work/stage4/s4_simulations.jsonl` | 6.8 MB |
| Cross-level mediation, all specs | `outputs/supplementary/fix_v3/fix_v3_unified_table3.csv` | — |
| ICC diagnostics | `outputs/supplementary/fix_v3/fix_v3_ICC_diagnostics.csv` | — |
| Session-level mediation | `outputs/supplementary/fix_v3/fix_v3_session_level_mediation.csv` | — |

---

## Installation

```bash
git clone https://github.com/your-org/patent-attribution-bias.git
cd patent-attribution-bias
pip install -r requirements.txt
```

Core Python dependencies:

```
numpy>=1.24        pandas>=2.0         scipy>=1.11
statsmodels>=0.14  scikit-learn>=1.3   matplotlib>=3.7
seaborn>=0.12      torch>=2.0          transformers>=4.35
openpyxl>=3.1      fastparquet>=2023.7  tqdm>=4.65
pyarrow>=12.0
```

R dependencies (for `Figures_all.R`):

```r
install.packages(c("ggplot2", "gridExtra", "grid", "scales"))
```

---

## Reproduction Walkthrough

Run stages in order. Each stage reads from the previous stage's `outputs/stage_work/` directory.

### Stage 0 — Ingest the B60W corpus and group applicants

```bash
python scripts/stage0_pipeline.py
# Loads b60w_data.xlsx; LLM-based applicant grouping → 925 corporate groups
# → outputs/stage_work/stage0/step3_applicant_groups.parquet
```

### Stage 0 (Supplementary) — Extract Appendix Table A1 crosswalk

```bash
python scripts/supplementary/extract_crosswalk_A1.py
# Reads step3_applicant_groups.parquet; applies headquarters nationality mapping
# Computes H1 contingency table and chi-square test
# → outputs/supplementary/crosswalk_A1.csv  (top-60 groups with nationality)
# → outputs/supplementary/h1_contingency_results.csv  (χ²=18,336.51, V=0.44)
```

### Stage 1 — Build the patent citation graph and extract node features

```bash
python scripts/stage1_graph_pipeline.py
python scripts/stage1-2_patch_features.py
# Corrects f3 (forward citation), f4 (backward citation), f5 (node age)
# → outputs/stage_work/stage1/s1_node_features_f8fixed.parquet
```

### Stage 2 — Forecast AV patent trends; identify structural breaks

```bash
python scripts/stage2_mamba_lstm.py
python scripts/stage2-2_chow_b60w60.py
# Chow test for B60W60 structural break (2017Q1)
```

### Stage 3 — SHAP attribution by nationality

```bash
python scripts/stage3_shap.py
# TF-SHAP scores under jurisdiction-based vs nationality-based coding
# → outputs/stage_work/stage3/s3_shap_level2_nationality.xlsx
```

### Stage 4 — Run multi-agent simulations; test H2 and H3

```bash
# Requires OpenAI-compatible API endpoint configured in config/agents_config.json
python scripts/stage4_sllm_multiagent.py
# 1,000 sessions × 4 agents × 3 rounds = 12,000 agent-rounds
# → outputs/stage_work/stage4/s4_simulations.jsonl

python scripts/stage4_postanalysis_v5c.py
# H2: convergence z-test + entropy Mann–Whitney
# H3: Baron–Kenny cluster-bootstrap mediation (agent-level primary pipeline)
# Generates Fig. 3
```

### Supplementary — Cross-level mediation, ICC correction, robustness

```bash
# Step 1: Root-cause diagnosis of z-statistic difference across specifications
python scripts/supplementary/diagnose_sobel_z_discrepancy.py
# → outputs/supplementary/sobel_diagnosis/sobel_diagnosis_summary.json

# Step 2: 2×2 Y×M decomposition + GEE + GLMM + session-level gold standard
python scripts/supplementary/supplementary_fix_v3.py
# Produces Table 4 source data and ICC diagnostics
# → outputs/supplementary/fix_v3/fix_v3_unified_table3.csv
# → outputs/supplementary/fix_v3/fix_v3_ICC_diagnostics.csv
# → outputs/supplementary/fix_v3/fix_v3_session_level_mediation.csv

# Step 3: Independent replication + session-label placebo permutation test
python scripts/supplementary/supplementary_robustness_REAL_v2.py
# Placebo: N = 100,000 permutations → p = 0.00016
```

### Figures — R script (Fig. 1, 2, 4, 5, A, B)

```r
# In R (working directory = repo root):
source("scripts/Figures_all.R")
# Output: outputs/fig1_theory_v6.png through figB_PM_comparison_v6.png
# Resolution: 300 DPI, publication-ready
```

---

## Methods Note — The z-Statistic Difference (§3.4)

Two z-statistics appear in the H3 mediation results. Both are correct; they describe different quantities within the cross-level design.

**2×2 Y×M decomposition** (`supplementary_fix_v3.py`):

| | M = `stance_direction_r3` | M = `stance_openness` |
|---|---|---|
| **Y = `probability`** (agent-level, continuous) | z = −15.686 ★ **PRIMARY** | z = −17.706 |
| **Y = `converged`** (session-shared, binary) | z = −8.925 | z = −9.033 |

Switching M while holding Y = probability: Δz = 2.02 — M is not the dominant driver.  
Switching Y while holding M = stance_direction: Δz = **6.76** — **Y operationalization is the dominant driver.**

The structural reason: `converged` is session-level (ICC = 1.000, N_eff = 1,000). `probability` is agent-level (ICC = 0.103, N_eff = 3,057). The precision difference reflects ICC structure, not mediator misspecification.

**Session-level gold standard** (n = 1,000, ICC-corrected): indirect = −0.062, z = −4.899, 95% CI [−0.088, −0.039], PM = 60.3%.

**Reporting convention.** z = −15.686 is the primary H3 statistic (Y = probability, M = stance_direction_r3, agent-level). z = −4.899 is the session-level robustness estimate. z = −9.033 is the agent-level robustness check under Y = converged. All three are reported in Table 4 with their specification labels.

---

## Multi-Agent Simulation Design

| Agent | Nationality | Strategic Persona |
|---|---|---|
| Toyota Group | JP | Cautious, incremental hybrid-to-EV roadmap; sensor-fusion depth |
| Hyundai Motor Group | KR | Aggressive EV-first market pursuit; software-defined vehicle push |
| Ford Motor Company | US | North American regulatory context; large-vehicle electrification |
| Bosch | DE | Tier-1 supplier standardization; modular platform strategy |

**Conditions:**
- `reversal_confirmed` (distortion salient): explicit briefing that JP patent attribution is systematically understated (n ≈ 488 sessions)
- `reversal_not_confirmed` (baseline): same underlying patent data, no distortion framing (n ≈ 512 sessions)
- Convergence criterion: `all_agents_converged = TRUE` at end of round 3

---

## Multiple Testing

Four primary tests (H1, H2 z-test, H2 entropy MW, H3 cluster-z). Session-level placebo permutation is reported separately.

| Test | p-raw | p-BH | p-Bonferroni | Rejected |
|---|---|---|---|---|
| H1: χ²(21) = 18,336.51 | < 10⁻³⁰⁰ | < 10⁻³⁰⁰ | < 10⁻³⁰⁰ | ✔ |
| H2: z-test | 1×10⁻⁴ | 1×10⁻⁴ | 4×10⁻⁴ | ✔ |
| H2: entropy (Mann–Whitney) | 1.5×10⁻³ | 1.5×10⁻³ | 6×10⁻³ | ✔ |
| H3: cluster-z (agent-level) | < 1×10⁻⁷ | < 4×10⁻⁷ | < 4×10⁻⁷ | ✔ |

All four tests survive both Benjamini–Hochberg (FDR) and Bonferroni correction.

---

## Appendix Table A1 — Group-to-Nationality Crosswalk (Top 60 Groups)

**Source:** `step3_applicant_groups.parquet`, researcher-constructed from publicly available corporate headquarters information.  
**Coverage:** Top 60 groups by patent count, covering > 85% of the 33,896 mapped patents.  
**Excluded:** Individual inventors, UNKNOWN entities, groups with < 10 patents.

| Rank | Corporate Group | Nationality | Patents (n) | Headquarters basis |
|---|---|---|---|---|
| 1 | Toyota Group | JP | 6,016 | Toyota Motor Corp., Aichi, Japan |
| 2 | Hyundai Motor Group | KR | 3,782 | Hyundai Motor Co., Seoul, Korea |
| 3 | Ford Group | US | 2,893 | Ford Motor Co., Dearborn, MI, USA |
| 4 | Honda Group | JP | 1,987 | Honda Motor Co., Tokyo, Japan |
| 5 | Bosch Group | DE | 1,952 | Robert Bosch GmbH, Stuttgart, Germany |
| 6 | GM Group | US | 1,881 | General Motors Co., Detroit, MI, USA |
| 7 | Nissan Group | JP | 1,771 | Nissan Motor Co., Yokohama, Japan |
| 8 | Waymo/Alphabet | US | 1,487 | Waymo LLC / Alphabet Inc., Mountain View, CA |
| 9 | Volkswagen Group | DE | 1,316 | Volkswagen AG, Wolfsburg, Germany |
| 10 | Mitsubishi Group | JP | 762 | Mitsubishi Motors Corp., Tokyo, Japan |
| 11 | Hitachi Group | JP | 749 | Hitachi Ltd., Tokyo, Japan |
| 12 | ZF Group | DE | 620 | ZF Friedrichshafen AG, Germany |
| 13 | Mercedes-Benz Group | DE | 547 | Mercedes-Benz Group AG, Stuttgart, Germany |
| 14 | Volvo Group | SE | 482 | AB Volvo, Gothenburg, Sweden |
| 15 | Continental Group | DE | 455 | Continental AG, Hanover, Germany |
| 16 | Eaton Group | US | 437 | Eaton Corporation (US-origin) |
| 17 | BMW Group | DE | 364 | Bayerische Motoren Werke AG, Munich, Germany |
| 18 | Mazda Group | JP | 343 | Mazda Motor Corp., Hiroshima, Japan |
| 19 | Subaru Group | JP | 322 | Subaru Corporation, Tokyo, Japan |
| 20 | Magna Group | CA | 294 | Magna International Inc., Aurora, Ontario |
| 21 | Tata Group | IN | 288 | Tata Motors Ltd., Mumbai, India |
| 22 | Renault Group | FR | 257 | Renault SA, Boulogne-Billancourt, France |
| 23 | Panasonic Group | JP | 223 | Panasonic Holdings Corp., Osaka, Japan |
| 24 | Bendix Group | US | 197 | Bendix Commercial Vehicle Systems (US ops) |
| 25 | Scania/Traton Group | SE | 196 | Scania AB / Traton SE, Södertälje, Sweden |
| 26 | Komatsu Group | JP | 188 | Komatsu Ltd., Tokyo, Japan |
| 27 | Intel Group | US | 184 | Intel Corporation, Santa Clara, CA |
| 28 | Schaeffler Group | DE | 178 | Schaeffler AG, Herzogenaurach, Germany |
| 29 | Denso Group | JP | 176 | DENSO Corporation, Kariya, Japan |
| 30 | Subaru Group (alt) | JP | 176 | Subaru Corporation, Tokyo, Japan |
| 31 | Cummins Group | US | 175 | Cummins Inc., Columbus, IN |
| 32 | LG Group | KR | 166 | LG Electronics Inc., Seoul, Korea |
| 33 | Delphi Group | US | 132 | Aptiv PLC (formerly Delphi; US-origin) |
| 34 | Yamaha Group | JP | 119 | Yamaha Motor Co., Shizuoka, Japan |
| 35 | Isuzu Group | JP | 119 | Isuzu Motors Ltd., Tokyo, Japan |
| 36 | Stellantis Group | NL | 113 | Stellantis N.V., Amsterdam, Netherlands |
| 37 | BorgWarner Group | US | 106 | BorgWarner Inc., Auburn Hills, MI |
| 38 | Valeo Group | FR | 100 | Valeo SA, Paris, France |
| 39 | Chrysler Group | US | 98 | US-origin legacy (now part of Stellantis) |
| 40 | Suzuki Group | JP | 83 | Suzuki Motor Corp., Hamamatsu, Japan |
| 41 | Fiat Group | IT | 83 | Fiat SpA / Stellantis (Italian-origin) |
| 42 | Geely Group | CN | 83 | Geely Automobile Holdings, Hangzhou, China |
| 43 | Harman Group | US | 80 | Harman International, Stamford, CT |
| 44 | Flextronics Group | SG | 75 | Flex Ltd., Singapore |
| 45 | Kubota Group | JP | 71 | Kubota Corporation, Osaka, Japan |
| 46 | NIO Group | CN | 64 | NIO Inc., Shanghai, China |
| 47 | Oshkosh Group | US | 64 | Oshkosh Corporation, Oshkosh, WI |
| 48 | Visteon Group | US | 62 | Visteon Corporation, Van Buren Township, MI |
| 49 | Mobileye Group | US | 58 | Mobileye (Intel subsidiary), Santa Clara |
| 50 | Tesla Group | US | 55 | Tesla Inc., Austin, TX |
| 51 | Samsung Group | KR | 54 | Samsung Electronics Co., Suwon, Korea |
| 52 | Qualcomm Group | US | 51 | Qualcomm Inc., San Diego, CA |
| 53 | NVIDIA Group | US | 49 | NVIDIA Corporation, Santa Clara, CA |
| 54 | Aisin Group | JP | 47 | Aisin Corporation, Kariya, Japan |
| 55 | Stradvision Group | KR | 43 | StradVision Inc., Seoul, Korea |
| 56 | BYD Group | CN | 41 | BYD Co. Ltd., Shenzhen, China |
| 57 | Baidu Group | CN | 38 | Baidu Inc., Beijing, China |
| 58 | Huawei Group | CN | 35 | Huawei Technologies Co., Shenzhen, China |
| 59 | Furukawa Group | JP | 2 | Furukawa Electric Co., Tokyo, Japan |
| 60 | Kia Group | KR | 6 | Kia Corporation, Seoul, Korea |

**Nationality codes:** JP = Japan, US = United States, DE = Germany, KR = Korea, SE = Sweden, FR = France, CN = China, CA = Canada, IN = India, IT = Italy, NL = Netherlands, SG = Singapore

**Reproducibility.** Merging `step3_applicant_groups.parquet` (field: `applicant_group_primary`) with the `nationality` column above — and then cross-tabulating by `Jurisdiction` (top 7 + Other) — fully reproduces the H1 chi-square test (χ²(21) = 18,336.51, V = 0.44). The extraction script is `scripts/supplementary/extract_crosswalk_A1.py`.

---

## Manuscript Conventions

**Verified statistics — use these exactly:**

| Statistic | Correct value | Do not use |
|---|---|---|
| JP home-filing rate | **6.2%** (n = 13,094) | ~~44.8%~~ |
| DE home-filing rate | **24.6%** (n = 5,581) | ~~95.4%~~ |
| US home-filing rate | **91.1%** (n = 9,021) | — |
| H1 chi-square | **χ²(21) = 18,336.51, V = 0.44** | ~~χ²(24) = 3,134.76, V = 0.13~~ |
| H3 primary z | **z = −15.686** (≈ −15.7) | ~~z = −15.3 alone~~ |
| H3 session-level PM | **60.3%** | ~~61.2%~~ |
| H3 PM reporting | **Always label level:** PM = 35.2% (agent-level) / PM = 60.3% (session-level) | ~~PM = 35.2% alone~~ |

**Do not use:** H1a, H1b, H2a, H2b as standalone confirmed hypothesis labels. Characterize the z = −15.686 vs. z = −4.899 difference as a "discrepancy" — it is a known structural property of the 2-1-2 cross-level design, explained by ICC(converged) = 1.000. Treat Appendix A composition results as confirmatory.

---

## License

MIT License. See [LICENSE](LICENSE). The underlying patent corpus is sourced from public patent databases; all derived datasets follow their respective terms of use.
