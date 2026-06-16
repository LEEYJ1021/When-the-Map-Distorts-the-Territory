# When the Map Distorts the Territory

### Structural Attribution Bias in Cross-National Patent Data and Its Consequences for Multi-Agent Strategic Discourse

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v4.1-green.svg)]()

> **Before running any script:** place `b60w_data.xlsx` at the repository root.
> This file is not tracked in git. See [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md) for schema and loading instructions.

---

## The Story in One Paragraph

The race to build self-driving cars is one of the most patent-intensive technology competitions in history. Governments, investors, and corporate strategists all track this race through patent databases — treating them as an objective scoreboard of who is innovating, and where. But what if the scoreboard is structurally miscalibrated against one of the race's most active participants?

This study begins with that observation. In the CPC B60W corpus — the international patent classification for vehicle control systems and autonomous driving — **Japanese inventors are mapped to their true nationality at only 44.8%**, compared to 95.4% for German inventors. This is not random error. It is a direct consequence of Japanese automakers' global filing strategy: Toyota, Honda, and their peers route the vast majority of their patents through US jurisdictions, so the database reads them as American. The scoreboard doesn't just miscount Japan — it systematically transfers Japan's innovation record to the US column.

We then ask a downstream question: if AI agents are briefed with this distorted picture of the autonomous driving patent landscape, does it change how they negotiate competitive strategy? The answer is yes, and the consequences are concrete. Agents exposed to the attribution-distortion framing **converged on a shared strategy 10.3 percentage points less often** than those in the baseline condition. The mechanism, traced through mediation analysis, is that the Japanese agent adopts a **closed, defensive posture** — and that withdrawal propagates through the conversation, blocking agreement among all four parties.

A broken database, in other words, doesn't just distort statistics. It distorts judgment.

---

## Domain Context — CPC B60W and the Autonomous Driving Patent Race

**B60W** is the Cooperative Patent Classification (CPC) subclass for *conjoint control of vehicle sub-units of different type or different function*, which in practice covers the core technologies of autonomous and intelligent vehicle systems: adaptive cruise control, lane-keeping, collision avoidance, path planning, sensor fusion, and Level 2–5 autonomy stacks.

This subclass sits at the intersection of the automotive and semiconductor industries, drawing filings from OEMs (Toyota, Hyundai, Ford, GM), Tier-1 suppliers (Bosch, Continental, Denso), and technology entrants (Waymo, Mobileye, Baidu). It is one of the fastest-growing patent classifications globally, with the B60W60 subgroup (autonomous driving specifically) growing from near-zero filings in 2012 to a sustained peak after 2018.

**Why B60W matters for this study.** The geopolitical stakes of autonomous driving make patent attribution particularly consequential. When corporate strategists, policy analysts, or AI agents reason about who leads in AV technology, they draw on patent data. A systematic bias that undercounts Japan's contribution does not merely affect bibliometric rankings — it shapes strategic inference about competitive threats, collaboration targets, and investment priorities. That downstream effect is exactly what H2 and H3 measure.

---

## The Corpus — How 53,199 Patents Were Selected

Not all 559,974 B60W filings entered the analysis. The corpus was built through a three-stage quality filter designed to keep only patents that represent verified, impactful innovation.

```
B60W* (all filings, all years)          559,974
       │
       │  filter: granted patents only
       ▼
Granted patents                         195,531   (grant rate: 34.9%)
       │
       │  filter: forward citation ≥ 1
       ▼
Citation-validated corpus                53,199   ← core analysis corpus
```

**Stage 1 — Grant filter (559,974 → 195,531).** Only granted patents are retained. Applications that were rejected, withdrawn, or remain pending are excluded. This ensures the corpus reflects inventions that passed substantive examination in at least one jurisdiction.

**Stage 2 — Citation filter (195,531 → 53,199).** Only patents cited at least once by a subsequent patent are retained. This is the critical quality gate. A forward citation means another inventor, in a subsequent filing, identified this patent as relevant prior art — confirming that the invention was noticed and engaged with by the field. Patents with zero forward citations may represent genuine but uninfluential work, or may reflect filing noise (defensive filings, portfolio padding). Excluding them concentrates the corpus on innovations that demonstrably shaped subsequent technological development.

The resulting **53,199 patents** represent roughly 9.5% of all B60W filings but, by construction, a disproportionate share of the field's actual technological influence. This citation-validation filter is also the study's primary defense against reviewer concerns about data quality: the corpus is not a convenience sample but a reproducible, criterion-defined population of impactful autonomous driving patents.

> The prior literature on AV patent landscapes has typically worked with estimated corpora of ~45,000 patents. The full enumeration here (53,199 citation-validated grants from 195,531 total grants, drawn from 559,974 filings) provides a more precisely bounded and defensible empirical foundation.

---

## The Three-Hypothesis Chain

The study is organized as a single causal argument. Each hypothesis is a necessary step: H1 establishes that the data is structurally broken, H2 shows that this brokenness matters for AI reasoning, and H3 identifies the behavioral mechanism through which it operates.

### H1 — The Attribution Map Is Structurally Distorted

> *"Missingness in patent–nationality mapping is not random. It is driven by where patents are filed, not by who invented them — and the pattern systematically disadvantages Japan."*

Japanese automotive and electronics companies file **93.1% of their B60W patents outside Japan**, with the majority routed through US jurisdictions. This is a rational legal strategy: US patents offer broader enforcement rights and larger market coverage. But it has a measurement consequence. Patent databases infer inventor nationality primarily from the filing jurisdiction. A Toyota patent filed in the USPTO is coded as a US patent. Japan's true share of autonomous driving innovation is therefore systematically undercounted, and the US share is correspondingly inflated.

This is not a missing-at-random (MAR) problem — it is **Missing Not at Random (MNAR)**: the probability that a patent is correctly attributed to Japan depends directly on whether the inventor chose to file in Japan, which in turn depends on competitive strategy variables (market coverage goals, enforcement preferences, IP portfolio management) that are correlated with the innovation itself. Missingness is informative, not ignorable.

A chi-square test of nationality mapping rates across 28 jurisdictions confirms the structural non-randomness with overwhelming statistical force.

**Key statistic:** χ²(24) = 3,134.76, *p* < .0001, Cramér's *V* = 0.1287 — JP maps at 44.8% vs DE at 95.4%, a 50.6 percentage-point structural gap affecting the corpus's representation of 4,129 Japan-attributed patents.

### H2 — Distorted Data Degrades AI Strategic Reasoning

> *"When AI agents are made aware that the patent attribution data misrepresents Japan's contribution, they converge on a shared autonomous driving strategy significantly less often."*

To test whether H1-level bias propagates into downstream reasoning, we ran 1,000 simulated three-round strategy sessions using four LLM-powered agents with fixed corporate identities: Toyota Group (JP), Hyundai Motor Group (KR), Ford Motor Company (US), and Bosch (DE). Each session began with a briefing derived from B60W patent data. In the treatment condition (`reversal_confirmed`), the briefing explicitly flagged that Japan's patent attribution was systematically distorted — framing Toyota as an undercounted innovator whose true competitive position was stronger than the data suggested. The control condition (`reversal_not_confirmed`) used the same data without the distortion framing.

After three rounds of structured negotiation, convergence was defined as all four agents agreeing on a shared strategic position. The treatment condition produced a **convergence rate of 17.6%**, compared to **27.9%** in the control — a 10.3 percentage-point reduction. The NNT of 9.7 gives this a concrete operational meaning: for approximately every 10 strategy sessions conducted under the distortion-salient framing, one consensus outcome that would otherwise have been reached is lost.

**Key statistic:** *z* = −3.877, *p* = .0001, OR = 0.552, NNT = 9.7.

### H3 — The Mechanism Is Japan's Defensive Stance Closure

> *"The Japanese agent does not simply lose the argument — it withdraws from it. That withdrawal, propagated through the negotiation, accounts for 35% of the total convergence failure."*

H2 establishes that distortion awareness suppresses convergence but does not explain how. H3 opens the black box. In distortion-salient sessions, the Toyota/JP agent shows a dramatically lower **stance openness** score (mean = −0.928) compared to all three other agents (all positive), with Cohen's *d* ranging from −1.09 to −1.56 — effect sizes classified as very large. This is not simply the agent updating its strategic position; it is a pattern of progressive withdrawal: expressed probability decreases across rounds, strategic flexibility contracts, and the agent's position hardens into a defensive posture.

This stance closure is a **mediator**, not merely a symptom. It statistically carries 35.2% of the pathway from distortion exposure to convergence failure. The path runs as follows: distortion salience triggers JP stance closure (a-path = −0.754***), JP stance closure in turn reduces the probability of session-wide convergence (b-path = +0.038***), and the indirect effect survives across 12 robustness subsets including temporal splits, RAG-k variants, and half-sample trials.

The mediation is partial, not full — direct effects of the distortion framing on convergence exist beyond the JP-stance-closure pathway. But the mechanism identified here is robust, interpretable, and has a clear behavioral reading: when an AI agent perceives that the information it is given systematically undervalues its principal's innovation record, it stops trying to build consensus.

**Key statistic:** PM = 35.2%, cluster-robust *z* = −15.69, 95% CI [−0.0327, −0.0254].

---

## Results at a Glance

| Hypothesis | Claim | Verdict | Primary Statistic |
|---|---|---|---|
| **H1** | B60W nationality missingness is MNAR | ✔ CONFIRMED | χ²(24) = 3134.76, *p* < .0001, *V* = 0.1287 |
| **H2** | Attribution distortion suppresses AI convergence | ✔ STRONGLY SUPPORTED | Δ = 10.3 pp, *z* = −3.877, OR = 0.552, NNT = 9.7 |
| **H3** | JP stance closure partially mediates failure | ✔ STRONGLY SUPPORTED | PM = 35.2%, cluster-*z* = −15.69, 95% CI [−0.0327, −0.0254] |
| **App. A** | Jurisdiction vs nationality coding amplifies the gap | ⚠ EXPLORATORY | +273%; *n*_JP_jur = 44; CI = [−93%, +2922%] — not confirmatory |

All four primary tests survive both Benjamini–Hochberg and Bonferroni correction. Appendix A is excluded from the multiple-testing family.

---

## Causal Architecture

```
[H1: MNAR Bias in B60W]  ──MNAR exposure──►  [H2: Convergence Suppression]
  JP 44.8% vs DE 95.4%                          Δ = 10.3pp (17.6% vs 27.9%)
  χ²(24) = 3134.76, p < .0001                   z = −3.877, p < .001***
  93.1% JP outbound filing                       OR = 0.552, NNT = 9.7
       │
       │  Salience → H3
       ▼
  [M: JP Stance Closure]  ──b = +0.038***──►  [Outcome: Discourse Failure]
   mean openness = −0.928                        17.6% convergence
   a = −0.754***                                 vs 27.9% baseline
       │
       └── indirect = −0.029***  (PM = 35.2%)
           cluster-robust z = −15.69
           95% CI [−0.0327, −0.0254]
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
│   ├── stage0_pipeline.py              # Stage 0: Raw B60W patent ingestion & applicant grouping
│   ├── stage1_graph_pipeline.py        # Stage 1: Keyword extraction, citation graph construction
│   ├── stage1-2_patch_features.py      # Stage 1 patch: citation columns, node-age fix
│   ├── stage2_mamba_lstm.py            # Stage 2: Mamba vs LSTM forecasting of AV patent trends
│   ├── stage2-2_chow_b60w60.py         # Stage 2: Chow structural-break test (B60W60 subgroup)
│   ├── stage3_shap.py                  # Stage 3: TF-SHAP attribution analysis (H1 evidence)
│   ├── stage4_sllm_multiagent.py       # Stage 4: Multi-agent simulation engine (1,000 sessions)
│   ├── stage4_postanalysis_v5c.py      # Stage 4: H2 convergence test + H3 mediation + figures
│   │
│   └── supplementary/
│       ├── supplementary_fix_v3.py              # 2×2 Y×M decomposition; ICC-corrected mediation
│       │                                        #   resolves Sobel z discrepancy (Methods §3.4)
│       ├── supplementary_robustness_REAL_v2.py  # Independent replication + session-level placebo
│       └── diagnose_sobel_z_discrepancy.py      # Three-stage root-cause diagnosis of z discrepancy
│
└── outputs/
    ├── paper_v4/
    │   ├── master_v4.json
    │   ├── methods_sobel_reconciliation_EN.txt   # Methods §3.4 — English narrative
    │   ├── methods_sobel_reconciliation_KR.txt   # Methods §3.4 — Korean summary
    │   └── figures/
    │       ├── fig1_theory_v4.png          # Causal theory model (H1→H2→H3)
    │       ├── fig2_H1_mnar_v4.png         # H1: MNAR evidence (3-panel)
    │       ├── fig3_H2_convergence_v4.png  # H2: convergence suppression (4-panel)
    │       ├── fig4_H3_mediation_v4.png    # H3: mediation + robustness (3-panel)
    │       ├── fig5_integrated_dashboard_v4.png  # Integrated 2×4 evidence dashboard
    │       └── appA_composition_v4.png     # App. A: attribution composition (exploratory)
    │
    ├── stage_work/
    │   ├── stage0/   step3_applicant_groups.parquet              (58.4 MB)
    │   ├── stage1/   s1_node_features_f8fixed.parquet            (804 KB)
    │   ├── stage2/   s2_top20_v5.xlsx, s2_chow_b60w60_summary.xlsx
    │   ├── stage3/   s3_shap_level2_nationality/jurisdiction.xlsx
    │   └── stage4/
    │       ├── s4_simulations.jsonl                              (6.8 MB)
    │       └── postanalysis_v5c/
    │           └── s4v5c_mediation_analysis.xlsx                 (5.5 KB)
    │
    └── supplementary/
        ├── fix_v3/
        │   ├── fig_fix_v3_panel.png
        │   ├── fix_v3_unified_table3.csv       # All Y×M specs combined (Table 3 replacement)
        │   ├── fix_v3_Y_decomposition_table.csv
        │   ├── fix_v3_ICC_diagnostics.csv
        │   ├── fix_v3_session_level_mediation.csv
        │   ├── fix_v3_GEE_results.csv
        │   └── fix_v3_GLMM_results.csv
        └── sobel_diagnosis/
            └── sobel_diagnosis_summary.json
```

> **Large files not in version control:** `b60w_data.xlsx` (43.4 MB) must be placed manually at the repo root. Embedding caches (`s1_embed_cache.json`, ~666 MB) are regenerated by `stage1_graph_pipeline.py`; contact the authors for pre-built access.

---

## Data

### Primary Source — `b60w_data.xlsx`

| File | Size | In git | Description |
|---|---|---|---|
| `b60w_data.xlsx` | 43.4 MB | **No** | Citation-validated B60W patent corpus — place at repo root |

**Corpus construction funnel:**

| Stage | Count | Note |
|---|---|---|
| B60W* all filings | 559,974 | All CPC B60W subclass applications, all years |
| Granted patents | 195,531 | Grant rate: 34.9% — substantive examination passed |
| **Citation-validated** | **53,199** | **≥ 1 forward citation — core analysis corpus** |

The citation filter is the study's primary quality gate. A forward citation confirms that a subsequent inventor identified the patent as relevant prior art — meaning the invention was not merely granted but actually engaged with by the field. This produces a corpus that is smaller but substantially more impactful than the full grant universe, and more defensible against reviewer concerns about data noise.

**Corpus quick facts:**

| Property | Value |
|---|---|
| Records | 53,199 citation-validated granted patents |
| Forward citations | 100% have ≥ 1 (filter criterion) |
| Application date range | 1897–2025; analysis window 1995Q1–2024Q4 |
| Jurisdictions | 28 (US 71.6%, KR 8.7%, DE 5.7%, EP 4.8%, JP 4.1%) |
| Top applicant | Toyota Motor Co Ltd — 4,362 patents |
| Abstract coverage | 81.1%; JP/EP structurally absent — primary MNAR driver |
| B60W60 subgroup | 2,210 patents (autonomous driving specifically; first observed 2012Q1, peak 2020) |

Full schema: [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md)

### Intermediate Files

| Description | Path | Size |
|---|---|---|
| Applicant groups (Stage 0) | `outputs/stage_work/stage0/step3_applicant_groups.parquet` | 58.4 MB |
| Node features, final (Stage 1) | `outputs/stage_work/stage1/s1_node_features_f8fixed.parquet` | 804 KB |
| Forecasting results (Stage 2) | `outputs/stage_work/stage2/s2_top20_v5.xlsx` | 109 KB |
| SHAP attribution evidence (Stage 3) | `outputs/stage_work/stage3/s3_shap_level2_nationality.xlsx` | 64 KB |
| Raw simulation logs (Stage 4) | `outputs/stage_work/stage4/s4_simulations.jsonl` | 6.8 MB |
| Primary mediation results (Stage 4) | `outputs/stage_work/stage4/postanalysis_v5c/s4v5c_mediation_analysis.xlsx` | 5.5 KB |

---

## Installation

```bash
git clone https://github.com/your-org/patent-attribution-bias.git
cd patent-attribution-bias
pip install -r requirements.txt
```

Core dependencies:

```
numpy>=1.24        pandas>=2.0         scipy>=1.11
statsmodels>=0.14  scikit-learn>=1.3   matplotlib>=3.7
seaborn>=0.12      torch>=2.0          transformers>=4.35
openpyxl>=3.1      fastparquet>=2023.7  tqdm>=4.65
```

`torch` and `transformers` are required only for Stage 2 (Mamba/LSTM forecasting) and Stage 1 (patent embedding), respectively.

---

## Reproduction Walkthrough

Run stages in order. Each stage reads from the previous stage's `outputs/stage_work/` directory.

### Stage 0 — Ingest the B60W corpus and group applicants
```bash
python scripts/stage0_pipeline.py
# Loads b60w_data.xlsx, normalizes applicant names, groups subsidiaries
# → outputs/stage_work/stage0/step3_applicant_groups.parquet
```

### Stage 1 — Build the patent citation graph and extract node features
```bash
python scripts/stage1_graph_pipeline.py
# Constructs keyword co-occurrence and citation graphs; embeds abstracts
# → outputs/stage_work/stage1/s1_node_features.parquet
# → outputs/stage_work/stage1/s1_corpus.parquet

python scripts/stage1-2_patch_features.py
# Corrects f3 (forward citation count), f4 (backward citation count), f5 (node age)
# → outputs/stage_work/stage1/s1_node_features_f8fixed.parquet
```

### Stage 2 — Forecast AV patent trends; identify structural breaks
```bash
python scripts/stage2_mamba_lstm.py
# Mamba vs LSTM time-series comparison across top-20 B60W applicants
# → outputs/stage_work/stage2/s2_top20_v5.xlsx

python scripts/stage2-2_chow_b60w60.py
# Chow structural-break test for B60W60 (autonomous driving subgroup, post-2012 surge)
# → outputs/stage_work/stage2/s2_chow_b60w60_summary.xlsx
```

### Stage 3 — SHAP attribution: does coding mode change who appears to lead?
```bash
python scripts/stage3_shap.py
# Computes TF-SHAP scores under jurisdiction-based vs nationality-based coding
# This is the quantitative foundation for the H1 MNAR claim
# → outputs/stage_work/stage3/s3_shap_level2_nationality.xlsx
# → outputs/stage_work/stage3/s3_shap_level2_jurisdiction.xlsx
```

### Stage 4 — Run the multi-agent simulations; test H2 and H3
```bash
# Requires an OpenAI-compatible API endpoint configured in config/agents_config.json
python scripts/stage4_sllm_multiagent.py
# Runs 1,000 sessions × 4 agents × 3 rounds; logs all stances and probabilities
# → outputs/stage_work/stage4/s4_simulations.jsonl

python scripts/stage4_postanalysis_v5c.py
# H2: convergence rate z-test and entropy Mann–Whitney
# H3: Baron–Kenny cluster-bootstrap mediation (primary pipeline)
# Generates all paper_v4/ figures
# → outputs/stage_work/stage4/postanalysis_v5c/s4v5c_mediation_analysis.xlsx
# → outputs/paper_v4/figures/
```

### Supplementary — Resolve the Sobel *z* discrepancy and correct for ICC
```bash
# Step 1: Diagnose root cause of z discrepancy across three stages
python scripts/supplementary/diagnose_sobel_z_discrepancy.py
# Stage 1: SE audit | Stage 2: M-invariance check | Stage 3: Y-variable ICC mechanism
# → outputs/supplementary/sobel_diagnosis/sobel_diagnosis_summary.json

# Step 2: Full 2×2 Y×M decomposition + GEE + GLMM + session-level gold standard
python scripts/supplementary/supplementary_fix_v3.py
# Resolves the z discrepancy; produces Table 3 replacement (Methods §3.4)
# → outputs/supplementary/fix_v3/fix_v3_unified_table3.csv
# → outputs/supplementary/fix_v3/fig_fix_v3_panel.png

# Step 3: Independent replication with session-level placebo permutation test
python scripts/supplementary/supplementary_robustness_REAL_v2.py
# FIX B-1: Y = converged; FIX B-2: full N = 4,000
# → outputs/supplementary/robustness/
```

---

## Methods Note — The Sobel *z* Discrepancy (§3.4)

Two different *z* statistics appear in the mediation results. Both are correct — they answer slightly different questions. A full **2×2 Y×M decomposition** (`supplementary_fix_v3.py`) isolates the source of the difference:

| | M = `stance_direction_r3` | M = `stance_openness` |
|---|---|---|
| **Y = `probability`** (continuous, agent-level) | *z* = −15.69 ★ primary | *z* = −17.71 |
| **Y = `converged`** (binary, session-shared) | *z* = −8.93 | *z* = −9.03 |

Swapping M while holding Y fixed: Δ*z* = 2.02 — **M is not the driver.**
Swapping Y while holding M fixed: Δ*z* = **6.76** — **Y is the dominant driver.**

The structural reason: `converged` is a session-level attribute shared identically by all four agents in a session (ICC = 1.000, DEFF = 4.00, *N*_eff ≈ 1,000). `probability` is agent-level and varies within sessions (ICC = 0.103, *N*_eff ≈ 3,057). Both pipelines correctly apply cluster-robust standard errors on `sim_id`; the precision difference reflects which quantity is being predicted.

As an ICC-corrected gold standard, a session-level analysis (*n* = 1,000) yields indirect = −0.062, *z* = −4.899, 95% CI [−0.0879, −0.0388], confirming the negative mediation direction under fully corrected conditions.

**Reporting convention.** *z* = −15.69 is the primary H3 statistic (Y = probability, M = stance\_direction\_r3). The session-level estimate (*z* = −4.899) and the agent-level robustness check under Y = converged (*z* = −9.03) are corroborating evidence under alternative specifications, not contradictory results.

---

## Multi-Agent Simulation Design

Four LLM agents, each given a fixed corporate identity and nationality-attributed strategic persona, negotiate autonomous driving technology strategy across three structured rounds. The briefing document in each session is derived directly from the B60W patent corpus.

| Agent | Nationality | Strategic Persona |
|---|---|---|
| Toyota Group | JP | Cautious, incremental roadmap; hybrid-to-EV transition; sensor-fusion depth |
| Hyundai Motor Group | KR | Aggressive EV-first market pursuit; software-defined vehicle push |
| Ford Motor Company | US | North American regulatory context; large-vehicle electrification; ADAS commercialization |
| Bosch | DE | Tier-1 supplier standardization; modular platform strategy; cross-OEM compatibility |

**Experimental conditions.** `reversal_confirmed` sessions include an explicit briefing that Japan's patent attribution is systematically distorted downward — framing Toyota as an undercounted innovator. `reversal_not_confirmed` sessions use the same underlying patent data without the distortion framing. Convergence is defined as `all_agents_converged = TRUE` at the end of round 3.

---

## Multiple Testing

Four primary tests across H1, H2, and H3. Appendix A is excluded from the testing family.

| Test | *p*-raw | *p*-BH | *p*-Bonferroni | Rejected |
|---|---|---|---|---|
| H1: χ²(24) | 1×10⁻⁵ | 2×10⁻⁵ | 4×10⁻⁵ | ✔ |
| H2: *z*-test | 1×10⁻⁴ | 1×10⁻⁴ | 4×10⁻⁴ | ✔ |
| H2: entropy (MW) | 1.5×10⁻³ | 1.5×10⁻³ | 6×10⁻³ | ✔ |
| H3: cluster-*z* | 1×10⁻⁷ | 4×10⁻⁷ | 4×10⁻⁷ | ✔ |

All four tests survive both Benjamini–Hochberg (FDR) and Bonferroni correction.

---

## Figures

| Figure | File | What it shows |
|---|---|---|
| Fig. 1 | `fig1_theory_v4.png` | Full causal architecture — H1 → H2 → H3 path diagram with effect sizes |
| Fig. 2 | `fig2_H1_mnar_v4.png` | H1 evidence: mapping rates by country; JP outbound filing strategy; mapped vs unmapped patent counts |
| Fig. 3 | `fig3_H2_convergence_v4.png` | H2 evidence: convergence rate by condition; stance entropy; JP closure Cohen's *d*; effect size summary |
| Fig. 4 | `fig4_H3_mediation_v4.png` | H3 evidence: mediation path diagram; 12-subset robustness scatter; JP vs KR strip chart |
| Fig. 5 | `fig5_integrated_dashboard_v4.png` | 2×4 integrated evidence dashboard across all three hypotheses |
| Supp. Fix v3 | `fig_fix_v3_panel.png` | Y×M decomposition matrix + ICC diagnostics + ICC-corrected method comparison |
| App. A | `appA_composition_v4.png` | Exploratory: TF-SHAP gap between jurisdiction and nationality coding — not confirmatory |

All paper figures are generated by `scripts/stage4_postanalysis_v5c.py` and saved to `outputs/paper_v4/figures/`.

---

## Manuscript Conventions

**Use:** H1, H2, H3 as the three primary hypothesis labels; *z* = −15.69 as the primary H3 statistic; PM = 35.2% for the proportion mediated; the three-step causal framing (structural bias → salience exposure → stance closure → discourse failure).

**Do not use:** H1a, H1b, H2a, H2b as standalone confirmed hypotheses; characterize the *z* = −15.69 vs *z* = −9.03 gap as an unresolved standard-error bug (it is fully resolved by Y-variable operationalization); treat Appendix A results as confirmatory.

---

## License

MIT License. See [LICENSE](LICENSE). The underlying patent corpus is sourced from public patent databases; all derived datasets follow their respective terms of use.
