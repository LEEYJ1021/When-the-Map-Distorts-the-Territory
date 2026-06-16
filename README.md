# When the Map Distorts the Territory
### Structural Attribution Bias in Cross-National Patent Data and Its Consequences for Multi-Agent Strategic Discourse

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v4.1-green.svg)]()

---

## Overview

This repository contains the full reproducibility package for a study demonstrating how structural measurement bias in cross-national patent databases propagates through an autonomous multi-agent strategic discourse system, ultimately suppressing inter-agent convergence. The study spans a three-hypothesis causal chain:

**H1 → H2 → H3: Structural Bias → Salience Exposure → Stance Closure → Discourse Failure**

The empirical foundation is the CPC B60W autonomous driving patent corpus (1995–2024, *N* = 47,281), combined with 1,000 simulated multi-agent strategy sessions involving four nationality-attributed corporate agents (Toyota/JP, Hyundai/KR, Ford/US, Bosch/DE).

---

## Causal Architecture

```
[H1: MNAR Bias]  ──MNAR exposure──►  [H2: Convergence Suppression]
  JP 44.8% vs                           Δ = 10.3pp (17.6% vs 27.9%)
  DE 95.4%                              z = −3.877, p < .001***
  χ²(24)=3134.76                        OR = 0.552, NNT = 9.7
       │
       │ Salience → H3
       ▼
  [M: JP Stance Closure]  ──b = +0.038***──►  [Outcome: Discourse Failure]
   mean openness = −0.928
   a = −0.754***
       │
       └── indirect = −0.029***  (PM = 35.2%)
           cluster-robust z = −15.69
           95% CI [−0.0327, −0.0254]
```

---

## Key Results

| Hypothesis | Status | Primary Statistic |
|---|---|---|
| **H1** Structural MNAR Missingness | ✔ CONFIRMED | χ²(24) = 3134.76, *p* < .0001, *V* = 0.1287 |
| **H2** Convergence Suppression | ✔ STRONGLY SUPPORTED | Δ = 10.3 pp, *z* = −3.877, *p* = .0001, OR = 0.552 |
| **H3** Partial Mediation (JP stance closure) | ✔ STRONGLY SUPPORTED | PM = 35.2%, cluster-*z* = −15.69***, 95% CI [−0.0327, −0.0254] |
| **Appendix A** Attribution Composition Patterns | ⚠ EXPLORATORY | +273% amplification; *n*_JP_jur = 44; CI = [−93%, +2922%] — not confirmatory |

All four primary tests (H1 χ², H2 z-test, H2 entropy MW, H3 cluster-*z*) survive both Benjamini–Hochberg and Bonferroni correction. Appendix A is excluded from the multiple-testing family.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── ── Pipeline Scripts ──────────────────────────────────────────────
├── stage0_pipeline.py              # Raw patent data ingestion & applicant grouping
├── stage1_graph_pipeline.py        # Keyword extraction, graph construction, node features
├── stage1-2_patch_features.py      # Feature patch: f3/f4 citation cols, f5 node-age fix
├── stage2_mamba_lstm.py            # Mamba vs LSTM time-series forecasting (H1a/H1b/H1c)
├── stage2-2_chow_b60w60.py         # Chow structural-break tests (B60W60 corpus)
├── stage3_shap.py                  # TF-SHAP attribution analysis (jurisdiction vs nationality)
├── stage4_sllm_multiagent.py       # Multi-agent simulation engine (1,000 sessions × 4 agents)
├── stage4_postanalysis_v5c.py      # Primary post-analysis pipeline (H2, H3 mediation)
│
├── ── Supplementary / Diagnostic Scripts ───────────────────────────
├── supplementary_fix_v3.py         # ★ 2×2 Y×M decomposition; ICC-corrected mediation
│                                   #   (resolves Sobel z discrepancy; Methods §3.4)
├── supplementary_robustness_REAL_v2.py  # Independent replication + session-level placebo
│                                        # (FIX B-1: Y=converged; FIX B-2: N=4,000)
├── diagnose_sobel_z_discrepancy.py # Three-stage root-cause diagnosis of z discrepancy
│                                   # Stage 1: SE audit | Stage 2: M-invariance |
│                                   # Stage 3: Y-variable ICC mechanism
│
├── ── Configuration ─────────────────────────────────────────────────
├── agents_config.json              # Agent personas (JP/KR/US/DE) for simulation
│
├── ── Paper Outputs (v4) ────────────────────────────────────────────
├── paper_output_v4/
│   ├── master_v4.json                          # Full metadata & manuscript rules
│   ├── methods_sobel_reconciliation_EN.txt     # Methods §3.4 full English narrative
│   └── figures/
│       ├── fig1_theory_v4.png                  # Causal theory model (H1→H2→H3)
│       ├── fig2_H1_mnar_v4.png                 # H1 MNAR evidence (3-panel)
│       ├── fig3_H2_convergence_v4.png          # H2 convergence suppression (4-panel)
│       ├── fig4_H3_mediation_v4.png            # H3 mediation + robustness (3-panel)
│       ├── fig5_integrated_dashboard_v4.png    # Integrated 2×4 evidence dashboard
│       └── appA_composition_v4.png             # Appendix A exploratory (3-panel)
│
├── ── Stage-Specific Supplementary Outputs ─────────────────────────
└── stage4_work/fix_v3_output/
    ├── fig_fix_v3_panel.png                    # 4-panel fix summary figure
    ├── fix_v3_unified_table3.csv               # Table 3 replacement (all Y×M specs)
    ├── fix_v3_Y_decomposition_table.csv        # 2×2 Y×M z-matrix
    ├── fix_v3_ICC_diagnostics.csv              # ICC / DEFF per variable
    ├── fix_v3_session_level_mediation.csv      # Session-level gold standard
    ├── fix_v3_GEE_results.csv                  # GEE population-average estimates
    ├── fix_v3_GLMM_results.csv                 # GLMM random-intercept estimates
    └── fix_v3_reviewer_response.txt            # Reviewer response text (EN)
```

---

## Data Requirements

| File | Size | Description |
|---|---|---|
| `b60w_data.xlsx` | 43.4 MB | Source patent corpus (CPC B60W, 1995–2024) — **not tracked in git** |
| `stage4_work/s4_simulations.jsonl` | 6.8 MB | Raw simulation logs (1,000 sessions × 4 agents) |
| `stage0_work/step3_applicant_groups.parquet` | 58.4 MB | Stage 0 final output |
| `stage1_work/s1_node_features_f8fixed.parquet` | 804 KB | Stage 1 final output (f8-patched) |
| `stage2_work/s2_top20_v5.xlsx` | 109 KB | Stage 2 core results |
| `stage3_work/s3_shap_level2_nationality.xlsx` | 64 KB | SHAP H1–H3 evidence |
| `stage4_work/postanalysis_v5c/s4v5c_mediation_analysis.xlsx` | 5.5 KB | Primary mediation pipeline |

> The raw patent corpus (`b60w_data.xlsx`) and large embedding caches (`s1_embed_cache.json`, ~666 MB) are excluded from version control due to size constraints. Contact the authors for data access.

---

## Installation

```bash
git clone https://github.com/your-org/patent-attribution-bias.git
cd patent-attribution-bias
pip install -r requirements.txt
```

**Core dependencies:**

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
statsmodels>=0.14
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
torch>=2.0          # for Mamba / LSTM (stage2)
transformers>=4.35  # for embedding (stage1)
openpyxl>=3.1
fastparquet>=2023.7
tqdm>=4.65
```

---

## Reproduction Walkthrough

Execute stages in order. Each stage reads from the previous stage's output directory.

### Stage 0 — Data Ingestion & Applicant Grouping
```bash
python stage0_pipeline.py
# Output: stage0_work/step3_applicant_groups.parquet
```

### Stage 1 — Graph Construction & Node Features
```bash
python stage1_graph_pipeline.py
# Output: stage1_work/s1_node_features.parquet, s1_corpus.parquet

python stage1-2_patch_features.py
# Fixes f3 (forward citation), f4 (backward citation), f5 (node age)
# Output: stage1_work/s1_node_features_fixed.parquet
```

### Stage 2 — Mamba vs LSTM Forecasting
```bash
python stage2_mamba_lstm.py
# Output: stage2_work/s2_top20_v5.xlsx, checkpoints/

python stage2-2_chow_b60w60.py
# Chow structural-break tests
# Output: stage2_work/s2_chow_b60w60_summary.xlsx
```

### Stage 3 — SHAP Attribution Analysis
```bash
python stage3_shap.py
# Output: stage3_work/s3_shap_level2_nationality.xlsx
#         stage3_work/s3_shap_level2_jurisdiction.xlsx
```

### Stage 4 — Multi-Agent Simulation & Post-Analysis
```bash
# Run 1,000 simulations (requires OpenAI-compatible API endpoint)
python stage4_sllm_multiagent.py

# Primary post-analysis: H2 convergence, H3 mediation
python stage4_postanalysis_v5c.py
# Output: stage4_work/postanalysis_v5c/s4v5c_mediation_analysis.xlsx
```

### Supplementary — Sobel z Reconciliation & ICC Fix
```bash
# Step 1: Three-stage root-cause diagnosis
python diagnose_sobel_z_discrepancy.py
# Output: stage4_work/sobel_diagnosis/sobel_diagnosis_summary.json

# Step 2: Full 2×2 Y×M decomposition + ICC-corrected methods
python supplementary_fix_v3.py
# Output: stage4_work/fix_v3_output/fix_v3_unified_table3.csv
#         stage4_work/fix_v3_output/fig_fix_v3_panel.png

# Step 3: Independent replication
python supplementary_robustness_REAL_v2.py
# Output: stage4_work/supplementary_robustness_output/
```

### Figure & Manuscript Generation
```bash
python stage4_postanalysis_v5c.py  # also generates all paper_output_v4/ figures
```

---

## Methods §3.4 — Resolved Sobel *z* Discrepancy

An apparent discrepancy between two reported Sobel *z* statistics is fully resolved by a **2×2 Y×M decomposition** (`supplementary_fix_v3.py`):

| Y specification | M = stance_direction_r3 | M = stance_openness |
|---|---|---|
| **Y = probability** (continuous, agent-level) | z = −15.69 ★ | z = −17.71 |
| **Y = converged** (binary, session-shared) | z = −8.93 | z = −9.03 |

- **M-swap** (same Y = probability): Δ*z* = 2.02 — M is not the driver
- **Y-swap** (same M = stance_direction): Δ*z* = **6.76** — **Y is the dominant driver**

The mechanism is straightforward: `converged` is a session-level attribute (ICC = 1.000, DEFF = 4.00, *N*_eff ≈ 1,000), whereas `probability` varies across agents within a session (ICC = 0.103, *N*_eff ≈ 3,057). Both pipelines apply cluster-robust standard errors on `sim_id` correctly; the precision difference reflects which quantity is being predicted.

As an ICC = 1.000 gold standard, a session-level aggregation analysis (*n* = 1,000 sessions) yields: indirect = −0.062, *z* = −4.899, 95% CI [−0.0879, −0.0388], confirming the negative mediation direction (*p* < .001) under ICC-corrected conditions.

**Reporting convention:** *z* = −15.69 is the primary H3 statistic. The session-level estimate (*z* = −4.899) and the robustness check under Y = converged (*z* = −9.03) are corroborating evidence, not conflicting results.

---

## Figures

| Figure | File | Description |
|---|---|---|
| Fig. 1 | `fig1_theory_v4.png` | Causal architecture: H1 → H2 → H3 theory model |
| Fig. 2 | `fig2_H1_mnar_v4.png` | H1 structural MNAR evidence (mapping rates, filing strategy, mapped vs unmapped) |
| Fig. 3 | `fig3_H2_convergence_v4.png` | H2 convergence suppression (rate, entropy, JP stance closure, effect sizes) |
| Fig. 4 | `fig4_H3_mediation_v4.png` | H3 mediation path diagram, 12-subset robustness, JP vs KR strip chart |
| Fig. 5 | `fig5_integrated_dashboard_v4.png` | Integrated 2×4 evidence dashboard |
| Supp. Fix v3 | `fig_fix_v3_panel.png` | Y×M decomposition matrix + ICC diagnostics (4-panel) |
| App. A | `appA_composition_v4.png` | Attribution composition patterns — exploratory only |

---

## Multi-Agent Simulation Design

Simulations use four nationality-attributed corporate agents configured in `agents_config.json`:

| Agent | Nationality | Persona Focus |
|---|---|---|
| Toyota Group | JP | Cautious incremental technology roadmap |
| Hyundai Motor Group | KR | Aggressive market-pursuit strategy |
| Ford Motor Company | US | North American regulatory environment & large-vehicle electrification |
| Bosch | DE | Tier-1 supplier standardization and modularity |

Each session proceeds over three structured rounds. The `reversal_confirmed` condition exposes agents to attribution-distortion-salient patent data; `reversal_not_confirmed` serves as baseline. Convergence is defined as `all_agents_converged = TRUE` for a session.

---

## Multiple Testing

Four primary tests; Appendix A excluded from the family.

| Test | *p*-raw | *p*-BH | *p*-Bonferroni | Rejected |
|---|---|---|---|---|
| H1: χ²(24) | 1×10⁻⁵ | 2×10⁻⁵ | 4×10⁻⁵ | ✔ |
| H2: z-test | 1×10⁻⁴ | 1×10⁻⁴ | 4×10⁻⁴ | ✔ |
| H2: entropy (MW) | 1.5×10⁻³ | 1.5×10⁻³ | 6×10⁻³ | ✔ |
| H3: cluster-z | 1×10⁻⁷ | 4×10⁻⁷ | 4×10⁻⁷ | ✔ |

All four tests survive both corrections.

---

## Manuscript Rules

The following conventions are enforced throughout the manuscript and all output scripts:

**Use:** H1, H2, H3; *z* = −15.69; PM = 35.2%; self-consistent mediation chain.

**Do not use:** H1a, H1b, H2a, H2b as standalone confirmed hypotheses; describe the *z* = −16.20 vs *z* = −9.19 gap as an unresolved SE bug (it is resolved); claim Appendix A as confirmatory.

---

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details. The underlying patent corpus is sourced from public patent databases; all derived datasets follow their respective terms of use.
