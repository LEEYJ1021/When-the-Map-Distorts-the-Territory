# When the Map Distorts the Territory

### Structural Attribution Bias in Cross-National Patent Data and Its Consequences for Multi-Agent Strategic Discourse

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-v4.1-green.svg)]()

> **Before running any script:** place `b60w_data.xlsx` at the repository root.
> This file is not tracked in git. See [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md) for schema and loading instructions.

---

## The Story in One Paragraph

Patent databases are widely used as proxies for innovation. But what happens when the database itself is systematically broken — not randomly, but in a way that consistently underrepresents one country's inventors? This study starts with that question and follows it to a surprising destination: **a broken database doesn't just distort statistics; it distorts the judgment of AI agents negotiating strategy in real time.**

We show that Japan's inventors are mapped to their true nationality at only **44.8%**, compared to **95.4%** for Germany — a gap that is not random noise but a structural artifact of how Japanese firms route patents through US jurisdictions. We then ask: if AI agents are briefed with this distorted data and told "Japanese innovation is being systematically misread," does that awareness change how they negotiate? The answer is yes, and not for the better. Agents exposed to the attribution-distortion framing **converged on a shared strategy 10 percentage points less often** than those who were not. The mechanism, traced through mediation analysis, is that the Japanese agent adopts a **closed, defensive stance** — and that closure propagates through the conversation, blocking agreement.

---

## The Three-Hypothesis Chain

The study is organized as a single causal argument across three hypotheses. Each one is a necessary step in the chain.

### H1 — The Database Is Structurally Broken

> *"Missingness in patent–nationality mapping is not random; it is driven by where patents are filed, not by who invented them."*

Japanese companies file **93.1% of their patents outside Japan**, mostly routing through the US. Because patent databases primarily map nationality from the filing jurisdiction, this creates a systematic gap: Toyota's US-filed patents often appear as "US patents" rather than "JP patents." The result is that Japan's true innovation share in autonomous driving is dramatically undercounted.

This is not a missing-at-random (MAR) problem — it is **Missing Not at Random (MNAR)**, meaning the probability of missingness is directly tied to the substantive variable being measured. A chi-square test across 28 jurisdictions confirms the non-randomness with overwhelming evidence.

**Key statistic:** χ²(24) = 3,134.76, *p* < .0001, Cramér's *V* = 0.1287 — JP maps at 44.8% vs DE at 95.4%, a 50.6 percentage-point structural gap.

### H2 — Knowing the Database Is Broken Makes AI Agents Negotiate Worse

> *"When agents are made aware of attribution distortion, they converge on a shared strategy significantly less often."*

We ran 1,000 simulated three-round strategy sessions with four LLM-powered agents representing Toyota (JP), Hyundai (KR), Ford (US), and Bosch (DE). Half the sessions included a briefing that the patent attribution data was distorted against Japan ("reversal-confirmed"); half did not ("reversal-not-confirmed"). We then measured whether all four agents converged on a shared strategic position by the end of round 3.

The distortion-salient condition produced a **convergence rate of 17.6%**, compared to **27.9%** in the baseline — a 10.3 percentage-point drop. This is not a small effect: the Number Needed to Treat (NNT = 9.7) means that for roughly every 10 strategy sessions that include the distortion framing, one convergence that would otherwise have occurred is lost.

**Key statistic:** *z* = −3.877, *p* = .0001, OR = 0.552, NNT = 9.7.

### H3 — The Japanese Agent's Defensive Closure Explains Why

> *"The Japanese agent is not simply outvoted — it withdraws. And that withdrawal mediates 35% of the total convergence failure."*

To understand the mechanism behind H2, we look inside the sessions. The Japanese agent in distortion-salient conditions shows a dramatically lower **stance openness** score (mean = −0.928) compared to all other agents (all positive). This closure — a pattern of decreasing expressed probability and contracting strategic flexibility across rounds — is not just a symptom. It is a **mediator**: it statistically carries 35.2% of the pathway from distortion exposure to convergence failure.

The mediation is partial (not full), meaning the distortion framing also has direct effects on convergence beyond what runs through the Japanese agent's stance. But the stance-closure mechanism is robust across 12 subsets of the data and survives ICC correction.

**Key statistic:** PM = 35.2%, cluster-robust *z* = −15.69, 95% CI [−0.0327, −0.0254].

---

## Results at a Glance

| Hypothesis | Claim | Verdict | Primary Statistic |
|---|---|---|---|
| **H1** | Patent missingness is MNAR, not random | ✔ CONFIRMED | χ²(24) = 3134.76, *p* < .0001, *V* = 0.1287 |
| **H2** | Attribution distortion suppresses AI convergence | ✔ STRONGLY SUPPORTED | Δ = 10.3 pp, *z* = −3.877, OR = 0.552 |
| **H3** | JP stance closure partially mediates failure | ✔ STRONGLY SUPPORTED | PM = 35.2%, cluster-*z* = −15.69 |
| **App. A** | Jurisdiction vs nationality coding amplifies gap | ⚠ EXPLORATORY | +273%; CI = [−93%, +2922%] — not confirmatory |

All four primary tests survive both Benjamini–Hochberg and Bonferroni correction. Appendix A is excluded from the multiple-testing family.

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

## The Empirical Foundation

Everything in this study rests on two datasets used together.

**The patent corpus.** We use the CPC B60W subclass — the international patent classification for autonomous driving and advanced vehicle control — covering 47,281 granted patents from 1995Q1 to 2024Q4. The corpus comes from `b60w_data.xlsx` (53,199 records including pre-analysis-window patents, 37 columns). It is the source for H1 and for the SHAP attribution analysis that feeds the agent briefings.

**The simulation data.** We ran 1,000 three-round strategy sessions using an LLM-powered multi-agent framework (`stage4_sllm_multiagent.py`). Each session involved four agents with fixed nationality personas, a shared briefing document derived from the patent corpus, and a structured negotiation protocol. Sessions were randomly assigned to the distortion-salient or baseline condition. This produces 4,000 agent-round observations used in H2 and H3.

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
│   ├── stage0_pipeline.py              # Stage 0: Raw patent ingestion & applicant grouping
│   ├── stage1_graph_pipeline.py        # Stage 1: Keyword extraction, graph construction
│   ├── stage1-2_patch_features.py      # Stage 1 patch: citation columns, node-age fix
│   ├── stage2_mamba_lstm.py            # Stage 2: Mamba vs LSTM time-series forecasting
│   ├── stage2-2_chow_b60w60.py         # Stage 2: Chow structural-break tests
│   ├── stage3_shap.py                  # Stage 3: TF-SHAP attribution analysis (H1 evidence)
│   ├── stage4_sllm_multiagent.py       # Stage 4: Multi-agent simulation (1,000 sessions)
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
    │   ├── methods_sobel_reconciliation_EN.txt
    │   ├── methods_sobel_reconciliation_KR.txt
    │   └── figures/
    │       ├── fig1_theory_v4.png          # Causal theory model (H1→H2→H3)
    │       ├── fig2_H1_mnar_v4.png         # H1: MNAR evidence (3-panel)
    │       ├── fig3_H2_convergence_v4.png  # H2: convergence suppression (4-panel)
    │       ├── fig4_H3_mediation_v4.png    # H3: mediation + robustness (3-panel)
    │       ├── fig5_integrated_dashboard_v4.png
    │       └── appA_composition_v4.png
    │
    ├── stage_work/
    │   ├── stage0/   step3_applicant_groups.parquet         (58.4 MB)
    │   ├── stage1/   s1_node_features_f8fixed.parquet       (804 KB)
    │   ├── stage2/   s2_top20_v5.xlsx, s2_chow_b60w60_summary.xlsx
    │   ├── stage3/   s3_shap_level2_nationality/jurisdiction.xlsx
    │   └── stage4/
    │       ├── s4_simulations.jsonl                         (6.8 MB)
    │       └── postanalysis_v5c/
    │           └── s4v5c_mediation_analysis.xlsx            (5.5 KB)
    │
    └── supplementary/
        ├── fix_v3/
        │   ├── fig_fix_v3_panel.png
        │   ├── fix_v3_unified_table3.csv       # All Y×M specs combined
        │   ├── fix_v3_Y_decomposition_table.csv
        │   ├── fix_v3_ICC_diagnostics.csv
        │   ├── fix_v3_session_level_mediation.csv
        │   ├── fix_v3_GEE_results.csv
        │   ├── fix_v3_GLMM_results.csv
        │   └── fix_v3_reviewer_response.txt
        └── sobel_diagnosis/
            └── sobel_diagnosis_summary.json
```

> **Large files not in version control:** `b60w_data.xlsx` (43.4 MB) must be placed manually at the repo root. Embedding caches (`s1_embed_cache.json`, ~666 MB) are regenerated by `stage1_graph_pipeline.py`; contact the authors for pre-built access.

---

## Data

### Primary Source

| File | Size | In git | Description |
|---|---|---|---|
| `b60w_data.xlsx` | 43.4 MB | **No** | CPC B60W patent corpus — place at repo root |

| Property | Value |
|---|---|
| Records | 53,199 granted patents |
| Forward citations ≥ 1 | 100% |
| Analysis window | 1995Q1–2024Q4 |
| Jurisdictions | 28 (US 71.6%, KR 8.7%, DE 5.7%, EP 4.8%, JP 4.1%) |
| Top applicant | Toyota Motor Co Ltd (4,362 patents) |
| Abstract coverage | 81.1% — JP/EP structurally absent, the primary MNAR driver |
| B60W60 subgroup | 2,210 patents (first observed 2012Q1; peak 2020) |

Full schema: [`data/DATA_DESCRIPTION.md`](data/DATA_DESCRIPTION.md)

### Intermediate Files

| File | Path | Size |
|---|---|---|
| Applicant groups | `outputs/stage_work/stage0/step3_applicant_groups.parquet` | 58.4 MB |
| Node features (final) | `outputs/stage_work/stage1/s1_node_features_f8fixed.parquet` | 804 KB |
| Stage 2 core results | `outputs/stage_work/stage2/s2_top20_v5.xlsx` | 109 KB |
| SHAP evidence | `outputs/stage_work/stage3/s3_shap_level2_nationality.xlsx` | 64 KB |
| Simulation logs | `outputs/stage_work/stage4/s4_simulations.jsonl` | 6.8 MB |
| Primary mediation | `outputs/stage_work/stage4/postanalysis_v5c/s4v5c_mediation_analysis.xlsx` | 5.5 KB |

---

## Installation

```bash
git clone https://github.com/your-org/patent-attribution-bias.git
cd patent-attribution-bias
pip install -r requirements.txt
```

Core dependencies:

```
numpy>=1.24       pandas>=2.0        scipy>=1.11
statsmodels>=0.14 scikit-learn>=1.3  matplotlib>=3.7
seaborn>=0.12     torch>=2.0         transformers>=4.35
openpyxl>=3.1     fastparquet>=2023.7 tqdm>=4.65
```

`torch` and `transformers` are required only for Stage 2 (Mamba/LSTM) and Stage 1 (embedding), respectively.

---

## Reproduction Walkthrough

Run stages in order. Each stage reads from the previous stage's `outputs/stage_work/` directory.

### Stage 0 — Ingest patents and group applicants
```bash
python scripts/stage0_pipeline.py
# → outputs/stage_work/stage0/step3_applicant_groups.parquet
```

### Stage 1 — Build the citation graph and extract node features
```bash
python scripts/stage1_graph_pipeline.py
# → outputs/stage_work/stage1/s1_node_features.parquet

python scripts/stage1-2_patch_features.py
# Corrects f3 (forward citations), f4 (backward citations), f5 (node age)
# → outputs/stage_work/stage1/s1_node_features_f8fixed.parquet
```

### Stage 2 — Forecast patent trends; test for structural breaks
```bash
python scripts/stage2_mamba_lstm.py
# Mamba vs LSTM comparison across top-20 applicants
# → outputs/stage_work/stage2/s2_top20_v5.xlsx

python scripts/stage2-2_chow_b60w60.py
# Chow test for B60W60 structural break (autonomous driving subgroup)
# → outputs/stage_work/stage2/s2_chow_b60w60_summary.xlsx
```

### Stage 3 — SHAP attribution: jurisdiction coding vs nationality coding
```bash
python scripts/stage3_shap.py
# Produces H1 evidence: JP SHAP scores diverge by coding mode
# → outputs/stage_work/stage3/s3_shap_level2_nationality.xlsx
# → outputs/stage_work/stage3/s3_shap_level2_jurisdiction.xlsx
```

### Stage 4 — Run simulations; test H2 and H3
```bash
# Requires an OpenAI-compatible API endpoint
python scripts/stage4_sllm_multiagent.py
# → outputs/stage_work/stage4/s4_simulations.jsonl  (1,000 sessions × 4 agents)

python scripts/stage4_postanalysis_v5c.py
# Runs H2 z-test, H3 mediation, generates all paper figures
# → outputs/stage_work/stage4/postanalysis_v5c/s4v5c_mediation_analysis.xlsx
# → outputs/paper_v4/figures/
```

### Supplementary — Resolve the Sobel *z* discrepancy and correct for ICC
```bash
# Step 1: Diagnose the z discrepancy
python scripts/supplementary/diagnose_sobel_z_discrepancy.py
# → outputs/supplementary/sobel_diagnosis/sobel_diagnosis_summary.json

# Step 2: Full 2×2 Y×M decomposition + ICC-corrected mediation (Methods §3.4)
python scripts/supplementary/supplementary_fix_v3.py
# → outputs/supplementary/fix_v3/fix_v3_unified_table3.csv
# → outputs/supplementary/fix_v3/fig_fix_v3_panel.png

# Step 3: Independent replication with session-level placebo test
python scripts/supplementary/supplementary_robustness_REAL_v2.py
# → outputs/supplementary/robustness/
```

---

## Methods Note — The Sobel *z* Discrepancy (§3.4)

Two different *z* statistics appear in the mediation results and both are correct. The primary pipeline reports *z* = −15.69; an independent robustness check reports *z* = −9.03. These are not contradictory. A full **2×2 Y×M decomposition** (`supplementary_fix_v3.py`) shows why:

| | M = `stance_direction_r3` | M = `stance_openness` |
|---|---|---|
| **Y = `probability`** (continuous, agent-level) | *z* = −15.69 ★ | *z* = −17.71 |
| **Y = `converged`** (binary, session-shared) | *z* = −8.93 | *z* = −9.03 |

Swapping M while holding Y fixed produces Δ*z* = 2.02. Swapping Y while holding M fixed produces Δ*z* = **6.76**. The outcome variable — not the mediator — is the driver of the gap.

The reason is structural: `converged` is a session-level attribute shared by all four agents in a session (ICC = 1.000, DEFF = 4.00, *N*_eff ≈ 1,000). `probability` varies at the agent level (ICC = 0.103, *N*_eff ≈ 3,057). Both pipelines cluster on `sim_id` correctly; the precision difference reflects what is being predicted.

As an ICC-corrected gold standard, a session-level analysis (*n* = 1,000) yields: indirect = −0.062, *z* = −4.899, 95% CI [−0.0879, −0.0388] — same sign, same significance, different scale.

**Reporting convention:** *z* = −15.69 is the primary H3 statistic. *z* = −4.899 (session-level) and *z* = −9.03 (Y = converged, agent-level) are corroborating robustness checks under different specifications.

---

## Multi-Agent Simulation Design

Four LLM agents negotiate autonomous driving strategy across three structured rounds. Each agent is given a nationality-attributed corporate persona and a briefing document derived from the patent corpus.

| Agent | Nationality | Persona |
|---|---|---|
| Toyota Group | JP | Cautious, incremental technology roadmap |
| Hyundai Motor Group | KR | Aggressive market-pursuit strategy |
| Ford Motor Company | US | North American regulatory environment; large-vehicle electrification |
| Bosch | DE | Tier-1 supplier standardization and modularity |

**Conditions.** The `reversal_confirmed` condition includes explicit framing that JP patent attribution is systematically distorted. The `reversal_not_confirmed` condition is the baseline (no such framing). **Convergence** is defined as `all_agents_converged = TRUE` at the end of round 3.

---

## Multiple Testing

Four primary tests; Appendix A is excluded from the family.

| Test | *p*-raw | *p*-BH | *p*-Bonferroni | Rejected |
|---|---|---|---|---|
| H1: χ²(24) | 1×10⁻⁵ | 2×10⁻⁵ | 4×10⁻⁵ | ✔ |
| H2: *z*-test | 1×10⁻⁴ | 1×10⁻⁴ | 4×10⁻⁴ | ✔ |
| H2: entropy (MW) | 1.5×10⁻³ | 1.5×10⁻³ | 6×10⁻³ | ✔ |
| H3: cluster-*z* | 1×10⁻⁷ | 4×10⁻⁷ | 4×10⁻⁷ | ✔ |

All four tests survive both corrections.

---

## Figures

| Figure | File | What it shows |
|---|---|---|
| Fig. 1 | `fig1_theory_v4.png` | Full causal architecture — H1 → H2 → H3 path diagram |
| Fig. 2 | `fig2_H1_mnar_v4.png` | H1 evidence: mapping rates by country, filing strategy, mapped vs unmapped counts |
| Fig. 3 | `fig3_H2_convergence_v4.png` | H2 evidence: convergence rate, stance entropy, JP closure effect sizes |
| Fig. 4 | `fig4_H3_mediation_v4.png` | H3 evidence: mediation path diagram, 12-subset robustness scatter, JP vs KR strip chart |
| Fig. 5 | `fig5_integrated_dashboard_v4.png` | 2×4 integrated evidence dashboard across all three hypotheses |
| Supp. | `fig_fix_v3_panel.png` | Y×M decomposition matrix + ICC diagnostics (4-panel) |
| App. A | `appA_composition_v4.png` | Exploratory: jurisdiction vs nationality SHAP gap — not confirmatory |

All paper figures are generated by `scripts/stage4_postanalysis_v5c.py`.

---

## Manuscript Conventions

**Use:** H1, H2, H3 as the three-hypothesis labels; *z* = −15.69 as the primary H3 statistic; PM = 35.2% for the proportion mediated.

**Do not use:** H1a, H1b, H2a, H2b as standalone confirmed hypotheses; frame the *z* = −15.69 vs *z* = −9.03 difference as an unresolved bug (it is resolved by Y-variable operationalization); claim Appendix A results as confirmatory.

---

## License

MIT License. See [LICENSE](LICENSE). The underlying patent corpus is sourced from public patent databases; all derived datasets follow their respective terms of use.
