# Data Description: `b60w_data.xlsx`

**Source corpus for all pipeline stages.** Place this file at the repository root before running any stage script.

> Download path (local): `C:\Users\LG\Downloads\b60w_data.xlsx`  
> Target path in repo: `b60w_data.xlsx` (repo root)  
> File size: ~43.4 MB  
> Tracked in git: **NO** — add to `.gitignore`

---

## 1. File Overview

| Property | Value |
|---|---|
| Format | Microsoft Excel (.xlsx) |
| Sheet | `Sheet1` (single sheet) |
| Rows | 53,199 |
| Columns | 37 |
| Coverage | CPC B60W — Vehicle Powertrain Control / Autonomous Driving |
| Application date range | 1897-06-12 ~ 2025-09-12 (effective analysis window: 1995Q1–2024Q4) |
| Publication date range | 1858-08-03 ~ 2026-05-12 |
| Data provider | [Lens.org](https://www.lens.org) patent database |

**Corpus definition:** All 53,199 records have `Cited by Patent Count ≥ 1` (forward-citation-validated). No records with zero forward citations are present in this extract.

---

## 2. Column Schema

| # | Column Name | Dtype | Missing (n) | Missing (%) | Unique Values | Notes |
|---|---|---|---|---|---|---|
| 1 | `#` | int64 | 0 | 0.00% | 53,199 | Sequential row index |
| 2 | `Jurisdiction` | str | 0 | 0.00% | 28 | Filing country/jurisdiction (≠ applicant nationality) |
| 3 | `Kind` | str | 0 | 0.00% | 18 | Patent kind code (B1, B2, A1, etc.) |
| 4 | `Display Key` | str | 0 | 0.00% | 53,199 | Human-readable patent number (e.g., `US 7963884 B2`) |
| 5 | `Lens ID` | str | 0 | 0.00% | 53,199 | Lens.org unique identifier |
| 6 | `Publication Date` | object | 20 | 0.04% | 8,737 | Patent publication date |
| 7 | `Publication Year` | float64 | 20 | 0.04% | 146 | Derived from Publication Date |
| 8 | `Application Number` | object | 0 | 0.00% | 53,157 | National filing application number |
| 9 | `Application Date` | object | 128 | 0.24% | 11,959 | **Primary date for quarter labeling in all stages** |
| 10 | `Priority Numbers` | str | 122 | 0.23% | 47,639 | Priority claim numbers (`;; `-delimited) |
| 11 | `Earliest Priority Date` | object | 130 | 0.24% | 11,015 | Earliest priority filing date |
| 12 | `Title` | str | 14 | 0.03% | 42,606 | Patent title |
| 13 | `Abstract` | str | 10,038 | 18.87% | 40,078 | Full abstract text; used in Stage 1 embedding |
| 14 | `Applicants` | str | 2,571 | 4.83% | 11,905 | Filing applicant(s); `;; `-delimited multi-value |
| 15 | `Inventors` | str | 3,079 | 5.79% | 36,415 | Named inventors; `;; `-delimited |
| 16 | `Owners` | str | 17,767 | 33.40% | 27,285 | Current patent owner(s) with date stamp |
| 17 | `URL` | str | 0 | 0.00% | 53,199 | Lens.org record URL |
| 18 | `Document Type` | str | 0 | 0.00% | 1 | All records: `Granted Patent` |
| 19 | `Has Full Text` | str | 0 | 0.00% | 2 | `yes` / `no` |
| 20 | `Cites Patent Count` | int64 | 0 | 0.00% | 611 | Backward citation count (patent cites made) |
| 21 | `Cited by Patent Count` | int64 | 0 | 0.00% | 400 | Forward citation count (times cited by later patents) |
| 22 | `Simple Family Size` | int64 | 0 | 0.00% | 82 | Count of simple patent family members |
| 23 | `Simple Family Members` | str | 0 | 0.00% | 41,324 | Lens IDs of simple family; `;; `-delimited |
| 24 | `Simple Family Member Jurisdictions` | str | 0 | 0.00% | 4,411 | Jurisdictions of simple family |
| 25 | `Extended Family Size` | int64 | 0 | 0.00% | 138 | Count of extended patent family members |
| 26 | `Extended Family Members` | str | 0 | 0.00% | 40,084 | Lens IDs of extended family; `;; `-delimited |
| 27 | `Extended Family Member Jurisdictions` | str | 0 | 0.00% | 4,462 | Jurisdictions of extended family |
| 28 | `Sequence Count` | int64 | 0 | 0.00% | 2 | Biological sequence count (0 or 1; irrelevant here) |
| 29 | `CPC Classifications` | str | 0 | 0.00% | 40,466 | CPC codes; `;; `-delimited |
| 30 | `IPCR Classifications` | str | 967 | 1.82% | 42,421 | IPC-R codes; `;; `-delimited |
| 31 | `US Classifications` | object | 37,923 | 71.29% | 12,542 | USPC codes; sparse (only US-granted patents) |
| 32 | `NPL Citation Count` | int64 | 0 | 0.00% | 247 | Non-patent literature citation count |
| 33 | `NPL Resolved Citation Count` | int64 | 0 | 0.00% | 81 | NPL citations resolved to Lens IDs |
| 34 | `NPL Resolved Lens ID(s)` | str | 47,167 | 88.66% | 4,496 | Lens IDs of resolved NPL references |
| 35 | `NPL Resolved External ID(s)` | object | 47,167 | 88.66% | 4,510 | External IDs (DOIs, etc.) of resolved NPL |
| 36 | `NPL Citations` | str | 34,295 | 64.47% | 18,393 | Raw NPL citation strings |
| 37 | `Legal Status` | str | 0 | 0.00% | 4 | `ACTIVE` / `EXPIRED` / `INACTIVE` / `UNKNOWN` |

---

## 3. Key Variable Distributions

### 3.1 Jurisdiction (Filing Country)

> **Critical note:** `Jurisdiction` reflects where the patent was *filed*, not the applicant's nationality. Japanese companies (e.g., Toyota) predominantly file in the US, making `Jurisdiction = US` account for 71.6% of the corpus despite JP-origin assignees dominating many technology areas. Stage 0 resolves this via Owners-based nationality mapping.

| Jurisdiction | Count | Share (%) |
|---|---|---|
| US | 38,086 | 71.59% |
| KR | 4,608 | 8.66% |
| DE | 3,041 | 5.72% |
| EP | 2,550 | 4.79% |
| JP | 2,172 | 4.08% |
| CN | 921 | 1.73% |
| GB | 521 | 0.98% |
| RU | 374 | 0.70% |
| FR | 341 | 0.64% |
| AT | 111 | 0.21% |
| Other (18) | 474 | 0.89% |
| **Total** | **53,199** | **100%** |

Top-8 jurisdictions account for 98.3% of all records.

**Jurisdiction ≠ Nationality illustration:**

| Query | Count |
|---|---|
| Toyota-owned patents with Jurisdiction = US | 4,131 |
| Toyota-owned patents with Jurisdiction = JP | 0 |

This MNAR-generating mechanism is the structural driver of H1 (§2.1 of the paper).

---

### 3.2 Application Date & Quarter Labeling

| Statistic | Value |
|---|---|
| Records with parseable Application Date | 53,071 (99.76%) |
| Records with missing/unparseable date | 128 (0.24%) |
| Earliest application date | 1897-06-12 |
| Latest application date | 2025-09-12 |
| Total observed quarters | 493 |
| Mean patents per quarter | 107.6 |
| Median patents per quarter | 13.0 |
| Max patents per quarter | 1,122 |

**Analysis window (1995Q1–2024Q4):**

| Window | Records | Share (%) |
|---|---|---|
| 1995Q1–2024Q4 (main analysis) | — | — |
| 2010Q1–2024Q4 (60-quarter window, Stage 2) | 32,110 | 60.36% |
| Outside 2010Q1–2024Q4 | 20,961 | 39.40% |
| Missing quarter label | 128 | 0.24% |

Recent 10-year application counts (key growth period):

| Year | Patents Filed |
|---|---|
| 2015 | ~1,100 |
| 2016 | ~1,400 |
| 2017 | ~1,600 |
| 2018 | ~1,700 |
| 2019 | ~1,800 |
| 2020 | ~1,900 |
| 2021 | 1,695 |
| 2022 | 1,053 |
| 2023 | 419 |
| 2024 | 96 |

The 2022–2024 decline reflects publication lag; recently filed patents have not yet been published/indexed.

---

### 3.3 Top Applicants

Applicants field uses `;; ` as multi-value delimiter. 77.7% of patents have a single applicant; 17.5% are joint filings.

| Rank | Applicant (normalized) | Patents |
|---|---|---|
| 1 | TOYOTA MOTOR CO LTD | 4,362 |
| 2 | FORD GLOBAL TECH LLC | 2,793 |
| 3 | HYUNDAI MOTOR CO LTD | 2,151 |
| 4 | HONDA MOTOR CO LTD | 1,874 |
| 5 | NISSAN MOTOR | 1,559 |
| 6 | BOSCH GMBH ROBERT | 1,384 |
| 7 | KIA MOTORS CORP | 988 |
| 8 | GM GLOBAL TECH OPERATIONS INC | 956 |
| 9 | DENSO CORP | 953 |
| 10 | GM GLOBAL TECH OPERATIONS LLC | 729 |
| 11 | VOLKSWAGEN AG | 540 |
| 12 | WAYMO LLC | 502 |
| 13 | BAYERISCHE MOTOREN WERKE AG | 501 |
| 14 | AUDI AG | 457 |
| 15 | STATE FARM MUTUAL AUTO INSURANCE CO | 448 |

Applicant name variants (e.g., `HYUNDAI MOTOR CO LTD` vs `현대자동차주식회사`) are resolved to canonical group names in Stage 0 via the applicant-grouping pipeline (`stage0_pipeline.py`). This produces the four simulation agents: **Toyota Group (JP)**, **Hyundai Motor Group (KR)**, **Ford Group (US)**, **Bosch Group (DE)**.

---

### 3.4 Abstract & Title Coverage

| Variable | Present | Missing | Missing (%) | Notes |
|---|---|---|---|---|
| `Abstract` | 43,161 | 10,038 | 18.87% | Used in Stage 1 keyword extraction & embedding |
| `Title` | 53,185 | 14 | 0.03% | Fallback when Abstract missing |

**Abstract missingness is structurally non-random (MNAR):**

| Jurisdiction | Abstract Missing (%) |
|---|---|
| BE | 100.0% |
| EP | 100.0% |
| FR | 92.7% |
| JP | 92.6% |
| CH | 83.3% |
| TW | 75.8% |
| DE | ~58.9% |
| KR | ~23.8% |
| US | ~3.9% |

JP and EP abstracts are structurally absent due to database language/format conventions, not random omission. This pattern compounds the MNAR missingness documented in H1 (patent-nationality mapping).

Abstract length distribution (word count, among 43,161 non-missing):

| Statistic | Words |
|---|---|
| Mean | 122.8 |
| Std | 68.1 |
| 25th pct | 91 |
| Median | 123 |
| 75th pct | 146 |
| Max | 1,871 |
| Records with < 5 words | 686 |

Records with fewer than 5 words are treated as uninformative and excluded from Stage 1 embedding.

---

### 3.5 Citation Variables

#### Forward Citations (`Cited by Patent Count`)

| Statistic | Value |
|---|---|
| Min | 1 |
| Mean | 12.98 |
| Median | 3 |
| 75th pct | 10 |
| Max | 1,920 |
| Records with count = 0 | **0 (0.00%)** |

All 53,199 records satisfy the forward-citation filter (≥ 1). This is the primary corpus inclusion criterion.

#### Backward Citations (`Cites Patent Count`)

| Statistic | Value |
|---|---|
| Min | 0 |
| Mean | 23.56 |
| Median | 10 |
| 75th pct | 19 |
| Max | 1,658 |
| Records with count = 0 | 8,552 (16.08%) |

`Cites Patent Count` provides backward citation *counts* only; the referenced patent ID list is not included. Graph edge construction (Stage 1) therefore uses co-classification and co-applicant links rather than direct citation edges.

#### NPL Citations (`NPL Citation Count`)

| Statistic | Value |
|---|---|
| Mean | 3.43 |
| Median | 0 |
| 75th pct | 1 |
| Max | 737 |
| NPL Citations text missing | 34,295 (64.47%) |

---

### 3.6 CPC / IPC Classification Coverage

| Variable | Missing (n) | Missing (%) | Notes |
|---|---|---|---|
| `CPC Classifications` | 0 | 0.00% | Complete; `;; `-delimited multi-code |
| `IPCR Classifications` | 967 | 1.82% | Minor gaps; CPC used as fallback |
| `US Classifications` | 37,923 | 71.29% | Only US-granted patents; not used in analysis |

**B60W subgroup distribution (IPC-R basis):**

| Subgroup | Patent Count | Notes |
|---|---|---|
| B60W10 | 17,519 | Control of vehicle sub-units |
| B60W30 | 17,159 | Driving safety systems |
| B60W50 | 9,763 | Driver monitoring and interaction |
| B60W20 | 7,879 | Hybrid electric vehicles |
| B60W40 | 7,830 | Vehicle state estimation |
| B60W60 | 2,210 | **Autonomous driving** (introduced 2020) |
| B60W1 / B60W | 2 | Obsolete/generic codes |

33.2% of patents have zero B60W subgroups under IPC-R; 17,675 of these have B60W codes under CPC, enabling CPC-supplemented coverage for node feature `f6` (subgroup diversity).

**B60W60 emergence timeline** (used as structural break signal in Stage 2 Chow tests):

| Year | B60W60 Patents Filed |
|---|---|
| 2012 | 3 |
| 2013 | 2 |
| 2014 | 5 |
| 2015 | 2 |
| 2016 | 18 |
| 2017 | 70 |
| 2018 | 134 |
| 2019 | 196 |
| **2020** | **737** |
| 2021 | 507 |
| 2022 | 352 |
| 2023 | 144 |
| 2024 | 23 |

First observed quarter: 2012Q1. Rapid growth from 2019Q1; peak volume 2020 (coinciding with SAE Level 3+ regulatory activity globally).

---

### 3.7 Patent Family Size

| Statistic | Simple Family Size | Extended Family Size |
|---|---|---|
| Mean | 8.13 | 13.47 |
| Std | 11.15 | 45.29 |
| 25th pct | 3 | 4 |
| Median | 6 | 7 |
| 75th pct | 9 | 10 |
| Max | 152 | 894 |

Records with both Simple and Extended family size = 1 (no family members): 4,342 (8.2%). These singleton patents have no cross-jurisdictional protection and are flagged in Stage 1 node feature `f7` (family size).

---

### 3.8 Legal Status

| Status | Count | Share (%) |
|---|---|---|
| ACTIVE | 29,938 | 56.3% |
| EXPIRED | 15,552 | 29.2% |
| INACTIVE | 7,689 | 14.5% |
| UNKNOWN | 20 | 0.04% |

INACTIVE patents (~14.5%) are retained in the corpus for historical technology-trend analysis. Strategic salience interpretation (Stage 4) should note that ACTIVE status patents represent currently enforceable IP positions.

---

## 4. Data Quality Summary

| Check | Status | Detail |
|---|---|---|
| Quarter labeling (Application Date) | ✔ 99.76% feasible | 128 records unfilterable; excluded from Stage 1 |
| Applicant identification | ✔ 95.2% present | Individual inventor names mixed in; resolved in Stage 0 |
| Abstract coverage | ⚠ 81.1% present | JP/EP structurally missing (MNAR); Title used as fallback |
| Forward citation filter (≥ 1) | ✔ 100% satisfied | All 53,199 records meet corpus inclusion criterion |
| Backward citation IDs | ⚠ Count only | Graph edge construction uses co-classification links |
| B60W subgroup (node feature f6) | ⚠ 66.8% (IPC-R) | CPC supplementation raises coverage to ~100% |
| Family size variables | ✔ Complete | Both Simple and Extended present for all records |
| Jurisdiction vs. nationality | ⚠ Divergent | 93.1% of JP-origin patents filed under non-JP jurisdiction; Stage 0 re-maps via Owners field → MNAR driver (H1) |

---

## 5. Role in Pipeline

| Stage | How `b60w_data.xlsx` is used |
|---|---|
| **Stage 0** | Loaded directly; language detection, machine translation (non-EN), applicant-to-group normalization → `step3_applicant_groups.parquet` |
| **Stage 1** | `s1_corpus.parquet` derived from Stage 0 output; Abstract + Title used for keyword extraction and sentence-transformer embedding; `Cites Patent Count` / `Cited by Patent Count` → node features f3, f4 |
| **Stage 2** | Quarterly patent counts per keyword node (derived from Stage 1) used as time-series input to Mamba and LSTM models |
| **Stage 3** | SHAP analysis uses nationality groupings (derived from Stage 0 applicant mapping) |
| **Stage 4** | Agent personas (Toyota/JP, Hyundai/KR, Ford/US, Bosch/DE) grounded in top-applicant distributions from this corpus |

---

## 6. Loading Instructions

```python
import pandas as pd
from pathlib import Path

# Place b60w_data.xlsx at repository root
df = pd.read_excel(Path("b60w_data.xlsx"), sheet_name="Sheet1")

# Parse dates
for col in ["Application Date", "Publication Date", "Earliest Priority Date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Quarter label for time-series staging
df["app_quarter"] = df["Application Date"].dt.to_period("Q")

# Split multi-value fields
def split_multi(val):
    if pd.isna(val):
        return []
    return [x.strip() for x in str(val).split(";;") if x.strip()]

df["applicants_list"] = df["Applicants"].apply(split_multi)
df["cpc_codes"]       = df["CPC Classifications"].apply(split_multi)
df["ipc_codes"]       = df["IPCR Classifications"].apply(split_multi)

print(f"Loaded {len(df):,} records across {df['app_quarter'].nunique()} quarters")
```

---

## 7. Reproducing Stage 0 Output

```bash
# Ensure b60w_data.xlsx is at repo root, then:
python stage0_pipeline.py
# Runtime: ~15–25 min (translation cache accelerates subsequent runs)
# Output: stage0_work/step3_applicant_groups.parquet (58.4 MB)
```

The Stage 0 output (`step3_applicant_groups.parquet`) is the authoritative input for all downstream stages. Running `stage0_pipeline.py` is required before any other stage unless pre-built parquet files are available.
