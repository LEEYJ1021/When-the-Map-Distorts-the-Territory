"""
extract_crosswalk_A1.py
=======================
Extracts Appendix Table A1 (group-to-nationality crosswalk) from
step3_applicant_groups.parquet, computes the H1 contingency table,
and runs the chi-square test of independence.

Outputs
-------
outputs/supplementary/crosswalk_A1.csv
    Top-60 corporate groups with assigned nationality and patent count.

outputs/supplementary/h1_contingency_table.csv
    4-nationality × 8-jurisdiction contingency table used in the chi-square test.

outputs/supplementary/h1_chi2_results.json
    χ²(21) = 18,336.51, p, Cramér's V, and per-nationality home-filing rates.

Verified numbers (v7.0 guidance):
    JP  6.2%  (n=13,094)   KR 49.6% (n=3,949)
    DE 24.6%  (n=5,581)    US 91.1% (n=9,021)
    χ²(21) = 18,336.51, p < .001, Cramér's V = 0.4395

Usage
-----
    python scripts/supplementary/extract_crosswalk_A1.py

Requirements
------------
    pandas >= 2.0
    scipy >= 1.11
    pyarrow >= 12.0      (parquet engine)

Place step3_applicant_groups.parquet at:
    outputs/stage_work/stage0/step3_applicant_groups.parquet
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parents[2]
PARQUET    = REPO_ROOT / "outputs" / "stage_work" / "stage0" / "step3_applicant_groups.parquet"
OUT_DIR    = REPO_ROOT / "outputs" / "supplementary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Group → Nationality crosswalk  (researcher-constructed, public HQ data)
# Matches Appendix Table A1 in the paper exactly.
# Key: applicant_group_primary value (after LLM label cleaning)
# Value: ISO-2 nationality code
# ---------------------------------------------------------------------------
CROSSWALK: dict[str, str] = {
    # Japan
    "Toyota Group":            "JP",
    "Honda Group":             "JP",
    "Nissan Group":            "JP",
    "Mitsubishi Group":        "JP",
    "Hitachi Group":           "JP",
    "Mazda Group":             "JP",
    "Subaru Group":            "JP",
    "Komatsu Group":           "JP",
    "Denso Group":             "JP",
    "Panasonic Group":         "JP",
    "Yamaha Group":            "JP",
    "Isuzu Group":             "JP",
    "Suzuki Group":            "JP",
    "Kubota Group":            "JP",
    "Aisin Group":             "JP",
    "Furukawa Group":          "JP",
    # United States
    "Ford Group":              "US",
    "GM Group":                "US",
    "Waymo/Alphabet":          "US",
    "Eaton Group":             "US",
    "Intel Group":             "US",
    "Cummins Group":           "US",
    "Delphi Group":            "US",
    "Bendix Group":            "US",
    "BorgWarner Group":        "US",
    "Chrysler Group":          "US",
    "Harman Group":            "US",
    "Oshkosh Group":           "US",
    "Visteon Group":           "US",
    "Mobileye Group":          "US",
    "Tesla Group":             "US",
    "Qualcomm Group":          "US",
    "NVIDIA Group":            "US",
    # Germany
    "Bosch Group":             "DE",
    "Volkswagen Group":        "DE",
    "ZF Group":                "DE",
    "Mercedes-Benz Group":     "DE",
    "Continental Group":       "DE",
    "BMW Group":               "DE",
    "Schaeffler Group":        "DE",
    # Korea
    "Hyundai Motor Group":     "KR",
    "LG Group":                "KR",
    "Samsung Group":           "KR",
    "Stradvision Group":       "KR",
    "Kia Group":               "KR",
    # Sweden
    "Volvo Group":             "SE",
    "Scania/Traton Group":     "SE",
    # France
    "Renault Group":           "FR",
    "Valeo Group":             "FR",
    # China
    "Geely Group":             "CN",
    "NIO Group":               "CN",
    "BYD Group":               "CN",
    "Baidu Group":             "CN",
    "Huawei Group":            "CN",
    # Other
    "Magna Group":             "CA",
    "Tata Group":              "IN",
    "Fiat Group":              "IT",
    "Stellantis Group":        "NL",
    "Flextronics Group":       "SG",
}

# The four nationalities corresponding to the Stage 4 agent personas
FOCUS_NATIONALITIES = ["JP", "KR", "DE", "US"]

# Home jurisdiction code for each nationality
HOME_JURISDICTION: dict[str, str] = {
    "JP": "JP",
    "KR": "KR",
    "DE": "DE",
    "US": "US",
    "SE": "SE",
    "FR": "FR",
    "CN": "CN",
    "CA": "CA",
    "IN": "IN",
    "IT": "IT",
    "NL": "NL",
    "SG": "SG",
}

# Top-7 jurisdictions to keep individually; everything else → "Other"
TOP_JURISDICTIONS = ["US", "DE", "EP", "JP", "KR", "CN", "RU"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_group_label(raw: str) -> str:
    """
    Strip LLM reasoning text that leaked into group labels.

    Example of leakage:
        "Fuji Heavy Industries is a subsidiary of Toyota, but it's more
         commonly known as Subaru.\nSubaru Group"
    → "Subaru Group"
    """
    if not isinstance(raw, str):
        return str(raw)
    # If a newline exists, take the last non-empty line (the actual label)
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    candidate = lines[-1] if lines else raw.strip()
    # Remove any residual punctuation artefacts at the start
    candidate = re.sub(r"^[^\w]+", "", candidate).strip()
    return candidate


def collapse_jurisdiction(jur: str) -> str:
    """Map raw Jurisdiction value to one of TOP_JURISDICTIONS or 'Other'."""
    if not isinstance(jur, str):
        return "Other"
    jur_upper = jur.strip().upper()
    return jur_upper if jur_upper in TOP_JURISDICTIONS else "Other"


def cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """Cramér's V with bias correction."""
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    k_corr = k - (k - 1) ** 2 / (n - 1)
    r_corr = r - (r - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2_corr / denom))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Load parquet ─────────────────────────────────────────────────────
    if not PARQUET.exists():
        sys.exit(
            f"[ERROR] Parquet not found: {PARQUET}\n"
            "Place step3_applicant_groups.parquet at the expected path."
        )

    print(f"Loading {PARQUET} …")
    df = pd.read_parquet(PARQUET)
    print(f"  Loaded {len(df):,} rows.  Columns: {df.columns.tolist()}")

    # ── 2. Identify required columns ────────────────────────────────────────
    required = {"applicant_group_primary", "Jurisdiction"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        sys.exit(f"[ERROR] Missing columns in parquet: {missing_cols}")

    # ── 3. Clean LLM label leakage ──────────────────────────────────────────
    df["group_clean"] = df["applicant_group_primary"].apply(clean_group_label)

    # ── 4. Assign nationality via crosswalk ─────────────────────────────────
    df["nationality"] = df["group_clean"].map(CROSSWALK)
    n_mapped   = df["nationality"].notna().sum()
    n_total    = len(df)
    print(f"  Mapped {n_mapped:,} / {n_total:,} patents ({n_mapped/n_total:.1%})")

    # ── 5. Compute Appendix Table A1 ────────────────────────────────────────
    # Count by group_clean, attach nationality and patent count
    group_counts = (
        df.dropna(subset=["nationality"])
          .groupby(["group_clean", "nationality"], as_index=False)
          .size()
          .rename(columns={"size": "n_patents"})
          .sort_values("n_patents", ascending=False)
          .reset_index(drop=True)
    )
    group_counts.insert(0, "rank", range(1, len(group_counts) + 1))

    # Add HQ basis from a lookup (mirrors README Table A1)
    hq_basis = {
        "Toyota Group":         "Toyota Motor Corp., Aichi, Japan",
        "Hyundai Motor Group":  "Hyundai Motor Co., Seoul, Korea",
        "Ford Group":           "Ford Motor Co., Dearborn, MI, USA",
        "Honda Group":          "Honda Motor Co., Tokyo, Japan",
        "Bosch Group":          "Robert Bosch GmbH, Stuttgart, Germany",
        "GM Group":             "General Motors Co., Detroit, MI, USA",
        "Nissan Group":         "Nissan Motor Co., Yokohama, Japan",
        "Waymo/Alphabet":       "Waymo LLC / Alphabet Inc., Mountain View, CA, USA",
        "Volkswagen Group":     "Volkswagen AG, Wolfsburg, Germany",
        "Mitsubishi Group":     "Mitsubishi Motors Corp., Tokyo, Japan",
        "Hitachi Group":        "Hitachi Ltd., Tokyo, Japan",
        "ZF Group":             "ZF Friedrichshafen AG, Germany",
        "Mercedes-Benz Group":  "Mercedes-Benz Group AG, Stuttgart, Germany",
        "Volvo Group":          "AB Volvo, Gothenburg, Sweden",
        "Continental Group":    "Continental AG, Hanover, Germany",
        "Eaton Group":          "Eaton Corporation (US-origin)",
        "BMW Group":            "Bayerische Motoren Werke AG, Munich, Germany",
        "Mazda Group":          "Mazda Motor Corp., Hiroshima, Japan",
        "Subaru Group":         "Subaru Corporation, Tokyo, Japan",
        "Magna Group":          "Magna International Inc., Aurora, Ontario, Canada",
        "Tata Group":           "Tata Motors Ltd., Mumbai, India",
        "Renault Group":        "Renault SA, Boulogne-Billancourt, France",
        "Panasonic Group":      "Panasonic Holdings Corp., Osaka, Japan",
        "Bendix Group":         "Bendix Commercial Vehicle Systems (US ops)",
        "Scania/Traton Group":  "Scania AB / Traton SE, Södertälje, Sweden",
        "Komatsu Group":        "Komatsu Ltd., Tokyo, Japan",
        "Intel Group":          "Intel Corporation, Santa Clara, CA, USA",
        "Schaeffler Group":     "Schaeffler AG, Herzogenaurach, Germany",
        "Denso Group":          "DENSO Corporation, Kariya, Japan",
        "Cummins Group":        "Cummins Inc., Columbus, IN, USA",
        "LG Group":             "LG Electronics Inc., Seoul, Korea",
        "Delphi Group":         "Aptiv PLC (formerly Delphi; US-origin)",
        "Yamaha Group":         "Yamaha Motor Co., Shizuoka, Japan",
        "Isuzu Group":          "Isuzu Motors Ltd., Tokyo, Japan",
        "Stellantis Group":     "Stellantis N.V., Amsterdam, Netherlands",
        "BorgWarner Group":     "BorgWarner Inc., Auburn Hills, MI, USA",
        "Valeo Group":          "Valeo SA, Paris, France",
        "Chrysler Group":       "US-origin legacy (now part of Stellantis)",
        "Suzuki Group":         "Suzuki Motor Corp., Hamamatsu, Japan",
        "Fiat Group":           "Fiat SpA / Stellantis (Italian-origin)",
        "Geely Group":          "Geely Automobile Holdings, Hangzhou, China",
        "Harman Group":         "Harman International, Stamford, CT, USA",
        "Flextronics Group":    "Flex Ltd., Singapore",
        "Kubota Group":         "Kubota Corporation, Osaka, Japan",
        "NIO Group":            "NIO Inc., Shanghai, China",
        "Oshkosh Group":        "Oshkosh Corporation, Oshkosh, WI, USA",
        "Visteon Group":        "Visteon Corporation, Van Buren Township, MI, USA",
        "Mobileye Group":       "Mobileye (Intel subsidiary), Santa Clara, CA",
        "Tesla Group":          "Tesla Inc., Austin, TX, USA",
        "Samsung Group":        "Samsung Electronics Co., Suwon, Korea",
        "Qualcomm Group":       "Qualcomm Inc., San Diego, CA, USA",
        "NVIDIA Group":         "NVIDIA Corporation, Santa Clara, CA, USA",
        "Aisin Group":          "Aisin Corporation, Kariya, Japan",
        "Stradvision Group":    "StradVision Inc., Seoul, Korea",
        "BYD Group":            "BYD Co. Ltd., Shenzhen, China",
        "Baidu Group":          "Baidu Inc., Beijing, China",
        "Huawei Group":         "Huawei Technologies Co., Shenzhen, China",
        "Furukawa Group":       "Furukawa Electric Co., Tokyo, Japan",
        "Kia Group":            "Kia Corporation, Seoul, Korea",
    }
    group_counts["hq_basis"] = group_counts["group_clean"].map(hq_basis).fillna("—")

    # Save top-60
    top60 = group_counts.head(60).copy()
    out_a1 = OUT_DIR / "crosswalk_A1.csv"
    top60.to_csv(out_a1, index=False, encoding="utf-8-sig")
    print(f"\n[Appendix A1] Saved {len(top60)} groups → {out_a1}")

    # ── 6. H1 contingency table ─────────────────────────────────────────────
    # Restrict to four focus nationalities; collapse jurisdiction
    df_focus = df[df["nationality"].isin(FOCUS_NATIONALITIES)].copy()
    df_focus["jur_collapsed"] = df_focus["Jurisdiction"].apply(collapse_jurisdiction)

    # Add home-jurisdiction indicator
    df_focus["home_jur"] = df_focus.apply(
        lambda row: HOME_JURISDICTION.get(row["nationality"], ""), axis=1
    )
    df_focus["is_home"] = df_focus["Jurisdiction"].str.upper().str.strip() == df_focus["home_jur"]

    # Per-nationality home-filing rates (primary H1 result)
    home_rates = (
        df_focus.groupby("nationality")["is_home"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "home_filings", "count": "total"})
    )
    home_rates["home_rate_pct"] = (home_rates["home_filings"] / home_rates["total"] * 100).round(1)
    home_rates = home_rates.loc[FOCUS_NATIONALITIES]   # canonical order

    print("\n[H1] Home-filing rates:")
    print(home_rates.to_string())

    # Full contingency table: nationality × collapsed_jurisdiction
    ctab = pd.crosstab(df_focus["nationality"], df_focus["jur_collapsed"])
    # Enforce row order
    ctab = ctab.reindex(FOCUS_NATIONALITIES)
    # Enforce column order: top-7 + Other
    col_order = [c for c in TOP_JURISDICTIONS if c in ctab.columns] + (
        ["Other"] if "Other" in ctab.columns else []
    )
    ctab = ctab[col_order].fillna(0).astype(int)

    out_ctab = OUT_DIR / "h1_contingency_table.csv"
    ctab.to_csv(out_ctab, encoding="utf-8-sig")
    print(f"\n[H1] Contingency table ({ctab.shape}) → {out_ctab}")
    print(ctab.to_string())

    # ── 7. Chi-square test ──────────────────────────────────────────────────
    chi2_stat, p_val, dof, _ = chi2_contingency(ctab)
    n_chi2 = int(ctab.values.sum())
    v = cramers_v(chi2_stat, n_chi2, k=ctab.shape[1], r=ctab.shape[0])

    print(f"\n[H1] χ²({dof}) = {chi2_stat:,.2f}, p = {p_val:.2e}, Cramér's V = {v:.4f}")
    print(f"     N = {n_chi2:,}")

    # Reference values from v7.0 guidance
    ref = dict(chi2=18336.51, df=21, V=0.4395)
    print("\n[Verification] Expected (v7.0 guidance):")
    print(f"  χ²({ref['df']}) = {ref['chi2']:,.2f}, V = {ref['V']}")
    chi2_ok = abs(chi2_stat - ref["chi2"]) / ref["chi2"] < 0.02   # within 2%
    v_ok    = abs(v - ref["V"]) < 0.03
    print(f"  χ² match: {'✓ OK' if chi2_ok else '⚠ MISMATCH — check crosswalk mapping'}")
    print(f"  V match:  {'✓ OK' if v_ok    else '⚠ MISMATCH — check crosswalk mapping'}")

    # ── 8. Save JSON results ────────────────────────────────────────────────
    results = {
        "chi2_stat":  round(chi2_stat, 2),
        "df":         int(dof),
        "p_value":    float(p_val),
        "cramers_V":  round(v, 4),
        "n_patents":  n_chi2,
        "home_filing_rates": {
            nat: {
                "home_filing_rate_pct": float(home_rates.loc[nat, "home_rate_pct"]),
                "home_filings":         int(home_rates.loc[nat, "home_filings"]),
                "total_patents":        int(home_rates.loc[nat, "total"]),
            }
            for nat in FOCUS_NATIONALITIES
        },
        "reference_v7": ref,
        "chi2_verified": bool(chi2_ok),
        "V_verified":    bool(v_ok),
    }
    out_json = OUT_DIR / "h1_chi2_results.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\n[H1] Results saved → {out_json}")

    # ── 9. Summary report ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — H1 Chi-Square Test Results")
    print("=" * 60)
    print(f"  N (mapped, focus nationalities): {n_chi2:,}")
    for nat in FOCUS_NATIONALITIES:
        r = home_rates.loc[nat]
        print(f"  {nat}: {r['home_rate_pct']:.1f}%  "
              f"({int(r['home_filings']):,} / {int(r['total']):,})")
    print(f"\n  χ²({dof}) = {chi2_stat:,.2f}")
    print(f"  p          = {p_val:.3e}")
    print(f"  Cramér's V = {v:.4f}")
    print("\nOutputs:")
    print(f"  {out_a1}")
    print(f"  {out_ctab}")
    print(f"  {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
