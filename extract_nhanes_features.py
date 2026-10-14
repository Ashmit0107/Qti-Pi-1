"""
ml/extract_nhanes_features.py
================================
Turns the raw NHANES .XPT files downloaded by ml/download_nhanes.py into a
single clean, real, lab-measured stroke-risk dataset:

    ml/data/nhanes_stroke.csv

Merge key: SEQN (unique NHANES participant ID), joined across files WITHIN
each 2-year cycle, then all cycles are stacked to reach 10,000+ rows.

Round 3 (NHANES cycle expansion): now covers D/E/F (2005-2010) + G/H/I
(2011-2016) + P (2017-March 2020 pre-pandemic) = 7 cycles total, roughly
doubling the prior 4-cycle (2011-2018) sample. The P cycle REPLACES the old
J (2017-2018)-only cycle rather than adding to it -- see the safety guard in
main() below for why (P re-releases the same 2017-2018 participants).

Target:
    MCQ160F ("Ever told... had a stroke") -> stroke = 1 (yes) / 0 (no).
    Values 7 (Refused) and 9 (Don't know) are dropped, not coerced to 0 --
    silently mapping "don't know" to "no stroke" would quietly poison the
    label and inflate apparent recall on a corrupted target.

Features:
    age                 RIDAGEYR   (Demographics, in years)
    sex                 RIAGENDR   (Demographics, 1=Male 2=Female -> 0/1)
    bmi                 BMXBMI     (Body Measures, kg/m^2, LAB-MEASURED)
    avg_glucose_level   LBXGLU     (Glucose, mg/dL, fasting LAB-MEASURED)
    cholesterol         LBXTC      (Total Cholesterol, mg/dL, LAB-MEASURED)
    hypertension        derived from BPX (BPXO for the 2017-2020 cycle,
                        oscillometric device -- auscultatory BPX measurement
                        ended after 2017-2018) measured blood pressure
                        readings (mean of up to 3 systolic/diastolic
                        readings; hypertensive if mean systolic >= 140 or
                        mean diastolic >= 90). Using the MEASURED reading
                        instead of a self-reported yes/no questionnaire
                        item is intentional -- it is the more clinically
                        reliable of the two.
    heart_disease       MCQ160B ("Ever told... had heart disease") -> 0/1,
                        included because it's a well-established stroke
                        comorbidity and mirrors the Kaggle schema's field
                        of the same intent.

Rows with a missing/refused/don't-know target are dropped (see above).
Rows with missing feature values are imputed with that cycle-stacked
column's median (reported in the printed summary) rather than dropped,
to avoid throwing away otherwise-valid stroke-history rows over one
missing lab value.

Usage:
    cd "C:\\SIH\\Qti π"
    python ml/extract_nhanes_features.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NHANES_DIR = os.path.join(BASE_DIR, "data", "nhanes")
OUT_CSV = os.path.join(BASE_DIR, "data", "nhanes_stroke.csv")

# file_key -> (filename prefix, [columns to keep besides SEQN])
FILE_SPECS = {
    "demographics": ("DEMO", ["RIDAGEYR", "RIAGENDR"]),
    "body_measures": ("BMX", ["BMXBMI", "BMXWAIST"]),
    "glucose": ("GLU", ["LBXGLU"]),
    "cholesterol": ("TCHOL", ["LBXTC"]),
    "blood_pressure": ("BPX", [
        "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3",
    ]),
    "questionnaire": ("MCQ", ["MCQ160F", "MCQ160B"]),
        "smoking": ("SMQ", ["SMQ020"]),
        "activity": ("PAQ", ["PAQ605"]),
        "diabetes": ("DIQ", ["DIQ010"]),
        "hba1c": ("GHB", ["LBXGH"]),
        "bp_questionnaire": ("BPQ", ["BPQ020"]),
}


def _find_file(cycle_dir: str, prefix: str) -> str | None:
    # Two possible naming styles: suffix-style "{prefix}_{cycle}.XPT" (e.g.
    # DEMO_D.XPT, used by every normal cycle) and prefix-style
    # "{cycle}_{prefix}.XPT" (e.g. P_DEMO.XPT, used only by the 2017-2020
    # pre-pandemic release). Try both; a cycle folder only ever contains one
    # style, so this is unambiguous in practice.
    matches = glob.glob(os.path.join(cycle_dir, f"{prefix}_*.XPT"))
    if not matches:
        matches = glob.glob(os.path.join(cycle_dir, f"*_{prefix}.XPT"))
    return matches[0] if matches else None


def _read_xpt(path: str, cols: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.read_sas(path, format="xport")
    except Exception as e:
        print(f"  FAILED to read {path}: {e}", file=sys.stderr)
        return None
    keep = ["SEQN"] + [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  note: {os.path.basename(path)} missing expected columns {missing} (renamed/absent this cycle)")
    return df[keep]


def _load_cycle(cycle_dir: str, cycle_label: str) -> pd.DataFrame | None:
    merged = None
    for key, (prefix, cols) in FILE_SPECS.items():
        # Round 3: the 2017-2020 pre-pandemic cycle measures blood pressure
        # oscillometrically (file BPXO, columns BPXOSY1/BPXODI1/...) instead of
        # auscultatorily (file BPX, columns BPXSY1/BPXDI1/...) -- auscultatory
        # BP measurement ended after the 2017-2018 cycle. Use the right
        # prefix/columns for this one cycle+key combo, then rename the result
        # back to the BPX-style names so _derive_hypertension() below (and
        # everything downstream) doesn't need to know the difference.
        use_prefix, use_cols = prefix, cols
        bpxo_rename = None
        if key == "blood_pressure" and cycle_label == "2017-2020":
            use_prefix = "BPXO"
            use_cols = ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"]
            bpxo_rename = {
                "BPXOSY1": "BPXSY1", "BPXOSY2": "BPXSY2", "BPXOSY3": "BPXSY3",
                "BPXODI1": "BPXDI1", "BPXODI2": "BPXDI2", "BPXODI3": "BPXDI3",
            }

        path = _find_file(cycle_dir, use_prefix)
        if path is None:
            print(f"  [{cycle_label}] skipping {key}: no {use_prefix}_*.XPT file found in {cycle_dir}")
            continue
        part = _read_xpt(path, use_cols)
        if part is None:
            continue
        if bpxo_rename:
            part = part.rename(columns=bpxo_rename)
        merged = part if merged is None else merged.merge(part, on="SEQN", how="outer")
    if merged is not None:
        merged["nhanes_cycle"] = cycle_label
    return merged


def _derive_hypertension(df: pd.DataFrame) -> pd.Series:
    sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3"] if c in df.columns]
    dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3"] if c in df.columns]
    mean_sys = df[sys_cols].mean(axis=1) if sys_cols else pd.Series(np.nan, index=df.index)
    mean_dia = df[dia_cols].mean(axis=1) if dia_cols else pd.Series(np.nan, index=df.index)
    # A diastolic reading of 0 in NHANES BPX means "no audible sound / not applicable",
    # not an actual diastolic pressure of zero -- treat as missing, not hypotensive.
    mean_dia = mean_dia.replace(0, np.nan)
    hyper = ((mean_sys >= 140) | (mean_dia >= 90)).astype("float")
    hyper[mean_sys.isna() & mean_dia.isna()] = np.nan
    return hyper


def main():
    if not os.path.isdir(NHANES_DIR):
        print(f"ERROR: {NHANES_DIR} not found. Run `python ml/download_nhanes.py` first.", file=sys.stderr)
        sys.exit(1)

    cycle_dirs = sorted(
        d for d in glob.glob(os.path.join(NHANES_DIR, "*")) if os.path.isdir(d)
    )
    if not cycle_dirs:
        print(f"ERROR: no cycle subfolders found under {NHANES_DIR}.", file=sys.stderr)
        sys.exit(1)

    # SAFETY GUARD: the 2017-2020 (P, pre-pandemic) release re-releases the SAME
    # participants (identical SEQNs) as the old 2017-2018 (J) cycle, merged with
    # the partial 2019-2020 collection under updated sample weights. Stacking
    # both would silently double-count every real 2017-2018 person's stroke
    # history. download_nhanes.py only ever ADDS cycle folders, so a stale
    # 2017-2018/ folder from a prior run can still be sitting on disk here --
    # this guard drops it whenever 2017-2020/ is also present, rather than
    # relying on the user to remember to delete it.
    cycle_labels = {os.path.basename(d) for d in cycle_dirs}
    if "2017-2020" in cycle_labels and "2017-2018" in cycle_labels:
        print(
            "NOTE: both ml/data/nhanes/2017-2018/ (old J cycle) and ml/data/nhanes/2017-2020/ "
            "(new P pre-pandemic cycle, which already includes the 2017-2018 participants under "
            "updated weights) were found. Excluding 2017-2018/ from this run to avoid "
            "double-counting the same real people. Safe to delete that old folder."
        )
        cycle_dirs = [d for d in cycle_dirs if os.path.basename(d) != "2017-2018"]

    all_cycles = []
    for cycle_dir in cycle_dirs:
        cycle_label = os.path.basename(cycle_dir)
        print(f"Loading cycle {cycle_label} ...")
        merged = _load_cycle(cycle_dir, cycle_label)
        if merged is None or len(merged) == 0:
            print(f"  WARNING: no usable data for cycle {cycle_label}, skipping")
            continue
        all_cycles.append(merged)

    if not all_cycles:
        print("ERROR: no cycles produced any data. Check ml/data/nhanes/ contents.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(all_cycles, ignore_index=True, sort=False)
    print(f"\nStacked {len(cycle_dirs)} cycles -> {len(df)} total participant rows before cleaning.")

    # Extra safety net (beyond the 2017-2018-vs-2017-2020 guard above): confirm no
    # single SEQN appears under two different cycle labels in the stacked data,
    # since that would mean the same real participant got counted twice.
    dup_seqn_across_cycles = df.groupby("SEQN")["nhanes_cycle"].nunique()
    n_cross_cycle_dupes = int((dup_seqn_across_cycles > 1).sum())
    if n_cross_cycle_dupes:
        print(
            f"WARNING: {n_cross_cycle_dupes} SEQN values appear under more than one cycle label "
            "-- this should not happen with the current cycle set and may indicate a real "
            "double-counting bug. Investigate before trusting this dataset.",
            file=sys.stderr,
        )
    else:
        print("Cross-cycle duplicate-participant check: PASS (no SEQN found under more than one cycle).")

    # --- Target: MCQ160F (stroke history) ---
    if "MCQ160F" not in df.columns:
        print("ERROR: MCQ160F (stroke history) column not found in any cycle -- cannot build target.", file=sys.stderr)
        sys.exit(1)
    before = len(df)
    df = df[df["MCQ160F"].isin([1.0, 2.0])].copy()  # drop 7=Refused, 9=Don't know, NaN
    print(f"Dropped {before - len(df)} rows with missing/refused/unknown stroke-history answer.")
    df["stroke"] = (df["MCQ160F"] == 1.0).astype(int)

    # --- heart_disease comorbidity flag ---
    if "MCQ160B" in df.columns:
        df["heart_disease"] = df["MCQ160B"].map({1.0: 1, 2.0: 0}).astype("float")
    else:
        df["heart_disease"] = np.nan

    # --- hypertension, derived from measured BP, with self-reported fallback ---
    df["hypertension"] = _derive_hypertension(df)
    if "BPQ020" in df.columns:
        # BPQ020: 1=Yes told high BP, 2=No. Only fills the gap where the
        # MEASURED reading is missing -- doesn't override real measured values.
        self_reported_htn = df["BPQ020"].map({1.0: 1, 2.0: 0})
        n_recovered = df["hypertension"].isna().sum()
        df["hypertension"] = df["hypertension"].fillna(self_reported_htn)
        n_recovered -= df["hypertension"].isna().sum()
        print(f"Recovered {n_recovered} hypertension values from self-report (BPQ020) where measured BP was missing.")

    # New lifestyle/diagnostic features. NHANES coding: 1=Yes, 2=No,
    # 7=Refused, 9=Don't know -> keep Yes/No as 1/0, treat anything else as missing.
    df["smoking"] = df["SMQ020"].map({1.0: 1, 2.0: 0}) if "SMQ020" in df.columns else np.nan
    df["physical_activity"] = df["PAQ605"].map({1.0: 1, 2.0: 0}) if "PAQ605" in df.columns else np.nan
    df["diabetes"] = df["DIQ010"].map({1.0: 1, 2.0: 0, 3.0: 0}) if "DIQ010" in df.columns else np.nan

    # --- rename / select final feature columns ---
    df = df.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",       # 1=Male, 2=Female (NHANES coding) -> kept numeric, see note below
        "BMXBMI": "bmi",
        "BMXWAIST":"waist_circumference",
        "LBXGLU": "avg_glucose_level",
        "LBXTC": "cholesterol",
        "LBXGH": "hba1c",
    })
    # NHANES sex coding is 1=Male, 2=Female; remap to 0/1 (0=Male, 1=Female) to match
    # the convention used elsewhere in this project's stroke feature set.
    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1.0: 0, 2.0: 1})

    feature_cols = ["age", "sex", "bmi", "avg_glucose_level", "cholesterol",
                     "hypertension", "heart_disease", "smoking", "physical_activity", "diabetes","hba1c","waist_circumference"]
                     
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    final = df[["SEQN", "nhanes_cycle"] + feature_cols + ["stroke"]].copy()

    # --- impute missing feature values with the cycle-stacked median, report how much ---
    print("\nMissing-value summary before imputation:")
    for col in feature_cols:
        n_missing = final[col].isna().sum()
        pct = 100.0 * n_missing / len(final) if len(final) else 0.0
        print(f"  {col}: {n_missing} missing ({pct:.1f}%)")
        if n_missing > 0:
            final[f"{col}_was_missing"] = final[col].isna().astype(int)
            median_val = final[col].median()
            final[col] = final[col].fillna(median_val)

    final = final.drop_duplicates(subset=["SEQN", "nhanes_cycle"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    final.to_csv(OUT_CSV, index=False)

    n_pos = int(final["stroke"].sum())
    n_total = len(final)
    print(f"\nWrote {n_total} rows ({n_pos} stroke=1, {n_total - n_pos} stroke=0, "
          f"{100.0 * n_pos / n_total:.2f}% positive rate) to {OUT_CSV}")
    print("Next: ml/data_preprocessing.py load_stroke() will use this file automatically if present.")


if __name__ == "__main__":
    main()
