"""
ml/download_nhanes.py
========================
Downloads the REAL National Health and Nutrition Examination Survey (NHANES)
data files needed to build a genuine, lab-measured stroke-risk dataset --
replacing the weaker BRFSS (binary lifestyle flags) / small 5,110-row Kaggle
options for stroke_rf.

Why NHANES: it's a CDC-run national survey where glucose, cholesterol, and
BMI are LAB-DRAWN / physically measured (not self-reported), and stroke
history comes from a validated questionnaire item (MCQ160F). Combining
multiple 2-year survey cycles gives 10,000+ real participant records.

This requires internet access on the machine you run it on.

Source: https://wwwn.cdc.gov/nchs/nhanes/
Public domain, U.S. government data -- no license/citation restrictions
beyond standard NHANES analytic guidelines (see DATASETS.md).

Usage:
    cd "C:\\SIH\\Qti π"
    python ml/download_nhanes.py

Downloads 7 cycles (2005-2006, 2007-2008, 2009-2010, 2011-2012, 2013-2014,
2015-2016, 2017-March 2020 pre-pandemic) x 11 file types (Demographics, Body
Measures, Glucose, Cholesterol, Blood Pressure, Questionnaire, Smoking,
Activity, Diabetes, HbA1c, BP Questionnaire) = 77 .XPT files into
ml/data/nhanes/. The 2017-March 2020 cycle (suffix P) REPLACES the older
2017-2018-only cycle (suffix J) rather than adding to it, since P already
contains those same participants under updated sample weights -- see the
CYCLES comment below for why.
"""
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NHANES_DIR = os.path.join(BASE_DIR, "data", "nhanes")

# (cycle_label, filename suffix letter, begin_year path segment) -- NHANES's real download
# URL uses the 4-digit BEGIN year of the cycle as a path segment (e.g. "2011" for
# 2011-2012), not the "YYYY-YYYY" cycle label used elsewhere on the site. Confirmed
# against CDC's own file-download tutorials/scripts, e.g.:
#   https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.XPT
#
# Round 3 (NHANES cycle expansion): added D/E/F (2005-2010, same suffix-style naming
# as G/H/I/J) and P (2017-March 2020 "pre-pandemic" combined release). The P cycle
# is used IN PLACE OF the old J (2017-2018) cycle, not in addition to it -- P re-releases
# the SAME 2017-2018 participants (identical SEQNs) merged with the partial 2019-2020
# collection under a single set of sample weights. Keeping both J and P would silently
# double-count every real 2017-2018 participant's stroke history. See plan
# 07321d10-50c7-4d38-b69a-9b3630208417 step 1 for the user-approved resolution.
CYCLES = [
    ("2005-2006", "D", "2005"),
    ("2007-2008", "E", "2007"),
    ("2009-2010", "F", "2009"),
    ("2011-2012", "G", "2011"),
    ("2013-2014", "H", "2013"),
    ("2015-2016", "I", "2015"),
    ("2017-2020", "P", "2017"),  # replaces the old ("2017-2018", "J", "2017") entry
]

# The P (2017-March 2020 pre-pandemic) cycle uses a PREFIX naming scheme
# (P_DEMO.XPT, P_BMX.XPT, ...) instead of the SUFFIX scheme every other cycle
# uses (DEMO_J.XPT, BMX_J.XPT, ...). Confirmed against CDC's own file listing,
# e.g. https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_HUQ.htm
PREFIX_STYLE_SUFFIXES = {"P"}

# file_key -> base NHANES file-name component (without suffix/prefix -- that's
# added per-cycle in main(), since the P cycle uses a different naming style
# than every other cycle -- see PREFIX_STYLE_SUFFIXES above).
FILES = {
    "demographics": "DEMO",   # age, sex
    "body_measures": "BMX",   # BMI (measured)
    "glucose": "GLU",         # fasting plasma glucose (lab)
    "cholesterol": "TCHOL",   # total cholesterol (lab)
    "blood_pressure": "BPX",  # measured blood pressure
    "questionnaire": "MCQ",   # MCQ160F = stroke history, MCQ160B = heart disease
        "smoking": "SMQ",
        "activity": "PAQ",
        "diabetes": "DIQ",
        "hba1c": "GHB",
        "bp_questionnaire": "BPQ",
}

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{begin_year}/DataFiles/{filename}"


def _is_valid_xpt(path: str) -> bool:
    """NHANES .XPT files are SAS XPORT transport files, which always start
    with a fixed ASCII header beginning 'HEADER RECORD'. CDC's site returns
    an HTML 404 page (status 200, not an HTTP error) for a wrong/renamed
    URL, so a plain urlretrieve() success is NOT proof the file is real --
    we must check the actual bytes."""
    try:
        with open(path, "rb") as f:
            head = f.read(20)
        return head.startswith(b"HEADER RECORD")
    except Exception:
        return False


def _download(url: str, dest: str) -> bool:
    if os.path.exists(dest):
        if _is_valid_xpt(dest):
            print(f"  already have {os.path.basename(dest)}, skipping")
            return True
        print(f"  {os.path.basename(dest)} exists but is not a valid XPT file (stale 404 page?) -- re-downloading")
        os.remove(dest)
    try:
        print(f"  downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; QtiPiResearch/1.0)"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        if not _is_valid_xpt(dest):
            print(f"  FAILED: {url} returned a non-XPT payload (likely a 404/redirect page, not the real file)", file=sys.stderr)
            os.remove(dest)
            return False
        return True
    except Exception as e:
        print(f"  FAILED: {url} ({e})", file=sys.stderr)
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    os.makedirs(NHANES_DIR, exist_ok=True)
    print(f"Downloading NHANES cycles {[c for c, _, _ in CYCLES]} to {NHANES_DIR} ...")
    print("This calls CDC's servers directly and requires internet access.")

    any_failed = False
    for cycle, suffix, begin_year in CYCLES:
        cycle_dir = os.path.join(NHANES_DIR, cycle)
        os.makedirs(cycle_dir, exist_ok=True)
        for key, base_name in FILES.items():
            # NHANES switched blood-pressure measurement from auscultatory
            # (mercury sphygmomanometer, file BPX) to oscillometric (file
            # BPXO) starting with the 2017-March 2020 pre-pandemic release --
            # auscultatory BP measurement ended after the 2017-2018 cycle.
            # Confirmed via CDC's own P_BPXO.htm documentation page.
            if key == "blood_pressure" and suffix in PREFIX_STYLE_SUFFIXES:
                base_name = "BPXO"
            if suffix in PREFIX_STYLE_SUFFIXES:
                filename = f"{suffix}_{base_name}.XPT"  # e.g. P_DEMO.XPT, P_BPXO.XPT
            else:
                filename = f"{base_name}_{suffix}.XPT"  # e.g. DEMO_J.XPT
            url = BASE_URL.format(begin_year=begin_year, filename=filename)
            dest = os.path.join(cycle_dir, filename)
            ok = _download(url, dest)
            any_failed = any_failed or not ok

    if any_failed:
        print(
            "\nSome files failed to download -- this can happen if CDC renamed a file for a "
            "given cycle (rare) or a firewall/proxy blocked cdc.gov. Check the cycle folders "
            f"under {NHANES_DIR} and see https://wwwn.cdc.gov/nchs/nhanes/ to fetch any missing "
            "file manually into the matching cycle subfolder. extract_nhanes_features.py will "
            "still work fine using whichever cycles/files did succeed.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone. Files saved under {NHANES_DIR}/<cycle>/")
    print("Next: run `python ml/extract_nhanes_features.py` to build ml/data/nhanes_stroke.csv.")


if __name__ == "__main__":
    main()
