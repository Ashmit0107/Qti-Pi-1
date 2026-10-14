"""
ml/download_mitbih.py
=======================
Downloads the REAL MIT-BIH Arrhythmia Database (PhysioNet) so the arrhythmia
model can be trained on genuine annotated heartbeats instead of the
synthetic placeholder.

This requires internet access on the machine you run it on (this repo's
Thrilia dev workspace has none, which is why this is a separate script for
you to run locally rather than something already executed for you).

Source: https://physionet.org/content/mitdb/1.0.0/
License: Open Data Commons Attribution License (ODC-By) — free to use with
citation. Citation is already recorded in ml/DATASETS.md.

Usage:
    cd "C:\\SIH\\Qti π"
    pip install wfdb   # if not already installed via requirements.txt
    python ml/download_mitbih.py

Downloads all 48 records (~100MB total) into ml/data/mitbih/.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MITBIH_DIR = os.path.join(BASE_DIR, "data", "mitbih")


def main():
    try:
        import wfdb
    except ImportError:
        print(
            "ERROR: the 'wfdb' package is not installed. Run:\n"
            "    pip install wfdb\n"
            "or 'pip install -r requirements.txt' (already lists it), then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(MITBIH_DIR, exist_ok=True)
    print(f"Downloading MIT-BIH Arrhythmia Database (48 records, ~100MB) to {MITBIH_DIR} ...")
    print("This calls PhysioNet's servers directly and requires internet access.")
    try:
        wfdb.dl_database("mitdb", dl_dir=MITBIH_DIR)
    except Exception as e:
        print(
            f"ERROR: download failed ({e}).\n"
            "Common causes: no internet access from this machine/session, or a firewall/proxy "
            "blocking physionet.org. You can also download manually from "
            "https://physionet.org/content/mitdb/1.0.0/ and place the .dat/.hea/.atr files "
            f"directly into {MITBIH_DIR}.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_files = len([f for f in os.listdir(MITBIH_DIR) if f.endswith(".hea")])
    print(f"Done. {n_files} records downloaded to {MITBIH_DIR}.")
    print("Next: run `python ml/extract_arrhythmia_features.py` to build ml/data/arrhythmia.csv.")


if __name__ == "__main__":
    main()
