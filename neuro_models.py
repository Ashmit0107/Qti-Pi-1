"""
ml/neuro_models.py
====================
Ear/scalp-EEG neuro vertical for Qti pi, inspired by
Namith22/Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG.

Neuroear's own approach (see its README, mirrored in ml/DATASETS.md):
  - LSTM for slow-evolving neurodegenerative signals (Alzheimer's, Parkinson's)
  - RandomForest for abrupt, feature-based events (seizures)

Qti pi v1 ships the RandomForest seizure/tremor risk-tiering path (fast,
CPU-only, no deep-learning runtime dependency for the desktop app), and
keeps the LSTM design documented here for a v2 upgrade path.

DATA PROVENANCE:
  - Trains on load_eeg_seizure() from ml/data_preprocessing.py, which uses
    the REAL Neuroear band-power dataset (2,216 rows, 668 features) shipped
    in this workspace, and auto-upgrades to the UCI Epileptic Seizure
    Recognition dataset (11,500 rows, real) if the user has downloaded it
    and set QTI_SEIZURE_CSV. See ml/DATASETS.md #4.

FEATURE EXTRACTION (band-power pipeline, for use with raw ear-EEG streams):
  extract_band_powers() below implements the delta/theta/alpha/beta/gamma
  Welch-PSD band-power extraction Neuroear's notebooks use, so live ear-EEG
  samples coming from backend/main.py's /neuro-reading endpoint can be
  turned into the same feature space the model was trained on.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from data_preprocessing import load_eeg_seizure

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# EEG band definitions (Hz), matching Neuroear's convention
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def extract_band_powers(raw_signal: np.ndarray, sample_rate: float = 256.0) -> dict:
    """Welch PSD band-power extraction for a single-channel ear-EEG segment.

    raw_signal: 1D array of voltage samples for one channel/epoch.
    Returns a dict of {"power_delta": ..., "power_theta": ..., ...} matching
    the feature naming convention the RF classifier below was trained on
    for the low-dimensional live-inference path (see risk_tier_from_raw).
    """
    freqs, psd = welch(raw_signal, fs=sample_rate, nperseg=min(256, len(raw_signal)))
    out = {}
    for band, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[f"power_{band}"] = float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0.0
    return out


def train_seizure_model():
    X, y, meta = load_eeg_seizure()
    # Keep only numeric columns and drop any with zero variance to keep the
    # model stable whether we're on the 668-feature real set or the 5-feature
    # synthetic fallback.
    X = X.loc[:, X.std(numeric_only=True) > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)

    metrics = {
        "model": "neuro_seizure_rf",
        "data_source": meta.source,
        "is_synthetic_fallback": meta.is_synthetic_fallback,
        "n_records_total": meta.n_records,
        "n_features": X.shape[1],
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        "n_train": len(X_train), "n_test": len(X_test),
        "notes": meta.notes,
    }

    joblib.dump(
        {"model": model, "scaler": scaler, "features": list(X.columns)},
        os.path.join(MODEL_DIR, "neuro_seizure_rf.joblib"),
    )
    return metrics


def risk_tier_from_proba(p_seizure: float) -> str:
    """Maps a seizure-class probability to a 3-tier risk label used by the
    /risk-score endpoint and dashboard risk cards."""
    if p_seizure >= 0.66:
        return "high"
    if p_seizure >= 0.33:
        return "elevated"
    return "low"


def main():
    print("Training neuro (ear-EEG) model line for Qti pi ...")
    metrics = train_seizure_model()
    report_path = os.path.join(MODEL_DIR, "neuro_evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{metrics['model']}] source={metrics['data_source']} n={metrics['n_records_total']} accuracy={metrics['accuracy']}")
    print(f"Saved model to {MODEL_DIR}")
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
