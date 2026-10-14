# Dataset Sources — Qti π ML Models

This project trains on **real, publicly documented clinical/EEG datasets**, not synthetic data. Local copies audited in `C:\SIH` are listed where present; canonical sources (used to scale any dataset to 10,000+ records) are cited for every model.

## Cardiac

### 1. Heart Disease risk (RF classifier)
- **Local file found:** `C:\SIH\heart.csv` (918 rows, 11 features + target) — the standard combined **UCI Heart Failure Prediction dataset** (merger of Cleveland, Hungarian, Switzerland, VA Long Beach + Stalog heart datasets).
  Source: UCI Machine Learning Repository / Kaggle mirror "Heart Failure Prediction Dataset" — https://archive.ics.uci.edu/dataset/45/heart+disease
- **Scale-to-10k+ source (recommended for production training):** CDC **BRFSS 2015 "Heart Disease Health Indicators Dataset"** — 253,680 survey respondents, 21 features, binary heart-disease label.
  Source: CDC Behavioral Risk Factor Surveillance System, redistributed on Kaggle as `heart_disease_health_indicators_BRFSS2015.csv` — https://www.cdc.gov/brfss/annual_data/annual_2015.html
  License: US Government public data (CDC), Kaggle mirror is free to use.

### 2. Stroke risk (RF classifier)
- **Local file found:** `C:\SIH\healthcare-dataset-stroke-data.csv` (5,110 rows, 12 features) — Kaggle **"Stroke Prediction Dataset"** (author: fedesoriano), attributed to a de-identified clinical stroke-risk study.
  Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Scale-to-10k+ source:** combine with the **stroke indicator column of the CDC BRFSS 2015 dataset** above (same 253,680-row survey includes a self-reported stroke history field), giving a combined real-world sample well past 10,000 records.

### 3. Arrhythmia classification (KNN, 13-class beat types)
- **Local reference repo found:** `C:\SIH\Cardiac_Arrhythmia_Detection_And_Classification-main` (ECG image + calibration pipeline referencing Kaggle ECG datasets).
- **Canonical source:** **MIT-BIH Arrhythmia Database** (PhysioNet) — 48 half-hour two-lead ECG recordings, ~110,000 annotated beats across 15 rhythm classes, collapsed to Qti π's 13-class scheme.
  Source: https://physionet.org/content/mitdb/1.0.0/ (PhysioNet / ODC-By license, free with citation)
- **Alternative tabular source:** **PTB-XL ECG dataset** — 21,799 clinical 12-lead ECGs from 18,869 patients.
  Source: https://physionet.org/content/ptb-xl/1.0.3/

## Neuro (ear/scalp EEG)

### 4. Seizure risk tiering (RF classifier)
- **Local reference found:** `C:\SIH\Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG-main\eeg(seizure).ipynb` uses a 2,216-row, 668-feature band-power dataset (per-channel delta/theta/alpha/beta/gamma power + wavelet STD features) derived from scalp EEG, label `epileptic_`.
- **Canonical 10k+ source:** UCI **"Epileptic Seizure Recognition"** dataset — 11,500 rows, 178 EEG samples/segment, derived from the Bonn University EEG database (5 classes, collapsed to binary/tiered seizure risk).
  Source: https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
- **Real ear/scalp EEG waveform source (for realistic raw-signal simulation of ear-EEG channels):** **CHB-MIT Scalp EEG Database** (PhysioNet) — 23 pediatric subjects, ~980 hours of continuous EEG, 198 seizure events.
  Source: https://physionet.org/content/chbmit/1.0.0/

### 5. Tremor / Parkinson's & Alzheimer's risk context (used to inform feature design, not a separate served endpoint in v1)
- **Local reference found:** `C:\SIH\Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG-main\eeg_alz_dataset.csv` (real EEG band-power feature set used in that project's Alzheimer's LSTM notebook).

## Why local files are used directly where available
This Thrilia workspace has **no internet access**, so `ml/data_preprocessing.py` is written to:
1. Look for the above files at their real local paths first (`C:\SIH\heart.csv`, `C:\SIH\healthcare-dataset-stroke-data.csv`, `C:\SIH\Neuroear-...\eeg_alz_dataset.csv`), so the user immediately gets **genuine data** in the shipped repo.
2. Fall back to documented download instructions (`ml/download_data.py` prints the exact URLs above) for the larger BRFSS/PhysioNet/UCI-11.5k sources when the user wants to retrain at full 10,000+ scale.
3. Never silently substitute synthetic data for a real dataset without printing a loud `[SYNTHETIC FALLBACK]` warning to the console and the evaluation report JSON.
