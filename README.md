# Qti π

A cardiac + neuro (ear-EEG) health-companion prototype: a FastAPI backend, a static HTML/JS dashboard, and a pywebview desktop wrapper — all served from one local process, no build step required.


> ⚠️ **Screening aid prototype, not a diagnostic device.** 

---

## What it does

Qti π ingests vitals and ear-EEG readings, runs them through locally-trained ML models, and combines the results with rule-based and environmental signals into a single live risk score. If risk crosses a threshold (or the user presses the SOS button), it can email every configured emergency contact with a full risk breakdown and a live GPS location link. A multilingual voice assistant (Gemini Live) and a rule-based text fallback are both available for querying that risk data conversationally.

Everything degrades gracefully: SOS, voice, weather, and location are all optional integrations. Without any configuration, the app runs fully offline on the built-in rule-based assistant and mock weather data.

---

## Architecture

```
Qti π/
├── backend/            FastAPI app — vitals/neuro ingestion, risk engine, SOS, assistant, voice
│   ├── main.py          API routes + static file serving
│   ├── risk_engine.py   Combined cardiac + neuro + rule-based + environmental risk scoring
│   ├── quantum_engine.py Independent quantum-kernel screening pillar (loads ml/quantum/ model)
│   ├── assistant.py     Rule-based text assistant fallback
│   ├── gemini_live_ws.py Multilingual Gemini Live voice session (WebSocket)
│   ├── email_alert.py   SOS email dispatch via SMTP
│   ├── weather.py       OpenWeather integration + mock fallback
│   ├── database.py      SQLite persistence (qti_pi.db)
│   └── .env.example     Template for optional integration credentials
├── frontend/            Static dashboard (HTML/JS/CSS, no build step)
│   ├── index.html
│   ├── app.js
│   └── style.css
├── ml/                   Model training pipeline and datasets
│   ├── cardiac_models.py         Trains heart-disease, stroke, arrhythmia models
│   ├── neuro_models.py           Trains seizure-risk model
│   ├── skin_cancer_model.py      Skin lesion screening — DualStreamHybridQCNN (MobileNetV2 + 4-qubit PennyLane layer, HAM10000, 7-class)
│   ├── data_preprocessing.py     Dataset loading/cleaning + feature engineering
│   ├── rule_based.py             Rule-based risk heuristics (heat/dehydration/fatigue/fall)
│   ├── extract_arrhythmia_features.py / extract_nhanes_features.py
│   ├── download_mitbih.py / download_nhanes.py
│   ├── quantum/                  Independent QML pillar — 4 architectures, see below
│   │   ├── quantum_features.py           Angle-encoding pipeline; domain-split cardiac/neuro datasets
│   │   ├── quantum_model.py              Quantum Kernel SVM (QSVM) + feature-map comparison tool
│   │   ├── vqc_model.py                  Variational Quantum Classifier (VQC)
│   │   ├── qnn_model.py                   Hybrid Quantum Neural Network (QNN)
│   │   ├── ensemble_cardiac.py            Classical (RF) + QSVM stacked ensemble for cardiac
│   │   ├── classical_baseline_cardiac.py  Classical control experiment (proves signal is learnable)
│   │   └── ibm_quantum_client.py          Opt-in real IBM Quantum Cloud hardware verification
│   ├── data/                     Local datasets (MIT-BIH, NHANES, arrhythmia CSVs)
│   ├── saved_models/             Trained .joblib models + evaluation reports
│   └── DATASETS.md               Full dataset provenance and licensing
├── desktop_app.py        Patient Dashboard — pywebview wrapper that runs the same backend as a native desktop window
├── requirements.txt
├── HOW_TO_RUN.md          Step-by-step setup and run instructions
├── ML_MODELS_EXPLAINED.md Deep dive on the 3 classical ML models, tuning history, and Q&A defense
├── detailed_ML.md         Exhaustive reference for every model, classical + quantum (QSVM/VQC/QNN/ensemble)
└── README.md              This file
```

---

## Quick start

```bash
cd "C:\SIH\Qti π"
pip install -r requirements.txt

# (optional) train the models — pre-trained .joblib files already ship in ml/saved_models/
python ml/cardiac_models.py
python ml/neuro_models.py
python ml/quantum/quantum_model.py   # separate quantum-kernel screening pillar

# run the backend
uvicorn backend.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/** — the backend serves the dashboard directly.

To run the **Patient Dashboard** as a native desktop window instead of a browser:

```bash
python desktop_app.py
```

(Clinicians use the separate Clinician Console at `http://127.0.0.1:8000/doctor.html`. Patients, doctors and the mobile app (`/m/`) all sign up / sign in with an emailed 6-digit code and can reset a forgotten password from the sign-in screen.)

Recording a demo? See **[Demo video guide](#demo-video-guide--what-to-show)** below.

For full setup detail (datasets, optional integrations, env vars), see **[HOW_TO_RUN.md](HOW_TO_RUN.md)**.

---

## ML models

Three models, all trained on real, publicly documented datasets (no synthetic or fabricated data ships in the trained weights):

| Model | Predicts | Dataset | Algorithm | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| `arrhythmia_hgb_5class` | AAMI 5-class heartbeat rhythm | MIT-BIH Arrhythmia Database (109,494 beats) | HistGradientBoosting | 94.04% | — |
| `heart_disease_hgb` | Heart disease risk | UCI Heart Failure Prediction (918 rows) | HistGradientBoosting | 85.33% | 0.9154 |
| `stroke_hgb` | Stroke risk | CDC NHANES, multi-cycle (43,346 rows) | HistGradientBoosting | 69.78% | 0.8032 |

A 4th component, seizure risk (ear-EEG band-power features), runs as a Random Forest classifier trained on scalp-EEG-derived data (96.87% accuracy on the UCI Epileptic Seizure Recognition dataset).

A 5th, independent screening vertical covers **skin lesion classification**: a user-trained **DualStreamHybridQCNN** PyTorch model (a MobileNetV2 backbone feeding a classical residual branch and a 4-qubit PennyLane quantum branch with 3-round data re-uploading), ported from the original training notebook so the saved state dict loads 1:1. Trained on **HAM10000** (7-class: akiec, bcc, bkl, df, mel, nv, vasc), it takes an uploaded JPG/PNG lesion photo and returns a predicted class, confidence, all 7 class probabilities, and a risk banner (elevated/high concern for akiec/bcc/mel). The weights ship as `ml/saved_models/skin_cancer_dualstream_qcnn.pt` (a PyTorch state dict); on the notebook's own held-out test split it scores 86.12% accuracy and 0.9666 macro one-vs-rest ROC-AUC (see `skin_cancer_dualstream_qcnn_evaluation_report.json`).

Full methodology, tuning history (including the stroke model's 3-round tuning story), and a judge-defense Q&A live in **[ML_MODELS_EXPLAINED.md](ML_MODELS_EXPLAINED.md)**. The complete exhaustive reference — every classical and quantum model, architecture, dataset, and result — lives in **[detailed_ML.md](detailed_ML.md)**. Dataset sources and licenses are documented in **[ml/DATASETS.md](ml/DATASETS.md)**.

### Quantum ML screening (separate pillar)

Alongside the classical models above, Qti π ships a **fully independent QML pillar** (`ml/quantum/`) — different code, different saved models, different endpoints, never merged with cardiac/neuro logic. Four architectures were built and compared:

1. **Quantum Kernel SVM (QSVM)** — real per-row features (`heart_rate`, `power_alpha` on the combined domain, or the full 8 real MIT-BIH features on the cardiac-only domain) → `StandardScaler` → `PCA(N_QUBITS)` → tanh-bounded rotation angles → a Qiskit `ZZFeatureMap` → a quantum kernel `K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²` computed via exact `Statevector` simulation → `SVC(kernel="precomputed")`. Combined domain: 0.625 accuracy / 0.568 ROC-AUC. Cardiac-only: near/below chance (0.49–0.57 AUC).
2. **Variational Quantum Classifier (VQC)** — a from-scratch trainable circuit (RY/RZ encoding + entangling ansatz), optimized by numerical gradient descent. Strongest quantum result in the project on the **neuro domain: 0.866 ROC-AUC**; cardiac sits below chance (0.44 AUC).
3. **Hybrid Quantum Neural Network (QNN)** — reads out every qubit into a small classical linear layer, trained end-to-end. Neuro: 0.859 ROC-AUC; cardiac: ~0.53 ROC-AUC.
4. **Classical-quantum ensemble** — a Random Forest + QSVM blend (3-way split, no leakage) for cardiac, where RF does the heavy lifting: **0.891 ROC-AUC**, beating RF alone (0.891 vs 0.8905) on every metric even though QSVM alone is near chance.

An opt-in **"Verify on IBM Quantum"** action submits the QSVM circuit to real IBM Quantum Cloud hardware via `qiskit-ibm-runtime`, for real measured (shot-based) confirmation — never on the default screening path.

**Honest summary:** quantum models show a genuine advantage on EEG-derived seizure detection (neuro), while on cardiac arrhythmia the individual quantum models sit at/below chance — a diagnosed, known NISQ-era limitation (small qubit counts on a heterogeneous label), not a bug — and the cardiac ensemble recovers classical-level accuracy with Random Forest doing the heavy lifting. Full details, including the leakage bug that was found and fixed in the feature pipeline, are in **[detailed_ML.md](detailed_ML.md)**.

See **HOW_TO_RUN.md, step 3b** for training and setting up `IBM_QUANTUM_TOKEN`.

---

## API summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/vitals` | Ingest cardiac/general vitals |
| GET | `/vitals/history` | Vitals history (dashboard KPIs) |
| POST | `/neuro-reading` | Ingest ear-EEG band-power / tremor reading, runs seizure model |
| GET | `/neuro-reading/history` | Neuro reading history |
| GET | `/risk-score` | Combined cardiac + neuro + rule-based + environmental risk (`?lat=&lon=`) |
| GET | `/risk-score/history` | Risk-score trend history |
| GET | `/alerts` | Active alerts derived from current risk (`?lat=&lon=`) |
| POST | `/sos` | Trigger SOS: narration + SMTP email to every emergency contact, with live location + risk breakdown |
| GET / POST / DELETE | `/emergency-contacts` | List / add / remove emergency contacts |
| POST | `/assistant/query` | Text Q&A grounded in live risk/alerts |
| WS | `/ws/voice` | Multilingual Gemini Live voice session; falls back cleanly if `GEMINI_API_KEY` isn't set |
| POST | `/quantum-screening` | Quantum-kernel healthy/unhealthy screening + cardiac/neuro leaning |
| GET | `/quantum-screening/history` | Quantum screening history |
| POST | `/quantum-screening/verify-ibm` | Opt-in real IBM Quantum Cloud hardware verification |
| POST | `/predict-skin` | Skin lesion screening — upload a JPG/PNG, runs the DualStreamHybridQCNN model, returns class/confidence/risk |
| GET | `/predict-skin/history` | Skin screening history |
| GET | `/` | Health check + serves the dashboard |

---

## Optional integrations

All configured via `backend/.env` (copy from `backend/.env.example`). Every integration is opt-in — nothing else breaks if any of these are left blank.

- **Gemini Live voice** (`GEMINI_API_KEY`) — real-time multilingual voice chat (Hindi, English, Bengali, Tamil, Telugu, Marathi, and more), auto-detecting the spoken language.
- **SOS email (SMTP)** (`SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, etc.) — emails every emergency contact with narration, risk breakdown, and a live Google Maps location link when SOS is triggered.
- **Weather** (`OPENWEATHER_API_KEY`, `MOCK_CITY`) — real weather data for environmental risk context; falls back to mock data for `MOCK_CITY` if unset.
- **Live location** — no configuration needed; the browser requests permission on first load.
- **IBM Quantum Cloud verification** (`IBM_QUANTUM_TOKEN`) — enables the opt-in "Verify on IBM Quantum" action on the Quantum Screening tab, which submits the screening's circuit to real IBM quantum hardware. Regular quantum screening works without it (local simulation).

---

## Tech stack

- **Backend**: FastAPI, Pydantic, SQLite
- **ML**: scikit-learn, pandas, numpy, scipy, joblib, wfdb (for MIT-BIH/PTB-XL ECG parsing)
- **Deep learning (skin screening)**: PyTorch + PennyLane (DualStreamHybridQCNN: MobileNetV2 backbone + 4-qubit quantum layer), Pillow (image preprocessing)
- **Quantum ML**: qiskit (ZZFeatureMap + Statevector kernel simulation), qiskit-ibm-runtime (real IBM Quantum Cloud hardware verification)
- **Frontend**: static HTML/JS/CSS, no build step
- **Desktop**: pywebview
- **Voice**: google-genai (Gemini Live)

See `requirements.txt` for exact version constraints.
