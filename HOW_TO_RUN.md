# HOW TO RUN — Qti π

Qti π is a cardiac + neuro (ear-EEG) health-companion prototype: a FastAPI
backend, a static HTML/JS dashboard, and a pywebview desktop wrapper — all
running from one live local server, no build step. It ports the full
feature set of [Ashmit0107/Health-Companion](https://github.com/Ashmit0107/Health-Companion)
(cardiac ML, SOS-via-SMTP, multilingual Gemini Live voice, emergency
contacts) and adds a neuro/ear-EEG vertical inspired by
[Namith22/Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG](https://github.com/Namith22/Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG).

---

## 1. Install dependencies

```bash
cd "C:\SIH\Qti π"
pip install -r requirements.txt
```

This installs FastAPI, scikit-learn, pandas, pywebview, and `google-genai`
(only used if you configure `GEMINI_API_KEY` — see step 4).

---

## 2. Get the datasets (optional but recommended)

Two real datasets are **already present** in `C:\SIH` and used automatically:

| Model | Local file | Rows | Source |
|---|---|---|---|
| Heart-disease risk | `C:\SIH\heart.csv` | 918 | UCI Heart Failure Prediction |
| Stroke risk | `C:\SIH\healthcare-dataset-stroke-data.csv` | 5,110 | Kaggle Stroke Prediction Dataset (fedesoriano) |
| Seizure risk | `C:\SIH\Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG-main\eeg(seizure).ipynb` data | 2,216 | Neuroear repo (scalp EEG band-power) |

To train at full **10,000+ record** scale with the canonical sources cited
to you, download these (all free, citation-only licenses) and drop them at
the paths `ml/DATASETS.md` documents:

- **CDC BRFSS 2015 "Heart Disease Health Indicators"** (253,680 rows) —
  https://www.cdc.gov/brfss/annual_data/annual_2015.html
  Cleaned mirror: https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- **MIT-BIH Arrhythmia Database** — https://physionet.org/content/mitdb/1.0.0/ (ODC-By license)
- **PTB-XL ECG Dataset** (21,799 ECGs) — https://physionet.org/content/ptb-xl/1.0.3/
- **UCI "Epileptic Seizure Recognition"** (11,500 rows) —
  https://archive.ics.uci.edu/dataset/388/epileptic+seizure+recognition
  Mirror: https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition
- **CHB-MIT Scalp EEG Database** — https://physionet.org/content/chbmit/1.0.0/

Full provenance, license notes, and exactly which column headers each
loader expects are in **`ml/DATASETS.md`**. If you skip this step, the app
still runs fine on the smaller real datasets already included — training
scripts print a `[SYNTHETIC FALLBACK]` warning only if a real file is
genuinely missing.

---

## 3. Train the ML models

```bash
python ml/cardiac_models.py    # heart-disease RF, stroke RF, arrhythmia KNN
python ml/neuro_models.py      # seizure-risk RF (band-power features)
```

Each script prints a JSON evaluation report and saves `.joblib` model files
to `ml/saved_models/`. Re-run any time you add a bigger dataset from step 2
— the backend hot-loads whatever's in `ml/saved_models/` on next request.

---

## 4. Configure optional integrations

Copy the example env file and fill in what you have:

```bash
copy backend\.env.example backend\.env
```

### Multilingual voice assistant (Gemini Live)
```
GEMINI_API_KEY=your_key_here
```
Get a key at https://aistudio.google.com/apikey. With this set, the
**Assistant tab → Start Voice Chat** button opens a real full-duplex voice
session: speak in Hindi, English, Bengali, Tamil, Telugu, Marathi, or any
other language, and Qti π auto-detects and replies in that same language,
grounded in your live cardiac + neuro + environmental risk data. **Without
a key**, voice chat shows a clear one-line error and the text assistant
(rule-based fallback) keeps working normally — nothing else breaks.

### SOS emails (SMTP)
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@example.com
SMTP_PASSWORD=your_gmail_app_password
SMTP_FROM=you@example.com
SMTP_FROM_NAME=Qti pi SOS
```
For Gmail, generate an **App Password** (not your normal password) at
https://myaccount.google.com/apppasswords. With SMTP configured, pressing
the **SOS button** emails *every* contact you've added under
**Settings → Emergency Contacts** with: the reason, an assistant narration,
a full risk breakdown (cardiac + neuro/seizure + heat/dehydration/fatigue/
fall), and your **live GPS location as a Google Maps link**. **Without
SMTP configured**, SOS still logs the event and shows the narration in the
app — it just returns a `skipped: SMTP credentials not configured` status
per contact instead of sending, and never blocks or errors out.

### Live location
No configuration needed — the browser will ask for location permission on
first load. It powers: the live-location badge in the topbar, the Google
Maps link attached to every SOS email, and grounding for weather/risk
context in `/risk-score`, `/assistant/query`, and voice chat.

### Weather (optional)
```
OPENWEATHER_API_KEY=your_key_here
MOCK_CITY=New Delhi
```
Leave blank to use built-in mock weather data for `MOCK_CITY`.

---

## 5. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/** in a browser — the backend serves the
dashboard directly, no separate frontend server needed.

---

## 6. Run the desktop app (optional)

Instead of a browser, open Qti π as a native window:

```bash
python desktop_app.py
```

This starts the same `uvicorn backend.main:app` server as a subprocess
(reusing one if already running) and opens a pywebview window pointing at
it. No build/bundle step — edit `frontend/*` and just refresh (Ctrl+R) or
restart the window; edit `backend/*.py` and restart.

---

## 7. API reference summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/vitals` | Ingest cardiac/general vitals |
| GET | `/vitals/history` | Vitals history (dashboard KPIs) |
| POST | `/neuro-reading` | Ingest ear-EEG band-power / tremor reading, runs seizure model |
| GET | `/neuro-reading/history` | Neuro reading history |
| GET | `/risk-score` | Combined cardiac + neuro + rule-based + environmental risk (`?lat=&lon=`) |
| GET | `/risk-score/history` | Risk-score trend history |
| GET | `/alerts` | Active alerts derived from current risk (`?lat=&lon=`) |
| POST | `/sos` | Trigger SOS: narration + **real SMTP email to every emergency contact**, with live location + full risk breakdown |
| GET | `/emergency-contacts` | List emergency contacts |
| POST | `/emergency-contacts` | Add an emergency contact (name, phone, relation, email) |
| DELETE | `/emergency-contacts/{id}` | Remove a contact |
| POST | `/assistant/query` | Text Q&A grounded in live risk/alerts (`lat`,`lon` in body) |
| WS | `/ws/voice` | **Real multilingual Gemini Live voice session** (`?lat=&lon=`); falls back cleanly with an error frame if `GEMINI_API_KEY` isn't set |
| GET | `/` | Health check + serves the dashboard |

---

## Notes

- This is a **screening aid prototype, not a diagnostic device** — always
  consult a medical professional for anything serious.
- All ML training happens **on your machine** — this repo ships code and
  documented data sources, not pre-trained weights, so every model's
  provenance is auditable (see `ml/DATASETS.md`).
- SOS/voice/location all degrade gracefully when unconfigured: the app is
  always usable offline, and each integration only activates once you add
  its credentials.
