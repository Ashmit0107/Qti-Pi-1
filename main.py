"""
backend/main.py — Qti pi FastAPI backend.

Endpoints:
  POST /vitals               — ingest cardiac/general sensor readings
  POST /neuro-reading          — ingest ear-EEG band-power / tremor readings, runs seizure model
  GET  /risk-score              — combined cardiac + neuro + rule-based + environmental risk
  GET  /alerts                   — active alerts derived from the current risk score
  POST /sos                       — trigger SOS narration (+ optional email hook)
  GET/POST/DELETE /emergency-contacts — manage emergency contacts
  POST /assistant/query            — text Q&A grounded in live risk/alerts
  WS   /ws/voice                    — voice-assistant stub (see docstring below)
  GET  /vitals/history, /neuro-reading/history, /risk-score/history — for dashboard charts
  static /                           — serves frontend/

Adapted from Ashmit0107/Health-Companion's backend/main.py (audited at
C:\\SIH\\health_companion\\backend\\main.py), extended with the neuro
vertical inspired by Namith22/Neuroear-AI-Powered-Neurological-Diagnosis-via-Ear-EEG.

Run:
    cd "C:\\SIH\\Qti π"
    pip install -r requirements.txt
    uvicorn backend.main:app --reload --port 8000
"""
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))

import database as db  # noqa: E402
import weather  # noqa: E402
import risk_engine  # noqa: E402
import assistant  # noqa: E402
import email_alert  # noqa: E402
import gemini_live_ws  # noqa: E402
from neuro_models import extract_band_powers  # noqa: E402

db.init_db()

app = FastAPI(title="Qti pi API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # prototype only — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Schemas ----------------

class VitalsIn(BaseModel):
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    body_temp: Optional[float] = None
    activity: Optional[str] = None
    sleep_hours: Optional[float] = None
    accel_peak_g: Optional[float] = None
    accel_post_variance_g: Optional[float] = None
    bp_systolic: Optional[float] = None
    age: Optional[float] = None
    sex: Optional[int] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    qrs: Optional[float] = None
    qt: Optional[float] = None
    t_wave: Optional[float] = None
    pr: Optional[float] = None
    p_wave: Optional[float] = None


class NeuroReadingIn(BaseModel):
    # Either send pre-computed band powers...
    power_delta: Optional[float] = None
    power_theta: Optional[float] = None
    power_alpha: Optional[float] = None
    power_beta: Optional[float] = None
    power_gamma: Optional[float] = None
    # ...or a raw single-channel signal + sample rate, and the backend extracts them.
    raw_signal: Optional[list[float]] = None
    sample_rate_hz: Optional[float] = 256.0
    # Tremor-specific fields (from accelerometer/gyroscope on the ear-worn device)
    dominant_tremor_freq_hz: Optional[float] = None
    tremor_amplitude_g: Optional[float] = None


class SosIn(BaseModel):
    reason: Optional[str] = "Manual SOS button pressed"
    lat: Optional[float] = None
    lon: Optional[float] = None


class EmergencyContactIn(BaseModel):
    name: str
    phone: str
    relation: Optional[str] = ""
    email: Optional[str] = ""


class AssistantQueryIn(BaseModel):
    query: str
    lat: Optional[float] = None
    lon: Optional[float] = None


# ---------------- Root / health ----------------
# NOTE: this is intentionally NOT registered at "/" — StaticFiles is mounted
# at "/" below to serve the dashboard's index.html, and an explicit @app.get("/")
# route would shadow it (FastAPI matches explicit routes before mounts), so
# the browser would see this JSON instead of the dashboard. Health check lives
# at /api/status instead.
@app.get("/api/status")
def root():
    return {"status": "ok", "service": "Qti pi API"}


# ---------------- Vitals (cardiac / general) ----------------

@app.post("/vitals")
def post_vitals(payload: VitalsIn):
    rid = db.insert_vitals(payload.dict())
    return {"ok": True, "id": rid}


@app.get("/vitals/history")
def get_vitals_history(limit: int = 50):
    return {"history": db.get_vitals_history(limit)}


# ---------------- Neuro readings (ear-EEG) ----------------

@app.post("/neuro-reading")
def post_neuro_reading(payload: NeuroReadingIn):
    data = payload.dict(exclude={"raw_signal", "sample_rate_hz"})
    if payload.raw_signal:
        bands = extract_band_powers(payload.raw_signal, payload.sample_rate_hz or 256.0)
        data.update(bands)
    # Compute a quick seizure-risk snapshot for this single reading so the
    # frontend can show an immediate tier without waiting for /risk-score.
    from risk_engine import _predict_seizure  # local import avoids circular import at module load
    seizure = _predict_seizure(data)
    data["seizure_risk_score"] = seizure.get("confidence")
    data["seizure_risk_tier"] = seizure.get("risk_level")
    data["model_source"] = "neuro_seizure_rf"
    rid = db.insert_neuro_reading(data)
    return {"ok": True, "id": rid, "seizure_risk": seizure}


@app.get("/neuro-reading/history")
def get_neuro_history(limit: int = 50):
    return {"history": db.get_neuro_history(limit)}


# ---------------- Risk score / alerts ----------------

@app.get("/risk-score")
def get_risk_score(city: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None):
    vitals = db.get_latest_vitals() or {}
    neuro = db.get_latest_neuro_reading() or {}
    environment = weather.get_environment(city, lat, lon)
    risk = risk_engine.compute_risk(vitals, neuro, environment)
    db.insert_risk_snapshot(risk["overall"]["score"], risk["overall"]["risk_level"], risk)
    return risk


@app.get("/risk-score/history")
def get_risk_history(limit: int = 50):
    return {"history": db.get_risk_history(limit)}


@app.get("/alerts")
def get_alerts(city: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None):
    vitals = db.get_latest_vitals() or {}
    neuro = db.get_latest_neuro_reading() or {}
    environment = weather.get_environment(city, lat, lon)
    risk = risk_engine.compute_risk(vitals, neuro, environment)
    alerts = risk_engine.build_alerts(risk, environment)
    return {"alerts": alerts, "count": len(alerts)}


# ---------------- SOS ----------------

@app.post("/sos")
def post_sos(payload: SosIn):
    vitals = db.get_latest_vitals() or {}
    neuro = db.get_latest_neuro_reading() or {}
    contacts = db.list_emergency_contacts()
    narration = assistant.narrate_sos(payload.reason, vitals, contacts)
    rid = db.insert_sos(payload.reason, str(vitals), narration)

    # Full risk/condition breakdown (cardiac + neuro/seizure + rule-based) for
    # the SOS email, using the SOS payload's own live lat/lon (captured by the
    # browser at SOS time) so the email reflects real-time weather + location
    # at the user's actual emergency location.
    environment = weather.get_environment(None, payload.lat, payload.lon)
    risk = risk_engine.compute_risk(vitals, neuro, environment)

    email_statuses = email_alert.send_sos_emails(
        contacts, payload.reason, narration, risk, lat=payload.lat, lon=payload.lon
    )
    return {
        "ok": True,
        "id": rid,
        "narration": narration,
        "contacts": contacts,
        "email_statuses": email_statuses,
        "location": {"lat": payload.lat, "lon": payload.lon},
    }


# ---------------- Emergency contacts (multi-contact CRUD) ----------------

@app.get("/emergency-contacts")
def get_emergency_contacts():
    return {"contacts": db.list_emergency_contacts()}


@app.post("/emergency-contacts")
def post_emergency_contacts(payload: EmergencyContactIn):
    if not payload.name.strip() or not payload.phone.strip():
        raise HTTPException(status_code=400, detail="name and phone are required")
    contact = db.add_emergency_contact(
        payload.name.strip(), payload.phone.strip(),
        (payload.relation or "").strip(), (payload.email or "").strip(),
    )
    return {"ok": True, "contact": contact}


@app.delete("/emergency-contacts/{contact_id}")
def delete_emergency_contact(contact_id: int):
    if not db.delete_emergency_contact(contact_id):
        raise HTTPException(status_code=404, detail="contact not found")
    return {"ok": True}


# ---------------- Assistant ----------------

@app.post("/assistant/query")
def post_assistant_query(payload: AssistantQueryIn):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    vitals = db.get_latest_vitals() or {}
    neuro = db.get_latest_neuro_reading() or {}
    environment = weather.get_environment(None, payload.lat, payload.lon)
    risk = risk_engine.compute_risk(vitals, neuro, environment)
    alerts = risk_engine.build_alerts(risk, environment)
    context = {"risk_score": risk, "alerts": alerts}
    response_text = assistant.answer_query(payload.query, context)
    db.log_assistant(payload.query, response_text)
    return {"response": response_text}


# ---------------- Voice stub ----------------

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """Real full-duplex Gemini Live voice session (see gemini_live_ws.py).
    Grounds the assistant in the same live risk/alerts context as
    /assistant/query (cardiac + neuro/seizure + rule-based + environmental).
    lat/lon are passed as query params on the WebSocket URL
    (ws://.../ws/voice?lat=..&lon=..) since a WS handshake has no JSON body —
    the frontend attaches the browser's live Geolocation coords there so the
    voice assistant is grounded in real-time weather/location too. If
    GEMINI_API_KEY isn't configured, gemini_live_ws sends a clear error frame
    and closes; the frontend falls back to the text /assistant/query path."""
    try:
        lat = float(websocket.query_params.get("lat")) if websocket.query_params.get("lat") else None
        lon = float(websocket.query_params.get("lon")) if websocket.query_params.get("lon") else None
    except (TypeError, ValueError):
        lat, lon = None, None
    vitals = db.get_latest_vitals() or {}
    neuro = db.get_latest_neuro_reading() or {}
    environment = weather.get_environment(None, lat, lon)
    risk = risk_engine.compute_risk(vitals, neuro, environment)
    alerts = risk_engine.build_alerts(risk, environment)
    context = {"risk_score": risk, "alerts": alerts}
    await gemini_live_ws.handle_voice_ws(websocket, context)



_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
