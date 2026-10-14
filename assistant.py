"""
backend/assistant.py — Qti pi

Lightweight text assistant. Adapted from Ashmit0107/Health-Companion's
backend/assistant.py (audited at C:\\SIH\\health_companion\\backend\\assistant.py):
if GEMINI_API_KEY is set, calls Gemini; otherwise falls back to a
deterministic rule-based explainer grounded in the live risk/alerts context,
so the assistant chat always works offline / without an API key.
"""
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

SYSTEM_PROMPT = (
    "You are the Qti pi assistant embedded in a personal cardiac + neuro "
    "safety wearable app. You explain the user's current vitals, EEG-derived "
    "neuro readings, risk scores, and alerts in plain, reassuring language, "
    "and give short, practical safety guidance. You are not a doctor — for "
    "anything serious, tell the user to seek medical help. IMPORTANT: the "
    "'confidence' percentages attached to heart-disease/stroke/seizure screens "
    "are the ML model's own certainty in its prediction (how sure it is about "
    "the label it picked, based on patterns in its training data) — they are "
    "NOT the clinical probability that the user actually has the condition. "
    "Never phrase a confidence percentage as 'you have an X% chance of Y'; "
    "instead say things like 'the screening model flagged this as elevated "
    "risk, with X% confidence in that call'. Keep responses to 2-4 sentences."
)


def _rule_based_fallback(query: str, context: dict) -> str:
    q = query.lower()
    risk = context.get("risk_score", {})

    if "heat" in q:
        hs = risk.get("heat_stress", {})
        return f"Your heat-stress risk is currently {hs.get('risk_level', 'unknown')} ({hs.get('reason', 'no data yet')}). Stay hydrated and avoid direct sun between 12-4 PM."
    if "dehydrat" in q or "water" in q or "thirst" in q:
        dh = risk.get("dehydration", {})
        return f"Dehydration risk is {dh.get('risk_level', 'unknown')} ({dh.get('reason', 'no data yet')}). Drink water regularly, even before you feel thirsty."
    if "heart" in q or "cardiac" in q:
        hr = risk.get("heart_risk", {})
        conf = hr.get("confidence")
        conf_note = f" (the screening model is {conf*100:.0f}% confident in this call — not a clinical probability)" if conf is not None else ""
        return f"Your latest heart-disease screening flagged {hr.get('risk_level', 'unknown')} risk ({hr.get('label', 'no data yet')}){conf_note}. If you feel chest pain or palpitations, seek medical attention."
    if "stroke" in q:
        sr = risk.get("stroke_risk", {})
        conf = sr.get("confidence")
        conf_note = f" (the screening model is {conf*100:.0f}% confident in this call — not a clinical probability)" if conf is not None else ""
        return f"Your stroke-risk screening flagged {sr.get('risk_level', 'unknown')} risk ({sr.get('label', 'no data yet')}){conf_note}."
    if "seizure" in q or "eeg" in q or "neuro" in q:
        ez = risk.get("seizure_risk", {})
        conf = ez.get("confidence")
        conf_note = f" The model estimates a {conf*100:.0f}% seizure probability from this reading's band-power pattern (a screening estimate, not a diagnosis)." if conf is not None else ""
        return f"Your latest EEG-derived seizure risk tier is {ez.get('risk_level', 'unknown')} ({ez.get('label', 'no reading yet')}).{conf_note} If you notice unusual movements or confusion, seek medical help."
    if "tremor" in q or "parkinson" in q:
        tr = risk.get("tremor", {})
        return f"Tremor risk is {tr.get('risk_level', 'unknown')} ({tr.get('reason', 'no data yet')})."
    if "fall" in q:
        fl = risk.get("fall", {})
        return f"Fall-detection status: {fl.get('risk_level', 'unknown')} ({fl.get('reason', 'no data yet')})."
    if "sos" in q or "emergency" in q:
        return "You can press the SOS button on the dashboard at any time to notify your saved emergency contacts with your current vitals and location."
    overall = risk.get("overall", {})
    return f"Your overall risk level is currently {overall.get('risk_level', 'unknown')} (score {overall.get('score', 'n/a')}/100). Ask me about heart, stroke, seizure, tremor, heat, dehydration, fatigue, or fall risk for more detail."


def _gemini_call(query: str, context: dict) -> str | None:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai  # optional dependency
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nUser question: {query}"
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception:
        return None  # silently fall back to rule-based


def answer_query(query: str, context: dict) -> str:
    gemini_answer = _gemini_call(query, context)
    return gemini_answer if gemini_answer else _rule_based_fallback(query, context)


def narrate_sos(reason: str, vitals: dict, contacts: list[dict]) -> str:
    names = ", ".join(c.get("name", "contact") for c in contacts) if contacts else "no saved contacts"
    hr = vitals.get("heart_rate", "unknown")
    temp = vitals.get("body_temp", "unknown")
    return (
        f"SOS triggered: {reason}. Latest vitals — heart rate {hr} bpm, body temperature {temp}C. "
        f"Notifying: {names}."
    )
