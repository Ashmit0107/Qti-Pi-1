"""
email_alert.py — sends a REAL SOS email via SMTP to every configured
emergency contact. Ported from Ashmit0107/Health-Companion's backend/email_alert.py
(audited at C:\\SIH\\health_companion\\backend\\email_alert.py), extended to
report Qti pi's neuro/seizure risk vertical alongside the cardiac/rule-based one.

This performs a genuine SMTP send (not a simulation/log-only stub): with
valid credentials it will actually deliver an email through your SMTP
provider. Requires, set in backend/.env:

    SMTP_HOST=smtp.gmail.com          # or your provider's SMTP host
    SMTP_PORT=587                     # 587 = STARTTLS, 465 = implicit TLS
    SMTP_USER=you@example.com         # login/sender account
    SMTP_PASSWORD=your_app_password   # Gmail: use an App Password, not your normal password
    SMTP_FROM=you@example.com         # optional, defaults to SMTP_USER
    SMTP_FROM_NAME=Qti pi             # optional display name

The recipients are the emergency contacts' email addresses (Settings tab).
If SMTP isn't configured, or no contact/contact email is set, send_sos_email()
returns a clear "skipped: ..." status instead of silently pretending to
succeed — it never raises, so a mail-server outage never blocks the /sos
endpoint from returning the SOS narration to the app.
"""
import os
import smtplib
import ssl
from email.message import EmailMessage

SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = os.environ.get("SMTP_PORT", "587").strip()
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip() or SMTP_USER
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "Qti pi SOS").strip()


def _fmt_result(label: str, r: dict | None) -> str:
    if not r:
        return f"{label}: Unavailable"
    level = r.get("risk_level", "Unknown")
    reason = r.get("reason") or r.get("label") or ""
    return f"{label}: {level}" + (f" — {reason}" if reason else "")


def build_risk_report_text(risk: dict) -> str:
    """Plain-text breakdown of every condition in the risk payload (cardiac,
    neuro/seizure, and rule-based) for the email body."""
    if not risk:
        return "(No risk data available yet — no vitals/neuro readings have been submitted this session.)"
    lines = []
    overall = risk.get("overall", {})
    lines.append(f"OVERALL RISK: {overall.get('risk_level', 'Unknown')} (score {overall.get('score', '?')}/100)")
    lines.append(
        "NOTE: any '% confidence' figures below are the ML model's own "
        "certainty in the label it picked, based on training-data patterns — "
        "NOT a clinical probability that the person actually has the condition. "
        "Treat this as a screening aid alongside the narration and vitals below, "
        "not a diagnosis."
    )
    lines.append("")
    cardiac = risk.get("cardiac", {})
    lines.append(_fmt_result("Cardiac (ECG classification)", cardiac))
    lines.append(_fmt_result("Body Temperature", risk.get("body_temperature")))
    lines.append(_fmt_result("Heat Stress", risk.get("heat_stress")))
    lines.append(_fmt_result("Dehydration", risk.get("dehydration")))
    lines.append(_fmt_result("Fatigue", risk.get("fatigue")))
    lines.append(_fmt_result("Fall Detection", risk.get("fall")))
    # Neuro vertical — Qti pi addition beyond Health-Companion.
    neuro = risk.get("neuro", {}) or risk.get("seizure", {})
    if neuro:
        lines.append(_fmt_result("Neuro (Seizure Risk)", neuro))
    tremor = risk.get("tremor")
    if tremor:
        lines.append(_fmt_result("Tremor", tremor))
    disease = risk.get("disease_risk", {})
    if disease:
        lines.append(
            f"Disease Risk (heart/stroke): {disease.get('risk_level', 'Unknown')} — "
            f"{disease.get('label', '')}".rstrip(" —")
        )
    env = risk.get("environment", {})
    if env:
        lines.append("")
        lines.append(
            f"Environment (at time of alert): {env.get('temp_c', '?')}C, "
            f"{env.get('humidity_pct', '?')}% humidity, AQI {env.get('aqi', '?')}"
        )
    return "\n".join(lines)


def build_location_text(lat: float | None, lon: float | None) -> str:
    if lat is None or lon is None:
        return "Live location: not available (user denied/no GPS fix at SOS time)."
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    return f"Live location: {lat}, {lon}\nGoogle Maps: {maps_link}"


def send_sos_email(contact: dict | None, reason: str, narration: str, risk: dict,
                    lat: float | None = None, lon: float | None = None) -> str:
    """Returns a short human-readable status string. Never raises — the
    caller (/sos endpoint) must always get a response even if the email
    itself fails, since SOS narration/logging must not be blocked by an
    external mail-server outage.
    """
    if not contact or not contact.get("email"):
        return "skipped: no emergency contact email configured (set one in Settings)"

    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        return "skipped: SMTP credentials not configured in backend/.env"

    try:
        port = int(SMTP_PORT)
    except ValueError:
        port = 587

    subject = f"[Qti pi] SOS Alert — {contact.get('name', 'Emergency Contact')}"
    body = (
        f"An SOS was triggered on the Qti pi app.\n\n"
        f"Reason: {reason}\n\n"
        f"--- Assistant narration ---\n{narration}\n\n"
        f"--- Risk & condition breakdown ---\n{build_risk_report_text(risk)}\n\n"
        f"--- Location ---\n{build_location_text(lat, lon)}\n\n"
        f"This is an automated message from the Qti pi prototype."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM}>"
    msg["To"] = contact["email"]
    msg.set_content(body)

    try:
        if port == 465:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, port, context=context, timeout=15) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, port, timeout=15) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(msg)
        return f"sent: email delivered to {contact['email']}"
    except Exception as e:
        return f"failed: {e}"


def send_sos_emails(contacts: list[dict], reason: str, narration: str, risk: dict,
                     lat: float | None = None, lon: float | None = None) -> list[str]:
    """Notify EVERY configured emergency contact (multi-contact SOS).
    Returns one "Name: status" string per contact so the caller/UI can show
    exactly who was (or wasn't) successfully notified."""
    if not contacts:
        return ["skipped: no emergency contacts configured (add one in Settings)"]
    statuses = []
    for c in contacts:
        name = c.get("name") or "Unnamed contact"
        status = send_sos_email(c, reason, narration, risk, lat=lat, lon=lon)
        statuses.append(f"{name}: {status}")
    return statuses
