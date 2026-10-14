"""
backend/risk_engine.py — Qti pi

Combines cardiac ML models + the new neuro (ear-EEG) ML model + rule-based
detectors + environmental data into a single risk-score payload. Adapted
from Ashmit0107/Health-Companion's backend/risk_engine.py (audited at
C:\\SIH\\health_companion\\backend\\risk_engine.py), extended with the
neuro vertical.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
from rule_based import (  # noqa: E402
    assess_heat_stress, assess_dehydration, assess_fatigue,
    assess_body_temperature, detect_fall, assess_neuro_tremor, RiskResult,
)
from neuro_models import risk_tier_from_proba  # noqa: E402

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml", "saved_models")

_model_cache: dict = {}


def _load(name: str):
    if name not in _model_cache:
        path = os.path.join(ML_DIR, name)
        _model_cache[name] = joblib.load(path) if os.path.exists(path) else None
    return _model_cache[name]


def _result_to_dict(r: RiskResult) -> dict:
    return {"risk_level": r.risk_level, "score": r.score, "reason": r.reason}


def _predict_heart_disease(vitals: dict) -> dict:
    bundle = _load("heart_disease_rf.joblib")
    if bundle is None:
        return {"risk_level": "Unknown", "label": "Model not trained yet", "confidence": None}
    model, features = bundle["model"], bundle["features"]
    row = {f: vitals.get(f, 0) for f in features}
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    try:
        proba = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])
        conf = float(max(proba))
    except Exception as e:
        return {"risk_level": "Unknown", "label": "Unavailable", "confidence": None, "error": str(e)}
    label = "Heart disease risk" if pred == 1 else "No significant risk detected"
    level = "HIGH" if (pred == 1 and conf > 0.7) else ("MODERATE" if pred == 1 else "None")
    return {"risk_level": level, "label": label, "confidence": round(conf, 3)}


def _predict_stroke(vitals: dict) -> dict:
    # NOTE: train_stroke() (ml/cardiac_models.py) now saves to
    # "stroke_final_model.joblib", not "stroke_rf.joblib" -- this was
    # silently loading a stale/older artifact left over from before the
    # 3-round tuning pass. "stroke_rf.joblib" is kept as a fallback only in
    # case a machine hasn't re-run training since this fix.
    bundle = _load("stroke_final_model.joblib") or _load("stroke_rf.joblib")
    if bundle is None:
        return {"risk_level": "Unknown", "label": "Model not trained yet", "confidence": None}
    model, features = bundle["model"], bundle["features"]
    row = {f: vitals.get(f, 0) for f in features}
    X = pd.DataFrame([[row[f] for f in features]], columns=features)
    try:
        proba = model.predict_proba(X)[0]
        pred = int(model.predict(X)[0])
        conf = float(max(proba))
    except Exception as e:
        return {"risk_level": "Unknown", "label": "Unavailable", "confidence": None, "error": str(e)}
    label = "Stroke risk" if pred == 1 else "No significant risk detected"
    level = "HIGH" if (pred == 1 and conf > 0.7) else ("MODERATE" if pred == 1 else "None")
    return {"risk_level": level, "label": label, "confidence": round(conf, 3)}


def _predict_arrhythmia(vitals: dict) -> dict:
    """5-class (AAMI N/S/V/F/Q) arrhythmia screen from the ECG-interval fields
    already collected on VitalsIn (heart_rate, qrs, pr, qt, t_wave, p_wave).
    This model was trained (see ml/cardiac_models.py:train_arrhythmia()) but
    was never actually called from the risk engine -- wiring it in here so a
    submitted vitals reading produces all three cardiac screens, not two.
    qtc_bazett/rr_ratio aren't collected by the simple vitals form; they
    default to 0 (missing), which the model treats like any other feature it
    wasn't given -- this is a best-effort screen from partial ECG data, not a
    full clinical arrhythmia read."""
    bundle = _load("arrhythmia_final_model.joblib") or _load("arrhythmia_knn.joblib")
    if bundle is None:
        return {"risk_level": "Unknown", "label": "Model not trained yet", "confidence": None}
    model, scaler, features = bundle["model"], bundle.get("scaler"), bundle["features"]
    field_map = {"qrs_duration": "qrs", "pr_interval": "pr", "qt_interval": "qt"}
    row = {f: vitals.get(field_map.get(f, f), 0) or 0 for f in features}
    try:
        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        X_in = scaler.transform(X) if scaler is not None else X
        proba = model.predict_proba(X_in)[0]
        classes = list(model.classes_)
        pred = classes[int(np.argmax(proba))]
        conf = float(max(proba))
    except Exception as e:
        return {"risk_level": "Unknown", "label": "Unavailable", "confidence": None, "error": str(e)}
    class_names = {
        "N": "Normal rhythm", "S": "Supraventricular ectopic beat pattern",
        "V": "Ventricular ectopic beat pattern", "F": "Fusion beat pattern",
        "Q": "Unknown/paced beat pattern",
    }
    label = class_names.get(pred, pred)
    if pred == "N":
        level = "None"
    elif pred in ("S", "Q"):
        level = "MODERATE" if conf > 0.6 else "LOW"
    else:  # V or F -- clinically the two groups worth flagging highest
        level = "HIGH" if conf > 0.6 else "MODERATE"
    return {"risk_level": level, "label": label, "confidence": round(conf, 3), "beat_class": pred}


def _predict_seizure(neuro: dict) -> dict:
    bundle = _load("neuro_seizure_rf.joblib")
    if bundle is None or not neuro:
        return {"risk_level": "Unknown", "label": "Model not trained yet / no reading", "confidence": None}
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    row = {f: neuro.get(f, 0) or 0 for f in features}
    try:
        X = pd.DataFrame([[row[f] for f in features]], columns=features)
        X_s = scaler.transform(X)
        proba = model.predict_proba(X_s)[0]
        p_seizure = float(proba[1]) if len(proba) > 1 else float(proba[0])
        tier = risk_tier_from_proba(p_seizure)
    except Exception as e:
        return {"risk_level": "Unknown", "label": "Unavailable", "confidence": None, "error": str(e)}
    level_map = {"high": "HIGH", "elevated": "MODERATE", "low": "None"}
    return {"risk_level": level_map[tier], "label": f"Seizure risk tier: {tier}", "confidence": round(p_seizure, 3)}


def compute_risk(vitals: dict, neuro: dict, environment: dict, baseline_hr: float = 70.0) -> dict:
    heart_rate = vitals.get("heart_rate") or 75
    body_temp = vitals.get("body_temp") or 37.0
    sleep_hours = vitals.get("sleep_hours") if vitals.get("sleep_hours") is not None else 7.0
    accel_peak = vitals.get("accel_peak_g") or 0.0
    accel_var = vitals.get("accel_post_variance_g") if vitals.get("accel_post_variance_g") is not None else 1.0

    temp_c = environment.get("temp_c", 32)
    humidity = environment.get("humidity_pct", 50)
    aqi = environment.get("aqi")

    heat = assess_heat_stress(temp_c, humidity, aqi)
    dehydration = assess_dehydration(heart_rate, temp_c, humidity)
    fatigue = assess_fatigue(heart_rate, baseline_hr, sleep_hours, recent_activity_low=False)
    fall = detect_fall(accel_peak, accel_var)
    body_temperature = assess_body_temperature(body_temp)
    tremor = assess_neuro_tremor(
        neuro.get("dominant_tremor_freq_hz") or 0.0, neuro.get("tremor_amplitude_g") or 0.0
    )

    heart_risk = _predict_heart_disease(vitals)
    stroke_risk = _predict_stroke(vitals)
    arrhythmia_risk = _predict_arrhythmia(vitals)
    seizure_risk = _predict_seizure(neuro)

    def level_weight(level: str) -> float:
        return {"HIGH": 1.0, "MODERATE": 0.6, "LOW": 0.3, "None": 0.0, "Unknown": 0.2}.get(level, 0.2)

    weights = [
        (heat.risk_level, 0.10),
        (dehydration.risk_level, 0.08),
        (fatigue.risk_level, 0.08),
        (fall.risk_level, 0.12),
        (body_temperature.risk_level, 0.10),
        (tremor.risk_level, 0.10),
        (heart_risk.get("risk_level", "Unknown"), 0.13),
        (stroke_risk.get("risk_level", "Unknown"), 0.11),
        (arrhythmia_risk.get("risk_level", "Unknown"), 0.11),
        (seizure_risk.get("risk_level", "Unknown"), 0.13),
    ]
    overall_score = sum(level_weight(lvl) * w for lvl, w in weights) * 100
    if overall_score >= 70:
        overall_level = "HIGH"
    elif overall_score >= 40:
        overall_level = "MODERATE"
    elif overall_score >= 15:
        overall_level = "LOW"
    else:
        overall_level = "None"

    return {
        "heart_risk": heart_risk,
        "stroke_risk": stroke_risk,
        "arrhythmia_risk": arrhythmia_risk,
        "seizure_risk": seizure_risk,
        "heat_stress": _result_to_dict(heat),
        "dehydration": _result_to_dict(dehydration),
        "fatigue": _result_to_dict(fatigue),
        "fall": _result_to_dict(fall),
        "body_temperature": _result_to_dict(body_temperature),
        "tremor": _result_to_dict(tremor),
        "overall": {"risk_level": overall_level, "score": round(overall_score, 1)},
        "environment": environment,
    }


def build_alerts(risk: dict, environment: dict) -> list[str]:
    alerts = []
    bt = risk["body_temperature"]
    if bt["risk_level"] in ("MODERATE", "HIGH"):
        alerts.append(f"Body Temperature Alert ({bt['risk_level']}): {bt['reason']}. Monitor closely; seek medical care if it rises further.")
    hs = risk["heat_stress"]
    if hs["risk_level"] in ("MODERATE", "HIGH"):
        alerts.append(f"Heat-Wave Advisory ({hs['risk_level']}): {hs['reason']}. Stay hydrated, avoid outdoor activity 12-4 PM.")
    aqi = environment.get("aqi")
    if aqi and aqi > 150:
        alerts.append(f"Air Quality Advisory: AQI {aqi} (Unhealthy). Limit outdoor exertion.")
    dh = risk["dehydration"]
    if dh["risk_level"] in ("MODERATE", "HIGH"):
        alerts.append(f"Dehydration Risk ({dh['risk_level']}): {dh['reason']}. Increase fluid intake.")
    fatigue = risk["fatigue"]
    if fatigue["risk_level"] in ("MODERATE", "HIGH"):
        alerts.append(f"Fatigue Warning ({fatigue['risk_level']}): {fatigue['reason']}. Consider resting.")
    fall = risk["fall"]
    if fall["risk_level"] == "HIGH":
        alerts.append(f"Possible Fall Detected: {fall['reason']}. SOS may be triggered automatically.")
    tremor = risk["tremor"]
    if tremor["risk_level"] in ("MODERATE", "HIGH"):
        alerts.append(f"Tremor Advisory ({tremor['risk_level']}): {tremor['reason']}. Consider a neurology consult.")
    hr = risk["heart_risk"]
    if hr.get("risk_level") in ("MODERATE", "HIGH"):
        alerts.append(f"Heart-Disease Risk Advisory ({hr.get('risk_level')}): {hr.get('label')} (confidence {hr.get('confidence')}).")
    sr = risk["stroke_risk"]
    if sr.get("risk_level") in ("MODERATE", "HIGH"):
        alerts.append(f"Stroke Risk Advisory ({sr.get('risk_level')}): {sr.get('label')} (confidence {sr.get('confidence')}).")
    ar = risk["arrhythmia_risk"]
    if ar.get("risk_level") in ("MODERATE", "HIGH"):
        alerts.append(f"Arrhythmia Advisory ({ar.get('risk_level')}): {ar.get('label')} (confidence {ar.get('confidence')}).")
    ez = risk["seizure_risk"]
    if ez.get("risk_level") in ("MODERATE", "HIGH"):
        alerts.append(f"Seizure Risk Advisory ({ez.get('risk_level')}): {ez.get('label')} (confidence {ez.get('confidence')}).")
    return alerts
