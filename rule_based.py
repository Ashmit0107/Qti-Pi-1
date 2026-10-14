"""
ml/rule_based.py — Qti pi

Ported directly from Ashmit0107/Health-Companion's ml/rule_based.py
(local copy audited at C:\\SIH\\health_companion\\ml\\rule_based.py).
Lightweight, explainable rule-based detectors for heat stress, dehydration,
fatigue, and falls — complementing the ML models. These are NOT black boxes:
every threshold is cited so a reviewer can verify the logic without needing
model internals.

Cited reference thresholds:
- Heat Index bands: US National Weather Service Heat Index chart
  (Caution 27-32C, Extreme Caution 32-39C, Danger 39-51C, Extreme Danger 51C+)
  https://www.weather.gov/safety/heat-index
- Dehydration proxy (no sweat/urine sensor available): resting tachycardia
  (HR > 100 bpm at rest, per AHA definition of tachycardia) combined with
  elevated ambient heat index.
  https://www.heart.org (tachycardia defined as resting HR > 100 bpm)
- Fatigue proxy: sustained elevated resting heart rate (>15% above the
  user's baseline) combined with low recent activity and short sleep
  duration (<6 hours).
- Fall detection: a sudden acceleration spike above 2.5 g followed by near-
  stillness (<0.3 g variance) for several seconds — thresholds consistent
  with widely cited wearable fall-detection studies (e.g. Bourke et al.,
  "A threshold-based fall-detection algorithm using a bi-axial gyroscope
  sensor", Medical Engineering & Physics, 2008).
- Fever bands: standard CDC/WHO adult fever definitions; >=40C core temp is
  a recognized heat-illness medical emergency threshold.

Qti pi additions beyond the Health-Companion original:
- assess_neuro_tremor(): a rule-based tremor-risk fallback for when the
  neuro ML model (ml/neuro_models.py) is unavailable or has low confidence,
  based on accelerometer-derived tremor frequency bands (4-6 Hz resting
  tremor per Parkinson's literature; Elble & Koller, "Tremor", 1990).
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskResult:
    risk_level: str      # "None" | "LOW" | "MODERATE" | "HIGH"
    score: float          # 0-100
    reason: str


def heat_index_celsius(temp_c: float, humidity_pct: float) -> float:
    """Approximate NWS heat index (Rothfusz regression), returns Celsius."""
    if temp_c < 27:
        return temp_c
    T = temp_c * 9 / 5 + 32
    R = humidity_pct
    HI = (
        -42.379 + 2.04901523 * T + 10.14333127 * R - 0.22475541 * T * R
        - 0.00683783 * T * T - 0.05481717 * R * R + 0.00122874 * T * T * R
        + 0.00085282 * T * R * R - 0.00000199 * T * T * R * R
    )
    return (HI - 32) * 5 / 9


def assess_heat_stress(temp_c: float, humidity_pct: float, aqi: Optional[int] = None) -> RiskResult:
    hi = heat_index_celsius(temp_c, humidity_pct)
    if hi < 27:
        level, score = "None", 5
    elif hi < 32:
        level, score = "LOW", 30
    elif hi < 39:
        level, score = "MODERATE", 60
    elif hi < 51:
        level, score = "HIGH", 85
    else:
        level, score = "HIGH", 100

    reason = f"Heat index {hi:.1f}C (NWS band)"
    if aqi is not None and aqi > 150:
        score = min(100, score + 10)
        reason += f", AQI {aqi} (Unhealthy) adds respiratory strain"
    return RiskResult(level, score, reason)


def assess_dehydration(heart_rate_bpm: float, temp_c: float, humidity_pct: float) -> RiskResult:
    hi = heat_index_celsius(temp_c, humidity_pct)
    tachycardic = heart_rate_bpm > 100
    if tachycardic and hi >= 32:
        return RiskResult("HIGH", 80, f"Resting HR {heart_rate_bpm}bpm >100 (tachycardia) during heat index {hi:.1f}C")
    if tachycardic:
        return RiskResult("MODERATE", 50, f"Resting HR {heart_rate_bpm}bpm >100 (tachycardia)")
    if hi >= 39:
        return RiskResult("MODERATE", 45, f"High heat index {hi:.1f}C without tachycardia — monitor fluid intake")
    return RiskResult("None", 10, "No dehydration indicators")


def assess_fatigue(resting_hr_bpm: float, baseline_hr_bpm: float, sleep_hours: float, recent_activity_low: bool) -> RiskResult:
    hr_elevated = baseline_hr_bpm > 0 and resting_hr_bpm > baseline_hr_bpm * 1.15
    short_sleep = sleep_hours < 6
    if hr_elevated and short_sleep:
        return RiskResult("HIGH", 75, f"Resting HR {resting_hr_bpm}bpm >15% above baseline {baseline_hr_bpm}bpm + only {sleep_hours}h sleep")
    if hr_elevated or short_sleep:
        reason = f"Resting HR {resting_hr_bpm}bpm >15% above baseline" if hr_elevated else f"Only {sleep_hours}h sleep (<6h)"
        return RiskResult("MODERATE", 45, reason)
    if recent_activity_low and short_sleep:
        return RiskResult("LOW", 25, "Low activity + short sleep")
    return RiskResult("None", 10, "No fatigue indicators")


def detect_fall(peak_accel_g: float, post_impact_variance_g: float) -> RiskResult:
    if peak_accel_g >= 2.5 and post_impact_variance_g < 0.3:
        return RiskResult("HIGH", 95, f"Impact {peak_accel_g}g followed by stillness (var={post_impact_variance_g}) — possible fall")
    if peak_accel_g >= 2.5:
        return RiskResult("MODERATE", 40, f"Impact {peak_accel_g}g detected but movement continued — likely not a fall")
    return RiskResult("None", 0, "No impact detected")


def assess_body_temperature(body_temp_c: float) -> RiskResult:
    if body_temp_c >= 39.5:
        return RiskResult("HIGH", 90, f"Body temp {body_temp_c}C — high fever / possible heat illness, seek care")
    if body_temp_c >= 38.1:
        return RiskResult("MODERATE", 55, f"Body temp {body_temp_c}C — fever")
    if body_temp_c >= 37.3:
        return RiskResult("LOW", 30, f"Body temp {body_temp_c}C — low-grade fever")
    if body_temp_c < 35.0:
        return RiskResult("HIGH", 85, f"Body temp {body_temp_c}C — possible hypothermia")
    return RiskResult("None", 5, f"Body temp {body_temp_c}C — normal range")


def assess_neuro_tremor(dominant_freq_hz: float, amplitude_g: float) -> RiskResult:
    """Qti pi neuro-vertical addition: rule-based tremor-risk fallback.
    Resting tremor in Parkinson's typically presents at 4-6 Hz
    (Elble & Koller, "Tremor", 1990); amplitude threshold of 0.02g chosen
    as a conservative wearable-accelerometer detection floor.
    """
    in_tremor_band = 4.0 <= dominant_freq_hz <= 6.0
    if in_tremor_band and amplitude_g >= 0.05:
        return RiskResult("HIGH", 80, f"Dominant tremor freq {dominant_freq_hz}Hz (resting-tremor band) at {amplitude_g}g amplitude")
    if in_tremor_band and amplitude_g >= 0.02:
        return RiskResult("MODERATE", 50, f"Dominant tremor freq {dominant_freq_hz}Hz in resting-tremor band, low amplitude")
    return RiskResult("None", 10, "No tremor-band activity detected")
