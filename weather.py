"""
backend/weather.py — Qti pi

Ported near-verbatim from Ashmit0107/Health-Companion's backend/weather.py
(audited at C:\\SIH\\health_companion\\backend\\weather.py). Real-time
OpenWeatherMap + Nominatim reverse-geocoding when OPENWEATHER_API_KEY is
configured; otherwise a clearly-labelled mock reading so the app still runs
offline / without paid keys. The "source" field always tells the caller
which path was used.
"""
import os
import random
import requests

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()
MOCK_CITY = os.environ.get("MOCK_CITY", "New Delhi")

_OWM_AQI_TO_US_BAND = {1: 35, 2: 75, 3: 115, 4: 175, 5: 300}


def _mock_reading(reason: str = "no OPENWEATHER_API_KEY / no coordinates supplied") -> dict:
    return {
        "source": f"mock ({reason})",
        "city": MOCK_CITY,
        "temp_c": round(random.uniform(28, 40), 1),
        "humidity_pct": round(random.uniform(35, 70), 1),
        "aqi": random.randint(60, 200),
        "lat": None,
        "lon": None,
    }


def _reverse_geocode_nominatim(lat: float, lon: float) -> str | None:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14, "addressdetails": 1},
            headers={"User-Agent": "QtiPiSIHPrototype/1.0 (educational SIH prototype)"},
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        addr = data.get("address", {})
        locality = (
            addr.get("suburb") or addr.get("neighbourhood") or addr.get("village")
            or addr.get("town") or addr.get("city_district") or addr.get("city")
        )
        state = addr.get("state")
        if locality and state:
            return f"{locality}, {state}"
        if locality:
            return locality
        display_name = data.get("display_name")
        if display_name:
            parts = [p.strip() for p in display_name.split(",")]
            return ", ".join(parts[:2])
    except Exception:
        pass
    return None


def _reverse_geocode_owm(lat: float, lon: float) -> str | None:
    try:
        resp = requests.get(
            "http://api.openweathermap.org/geo/1.0/reverse",
            params={"lat": lat, "lon": lon, "limit": 1, "appid": OPENWEATHER_API_KEY},
            timeout=6,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            place = results[0]
            name, state = place.get("name"), place.get("state")
            return f"{name}, {state}" if name and state else name
    except Exception:
        pass
    return None


def _reverse_geocode(lat: float, lon: float) -> str | None:
    return _reverse_geocode_nominatim(lat, lon) or _reverse_geocode_owm(lat, lon)


def _fetch_live_by_coords(lat: float, lon: float) -> dict:
    weather_resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=6,
    )
    weather_resp.raise_for_status()
    w = weather_resp.json()
    accurate_city = _reverse_geocode(lat, lon)

    aqi_value = None
    try:
        aqi_resp = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY},
            timeout=6,
        )
        aqi_resp.raise_for_status()
        owm_band = aqi_resp.json()["list"][0]["main"]["aqi"]
        aqi_value = _OWM_AQI_TO_US_BAND.get(owm_band, 150)
    except Exception:
        aqi_value = None

    return {
        "source": "live",
        "city": accurate_city or w.get("name") or f"{lat:.3f},{lon:.3f}",
        "temp_c": w["main"]["temp"],
        "humidity_pct": w["main"]["humidity"],
        "aqi": aqi_value,
        "lat": lat,
        "lon": lon,
        "weather_desc": (w.get("weather") or [{}])[0].get("description"),
        "wind_speed_ms": w.get("wind", {}).get("speed"),
    }


def _fetch_live_by_city(city: str) -> dict:
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
        timeout=6,
    )
    resp.raise_for_status()
    data = resp.json()
    coords = data.get("coord", {})
    lat, lon = coords.get("lat"), coords.get("lon")
    if lat is not None and lon is not None:
        result = _fetch_live_by_coords(lat, lon)
        result["city"] = city
        return result
    return {
        "source": "live", "city": city, "temp_c": data["main"]["temp"],
        "humidity_pct": data["main"]["humidity"], "aqi": None, "lat": lat, "lon": lon,
    }


def get_environment(city: str | None = None, lat: float | None = None, lon: float | None = None) -> dict:
    if not OPENWEATHER_API_KEY:
        return _mock_reading("no OPENWEATHER_API_KEY configured in backend/.env")
    if lat is not None and lon is not None:
        try:
            return _fetch_live_by_coords(lat, lon)
        except Exception as e:
            fallback = _mock_reading(f"live call by coordinates failed: {e}")
            fallback["lat"], fallback["lon"] = lat, lon
            return fallback
    target_city = city or MOCK_CITY
    try:
        return _fetch_live_by_city(target_city)
    except Exception as e:
        return _mock_reading(f"live call by city '{target_city}' failed: {e}")
