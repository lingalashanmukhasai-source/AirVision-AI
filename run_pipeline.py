
# run_pipeline.py
import os
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")

KEY = os.getenv("OPENAQ_API_KEY", "").strip()
BASE = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": KEY, "Accept": "application/json"}
TIMEOUT = 35

# Start with Vijayawada. OpenAQ radius searches are limited to 25 km.
CENTER_LAT, CENTER_LON = 16.5062, 80.6480
RADIUS_M = 25000
HISTORY_DAYS = 365
MIN_TRAIN_ROWS = 100


def get_json(url, params=None, headers=None):
    response = requests.get(
        url, params=params, headers=headers or HEADERS, timeout=TIMEOUT
    )
    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:500]
        raise RuntimeError(
            f"HTTP {response.status_code} for {response.url}\n{detail}"
        )
    return response.json()


def results(payload):
    return payload.get("results", []) or []


def fetch_locations():
    payload = get_json(
        f"{BASE}/locations",
        {
            "coordinates": f"{CENTER_LAT:.4f},{CENTER_LON:.4f}",
            "radius": RADIUS_M,
            "limit": 100,
            "page": 1,
            "parameters_id": 2,  # OpenAQ PM2.5 parameter
        },
    )
    return results(payload)


def fetch_pm25_sensors(location_id):
    payload = get_json(
        f"{BASE}/locations/{location_id}/sensors",
        {"limit": 100, "page": 1},
    )
    found = []
    for sensor in results(payload):
        parameter = sensor.get("parameter") or {}
        name = str(parameter.get("name", "")).lower().replace(" ", "")
        if name in ("pm25", "pm2.5", "pm₂.₅"):
            found.append(sensor)
    return found


def fetch_daily_sensor(sensor_id, start_date, end_date):
    """Request aggregated daily values; never silently substitute fake values."""
    url = f"{BASE}/sensors/{sensor_id}/days"
    payload = get_json(
        url,
        {
            "date_from": start_date,
            "date_to": end_date,
            "limit": 1000,
            "page": 1,
        },
    )
    rows = []
    for item in results(payload):
        value = item.get("value")
        period = item.get("period") or item.get("datetime") or {}
        if isinstance(period, dict):
            stamp = (
                period.get("datetimeFrom")
                or period.get("datetimeTo")
                or period.get("datetime")
                or {}
            )
            if isinstance(stamp, dict):
                stamp = stamp.get("utc") or stamp.get("local") or ""
        else:
            stamp = str(period)

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value) or value < 0:
            continue
        rows.append({"date": str(stamp)[:10], "pm25": value})
    return rows


def fetch_weather_daily(lat, lon, start_date, end_date):
    """Historical weather reanalysis for the sensor's location."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    payload = get_json(
        url,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": (
                "temperature_2m_mean,relative_humidity_2m_mean,"
                "precipitation_sum,wind_speed_10m_max"
            ),
            "timezone": "UTC",
        },
        headers={"Accept": "application/json"},
    )
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    weather = pd.DataFrame({
        "date": daily["time"],
        "temperature": daily.get("temperature_2m_mean", []),
        "humidity": daily.get("relative_humidity_2m_mean", []),
        "precipitation": daily.get("precipitation_sum", []),
        "wind_speed": daily.get("wind_speed_10m_max", []),
    })
    return weather


def collect_history():
    if not KEY or KEY in ("YOUR_REAL_API_KEY", "PASTE_YOUR_REAL_KEY_HERE"):
        raise RuntimeError(
            "Set your real OPENAQ_API_KEY in the .env file before running."
        )

    end = pd.Timestamp.now(tz="UTC").date()
    start = end - pd.Timedelta(days=HISTORY_DAYS)
    start_s, end_s = str(start), str(end)

    print("Searching for real PM2.5 monitoring stations...")
    locations = fetch_locations()
    if not locations:
        raise RuntimeError(
            "No OpenAQ locations returned. Check API access or try a different region."
        )

    all_rows = []
    weather_cache = {}

    for i, location in enumerate(locations, start=1):
        loc_id = location.get("id")
        loc_name = location.get("name", f"Location {loc_id}")
        coords = location.get("coordinates") or {}
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        if loc_id is None or lat is None or lon is None:
            continue

        print(f"[{i}/{len(locations)}] {loc_name}: reading sensor metadata")
        try:
            sensors = fetch_pm25_sensors(int(loc_id))
        except Exception as exc:
            print(f"  Skipped sensor listing: {exc}")
            continue

        for sensor in sensors:
            sensor_id = sensor.get("id")
            if sensor_id is None:
                continue
            try:
                measurements = fetch_daily_sensor(sensor_id, start_s, end_s)
            except Exception as exc:
                print(f"  Sensor {sensor_id} history unavailable: {exc}")
                continue

            if not measurements:
                continue

            key = (round(float(lat), 4), round(float(lon), 4))
            if key not in weather_cache:
                try:
                    weather_cache[key] = fetch_weather_daily(
                        float(lat), float(lon), start_s, end_s
                    )
                except Exception as exc:
                    print(f"  Weather unavailable for {loc_name}: {exc}")
                    weather_cache[key] = pd.DataFrame()

            weather = weather_cache[key]
            for measurement in measurements:
                all_rows.append({
                    "location_id": int(loc_id),
                    "location_name": loc_name,
                    "sensor_id": int(sensor_id),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "date": measurement["date"],
                    "pm25": measurement["pm25"],
                    "weather_key": f"{key[0]},{key[1]}",
                })

    if not all_rows:
        raise RuntimeError(
            "No daily historical PM2.5 records were returned. The API may have "
            "limited coverage for these sensors or date parameters. No synthetic "
            "readings were created. Check OpenAQ sensor history in its API docs."
        )

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date", "pm25", "latitude", "longitude"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(["sensor_id", "date"])

    merged = []
    for weather_key, group in df.groupby("weather_key"):
        lat, lon = map(float, weather_key.split(","))
        weather = weather_cache.get((round(lat, 4), round(lon, 4)), pd.DataFrame())
        if not weather.empty:
            merged.append(group.merge(weather, on="date", how="left"))
        else:
            for col in ["temperature", "humidity", "precipitation", "wind_speed"]:
                group[col] = np.nan
            merged.append(group)

    df = pd.concat(merged, ignore_index=True)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date_dt"].dt.month
    df["day_of_year"] = df["date_dt"].dt.dayofyear
    df["day_of_week"] = df["date_dt"].dt.dayofweek

    # Only rows with observed PM2.5 and complete model features are used for training.
    feature_cols = [
        "latitude", "longitude", "temperature", "humidity",
        "precipitation", "wind_speed", "month", "day_of_year", "day_of_week"
    ]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    usable = df.dropna(subset=["pm25"] + feature_cols).copy()

    df.to_csv(DATA / "pm25_weather_history.csv", index=False)
    usable.to_csv(DATA / "training_dataset.csv", index=False)
    print(f"Saved {len(df)} historical rows.")
    print(f"Rows with complete weather and model features: {len(usable)}")
    print(f"Distinct locations: {usable['location_id'].nunique()}")
    return usable, feature_cols


def train_model(df, feature_cols):
    metrics_path = OUT / "metrics.json"
    model_path = OUT / "airvision_model.joblib"

    if len(df) < MIN_TRAIN_ROWS or df["location_id"].nunique() < 4:
        metrics = {
            "trained": False,
            "rows": int(len(df)),
            "locations": int(df["location_id"].nunique()) if not df.empty else 0,
            "message": (
                f"Need at least {MIN_TRAIN_ROWS} complete rows across at least "
                "4 distinct locations. No model/RMSE has been reported."
            ),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(metrics["message"])
        return

    X = df[feature_cols]
    y = df["pm25"].astype(float)
    groups = df["location_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    mae = float(mean_absolute_error(y_test, prediction))
    r2 = float(r2_score(y_test, prediction)) if len(y_test) >= 2 else None

    # Refit on all available rows after evaluation.
    model.fit(X, y)
    joblib.dump({"model": model, "features": feature_cols}, model_path)

    metrics = {
        "trained": True,
        "rows": int(len(df)),
        "locations": int(df["location_id"].nunique()),
        "held_out_locations": int(groups.iloc[test_idx].nunique()),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "features": feature_cols,
        "model": "ExtraTreesRegressor",
        "validation": "GroupShuffleSplit holding out entire locations",
        "warning": (
            "Results apply only to this dataset and test split. Extrapolating "
            "to distant places or different seasons may be inaccurate. This is "
            "an estimate, not a sensor measurement."
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nMODEL EVALUATION")
    print(f"Hold-out location RMSE: {rmse:.2f} µg/m³ (assuming source values are µg/m³)")
    print(f"Hold-out location MAE:  {mae:.2f}")
    print(f"R²: {r2}")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    try:
        training_df, features = collect_history()
        train_model(training_df, features)
    except Exception as error:
        print("\nAIRVISION ERROR")
        print(error)
        raise SystemExit(1)
