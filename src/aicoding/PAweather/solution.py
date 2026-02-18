"""NOAA weather data — monthly max temperature timeseries for Palo Alto.

Fetches daily weather reports from the NOAA Climate Data Online API for
Palo Alto, CA (1960–2000), computes monthly maximum temperatures, and
plots the timeseries.
"""

import json
import os

from dotenv import load_dotenv
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOAA CDO API base
NOAA_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
# Palo Alto station (GHCND)
PALO_ALTO_STATION = "GHCND:USW00023234"  # Palo Alto Airport


def save_weather_cache(records, cache_path):
    """Save weather records to a JSON cache file.

    Parameters
    ----------
    records : list of dict
        Each dict has 'date' and 'TMAX' keys.
    cache_path : str
        Path to write the cache file.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(records, f)


def load_weather_cache(cache_path):
    """Load weather records from a JSON cache file.

    Parameters
    ----------
    cache_path : str
        Path to the cache file.

    Returns
    -------
    list of dict or None
        Records if cache exists, None otherwise.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fetch_weather_data(api_key, start_year=1960, end_year=2000,
                       station_id=PALO_ALTO_STATION):
    """Fetch daily TMAX data from NOAA for a station.

    Parameters
    ----------
    api_key : str
        NOAA API token.
    start_year, end_year : int
        Year range (inclusive).
    station_id : str
        GHCND station identifier.

    Returns
    -------
    list of dict
        Each dict has 'date' and 'TMAX' keys.
    """
    headers = {"token": api_key}
    all_records = []

    for year in range(start_year, end_year + 1):
        offset = 1
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": station_id,
                "datatypeid": "TMAX",
                "startdate": f"{year}-01-01",
                "enddate": f"{year}-12-31",
                "units": "standard",
                "limit": 1000,
                "offset": offset,
            }
            try:
                resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=30)
                if resp.status_code != 200:
                    break
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break
                for record in results:
                    all_records.append({
                        "date": record["date"][:10],
                        "TMAX": record["value"],
                    })
                if len(results) < 1000:
                    break
                offset += 1000
            except (requests.RequestException, ValueError):
                break

    return all_records


def compute_monthly_max_temps(records):
    """Compute maximum temperature for each month from daily records.

    Parameters
    ----------
    records : list of dict
        Each dict has 'date' (YYYY-MM-DD) and 'TMAX'.

    Returns
    -------
    pandas.Series
        Monthly max temperatures indexed by year-month.
    """
    if not records:
        return pd.Series(dtype=float)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_max = df.groupby("year_month")["TMAX"].max()
    monthly_max = monthly_max.sort_index()
    return monthly_max


def plot_monthly_max_temps(monthly_max, save_path=None):
    """Plot timeseries of monthly maximum temperatures.

    Parameters
    ----------
    monthly_max : pandas.Series
        Monthly max temps indexed by period.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    if len(monthly_max) > 0:
        x = range(len(monthly_max))
        ax.plot(x, monthly_max.values, linewidth=0.8)
        # Label every 12th tick with the year
        step = max(1, len(monthly_max) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(p) for p in monthly_max.index[::step]], rotation=45, ha="right")

    ax.set_xlabel("Month")
    ax.set_ylabel("Max Temperature (°F)")
    ax.set_title("Monthly Maximum Temperature — Palo Alto, CA")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    load_dotenv()

    cache_path = os.path.join("data", "pa_weather_cache.json")
    records = load_weather_cache(cache_path)

    if records is not None:
        print(f"Loaded {len(records)} cached records from {cache_path}")
    else:
        api_key = os.environ.get("NOAA_API_KEY")
        if not api_key:
            print("Error: NOAA_API_KEY not found in environment or .env file.")
            exit(1)

        print("Fetching Palo Alto weather data (1960-2000)...")
        records = fetch_weather_data(api_key, start_year=1960, end_year=2000)
        print(f"Fetched {len(records)} daily records.")

        if records:
            save_weather_cache(records, cache_path)
            print(f"Saved cache to {cache_path}")

    monthly_max = compute_monthly_max_temps(records)
    print(f"Computed monthly max temps for {len(monthly_max)} months.")

    os.makedirs("outputs", exist_ok=True)
    fig = plot_monthly_max_temps(monthly_max, save_path="outputs/pa_weather.png")
    print("Saved plot to outputs/pa_weather.png")
    plt.close(fig)
