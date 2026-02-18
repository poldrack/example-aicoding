"""Tests for PAweather — NOAA weather API, monthly max temp timeseries."""

import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aicoding.PAweather.solution import (
    fetch_weather_data,
    compute_monthly_max_temps,
    plot_monthly_max_temps,
    save_weather_cache,
    load_weather_cache,
)


@pytest.fixture
def sample_weather_records():
    """Synthetic weather records for testing."""
    records = []
    for year in range(1960, 1963):
        for month in range(1, 13):
            for day in range(1, 29):
                records.append({
                    "date": f"{year}-{month:02d}-{day:02d}",
                    "TMAX": float(np.random.RandomState(year * 1000 + month * 100 + day).randint(50, 100)),
                })
    return records


class TestFetchWeatherData:
    @patch("aicoding.PAweather.solution.requests.get")
    def test_returns_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"date": "1960-01-01T00:00:00", "value": 55, "datatype": "TMAX"}]}
        mock_get.return_value = mock_response
        result = fetch_weather_data(api_key="fake_key", start_year=1960, end_year=1960)
        assert isinstance(result, list)

    @patch("aicoding.PAweather.solution.requests.get")
    def test_records_have_date_and_tmax(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"date": "1960-01-01T00:00:00", "value": 55, "datatype": "TMAX", "station": "X"}]
        }
        mock_get.return_value = mock_response
        result = fetch_weather_data(api_key="fake_key", start_year=1960, end_year=1960)
        if result:
            assert "date" in result[0]
            assert "TMAX" in result[0]


class TestComputeMonthlyMaxTemps:
    def test_returns_dict_or_series(self, sample_weather_records):
        result = compute_monthly_max_temps(sample_weather_records)
        assert hasattr(result, '__len__')

    def test_correct_number_of_months(self, sample_weather_records):
        result = compute_monthly_max_temps(sample_weather_records)
        # 3 years * 12 months = 36
        assert len(result) == 36

    def test_values_positive(self, sample_weather_records):
        result = compute_monthly_max_temps(sample_weather_records)
        if hasattr(result, 'values'):
            vals = list(result.values()) if isinstance(result, dict) else result.values
        else:
            vals = list(result)
        assert all(v > 0 for v in vals)

    def test_empty_records(self):
        result = compute_monthly_max_temps([])
        assert len(result) == 0


class TestPlotMonthlyMaxTemps:
    def test_creates_figure(self, sample_weather_records):
        monthly = compute_monthly_max_temps(sample_weather_records)
        fig = plot_monthly_max_temps(monthly)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, sample_weather_records, tmp_path):
        monthly = compute_monthly_max_temps(sample_weather_records)
        output = str(tmp_path / "weather.png")
        fig = plot_monthly_max_temps(monthly, save_path=output)
        assert os.path.exists(output)
        plt.close(fig)


class TestWeatherCache:
    def test_save_and_load_roundtrip(self, sample_weather_records, tmp_path):
        """Saving and loading should produce identical records."""
        cache_path = str(tmp_path / "weather_cache.json")
        save_weather_cache(sample_weather_records, cache_path)
        loaded = load_weather_cache(cache_path)
        assert len(loaded) == len(sample_weather_records)
        assert loaded[0]["date"] == sample_weather_records[0]["date"]
        assert loaded[0]["TMAX"] == sample_weather_records[0]["TMAX"]

    def test_load_nonexistent_returns_none(self):
        """Loading from a nonexistent path should return None."""
        result = load_weather_cache("/nonexistent/path/cache.json")
        assert result is None

    def test_save_creates_file(self, tmp_path):
        """Save should create the cache file."""
        cache_path = str(tmp_path / "cache.json")
        save_weather_cache([{"date": "2000-01-01", "TMAX": 65.0}], cache_path)
        assert os.path.exists(cache_path)


@pytest.mark.slow
class TestLiveAPI:
    def test_fetch_real_data(self):
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("NOAA_API_KEY")
        if not api_key:
            pytest.skip("NOAA_API_KEY not found in environment or .env")
        result = fetch_weather_data(api_key=api_key, start_year=1960, end_year=1960)
        assert isinstance(result, list)
