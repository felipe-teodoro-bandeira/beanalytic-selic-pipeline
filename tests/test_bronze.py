from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bronze.ingest import _DEFAULT_END, _DEFAULT_START, _row_count_bounds, fetch_selic_raw


def _mock_response(records):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = records
    return resp


def _make_records(n: int) -> list:
    return [{"data": "02/01/2020", "valor": "0.0267"}] * n


class TestRowCountBounds:
    def test_bounds_contain_actual_count_for_default_range(self):
        """Dynamic bounds for 2020-2024 must contain the real BCB row count (1255)."""
        lower, upper = _row_count_bounds(_DEFAULT_START, _DEFAULT_END)
        assert lower <= 1255 <= upper

    def test_lower_is_less_than_upper(self):
        lower, upper = _row_count_bounds("01/01/2023", "31/12/2023")
        assert lower < upper

    def test_bounds_scale_with_shorter_range(self):
        """A single quarter must have much lower bounds than the full 5-year range."""
        lower_full, _ = _row_count_bounds(_DEFAULT_START, _DEFAULT_END)
        _, upper_quarter = _row_count_bounds("01/01/2024", "31/03/2024")
        assert upper_quarter < lower_full


class TestFetchSelicRaw:
    def test_raises_on_empty_payload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        with patch("bronze.ingest.requests.get", return_value=_mock_response([])):
            with pytest.raises(ValueError, match="empty payload"):
                fetch_selic_raw()

    def test_raises_on_missing_schema_column(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        records = [{"data": "02/01/2020"}] * 1200  # missing 'valor'
        with patch("bronze.ingest.requests.get", return_value=_mock_response(records)):
            with pytest.raises(ValueError, match="missing expected columns"):
                fetch_selic_raw()

    def test_raises_when_count_below_lower_bound(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        lower, _ = _row_count_bounds(_DEFAULT_START, _DEFAULT_END)
        records = _make_records(lower - 1)
        with patch("bronze.ingest.requests.get", return_value=_mock_response(records)):
            with pytest.raises(ValueError, match="out of expected range"):
                fetch_selic_raw()

    def test_raises_when_count_exceeds_upper_bound(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        _, upper = _row_count_bounds(_DEFAULT_START, _DEFAULT_END)
        records = _make_records(upper + 1)
        with patch("bronze.ingest.requests.get", return_value=_mock_response(records)):
            with pytest.raises(ValueError, match="out of expected range"):
                fetch_selic_raw()

    def test_saves_parquet_on_valid_payload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        lower, upper = _row_count_bounds(_DEFAULT_START, _DEFAULT_END)
        n = (lower + upper) // 2
        with patch("bronze.ingest.requests.get", return_value=_mock_response(_make_records(n))):
            fetch_selic_raw()

        saved = pd.read_parquet(tmp_path / "selic_raw.parquet")
        assert len(saved) == n
        assert list(saved.columns) == ["data", "valor"]

    def test_custom_date_range_reflected_in_url(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        captured = []

        def mock_get(url, **kwargs):
            captured.append(url)
            lower, upper = _row_count_bounds("01/01/2023", "31/12/2023")
            return _mock_response(_make_records((lower + upper) // 2))

        with patch("bronze.ingest.requests.get", side_effect=mock_get):
            fetch_selic_raw(start="01/01/2023", end="31/12/2023")

        assert "dataInicial=01/01/2023" in captured[0]
        assert "dataFinal=31/12/2023" in captured[0]
