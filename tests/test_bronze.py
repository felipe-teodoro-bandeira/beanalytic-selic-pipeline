from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bronze.ingest import _MIN_EXPECTED_ROWS, fetch_selic_raw


def _mock_response(records):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = records
    return resp


def _make_records(n: int) -> list:
    return [{"data": "02/01/2020", "valor": "0.0267"}] * n


class TestFetchSelicRaw:
    def test_raises_on_empty_payload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        with patch("bronze.ingest.requests.get", return_value=_mock_response([])):
            with pytest.raises(ValueError, match="empty payload"):
                fetch_selic_raw()

    def test_raises_on_low_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        records = _make_records(_MIN_EXPECTED_ROWS - 1)
        with patch("bronze.ingest.requests.get", return_value=_mock_response(records)):
            with pytest.raises(ValueError, match="low record count"):
                fetch_selic_raw()

    def test_saves_parquet_on_valid_payload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        n = _MIN_EXPECTED_ROWS + 50
        with patch("bronze.ingest.requests.get", return_value=_mock_response(_make_records(n))):
            fetch_selic_raw()

        saved = pd.read_parquet(tmp_path / "selic_raw.parquet")
        assert len(saved) == n
        assert list(saved.columns) == ["data", "valor"]

    def test_returns_row_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr("bronze.ingest.BRONZE_PATH", tmp_path)
        n = _MIN_EXPECTED_ROWS + 100
        with patch("bronze.ingest.requests.get", return_value=_mock_response(_make_records(n))):
            assert fetch_selic_raw() == n
