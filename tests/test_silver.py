import pandas as pd
import pytest

from silver.transform import _validate_schema, transform_selic


def _write_bronze(tmp_path, df: pd.DataFrame):
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(bronze_dir / "selic_raw.parquet", index=False)
    return bronze_dir


def _make_df(dates: list[str], valores: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"data": dates, "valor": valores})


class TestValidateSchema:
    def test_raises_on_missing_valor_column(self):
        df = pd.DataFrame({"data": ["02/01/2020"]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_schema(df)

    def test_raises_on_missing_data_column(self):
        df = pd.DataFrame({"valor": ["0.0267"]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_schema(df)

    def test_passes_on_valid_schema(self, bronze_df):
        _validate_schema(bronze_df)  # must not raise


class TestTransformSelic:
    def test_parses_date_and_valor_types(self, tmp_path, bronze_df, monkeypatch):
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, bronze_df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        transform_selic()

        result = pd.read_parquet(silver_dir / "selic_trusted.parquet")
        assert result["data"].dtype == "datetime64[ns]"
        assert result["valor"].dtype == "float64"

    def test_adds_all_derived_columns(self, tmp_path, bronze_df, monkeypatch):
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, bronze_df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        transform_selic()

        result = pd.read_parquet(silver_dir / "selic_trusted.parquet")
        for col in ("ano", "mes", "ano_mes", "dia_semana"):
            assert col in result.columns

    def test_dia_semana_in_portuguese(self, tmp_path, bronze_df, monkeypatch):
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, bronze_df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        transform_selic()

        result = pd.read_parquet(silver_dir / "selic_trusted.parquet")
        pt_days = {"Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"}
        assert set(result["dia_semana"].unique()).issubset(pt_days)

    def test_output_is_sorted_ascending_by_date(self, tmp_path, monkeypatch):
        df = _make_df(["05/01/2020", "02/01/2020", "04/01/2020"], ["0.0267"] * 3)
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        transform_selic()

        result = pd.read_parquet(silver_dir / "selic_trusted.parquet")
        assert result["data"].is_monotonic_increasing

    def test_drops_rows_below_null_ratio_threshold(self, tmp_path, monkeypatch):
        """1 null in 200 rows = 0.5% < 1% threshold → drops the row, does not fail."""
        dates = list(pd.date_range("2020-01-02", periods=200, freq="B").strftime("%d/%m/%Y"))
        valor = ["0.0267"] * 199 + [""]
        df = pd.DataFrame({"data": dates, "valor": valor})
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        row_count = transform_selic()
        assert row_count == 199

    def test_raises_when_null_ratio_exceeds_threshold(self, tmp_path, monkeypatch):
        """20% empty strings → above 1% threshold → must fail."""
        dates = list(pd.date_range("2020-01-02", periods=10, freq="B").strftime("%d/%m/%Y"))
        valor = ["0.0267"] * 8 + ["", ""]
        df = pd.DataFrame({"data": dates, "valor": valor})
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        with pytest.raises(ValueError, match="exceeds threshold"):
            transform_selic()

    def test_quality_gate_valor_out_of_range(self, tmp_path, monkeypatch):
        df = _make_df(["02/01/2020"], ["99.9"])
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        with pytest.raises(ValueError, match="quality gate"):
            transform_selic()

    def test_quality_gate_unexpected_year(self, tmp_path, monkeypatch):
        df = _make_df(["02/01/2019"], ["0.0267"])
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        with pytest.raises(ValueError, match="unexpected years"):
            transform_selic()

    def test_quality_gate_duplicate_dates(self, tmp_path, monkeypatch):
        df = _make_df(["02/01/2020", "02/01/2020", "03/01/2020"], ["0.0267"] * 3)
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        with pytest.raises(ValueError, match="duplicate dates"):
            transform_selic()

    def test_quality_gate_date_gap_too_large(self, tmp_path, monkeypatch):
        df = _make_df(["02/01/2020", "20/01/2020"], ["0.0267", "0.0267"])
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        with pytest.raises(ValueError, match="gap of"):
            transform_selic()

    def test_passes_with_legitimate_holiday_gap(self, tmp_path, monkeypatch):
        """Christmas + New Year block (~11 days) must not trigger the gap gate."""
        df = _make_df(["24/12/2020", "04/01/2021"], ["0.0267", "0.0267"])
        silver_dir = tmp_path / "silver"
        monkeypatch.setattr("silver.transform.BRONZE_PATH", _write_bronze(tmp_path, df))
        monkeypatch.setattr("silver.transform.SILVER_PATH", silver_dir)

        transform_selic()  # must not raise
