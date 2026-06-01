import pandas as pd
import pytest


def _make_bronze_df(n: int = 20, valor: str = "0.0267", year: int = 2020) -> pd.DataFrame:
    dates = pd.date_range(f"{year}-01-02", periods=n, freq="B").strftime("%d/%m/%Y")
    return pd.DataFrame({"data": list(dates), "valor": [valor] * n})


@pytest.fixture
def bronze_df() -> pd.DataFrame:
    return _make_bronze_df()


@pytest.fixture
def silver_df() -> pd.DataFrame:
    df = _make_bronze_df()
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    df["dia_semana"] = df["data"].dt.day_name()
    return df
