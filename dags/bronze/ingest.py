import logging
import os
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BCB_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
_DEFAULT_START = "01/01/2020"
_DEFAULT_END = "31/12/2024"

BRONZE_PATH = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow")) / "data" / "bronze"

# Brazilian national holidays reduce weekday count by ~5% on average.
# Tolerance of ±10% accommodates year-to-year variation without false positives.
_HOLIDAY_FACTOR = 0.95
_TOLERANCE = 0.10


def _row_count_bounds(start: str, end: str) -> tuple[int, int]:
    """Derive expected row count bounds from the requested date range.

    Uses weekday count as a ceiling, then applies a holiday adjustment factor
    to approximate Brazilian business days, with ±10% tolerance.
    """
    start_dt = pd.to_datetime(start, format="%d/%m/%Y")
    end_dt = pd.to_datetime(end, format="%d/%m/%Y")
    weekdays = len(pd.bdate_range(start=start_dt, end=end_dt))
    expected = weekdays * _HOLIDAY_FACTOR
    return int(expected * (1 - _TOLERANCE)), int(expected * (1 + _TOLERANCE))


def _validate_api_schema(df: pd.DataFrame) -> None:
    """Fail fast if the BCB API response is missing expected columns."""
    required = {"data", "valor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"BCB API response missing expected columns: {missing}")


def fetch_selic_raw(start: str = _DEFAULT_START, end: str = _DEFAULT_END) -> int:
    """Consume BCB API and persist raw SELIC data as Parquet.

    Args:
        start: period start in dd/MM/yyyy format (default: 01/01/2020).
        end:   period end   in dd/MM/yyyy format (default: 31/12/2024).

    Raises:
        requests.HTTPError: on non-2xx responses.
        ValueError: on schema mismatch, empty payload, or row count out of range.

    Returns:
        Number of rows saved to Bronze.
    """
    url = f"{_BCB_BASE_URL}?formato=json&dataInicial={start}&dataFinal={end}"
    logger.info("Requesting SELIC data from BCB API (%s → %s)...", start, end)

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    records = response.json()
    if not records:
        raise ValueError("BCB API returned an empty payload")

    df = pd.DataFrame(records)
    logger.info("Fetched %d records from BCB API", len(df))

    _validate_api_schema(df)

    lower, upper = _row_count_bounds(start, end)
    if not (lower <= len(df) <= upper):
        raise ValueError(
            f"Row count {len(df)} out of expected range [{lower}, {upper}] "
            f"for period {start} → {end}"
        )

    BRONZE_PATH.mkdir(parents=True, exist_ok=True)
    output_path = BRONZE_PATH / "selic_raw.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")

    logger.info("Bronze saved → %s (%d rows)", output_path, len(df))
    return len(df)
