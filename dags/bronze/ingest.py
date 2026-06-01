import logging
import os
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BCB_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
    "?formato=json&dataInicial=01/01/2020&dataFinal=31/12/2024"
)
BRONZE_PATH = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow")) / "data" / "bronze"

# Minimum expected records: 2020-2024 has ~1250 business days
_MIN_EXPECTED_ROWS = 1000


def fetch_selic_raw() -> int:
    """Consume BCB API and persist raw SELIC data as Parquet.

    Raises:
        requests.HTTPError: on non-2xx responses.
        ValueError: on empty payload or unexpectedly low row count.

    Returns:
        Number of rows saved to Bronze.
    """
    logger.info("Requesting SELIC data from BCB API...")
    response = requests.get(BCB_URL, timeout=30)
    response.raise_for_status()

    records = response.json()

    if not records:
        raise ValueError("BCB API returned an empty payload")

    df = pd.DataFrame(records)
    logger.info("Fetched %d records from BCB API", len(df))

    if len(df) < _MIN_EXPECTED_ROWS:
        raise ValueError(
            f"Unexpectedly low record count: {len(df)} (expected >= {_MIN_EXPECTED_ROWS})"
        )

    BRONZE_PATH.mkdir(parents=True, exist_ok=True)
    output_path = BRONZE_PATH / "selic_raw.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")

    logger.info("Bronze saved → %s (%d rows)", output_path, len(df))
    return len(df)
