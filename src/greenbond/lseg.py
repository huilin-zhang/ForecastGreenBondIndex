"""Explicit opt-in API extraction with per-instrument cache and request ledger."""
import hashlib
import json
from pathlib import Path
import time

import pandas as pd

from .data import normalize


def extract(provider, instruments, start, end, cache: Path, retries=2):
    """No API session is opened here, enabling credential-free contract tests."""
    cache.mkdir(parents=True, exist_ok=True)
    columns, ledger = [], []
    names = [item["name"] for item in instruments]
    if len(names) != len(set(names)):
        raise ValueError("Instrument output names must be unique.")
    for item in instruments:
        key = hashlib.sha256(json.dumps([item, start, end], sort_keys=True).encode()).hexdigest()[:20]
        cached = cache / f"{key}.csv"
        status = {"name": item["name"], "ric": item["ric"], "cached": cached.exists()}
        if cached.exists():
            series = normalize(pd.read_csv(cached))
            status.update(rows=len(series), field="cached")
        else:
            series = None
            errors = []
            for field in item["fields"]:
                for attempt in range(retries + 1):
                    try:
                        response = provider.get_history(universe=item["ric"], fields=[field],
                                                        interval="daily", start=start, end=end)
                        if response is None or response.empty:
                            break
                        if response.shape[1] != 1:
                            raise ValueError("Expected one value column per requested instrument.")
                        response = response.copy()
                        response.columns = [item["name"]]
                        response.index.name = "date"
                        series = normalize(response.reset_index())
                        series = series.dropna(subset=[item["name"]])
                        if series.empty:
                            series = None
                            break
                        status.update(rows=len(series), field=field)
                        series.to_csv(cached, index_label="date")
                        break
                    except Exception as exc:
                        # Do not serialize error strings, which may contain credential material.
                        errors.append(type(exc).__name__)
                        if attempt < retries:
                            time.sleep(min(2 ** attempt, 4))
                if series is not None:
                    break
            if series is None:
                status.update(rows=0, error_types=errors)
        status["status"] = "ok" if series is not None else "failed"
        ledger.append(status)
        if series is not None:
            columns.append(series)
    (cache / "extraction.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    if any(row["status"] == "failed" for row in ledger):
        raise RuntimeError("Extraction incomplete; inspect extraction.json. Successful series are cached.")
    if not columns:
        raise ValueError("No instruments configured.")
    return pd.concat(columns, axis=1).sort_index()


def download(instruments_path: Path, start, end, cache: Path, config: Path):
    try:
        import lseg.data as ld
    except ImportError as exc:
        raise RuntimeError('Install the LSEG extra: pip install -e ".[lseg]"') from exc
    instruments = json.loads(instruments_path.read_text(encoding="utf-8"))["instruments"]
    if not config.is_file():
        raise ValueError("Provide a local LSEG configuration file.")
    pd.Timestamp(start), pd.Timestamp(end)
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Start date must precede end date.")
    ld.open_session(name="desktop.workspace", config_name=str(config))
    try:
        return extract(ld, instruments, start, end, cache)
    finally:
        ld.close_session()
