"""Keep test intermediates inside this project, using a fresh directory each run."""
from pathlib import Path
from uuid import uuid4


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    if config.option.basetemp is None:
        config.option.basetemp = str(reports / f"pytest-{uuid4().hex}")
