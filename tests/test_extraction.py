import json
from unittest.mock import Mock

import pandas as pd
import pytest

from greenbond.lseg import extract


def response():
    return pd.DataFrame({"value": [10., 11.]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))


def test_cache_prevents_repeated_requests(tmp_path):
    provider = Mock()
    provider.get_history.return_value = response()
    items = [{"name": "market", "ric": "TEST", "fields": ["Close"]}]
    first = extract(provider, items, "2020-01-01", "2020-01-02", tmp_path)
    second = extract(provider, items, "2020-01-01", "2020-01-02", tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert provider.get_history.call_count == 1
    assert json.loads((tmp_path / "extraction.json").read_text())[0]["cached"]


def test_field_fallback(tmp_path):
    provider = Mock()
    provider.get_history.side_effect = [pd.DataFrame(), response()]
    items = [{"name": "market", "ric": "TEST", "fields": ["Close", "Settlement"]}]
    frame = extract(provider, items, "2020-01-01", "2020-01-02", tmp_path, retries=0)
    assert len(frame) == 2
    ledger = json.loads((tmp_path / "extraction.json").read_text())
    assert ledger[0]["field"] == "Settlement"


def test_partial_failure_is_explicit_and_error_text_is_not_serialized(tmp_path):
    provider = Mock()
    provider.get_history.side_effect = [response(), RuntimeError("sensitive credential")]
    items = [{"name": "ok", "ric": "A", "fields": ["Close"]},
             {"name": "bad", "ric": "B", "fields": ["Close"]}]
    with pytest.raises(RuntimeError, match="incomplete"):
        extract(provider, items, "2020-01-01", "2020-01-02", tmp_path, retries=0)
    text = (tmp_path / "extraction.json").read_text()
    assert "sensitive credential" not in text
    assert "RuntimeError" in text
    assert len(list(tmp_path.glob("*.csv"))) == 1
