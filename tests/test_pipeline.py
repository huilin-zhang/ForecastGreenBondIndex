import json

import numpy as np
import pandas as pd
import pytest

from greenbond.data import synthetic_data
from greenbond.pipeline import run


def test_end_to_end_reports_score_identical_dates(tmp_path):
    output = tmp_path / "experiment"
    metrics = run(synthetic_data(180), output, source="synthetic", rf_trials=1)
    metadata = json.loads((output / "run.json").read_text())
    predictions = pd.read_csv(output / "predictions.csv")
    assert len(predictions) == metadata["splits"]["test_rows"]
    assert predictions["date"].iloc[0] == metadata["splits"]["test_start"]
    assert predictions["date"].iloc[-1] == metadata["splits"]["test_end"]
    assert not predictions.isna().any().any()
    assert np.isfinite(metrics.select_dtypes("number")).all().all()
    assert set(metrics["model"]) == {"Naive (last value)", "ARIMA", "Random Forest", "Random Forest (random search)"}
    assert len(list((output / "figures").glob("*.png"))) == 4
    assert "Synthetic demonstration" in (output / "report.md").read_text()
    with pytest.raises(ValueError, match="already exists"):
        run(synthetic_data(180), output)
