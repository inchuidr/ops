import numpy as np
import pandas as pd

from mlflowops import monitoring, training
from mlflowops.sample_data import generate


def test_psi_identical_is_zero_and_shift_is_large():
    ref_df = generate(2000, seed=1)[["monthly_charges", "contract"]]
    ref = training.build_reference(ref_df)
    same = generate(2000, seed=2)[["monthly_charges", "contract"]]
    rep = monitoring.drift_report(ref, same, 0.2, 30)
    assert not rep["drifted"] and all(f["psi"] < 0.1 for f in rep["features"])
    shifted = same.copy()
    shifted["monthly_charges"] += 40
    shifted["contract"] = "Month-to-month"
    rep = monitoring.drift_report(ref, shifted, 0.2, 30)
    assert rep["drifted"] and set(rep["drifted_features"]) == {"monthly_charges", "contract"}


def test_insufficient_data():
    ref = training.build_reference(pd.DataFrame({"a": np.arange(100.0)}))
    assert monitoring.drift_report(ref, pd.DataFrame({"a": [1.0]}), 0.2, 30)["status"] == "insufficient_data"
