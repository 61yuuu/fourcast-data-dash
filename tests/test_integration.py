"""Check the production factory, cross-member evaluation, and numeric boundary."""
import numpy as np
import pandas as pd
import pytest
import models
import evaluation
from member1_features import FeatureBuilder
from off_market_model import OffMarketModel
from list_aware_feature_models import FeatureRatioModel
from test_starter import make_listings


def test_factory_keeps_both_selected_models():
    assert isinstance(models.make_model("off_market"), OffMarketModel)
    assert models.make_model("off_market").variant == "history_text_boost"
    assert isinstance(models.make_model("list_aware"), FeatureRatioModel)
    assert models.make_model("list_aware").variant == "history"
    with pytest.raises(ValueError):
        models.make_model("invalid")


def test_missing_area_remains_numeric_in_fitted_features():
    rows = make_listings(20)
    rows.loc[0, "house_area"] = None
    builder = FeatureBuilder().fit(rows)
    X = builder.off_market_features(rows)
    assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
    assert not np.isinf(X.to_numpy(dtype=float)).any()


def test_evaluation_rejects_conflicting_truth(tmp_path):
    rows = make_listings(20)
    rows["month"] = "2026-01"
    rows["track"] = "off_market"
    rows["model"] = "a"
    rows["predicted_price"] = rows.sold_price
    other = rows.copy()
    other["model"] = "b"
    other.loc[0, "sold_price"] *= 2
    path = tmp_path / "predictions.csv"
    pd.concat([rows, other]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Conflicting sold_price"):
        evaluation.load_predictions(path)
