"""Team integration: Member 2 off-market and Member 3 selected list-aware."""
import json
from pathlib import Path
from off_market_model import OffMarketModel
from starter_models import ListPriceBaseline, NearestSalesBaseline, FeatureModel


def make_model(track):
    if track == "list_aware":
        from list_aware_feature_models import FeatureRatioModel
        return FeatureRatioModel("history")
    if track != "off_market":
        raise ValueError(track)
    path = Path(__file__).with_name("selected_off_market.json")
    if not path.exists():
        raise RuntimeError("Run experiment_off_market.py to select the off-market model first")
    config = json.loads(path.read_text())
    return OffMarketModel(config["variant"])


def model_name(track):
    """Model labels shared by validation, evaluation and provenance records."""
    if track == "list_aware":
        return "current_v3"
    if track == "off_market":
        config = json.loads(Path(__file__).with_name("selected_off_market.json").read_text())
        return config["variant"]
    raise ValueError(track)
