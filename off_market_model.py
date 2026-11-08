"""Off-market candidates: all inputs are available in test_off_market.csv.

Feature code is the corrected Member 1 implementation. Category vocabularies,
imputation and sales history are refitted inside each official monthly fold.
No list_price, AVM, actual sale date or days-on-market is used as a model input.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import features as starter_features
import member1_features

CANDIDATES = ("starter_boost", "member1_boost", "history_boost",
              "history_text_boost", "extra_trees")

# Fixed before the first experiment; not learned from validation/test outcomes.
TEXT_PATTERNS = {
    "renovated": r"renovat|fully redone|remodel|updated|upgraded|modern finish",
    "needs_work": r"needs? work|fixer|handyman|as.is|tlc|original condition|renovation opportunity",
    "new_kitchen": r"new kitchen|updated kitchen|renovated kitchen|quartz|granite|stainless",
    "new_floor": r"new.*floor|hardwood|refinished|newer flooring",
    "finished_basement": r"finished basement|finished lower level|basement apartment",
    "unfinished_basement": r"unfinished basement|unfinished lower",
    "separate_entry": r"separate entrance|separate entry|in.law|income potential",
    "view": r"unobstructed|panoramic|skyline|lake view|ravine view",
    "outdoor": r"balcony|terrace|patio|backyard|back yard|garden",
    "pool": r"\bpool\b",
    "corner": r"corner|end.unit",
    "parking": r"parking|garage|driveway",
    "transit": r"subway|transit|commut|\bttc\b|go station",
    "amenities": r"gym|concierge|amenities|fitness",
    "premium": r"luxury|luxurious|premium|custom.built|high.end",
    "move_in": r"turnkey|turn.key|move.in.ready|nothing to do|unpack",
}


def text_features(homes):
    remarks = homes["remarks"].fillna("").astype(str).str.lower()
    return pd.DataFrame({"text_" + name: remarks.str.contains(pattern, regex=True).astype(int)
                         for name, pattern in TEXT_PATTERNS.items()}, index=homes.index)


class OffMarketModel:
    def __init__(self, variant="history_text_boost"):
        if variant not in CANDIDATES:
            raise ValueError(variant)
        self.variant = variant
        self.categories = ["house_style", "municipality", "community"]
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        self.builder = member1_features.FeatureBuilder(include_history=False)
        self.history = member1_features.HistoricalSalesFeatures(windows=(90, 365), min_sales=5)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        if variant == "extra_trees":
            self.model = ExtraTreesRegressor(n_estimators=80, max_depth=24,
                min_samples_leaf=5, max_features=0.8, n_jobs=2, random_state=0)
        elif variant == "starter_boost":
            # Exact official FeatureModel estimator and original 13 features.
            self.model = HistGradientBoostingRegressor(random_state=0)
        else:
            self.model = HistGradientBoostingRegressor(max_iter=220, learning_rate=0.08,
                max_leaf_nodes=31, min_samples_leaf=25, l2_regularization=3.0,
                early_stopping=False, categorical_features=self.categories, random_state=0)

    def make_features(self, homes, fit=False):
        if self.variant == "starter_boost":
            X = starter_features.off_market_features(homes)
        elif self.variant == "extra_trees":
            if fit:
                self.builder.fit(homes)
            # One-hot categories avoid imposing a made-up ordering for ExtraTrees.
            X = self.builder.off_market_features(homes)
        else:
            X = member1_features.off_market_features(homes)
            raw = homes[self.categories].fillna("__MISSING__").astype(str)
            codes = self.encoder.fit_transform(raw) if fit else self.encoder.transform(raw)
            for i, col in enumerate(self.categories):
                X[col] = codes[:, i]
            if self.variant in ("history_boost", "history_text_boost"):
                history = self.history.fit_transform(homes) if fit else self.history.transform(homes)
                X = X.join(history)
            if self.variant == "history_text_boost":
                X = X.join(text_features(homes))
        return X

    def fit(self, train):
        X = self.make_features(train, fit=True)
        self.feature_names_ = list(X.columns)
        if self.variant == "extra_trees":
            X = self.imputer.fit_transform(X)
        self.model.fit(X, np.log(train["sold_price"].to_numpy(dtype=float)))
        return self

    def predict(self, homes):
        X = self.make_features(homes)
        if list(X.columns) != self.feature_names_:
            raise ValueError("Training/prediction feature schemas differ")
        if self.variant == "extra_trees":
            X = self.imputer.transform(X)
        prices = np.exp(self.model.predict(X))
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Predictions must be finite and positive")
        return prices
