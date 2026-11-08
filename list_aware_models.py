"""Fixed first-pass candidates, using the same 14 starter input variables."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import features


class ListAwareCandidate:
    def __init__(self, kind):
        self.kind = kind
        if kind == 'ridge':
            self.model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
        elif kind == 'ratio_tree':
            self.model = HistGradientBoostingRegressor(random_state=0)
        elif kind == 'catboost':
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                           loss_function='RMSE', random_seed=0, thread_count=2,
                                           verbose=False, allow_writing_files=False)
        else:
            raise ValueError(kind)

    def make_features(self, homes):
        X = features.list_aware_features(homes)
        if self.kind == 'ridge':
            # Linear model uses log price and log area to express proportional effects.
            X['list_price'] = np.log(X['list_price'])
            X['square_feet'] = np.log1p(X['square_feet'])
        return X

    def fit(self, train):
        y = (train['sold_price'] / train['list_price'] if self.kind == 'ratio_tree'
             else np.log(train['sold_price']))
        self.model.fit(self.make_features(train), y)
        return self

    def predict(self, homes):
        values = self.model.predict(self.make_features(homes))
        prices = values * homes['list_price'].to_numpy() if self.kind == 'ratio_tree' else np.exp(values)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('Model produced invalid prices')
        return prices
