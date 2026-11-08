"""Member 1 numeric/history features plus the established native category encoding.
Original features.py is deliberately unchanged to preserve previous experiments.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import member1_features
from list_aware_feature_models import HistoricalRatios

VARIANTS = ['member1_base', 'member1_our_history', 'member1_sales_history', 'member1_both_history']


class Member1RatioModel:
    def __init__(self, variant='member1_base'):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.categories = ['house_style', 'municipality', 'community']
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        self.our_history = HistoricalRatios()
        self.sales_history = member1_features.HistoricalSalesFeatures(windows=(90,365), min_sales=5,
                                                                     availability_lag_days=0)
        self.model = HistGradientBoostingRegressor(random_state=0, categorical_features=self.categories)

    def make_features(self, homes, fit=False):
        X = member1_features.list_aware_features(homes)
        raw = homes[self.categories].fillna('__MISSING__').astype(str)
        codes = self.encoder.fit_transform(raw) if fit else self.encoder.transform(raw)
        for i,col in enumerate(self.categories):
            X[col] = codes[:,i]
        if self.variant in ['member1_our_history','member1_both_history']:
            H = self.our_history.fit_transform(homes) if fit else self.our_history.transform(homes)
            X = X.join(H)
        if self.variant in ['member1_sales_history','member1_both_history']:
            H = self.sales_history.fit_transform(homes) if fit else self.sales_history.transform(homes)
            X = X.join(H)
        return X

    def fit(self, train):
        X=self.make_features(train,fit=True)
        self.feature_names_ = list(X.columns)
        self.model.fit(X, train.sold_price/train.list_price)
        return self

    def predict(self, homes):
        prices=self.model.predict(self.make_features(homes))*homes.list_price.to_numpy()
        if not np.isfinite(prices).all() or (prices<=0).any():
            raise ValueError('Invalid prices')
        return prices
