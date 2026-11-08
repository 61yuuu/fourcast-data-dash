"""Cumulative feature experiments for the fixed sale/list-ratio tree.
Historical target summaries for training rows are built from strictly earlier
sale months, not from the full training table. Prediction uses only fit-time sales.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
import features

VARIANTS = ['ratio_base', 'property', 'region', 'history']


class HistoricalRatios:
    min_count = 20
    groups = [('house_type_name',), ('municipality', 'house_type_name'),
              ('community', 'house_type_name')]

    def snapshot(self, train, cutoff):
        recent = train[(train.date_sold < cutoff) &
                       (train.date_sold >= cutoff - pd.DateOffset(months=3))].copy()
        recent['ratio'] = recent.sold_price / recent.list_price
        overall = float(recent.ratio.median()) if len(recent) else 1.0
        tables = [recent.groupby(list(keys)).ratio.agg(['median', 'count']) for keys in self.groups]
        return overall, len(recent), tables

    def apply_snapshot(self, homes, snapshot):
        median, n, tables = snapshot
        result = pd.DataFrame({'recent_ratio': median, 'recent_count': float(n)}, index=homes.index)
        for keys, table in zip(self.groups, tables):
            keys_index = (pd.Index(homes[keys[0]]) if len(keys) == 1 else
                          pd.MultiIndex.from_frame(homes[list(keys)]))
            matched = table.reindex(keys_index)
            good = matched['count'].fillna(0).to_numpy() >= self.min_count
            result.loc[good, 'recent_ratio'] = matched['median'].to_numpy()[good]
            result.loc[good, 'recent_count'] = matched['count'].to_numpy()[good]
        result['recent_count'] = np.log1p(result['recent_count'])
        return result

    def fit_transform(self, train):
        result = pd.DataFrame(index=train.index, columns=['recent_ratio', 'recent_count'], dtype=float)
        months = train.date_sold.dt.to_period('M')
        for month in sorted(months.unique()):
            idx = months == month
            result.loc[idx] = self.apply_snapshot(train.loc[idx], self.snapshot(train, month.start_time))
        self.cutoff = (train.date_sold.max().to_period('M') + 1).start_time
        self.prediction_snapshot = self.snapshot(train, self.cutoff)
        return result

    def transform(self, homes):
        return self.apply_snapshot(homes, self.prediction_snapshot)


class FeatureRatioModel:
    def __init__(self, variant='history'):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.categories = ([] if variant == 'ratio_base' else ['house_style'])
        if variant in ['region', 'history']:
            self.categories += ['municipality', 'community']
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        self.history = HistoricalRatios()
        self.model = HistGradientBoostingRegressor(random_state=0,
                    categorical_features=self.categories or None)

    def numeric_features(self, homes):
        X = features.list_aware_features(homes)
        if self.variant != 'ratio_base':
            X['garage'] = homes.garage
            X['lot_front'] = homes.lot_front
            X['lot_depth'] = homes.lot_depth
            X['lot_area'] = homes.lot_front * homes.lot_depth
            X['lot_front_missing'] = homes.lot_front.isna().astype(int)
            X['lot_depth_missing'] = homes.lot_depth.isna().astype(int)
        return X

    def make_features(self, homes, fit=False):
        X = self.numeric_features(homes)
        if self.categories:
            raw = homes[self.categories].fillna('__MISSING__').astype(str)
            codes = self.encoder.fit_transform(raw) if fit else self.encoder.transform(raw)
            for i, col in enumerate(self.categories):
                X[col] = codes[:, i]
        if self.variant == 'history':
            history = self.history.fit_transform(homes) if fit else self.history.transform(homes)
            X = X.join(history)
        return X

    def fit(self, train):
        self.model.fit(self.make_features(train, fit=True), train.sold_price / train.list_price)
        return self

    def predict(self, homes):
        prices = self.model.predict(self.make_features(homes)) * homes.list_price.to_numpy()
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError('Invalid prediction')
        return prices
