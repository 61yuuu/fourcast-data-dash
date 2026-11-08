"""EDIT ME: turn a table of homes into numbers a model can learn from.

Each function gets a table of homes with the columns of its test file (so never
sold_price or date_sold) and returns a table of numbers, one row per home.
FeatureModel in models.py uses them; switch it on in make_model() to try your features.

Ideas to try:
- lot size: lot_front * lot_depth (empty for condos, which is fine for tree models)
- garage, house_style, municipality
- how pricey a community is: the median price per square foot of its past sales
  (work it out from the training sales only, never from the homes you predict)
- how prices have moved recently in the area or for that property type
- words in the remarks, such as "renovated" or "needs work"
- off-market and list-aware can use different features: list-aware has list_price
"""
import pandas as pd
import numpy as np



PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]
CATEGORICAL_COLUMNS = ["municipality", "community", "house_style"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    text = house_area.astype('string').str.strip().str.replace(',', '', regex=False)
    parts = text.str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    result = (low + high) / 2
    return result.where((low > 0) & (high >= low) & np.isfinite(result)).astype(float)


def _number(series, *, positive=False):
    result = pd.to_numeric(series, errors='coerce').astype(float)
    return result.where(np.isfinite(result) & (result.gt(0) if positive else result.ge(0)))


def off_market_features(homes):
    """Features for homes with no list price."""
    X = pd.DataFrame(index=homes.index)
    X['square_feet'] = square_feet(homes['house_area'])
    X['log_square_feet'] = np.log1p(X['square_feet'])
    X['area_missing'] = X['square_feet'].isna().astype(int)
    text = homes['house_area'].astype('string')
    X['area_is_range'] = (text.str.contains('-', regex=False, na=False)
                          & X['square_feet'].notna()).astype(int)
    for source, name in [('bedrooms','bedrooms'), ('washrooms','washrooms'),
                         ('parking_total','parking'), ('garage','garage')]:
        X[name] = _number(homes[source])
    for col in ['lot_front','lot_depth']:
        X[col] = _number(homes[col], positive=True)
        X[col + '_missing'] = X[col].isna().astype(int)
    X['lot_area'] = (X['lot_front'] * X['lot_depth']).replace([np.inf,-np.inf], np.nan)
    X['log_lot_area'] = np.log1p(X['lot_area'])
    for col, limit in [('latitude',90), ('longitude',180)]:
        value = pd.to_numeric(homes[col], errors='coerce').astype(float)
        X[col] = value.where(np.isfinite(value) & value.between(-limit,limit))
    dates = pd.to_datetime(homes['date_listed'], errors='coerce')
    X['months'] = (dates - pd.Timestamp('2021-01-01')).dt.days / 30.44
    X['listed_month_sin'] = np.sin(2*np.pi*(dates.dt.month-1)/12)
    X['listed_month_cos'] = np.cos(2*np.pi*(dates.dt.month-1)/12)
    X['listed_date_missing'] = dates.isna().astype(int)
    for kind in PROPERTY_TYPES:
        X['is_' + kind] = homes['house_type_name'].eq(kind).fillna(False).astype(int)
    X['property_type_unknown'] = (~homes['house_type_name'].isin(PROPERTY_TYPES)).astype(int)
    return X


def list_aware_features(homes):
    """Features for homes that have a list price: everything above, plus the list price."""
    X = off_market_features(homes)
    X['list_price'] = _number(homes['list_price'], positive=True)
    X['log_list_price'] = np.log1p(X['list_price'])
    X['list_price_missing'] = X['list_price'].isna().astype(int)
    return X

class FeatureBuilder:
    """Fit on EACH training fold; reuse that builder for training/prediction.

    Learns only category vocabularies, plus optionally training-sales history.
    Output is all numeric, with fixed columns after fit. Missing and unseen
    categories have separate indicators, without invented numeric ranks.
    include_history=False is the EDA-based location/property first stage.
    """
    def __init__(self, include_history=False, windows=(90,365), min_sales=5,
                 availability_lag_days=0):
        self.include_history = include_history
        self.windows = windows
        self.min_sales = min_sales
        self.availability_lag_days = availability_lag_days

    def fit(self, train):
        self.vocabulary_ = {
            col: sorted(train[col].astype('string').dropna().unique().tolist())
            for col in CATEGORICAL_COLUMNS
        }
        self.history_ = (HistoricalSalesFeatures(
            self.windows, self.min_sales, self.availability_lag_days).fit(train)
                         if self.include_history else None)
        return self

    def _transform(self, homes, list_aware):
        if not hasattr(self, 'vocabulary_'):
            raise RuntimeError('Call FeatureBuilder.fit(train) before creating features')
        base = list_aware_features(homes) if list_aware else off_market_features(homes)
        columns = {col: base[col].to_numpy() for col in base}
        for col, vocabulary in self.vocabulary_.items():
            values = homes[col].astype('string')
            # Index-based names are unambiguous even if category strings contain punctuation.
            for number, category in enumerate(vocabulary):
                columns[f'cat_{col}_{number}'] = values.eq(category).fillna(False).to_numpy(dtype=np.int8)
            columns[f'cat_{col}_missing'] = values.isna().to_numpy(dtype=np.int8)
            columns[f'cat_{col}_unknown'] = (values.notna() & ~values.isin(vocabulary)).to_numpy(dtype=np.int8)
        if self.history_ is not None:
            historical = self.history_.transform(homes)
            columns.update({col: historical[col].to_numpy() for col in historical})
        return pd.DataFrame(columns, index=homes.index)

    def off_market_features(self, homes):
        return self._transform(homes, list_aware=False)

    def list_aware_features(self, homes):
        return self._transform(homes, list_aware=True)


class HistoricalSalesFeatures:
    """Trailing sale counts, mean prices and mean price per approximate sq ft.

    fit(train_sales) stores ONLY that fold's history. transform(homes) uses
    [listing_day - window, listing_day), measured by sale availability day.
    Same-day sales and the same ml_num are excluded. Dates are normalized to
    calendar days; timezone-aware dates are rejected to avoid implicit shifts.
    Set availability_lag_days if sales are published after the sale date.
    Sparse means remain NaN; broader groups are exposed separately, not filled
    from future/global target statistics. No category encoders are fitted here.
    """

    GROUPS = {
        'market': [],
        'type': ['house_type_name'],
        'community_type': ['municipality', 'community', 'house_type_name'],
    }

    def __init__(self, windows=(90, 365), min_sales=5, availability_lag_days=0):
        if not windows or any(not isinstance(w, (int, np.integer)) or w <= 0 for w in windows):
            raise ValueError('windows must contain positive integer day counts')
        if len(set(windows)) != len(windows):
            raise ValueError('windows must be unique')
        if not isinstance(min_sales, (int, np.integer)) or min_sales < 1:
            raise ValueError('min_sales must be a positive integer')
        if not isinstance(availability_lag_days, (int, np.integer)) or availability_lag_days < 0:
            raise ValueError('availability_lag_days must be a nonnegative integer')
        self.windows = tuple(windows)
        self.min_sales = min_sales
        self.availability_lag_days = availability_lag_days

    @staticmethod
    def _days(values):
        dates = pd.to_datetime(values, errors='coerce')
        if dates.dt.tz is not None:
            raise ValueError('Use timezone-naive local calendar dates for historical features')
        return dates.dt.normalize()

    @staticmethod
    def _keys(frame, columns):
        if not columns:
            return [()] * len(frame)
        # Missing geography must not become a fabricated shared neighborhood.
        return [tuple(row) if all(pd.notna(v) for v in row) else None
                for row in frame[columns].itertuples(index=False, name=None)]

    def fit(self, train_sales):
        """Fit a fresh history per validation fold; never pass validation/test sales."""
        sales = train_sales.reset_index(drop=True).copy()
        if sales['ml_num'].isna().any() or sales['ml_num'].duplicated().any():
            raise ValueError('Historical sales require nonmissing, unique ml_num')
        available = self._days(sales['date_sold']) + pd.Timedelta(days=self.availability_lag_days)
        price = pd.to_numeric(sales['sold_price'], errors='coerce')
        area = square_feet(sales['house_area'])
        ppsf = (price / area).replace([np.inf, -np.inf], np.nan)
        valid = available.notna() & np.isfinite(price) & price.gt(0)
        self.audit_ = {'input_rows': len(sales), 'excluded_rows': int((~valid).sum())}
        sales = sales.loc[valid].copy()
        sales['_available'] = available[valid].to_numpy(dtype='datetime64[ns]')
        sales['_price'] = price[valid].to_numpy(dtype=float)
        sales['_ppsf'] = ppsf[valid].to_numpy(dtype=float)
        self.groups_ = {}
        for scope, columns in self.GROUPS.items():
            buckets = {}
            for pos, key in enumerate(self._keys(sales, columns)):
                if key is not None:
                    buckets.setdefault(key, []).append(pos)
            groups = {}
            for key, positions in buckets.items():
                group = sales.iloc[positions].sort_values('_available', kind='stable')
                prices = group['_price'].to_numpy()
                per_foot = group['_ppsf'].to_numpy()
                values = np.column_stack([np.ones(len(group)), prices,
                                          np.isfinite(per_foot), np.nan_to_num(per_foot, nan=0)])
                cumulative = np.vstack([np.zeros(4), values.cumsum(axis=0)])
                ids = {identifier: pos for pos, identifier in enumerate(group['ml_num'])}
                groups[key] = (group['_available'].to_numpy(dtype='datetime64[ns]'),
                               cumulative, values, ids)
            self.groups_[scope] = groups
        return self

    def transform(self, homes):
        """Return fixed columns without reading the current homes' sale answers."""
        if not hasattr(self, 'groups_'):
            raise RuntimeError('Call fit(train_sales) before transform(homes)')
        frame = homes.reset_index(drop=True)
        days = self._days(frame['date_listed']).to_numpy(dtype='datetime64[ns]')
        identifiers = frame['ml_num'].to_numpy()
        output = {}
        for scope, columns in self.GROUPS.items():
            buckets = {}
            keys = self._keys(frame, columns)
            unknown = np.isnat(days) | np.array([key is None for key in keys], dtype=bool)
            for pos, key in enumerate(keys):
                if key is not None and not np.isnat(days[pos]):
                    buckets.setdefault(key, []).append(pos)
            for window in self.windows:
                # Unknown prediction date => unknown count; known date with no history => 0.
                count = np.where(unknown, np.nan, 0.0)
                price_mean = np.full(len(frame), np.nan)
                ppsf_count = count.copy()
                ppsf_mean = price_mean.copy()
                for key, positions in buckets.items():
                    history = self.groups_[scope].get(key)
                    if history is None:
                        continue
                    dates, cumulative, values, ids = history

                    q = days[positions]
                    right = np.searchsorted(dates, q, side='left')
                    left = np.searchsorted(dates, q - np.timedelta64(window, 'D'), side='left')
                    totals = cumulative[right] - cumulative[left]
                    for row, pos in enumerate(positions):
                        own = ids.get(identifiers[pos])
                        if own is not None and left[row] <= own < right[row]:
                            totals[row] -= values[own]
                    count[positions] = totals[:, 0]
                    ppsf_count[positions] = totals[:, 2]
                    enough = totals[:, 0] >= self.min_sales
                    enough_area = totals[:, 2] >= self.min_sales
                    means = np.full(len(positions), np.nan)
                    means[enough] = totals[enough, 1] / totals[enough, 0]
                    price_mean[positions] = means
                    means = np.full(len(positions), np.nan)
                    means[enough_area] = totals[enough_area, 3] / totals[enough_area, 2]
                    ppsf_mean[positions] = means
                prefix = f'hist_{scope}_{window}d'
                output[prefix + '_count'] = count
                output[prefix + '_mean_price'] = price_mean
                output[prefix + '_ppsf_count'] = ppsf_count
                output[prefix + '_mean_ppsf'] = ppsf_mean
        return pd.DataFrame(output, index=homes.index)

    def fit_transform(self, train_sales):
        """Training rows also receive only their strictly earlier history."""
        return self.fit(train_sales).transform(train_sales)


