import numpy as np
import pandas as pd
import pytest

import data
import member1_features as features
from test_starter import make_listings


def test_history_matches_direct_filter_with_duplicates_and_missing_dates():
    train = make_listings(90)
    # Missing/invalid observations must not contribute to price or area means.
    train.loc[1, 'house_area'] = 'bad'
    train.loc[2, 'sold_price'] = np.inf
    train.loc[3, 'date_sold'] = pd.NaT
    homes = make_listings(15, seed=3, first_id=500)
    homes = pd.concat([homes, train.iloc[[0]].assign(date_listed=pd.Timestamp('2026-06-01'))])
    homes.loc[homes.index[4], 'date_listed'] = pd.NaT
    homes.index = [7] * len(homes)
    before = train.copy(deep=True)
    history = features.HistoricalSalesFeatures(windows=(90, 365), min_sales=1).fit(train)
    got = history.transform(data.hide_answers(homes, 'off_market'))
    assert got.index.equals(homes.index)
    pd.testing.assert_frame_equal(train, before)
    for pos, (_, home) in enumerate(homes.iterrows()):
        for scope, columns in history.GROUPS.items():
            for window in history.windows:
                prefix = f'hist_{scope}_{window}d'
                if pd.isna(home.date_listed):
                    assert got.iloc[pos].isna().all()
                    continue
                day = home.date_listed.normalize()
                eligible = train.loc[
                    (train.date_sold.dt.normalize() < day)
                    & (train.date_sold.dt.normalize() >= day - pd.Timedelta(days=window))
                    & np.isfinite(train.sold_price) & train.sold_price.gt(0)
                    & train.ml_num.ne(home.ml_num)
                ]
                for col in columns:
                    eligible = eligible.loc[eligible[col] == home[col]]
                row = got.iloc[pos]
                assert row[prefix + '_count'] == len(eligible)
                expected = eligible.sold_price.mean()
                assert np.isclose(row[prefix + '_mean_price'], expected, equal_nan=True)
                per_foot = eligible.sold_price / features.square_feet(eligible.house_area)
                assert row[prefix + '_ppsf_count'] == per_foot.notna().sum()
                assert np.isclose(row[prefix + '_mean_ppsf'], per_foot.mean(), equal_nan=True)


def test_strict_day_boundaries_publication_lag_and_target_invariance():
    train = make_listings(4).assign(
        date_sold=pd.to_datetime(['2025-01-01', '2025-01-02', '2025-04-01', '2025-04-02']),
        sold_price=[100., 200., 300., 1e12], house_area='100')
    home = train.iloc[[0]].assign(ml_num='query', date_listed=pd.Timestamp('2025-04-01 15:00'))
    history = features.HistoricalSalesFeatures(windows=(90,), min_sales=1).fit(train)
    got = history.transform(home)
    assert got.iloc[0]['hist_market_90d_count'] == 2
    assert got.iloc[0]['hist_market_90d_mean_price'] == 150
    changed = home.assign(sold_price=1e20, date_sold=pd.Timestamp('2099-01-01'),
                          external_avm_estimate=1e20, list_price=1e20)
    pd.testing.assert_frame_equal(got, history.transform(changed))
    # A same-day source remains excluded even when it has a midnight timestamp.
    lagged = features.HistoricalSalesFeatures(windows=(90,), min_sales=1,
                                             availability_lag_days=89).fit(train)
    assert lagged.transform(home).iloc[0]['hist_market_90d_count'] == 1
    altered = train.copy()
    altered.loc[2:, 'sold_price'] = 9e15
    pd.testing.assert_frame_equal(got, history.fit(altered).transform(home))


def test_empty_history_sparse_groups_unknowns_and_refit():
    train = make_listings(10)
    homes = train.iloc[:2].assign(community='UNSEEN')
    with pytest.raises(RuntimeError):
        features.HistoricalSalesFeatures().transform(homes)
    history = features.HistoricalSalesFeatures().fit(train.iloc[:0])
    got = history.transform(homes)
    assert got.filter(like='_count').eq(0).all().all()
    assert got.filter(like='_mean_').isna().all().all()
    history.fit(train)
    history.fit(train.iloc[:0])
    pd.testing.assert_frame_equal(got, history.transform(homes))
    homes.loc[homes.index[0], 'community'] = None
    got = history.transform(homes)
    assert got.iloc[0].filter(like='community_type').isna().all()
    assert got.iloc[0]['hist_market_90d_count'] == 0
    with pytest.raises(ValueError, match='unique'):
        history.fit(pd.concat([train, train]))



def test_area_and_missing_lot_values():
    area = features.square_feet(pd.Series(['600-799','1,200','900-700','0','700+',None]))
    assert area.iloc[:2].tolist() == [699.5,1200.]
    assert area.iloc[2:].isna().all()
    homes = make_listings(3)
    homes['lot_front'] = [20.,np.nan,-1.]
    homes['lot_depth'] = [50.,np.nan,30.]
    X = features.off_market_features(homes)
    assert X.lot_area.iloc[0] == 1000
    assert X.lot_area.iloc[1:].isna().all()
    assert X.lot_front_missing.tolist() == [0,1,1]


def test_fitted_categories_unknowns_schema_and_answer_invariance():
    train = make_listings(50)
    builder = features.FeatureBuilder(include_history=True).fit(train)
    homes = train.iloc[:3].copy()
    homes.index = [9,9,1]
    homes['community'] = ['UNSEEN', None, 'Testville']
    original = homes.copy(deep=True)
    off = builder.off_market_features(data.hide_answers(homes,'off_market'))
    aware = builder.list_aware_features(data.hide_answers(homes,'list_aware'))
    pd.testing.assert_frame_equal(off,aware.drop(columns=['list_price','log_list_price','list_price_missing']))
    assert off.cat_community_unknown.tolist() == [1,0,0]
    assert off.cat_community_missing.tolist() == [0,1,0]
    assert off.index.equals(homes.index)
    assert list(off) == list(builder.off_market_features(train))
    changed = homes.assign(sold_price=1e20,list_price=1e20,date_sold=pd.Timestamp('2099-01-01'),
                           days_on_market=999,external_avm_estimate=1e20)
    pd.testing.assert_frame_equal(off,builder.off_market_features(changed))
    pd.testing.assert_frame_equal(homes, original)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in off.dtypes)
    assert not np.isinf(off.to_numpy(dtype=float)).any()
    assert not set(data.AFTER_SALE_COLUMNS) & set(off)
    with pytest.raises(RuntimeError): features.FeatureBuilder().off_market_features(homes)


def test_original_feature_model_accepts_both_apis():
    import models
    train = make_listings(200)
    builder = features.FeatureBuilder(include_history=True).fit(train)
    for track in ['off_market','list_aware']:
        for maker in [getattr(features,track+'_features'),getattr(builder,track+'_features')]:
            model = models.FeatureModel(maker).fit(train)
            homes = data.hide_answers(make_listings(8,seed=3,first_id=1000),track)
            predicted = model.predict(homes)
            assert len(predicted) == len(homes)
            assert np.all(np.isfinite(predicted) & (predicted>0))
