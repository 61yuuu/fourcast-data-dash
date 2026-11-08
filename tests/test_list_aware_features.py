import numpy as np
import pandas as pd
import pytest
import data
from list_aware_feature_models import HistoricalRatios, FeatureRatioModel
from test_starter import make_listings


def test_history_excludes_current_and_later_months():
    rows = make_listings(120)
    rows['date_sold'] = pd.to_datetime(['2025-01-15']*40+['2025-02-15']*40+['2025-03-15']*40)
    rows['house_type_name'] = 'Detached'
    rows['municipality'] = 'Town'
    rows['community'] = 'Community'
    rows['list_price'] = 100.0
    rows['sold_price'] = [90.0]*40 + [110.0]*40 + [300.0]*40
    h = HistoricalRatios()
    X = h.fit_transform(rows)
    np.testing.assert_allclose(X.iloc[:40].recent_ratio, 1.0)
    np.testing.assert_allclose(X.iloc[40:80].recent_ratio, 0.9)
    np.testing.assert_allclose(X.iloc[80:].recent_ratio, 1.0)
    changed = rows.copy()
    changed.loc[80:, 'sold_price'] = 1000000
    changed_X = HistoricalRatios().fit_transform(changed)
    pd.testing.assert_frame_equal(X, changed_X)
    predicted_features = h.transform(data.hide_answers(rows.iloc[:3], 'list_aware'))
    assert np.isfinite(predicted_features.to_numpy()).all()


def test_history_rare_group_uses_backoff_and_no_history_is_safe():
    rows = make_listings(30)
    rows['date_sold'] = pd.Timestamp('2025-01-15')
    rows['house_type_name'] = 'Detached'
    rows['municipality'] = 'Town'
    rows['community'] = ['Rare'] + ['Common']*29
    rows['list_price'] = 100.0
    rows['sold_price'] = [900.0]+[90.0]*29
    h=HistoricalRatios()
    h.fit_transform(rows)
    homes=data.hide_answers(rows.iloc[:1], 'list_aware').copy()
    homes['community']='Unseen'
    np.testing.assert_allclose(h.transform(homes).recent_ratio, 0.9)


@pytest.mark.parametrize('variant', ['ratio_base', 'property', 'region', 'history'])
def test_new_features_handle_missing_lots_and_unseen_categories(variant):
    train=make_listings(200)
    homes=data.hide_answers(make_listings(5,seed=13,first_id=9999),'list_aware')
    homes['community']='NeverSeen'
    homes['house_style']='UnseenStyle'
    model=FeatureRatioModel(variant).fit(train)
    prices=model.predict(homes)
    assert np.isfinite(prices).all() and (prices>0).all()
    # Even if forbidden columns are accidentally present, they cannot affect inference.
    polluted=homes.assign(sold_price=1e12,date_sold=pd.Timestamp('2099-01-01'),days_on_market=9999,
                          external_avm_estimate=1e12)
    np.testing.assert_allclose(prices,model.predict(polluted))
