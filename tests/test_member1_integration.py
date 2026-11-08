import numpy as np
import pandas as pd
import pytest
import data
from member1_features import square_feet, HistoricalSalesFeatures
from list_aware_member1_model import Member1RatioModel
from test_starter import make_listings


def history_fixture():
    rows=make_listings(8)
    rows['municipality']='Town'
    rows['house_type_name']='Detached'
    rows['community']=['A']*4+['B']*4
    rows['house_area']='1,000'
    rows['list_price']=100.0
    rows['sold_price']=[90.,110.,200.,300.,50.,70.,80.,90.]
    rows['date_sold']=pd.to_datetime(['2025-01-01','2025-01-10','2025-01-20','2025-02-01']*2)
    rows['date_listed']=pd.Timestamp('2025-01-20')
    return rows


def test_area_cleanup_handles_commas_and_invalid_ranges():
    a=square_feet(pd.Series(['1,200',' 600-799 ','900-700','0',None]))
    np.testing.assert_allclose(a.iloc[:2],[1200,699.5])
    assert a.iloc[2:].isna().all()


def test_history_retains_all_scopes_and_all_communities():
    rows=history_fixture()
    h=HistoricalSalesFeatures(windows=(90,),min_sales=1).fit(rows)
    assert set(h.groups_)=={'market','type','community_type'}
    assert len(h.groups_['community_type'])==2
    homes=data.hide_answers(rows.iloc[[2,6]],'list_aware')
    X=h.transform(homes)
    np.testing.assert_allclose(X.hist_community_type_90d_count,[2,2])
    np.testing.assert_allclose(X.hist_community_type_90d_mean_price,[100.,60.])
    np.testing.assert_allclose(X.hist_market_90d_mean_price,[80.,80.])
    np.testing.assert_allclose(X.hist_community_type_90d_mean_ppsf,[.1,.06])
    # Current-day and future transactions must not affect these features.
    changed=rows.copy()
    changed.loc[[2,3,6,7],'sold_price']=1e12
    Y=HistoricalSalesFeatures(windows=(90,),min_sales=1).fit(changed).transform(homes)
    pd.testing.assert_frame_equal(X,Y)


def test_history_window_and_self_exclusion_with_available_lag():
    rows=history_fixture().iloc[:4].copy()
    rows['date_listed']=pd.Timestamp('2025-01-15')
    # First row's own ID must not count even if its sale precedes the query.
    homes=data.hide_answers(rows.iloc[[0]],'list_aware')
    X=HistoricalSalesFeatures(windows=(10,),min_sales=1).fit(rows).transform(homes)
    assert X.hist_market_10d_count.iloc[0]==1
    assert X.hist_market_10d_mean_price.iloc[0]==110
    Y=HistoricalSalesFeatures(windows=(10,),min_sales=1,availability_lag_days=10).fit(rows).transform(homes)
    assert Y.hist_market_10d_count.iloc[0]==0


@pytest.mark.parametrize('variant',['member1_base','member1_our_history','member1_sales_history','member1_both_history'])
def test_integration_has_stable_schema_and_ignores_prediction_answers(variant):
    train=make_listings(250)
    homes=data.hide_answers(make_listings(5,seed=17,first_id=9000),'list_aware')
    homes['community']='Unseen'
    homes.loc[0,'house_area']='1,200'
    m=Member1RatioModel(variant).fit(train)
    p=m.predict(homes)
    assert np.isfinite(p).all() and (p>0).all()
    polluted=homes.assign(sold_price=1e15,date_sold=pd.Timestamp('2099-01-01'),external_avm_estimate=1e15)
    np.testing.assert_allclose(p,m.predict(polluted))
    assert list(m.make_features(homes))==m.feature_names_
