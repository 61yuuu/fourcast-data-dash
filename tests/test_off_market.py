import numpy as np
import pandas as pd
import pytest
import data
from member1_features import HistoricalSalesFeatures, square_feet
from off_market_model import OffMarketModel, CANDIDATES


def homes(n=220, seed=0):
    rng = np.random.default_rng(seed)
    sale = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(1, 500, n), unit="D")
    return pd.DataFrame({"ml_num": [f"{seed}-{i}" for i in range(n)], "date_sold": sale,
        "date_listed": sale-pd.to_timedelta(rng.integers(1, 60, n),unit="D"),
        "sold_price": rng.uniform(300000, 1500000, n), "list_price": rng.uniform(300000, 1500000,n),
        "house_type_name": rng.choice(["Detached", "Condo Apt"], n), "house_style": "Apartment",
        "municipality": "Town", "community": rng.choice(["A", "B"],n), "house_area": "600-799",
        "bedrooms": 2, "washrooms": 1, "parking_total": 1, "garage": 0., "lot_front": np.nan,
        "lot_depth": np.nan, "latitude": 43.7+rng.normal(0,.01,n), "longitude": -79.4+rng.normal(0,.01,n),
        "remarks": "Renovated kitchen with stainless appliances and balcony."})


@pytest.mark.parametrize("variant", CANDIDATES)
def test_prediction_ignores_forbidden_fields_and_batch_order(variant):
    train=homes(); test=data.hide_answers(homes(8,seed=12),"off_market")
    test["community"]="Unseen"; test.loc[0,"house_area"]=None
    m=OffMarketModel(variant).fit(train)
    p=m.predict(test)
    assert p.shape==(8,) and np.isfinite(p).all() and (p>0).all()
    polluted=test.assign(list_price=1e15,sold_price=1e15,date_sold=pd.Timestamp("2099-01-01"),
                         days_on_market=999, external_avm_estimate=1e15)
    np.testing.assert_allclose(p,m.predict(polluted))
    np.testing.assert_allclose(p,m.predict(test.iloc[::-1])[::-1])
    banned={"list_price","log_list_price","sold_price","date_sold","days_on_market","external_avm_estimate"}
    assert not banned.intersection(m.feature_names_)


def test_history_excludes_same_day_future_and_own_sale():
    rows=homes(6)
    rows["date_sold"]=pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03"]*2)
    rows["community"]=["A"]*3+["B"]*3
    rows["house_type_name"]="Detached"
    rows["date_listed"]=pd.Timestamp("2025-01-03")
    rows["sold_price"]=[100.,200.,1000.,400.,500.,2000.]
    query=data.hide_answers(rows.iloc[[2,5]],"off_market")
    h=HistoricalSalesFeatures(windows=(90,),min_sales=1).fit(rows)
    X=h.transform(query)
    np.testing.assert_allclose(X.hist_community_type_90d_mean_price,[150.,450.])
    assert set(h.groups_)=={"market","type","community_type"}
    changed=rows.copy(); changed.loc[[2,5],"sold_price"]=1e15
    pd.testing.assert_frame_equal(X,HistoricalSalesFeatures(windows=(90,),min_sales=1).fit(changed).transform(query))
    own=query.iloc[[0]].copy(); own["ml_num"]=rows.iloc[0].ml_num
    assert h.transform(own).hist_community_type_90d_count.iloc[0]==1


def test_corrected_area_parser():
    x=square_feet(pd.Series(["1,200"," 600-799 ","900-700","0",None]))
    np.testing.assert_allclose(x.iloc[:2],[1200.,699.5])
    assert x.iloc[2:].isna().all()
