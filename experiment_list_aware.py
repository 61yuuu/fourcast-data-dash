"""Fixed, reproducible comparison; no hyperparameter search or test-label access.
Run with OMP_NUM_THREADS=2 and HF_HUB_OFFLINE=1 after the data is cached.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import data
import features
import models
from list_aware_models import ListAwareCandidate
from run import VALIDATION_MONTHS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/list_aware_v2'
KINDS = ['list_price', 'starter_tree', 'ridge', 'ratio_tree', 'catboost']


def make_model(kind):
    if kind == 'list_price':
        return models.ListPriceBaseline()
    if kind == 'starter_tree':
        return models.FeatureModel(features.list_aware_features)
    return ListAwareCandidate(kind)


def score(g):
    err = (g.predicted_price - g.sold_price).to_numpy()
    ape = np.abs(err) / g.sold_price.to_numpy() * 100
    return dict(n=len(g), mdape=float(np.median(ape)), mape=float(np.mean(ape)),
                mae_cad=float(np.abs(err).mean()), rmse_cad=float(np.sqrt((err**2).mean())),
                median_signed_pct=float(np.median(err / g.sold_price.to_numpy() * 100)),
                over_20pct=float(np.mean(ape > 20) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    listings = data.load('listings')
    all_results = []
    timing = []
    for month in VALIDATION_MONTHS:
        train = listings[listings.date_sold < month.start_time]
        homes = listings[listings.date_sold.dt.to_period('M') == month]
        assert train.date_sold.max() < month.start_time
        inputs = data.hide_answers(homes, 'list_aware')
        for kind in KINDS:
            start = time.perf_counter()
            model = make_model(kind).fit(train)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predicted = np.round(model.predict(inputs))
            predict_seconds = time.perf_counter() - start
            assert np.isfinite(predicted).all() and (predicted > 0).all()
            result = homes[['ml_num', 'sold_price', 'list_price', 'house_type_name', 'community']].copy()
            result['month'] = str(month)
            result['model'] = kind
            result['predicted_price'] = predicted
            all_results.append(result)
            timing.append(dict(month=str(month), model=kind, fit_seconds=fit_seconds,
                               predict_seconds=predict_seconds, n_train=len(train)))
            # Keep completed folds if an external dependency fails later.
            result.to_csv(OUT / f'{kind}_{month}.csv', index=False)
            print(f'{month} {kind}: MdAPE {score(result)["mdape"]:.3f}%, fit {fit_seconds:.2f}s', flush=True)
    predictions = pd.concat(all_results, ignore_index=True)
    predictions.to_csv(OUT / 'validation_predictions.csv', index=False)
    pd.DataFrame(timing).to_csv(OUT / 'timing.csv', index=False)
    summary = pd.DataFrame([dict(model=k, **score(g)) for k, g in predictions.groupby('model')])
    summary = summary.sort_values(['mdape', 'mape'])
    summary.to_csv(OUT / 'summary.csv', index=False)
    for columns, name in [(['model', 'month'], 'monthly'), (['model', 'house_type_name'], 'by_property_type')]:
        report = [dict(zip(columns, keys), **score(g)) for keys, g in predictions.groupby(columns)]
        pd.DataFrame(report).to_csv(OUT / f'{name}.csv', index=False)
    # Price bands are descriptive only, never model inputs.
    predictions['price_band'] = pd.cut(predictions.sold_price, [0, 600000, 1000000, 2000000, np.inf],
                                      labels=['under_600k', '600k_to_1m', '1m_to_2m', 'over_2m'])
    pd.DataFrame([dict(model=k, price_band=str(b), **score(g)) for (k,b),g in
                  predictions.groupby(['model','price_band'], observed=True)]).to_csv(OUT / 'by_price.csv', index=False)
    winner = str(summary.iloc[0].model)
    (OUT / 'selection.json').write_text(json.dumps(dict(selected_model=winner,
        rule='Lowest pooled validation MdAPE; MAPE breaks ties. Not an independent unseen holdout.',
        validation_months=[str(x) for x in VALIDATION_MONTHS], candidates=KINDS), indent=2))
    # No access to hidden test outcomes; predict only after comparison is complete.
    test = data.load('list_aware')
    final = make_model(winner).fit(listings)
    pd.DataFrame({'ml_num':test.ml_num, 'predicted_price':np.round(final.predict(test))}).to_csv(
        OUT / 'submission_list_aware.csv', index=False)
    from check_submission import check_file
    errors = check_file(OUT / 'submission_list_aware.csv', test.ml_num.tolist())
    assert not errors, errors
    print(summary.to_string(index=False), flush=True)
    print(f'Selected {winner}; submission check OK ({len(test)} homes).', flush=True)


if __name__ == '__main__':
    main()
