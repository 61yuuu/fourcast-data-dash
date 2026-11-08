"""Compare baseline and FeatureModel on identical walk-forward validation homes.
Run after run.py validate. Results are written to reports/ and outputs/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models
import run

ROOT = Path(__file__).resolve().parent


def main():
    if not isinstance(models.make_model("list_aware"), models.FeatureModel):
        raise RuntimeError("First-round comparison only. Run experiment_list_aware.py for the new candidates.")
    predictions = pd.read_csv(ROOT / 'outputs/validation_predictions.csv')
    rows = predictions.loc[predictions['track'].eq('list_aware')].copy()
    listings = data.load('listings')
    rows = rows.merge(listings[['ml_num', 'list_price', 'community', 'municipality']],
                      on='ml_num', how='left', validate='one_to_one')
    assert rows['list_price'].notna().all()
    assert np.isfinite(rows['predicted_price']).all() and (rows['predicted_price'] > 0).all()
    rows['baseline_price'] = rows['list_price']
    summary = {}
    monthly = []
    for name, column in [('ListPriceBaseline', 'baseline_price'), ('FeatureModel', 'predicted_price')]:
        scored = rows.copy()
        scored['predicted_price'] = rows[column]
        summary[name] = run.metrics(scored)
        for month, group in scored.groupby('month'):
            errors = 100 * (group.predicted_price - group.sold_price).abs() / group.sold_price
            monthly.append(dict(model=name, month=month, n=len(group),
                                mdape=float(errors.median()), mape=float(errors.mean())))
    rows['baseline_ape_pct'] = 100 * (rows.baseline_price - rows.sold_price).abs() / rows.sold_price
    rows['model_ape_pct'] = 100 * (rows.predicted_price - rows.sold_price).abs() / rows.sold_price
    rows.to_csv(ROOT / 'outputs/list_aware_comparison_predictions.csv', index=False)
    pd.DataFrame(monthly).to_csv(ROOT / 'reports/list_aware_monthly.csv', index=False)
    (ROOT / 'reports/list_aware_comparison.json').write_text(json.dumps(summary, indent=2))
    lines = ['# List-aware: first FeatureModel experiment', '',
             'Validation: December 2025 to May 2026; train on sales before each validation month.',
             'Both models are evaluated on the same homes. No hyperparameter search was performed.', '',
             '| Model | Homes | MdAPE | MAPE |', '|---|---:|---:|---:|']
    for name, score in summary.items():
        lines.append(f"| {name} | {score['n']:,} | {score['mdape']:.2f}% | {score['mape']:.2f}% |")
    baseline = summary['ListPriceBaseline']['mdape']
    model = summary['FeatureModel']['mdape']
    lines += ['', f'MdAPE reduction: {baseline - model:.2f} percentage points ({100 * (baseline-model)/baseline:.1f}% relative).', '',
              'Features: numeric area (range midpoint), bedrooms, washrooms, parking, latitude, longitude,',
              'months since January 2021 (from listing date), six property-type indicators, list price.', '',
              'Model: HistGradientBoostingRegressor(random_state=0), default parameters, trained on log(sold_price).',
              'The sale price, sale date, days on market and external AVM estimate are not input features.', '',
              'Limitations: starter features only; no team-specific feature code was supplied. Validation is not the hidden test.',
              'Future-month prices may change; the official monthly protocol is retained without modification.',
              'Community names are included in the exported predictions for teammate error analysis, not as model inputs.', '',
              'Reproduce: run.py validate, run.py report, then compare_list_aware.py.']
    (ROOT / 'reports/list_aware_comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
