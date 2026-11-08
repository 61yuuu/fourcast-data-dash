"""Verify regenerated predictions against official data and archived team results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import data
import models
from check_submission import check_all

ROOT = Path(__file__).resolve().parent


def main():
    predictions = pd.read_csv(ROOT / 'outputs/validation_predictions.csv')
    listings = data.load('listings')
    months = pd.period_range('2025-12', '2026-05', freq='M')
    truth = listings[listings.date_sold.dt.to_period('M').isin(months)]
    results = {}
    for track, member in [('off_market', 'member2'), ('list_aware', 'member3')]:
        rows = predictions[predictions.track == track]
        if rows.ml_num.duplicated().any() or set(rows.ml_num) != set(truth.ml_num):
            raise ValueError(f'{track}: incomplete or duplicate validation IDs')
        joined = rows.merge(truth[['ml_num', 'sold_price', 'date_sold']], on='ml_num',
                            validate='one_to_one', suffixes=('', '_truth'))
        if not np.array_equal(joined.sold_price, joined.sold_price_truth):
            raise ValueError(f'{track}: saved outcomes disagree with official data')
        if not np.array_equal(joined.month, joined.date_sold.dt.to_period('M').astype(str)):
            raise ValueError(f'{track}: saved months disagree with official data')
        prices = rows.predicted_price.to_numpy()
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError(f'{track}: invalid validation prices')
        ape = 100 * (rows.predicted_price / rows.sold_price - 1).abs()
        reference = pd.read_csv(ROOT / 'reference_results' / member / 'validation_predictions.csv')
        reference = reference[reference.model == models.model_name(track)]
        old_ape = 100 * (reference.predicted_price / reference.sold_price - 1).abs()
        compared = rows.merge(reference[['ml_num', 'predicted_price']], on='ml_num',
                              validate='one_to_one', suffixes=('', '_reference'))
        results[track] = {'model': models.model_name(track), 'n': len(rows),
                          'mdape_pct': float(ape.median()), 'mape_pct': float(ape.mean()),
                          'reference_mdape_pct': float(old_ape.median()),
                          'mdape_change_percentage_points': float(ape.median() - old_ape.median()),
                          'predictions_differing_from_reference': int((compared.predicted_price != compared.predicted_price_reference).sum())}
    if not check_all(ROOT / 'submissions'):
        raise ValueError('Submission validation failed')
    result = {'validation': results, 'submission_check': 'passed',
              'environment': json.loads((ROOT / 'outputs/validation_metadata.json').read_text()),
              'note': 'Results are development validation. Dependency versions can change fitted predictions.'}
    (ROOT / 'docs/run_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
