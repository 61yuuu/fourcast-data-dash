"""Candidates must predict from real test schemas, including missing numeric area."""
import numpy as np
import pytest
import data
from list_aware_models import ListAwareCandidate
from test_starter import make_listings


@pytest.mark.parametrize('kind', ['ridge', 'ratio_tree', 'catboost'])
def test_candidate_predicts_unseen_test_schema_with_missing_area(kind):
    if kind == 'catboost':
        pytest.importorskip('catboost', reason='Optional experiment: install requirements-experiments.txt')
    train = make_listings(200)
    train.loc[0, 'house_area'] = None
    homes = data.hide_answers(make_listings(8, seed=12, first_id=5000), 'list_aware')
    homes.loc[0, 'house_area'] = None
    model = ListAwareCandidate(kind).fit(train)
    predicted = model.predict(homes)
    assert predicted.shape == (len(homes),)
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    # Prediction must not depend on row order or a home's position in a batch.
    reversed_prediction = model.predict(homes.iloc[::-1])[::-1]
    np.testing.assert_allclose(predicted, reversed_prediction)
