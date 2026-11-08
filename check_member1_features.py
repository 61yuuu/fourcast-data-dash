"""Usage: python check_member1_features.py TRAIN.parquet OFF_MARKET.parquet LIST_AWARE.parquet"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import member1_features as features


def main(paths):
    train, off, aware = [pd.read_parquet(p) for p in paths]
    builders = [features.FeatureBuilder().fit(train),
                features.FeatureBuilder(include_history=True).fit(train)]
    lines = ['# Actual compatibility checks / 实际兼容性检查', '',
             f'Training rows / 训练行数: {len(train):,}', '',
             '| Track | Rows | Stateless base | Fitted location/property | With history |',
             '|---|---:|---:|---:|---:|']
    for track, homes in [('off_market',off),('list_aware',aware)]:
        name = track + '_features'
        results = [getattr(features,name)(homes)]
        for builder in builders:
            X = getattr(builder,name)(homes)
            T = getattr(builder,name)(train.iloc[:100])
            assert list(X) == list(T)
            results.append(X)
        for X in results:
            assert X.index.equals(homes.index) and len(X) == len(homes)
            assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes)
            assert not np.isinf(X.to_numpy(dtype=float)).any()
            assert not {'sold_price','date_sold','days_on_market','external_avm_estimate',
                        'external_avm_estimate_date'} & set(X)
            if track == 'off_market':
                assert not {'list_price','log_list_price'} & set(X)
        lines.append(f'| {track} | {len(homes):,} | {results[0].shape[1]} | '
                     f'{results[1].shape[1]} | {results[2].shape[1]} |')
    for builder in builders:
        for name in ['off_market_features','list_aware_features']:
            X = getattr(builder,name)(train)
            assert not np.isinf(X.to_numpy(dtype=float)).any()
            assert len(X) == len(train) and X.index.equals(train.index)
    lines += ['', 'PASS: all-numeric output, schema, row/index preservation, no infinities,',
              'and answer-column exclusion; full training transforms passed for both tracks.',
              '通过：全数值输出、固定列、行索引保留、无无穷值、排除答案列；两 Track 完整训练转换通过。',
              '', 'No new accuracy scores were computed. / 本次没有重新计算预测精度。']
    Path(__file__).with_name('validation.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
