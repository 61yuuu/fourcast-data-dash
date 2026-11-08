"""Evaluate saved out-of-sample housing predictions. Does not train models.

Usage: python evaluation.py --input validation_predictions.csv --output results
Dependencies: pandas, numpy, matplotlib (already in starter environment).
Percent metrics are expressed in percent units, e.g. 7.26 means 7.26%.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_predictions(path, model_labels=None, metadata=None):
    df = pd.read_csv(path, dtype={'ml_num': str, 'month': str})
    required = ['ml_num', 'track', 'month', 'house_type_name',
                'sold_price', 'predicted_price']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.empty or df[required].isna().any().any():
        raise ValueError('Input is empty or contains missing required values.')
    if 'model' not in df:
        df['model'] = 'provided_model'
    if df['model'].isna().any():
        raise ValueError('Missing model labels.')
    for track, label in (model_labels or {}).items():
        mask = (df['track'] == track) & (df['model'] == 'provided_model')
        if not mask.any():
            raise ValueError(f'No unnamed predictions for track {track}; keep existing model names.')
        df.loc[mask, 'model'] = label
    if metadata:
        meta = pd.read_csv(metadata, dtype={'ml_num': str})
        if not {'ml_num', 'community'}.issubset(meta.columns):
            raise ValueError('Metadata requires ml_num and community columns.')
        if meta.ml_num.isna().any():
            raise ValueError('Metadata contains missing home identifiers.')
        if (meta.groupby('ml_num').community.nunique() > 1).any():
            raise ValueError('Conflicting communities for the same metadata home.')
        meta = meta[['ml_num', 'community']].dropna(subset=['community']).drop_duplicates('ml_num')
        df = df.merge(meta.rename(columns={'community': '_metadata_community'}),
                      on='ml_num', how='left', validate='many_to_one')
        if 'community' in df:
            conflict = (df.community.notna() & df._metadata_community.notna()
                        & df.community.ne(df._metadata_community))
            if conflict.any():
                raise ValueError('Prediction and metadata community labels disagree.')
            df['community'] = df.community.fillna(df._metadata_community)
        else:
            df['community'] = df._metadata_community
        df = df.drop(columns='_metadata_community')
    if 'community' in df:
        df['community'] = df.community.replace(r'^\s*$', np.nan, regex=True)
        if (df.groupby('ml_num').community.nunique() > 1).any():
            raise ValueError('Conflicting communities for the same prediction home.')
    for col in ['sold_price', 'predicted_price']:
        df[col] = pd.to_numeric(df[col], errors='raise')
        if not np.isfinite(df[col]).all() or (df[col] <= 0).any():
            raise ValueError(f'{col} must contain finite positive prices.')
    if not df['month'].str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('month must have YYYY-MM format.')
    pd.to_datetime(df['month'], format='%Y-%m', errors='raise')
    if df.duplicated(['model', 'track', 'ml_num']).any():
        raise ValueError('Duplicate model/track/home records.')
    # A home must have consistent outcomes and metadata across model versions.
    for col in ['sold_price', 'month', 'house_type_name']:
        if (df.groupby('ml_num')[col].nunique() > 1).any():
            raise ValueError(f'Conflicting {col} for the same home.')
    # Within each track compare identical validation homes only.
    for track, subset in df.groupby('track'):
        sets = [set(g.ml_num) for _, g in subset.groupby('model')]
        if any(s != sets[0] for s in sets[1:]):
            raise ValueError(f'Models in {track} use different validation homes.')
    df['signed_error_pct'] = 100 * (df.predicted_price / df.sold_price - 1)
    df['absolute_error_pct'] = df.signed_error_pct.abs()
    # One fixed set of observed-price boundaries for all model versions.
    # These outcome-based groups are diagnostic only, never predictors.
    homes = df.drop_duplicates('ml_num').copy()
    homes['price_decile'] = pd.qcut(homes.sold_price, 10, labels=False,
                                    duplicates='drop') + 1
    if homes.price_decile.isna().any():
        raise ValueError('Too few distinct prices to form price groups.')
    df = df.merge(homes[['ml_num', 'price_decile']], on='ml_num', validate='many_to_one')
    return df


def summarize(df, keys):
    return df.groupby(keys, observed=True).agg(
        n=('ml_num', 'size'),
        MdAPE_pct=('absolute_error_pct', 'median'),
        MAPE_pct=('absolute_error_pct', 'mean'),
        median_signed_error_pct=('signed_error_pct', 'median'),
        overprediction_pct=('signed_error_pct', lambda x: 100 * (x > 0).mean()),
        error_q25_pct=('absolute_error_pct', lambda x: x.quantile(.25)),
        error_q75_pct=('absolute_error_pct', lambda x: x.quantile(.75)),
        median_sold_price=('sold_price', 'median'),
        min_sold_price=('sold_price', 'min'),
        max_sold_price=('sold_price', 'max'),
    ).reset_index()


def make_plot(table, group, dest):
    categories = sorted(table[group].unique())
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    lines = ['-', '--', '-.', ':']
    styles = { (model, track): i for track, t in table.groupby('track')
               for i, model in enumerate(sorted(t.model.unique())) }
    for (model, track), g in table.groupby(['model', 'track']):
        g = g.set_index(group).reindex(categories)
        label = f'{track} / {model}'
        i = styles[(model, track)]
        style = dict(marker=markers[i % len(markers)],
                     linestyle='None' if group == 'house_type_name' else lines[i % len(lines)])
        color = {'off_market': '#2878c8', 'list_aware': '#ef6c36'}.get(track)
        axes[0].plot(range(len(categories)), g.MdAPE_pct, label=label, color=color, **style)
        axes[1].plot(range(len(categories)), g.median_signed_error_pct, label=label, color=color, **style)
    axes[0].set_ylabel('MdAPE (%)')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Median signed error (%)')
    axes[1].axhline(0, color='black', linewidth=.8)
    for ax in axes:
        ax.grid(axis='y', alpha=.25)
    names = {'house_type_name': 'Property type', 'month': 'Validation month',
             'price_decile': 'Observed sold-price group (1 = cheapest)'}
    axes[1].set_xlabel(names[group])
    axes[1].set_xticks(range(len(categories)), [str(x) for x in categories],
                       rotation=25 if group != 'price_decile' else 0, ha='right')
    fig.suptitle(f'Validation errors by {names[group].lower()}')
    fig.tight_layout()
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='outputs/validation_predictions.csv')
    parser.add_argument('--output', default='evaluation_results')
    parser.add_argument('--metadata', help='Optional CSV with ml_num and community, e.g. member3_selected.csv')
    parser.add_argument('--model-label', action='append', default=[], metavar='TRACK=NAME',
                        help='Name predictions with a missing/default model label; repeat for each track.')
    parser.add_argument('--min-community-n', type=int, default=30)
    args = parser.parse_args()
    if args.min_community_n < 1:
        parser.error('--min-community-n must be positive.')
    labels = {}
    for item in args.model_label:
        track, sep, label = item.partition('=')
        if not sep or not track.strip() or not label.strip() or track in labels:
            parser.error('--model-label requires a unique TRACK=NAME.')
        labels[track] = label.strip()
    df = load_predictions(args.input, labels, args.metadata)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    overall = summarize(df, ['model', 'track'])
    overall.to_csv(out / 'overall_metrics.csv', index=False)
    notes = ['# Validation summary', '',
             f'Source: {Path(args.input).name}',
             f'Month range: {df.month.min()} to {df.month.max()}',
             f'Rows: {len(df):,}; unique homes: {df.ml_num.nunique():,}', '',
             '## Overall results', '']
    for _, r in overall.iterrows():
        notes.append(f'- {r["model"]} / {r.track}: n={r.n:,}, '
                     f'MdAPE={r.MdAPE_pct:.2f}%, MAPE={r.MAPE_pct:.2f}%, '
                     f'median signed error={r.median_signed_error_pct:+.2f}%.')
    for group in ['house_type_name', 'price_decile', 'month']:
        table = summarize(df, ['model', 'track', group])
        table.to_csv(out / f'errors_by_{group}.csv', index=False)
        make_plot(table, group, out / f'errors_by_{group}.png')
        notes.extend(['', f'## Highest median absolute error by {group}', ''])
        for (model, track), g in table.groupby(['model', 'track']):
            r = g.loc[g.MdAPE_pct.idxmax()]
            notes.append(f'- {model} / {track}: {r[group]}, n={r.n:,}, '
                         f'MdAPE={r.MdAPE_pct:.2f}%, '
                         f'median signed error={r.median_signed_error_pct:+.2f}%.')
    notes.extend(['', '## Community diagnostics', ''])
    if 'community' in df and df.community.notna().any():
        community = summarize(df, ['model', 'track', 'community'])
        community = community.sort_values(['model', 'track', 'MdAPE_pct'],
                                          ascending=[True, True, False])
        community.to_csv(out / 'errors_by_community.csv', index=False)
        eligible = community[community.n >= args.min_community_n]
        eligible.to_csv(out / 'errors_by_community_eligible.csv', index=False)
        notes.append(f'Ranking includes communities with at least {args.min_community_n} homes per model/track.')
        notes.append(f'Rows missing community (excluded from this analysis): {df.community.isna().sum():,}.')
        if args.metadata:
            notes.append(f'Community metadata source: {Path(args.metadata).name} (joined by ml_num).')
        for (model, track), g in community.groupby(['model', 'track']):
            ranked = g[g.n >= args.min_community_n]
            notes.extend(['', f'### {track} / {model}', '',
                          f'{len(g)} communities; {len(ranked)} meet the sample-size threshold.', ''])
            for _, r in ranked.head(10).iterrows():
                notes.append(f'- {r.community}: n={r.n:,}, MdAPE={r.MdAPE_pct:.2f}%, '
                             f'MAPE={r.MAPE_pct:.2f}%, median signed error={r.median_signed_error_pct:+.2f}%.')
    else:
        notes.append('Not evaluated: no community labels. Supply --metadata with ml_num and community.')
    notes.extend(['', '## Interpretation and limits', '',
        '- Positive signed error means overprediction; negative means underprediction.',
        '- Q25 and Q75 describe the middle 50% of absolute errors, not confidence intervals.',
        '- Price groups use observed outcomes for diagnosis only. Never use them as input features.',
        '- Check subgroup sample sizes. Small groups can have unstable error summaries.',
        '- Community names are used as supplied; confirm they uniquely identify your intended geographic groups.',
        '- The CSV cannot prove temporal separation or absence of leakage. Review training and feature code.',
        '- Matching homes do not prove matching training cutoffs. Confirm the same validation protocol with teammates.',
        '- Repeated tuning on these months makes them development results, not an untouched final test.',
        '- Synthetic-data performance does not establish accuracy on real Toronto transactions.',
        '- No confidence intervals are produced by this script. Transactions may share monthly market effects.',
    ])
    (out / 'validation_summary.md').write_text('\n'.join(notes) + '\n', encoding='utf-8')
    print(overall.to_string(index=False))
    print(f'Wrote results to {out.resolve()}')


if __name__ == '__main__':
    main()
