"""Compare real frontier results with the placebo null (task 7).

Run:   python placebo_summary.py [frontier_<tag>.csv ...]   (default: frontier_dr_rf.csv)
Input: frontier_dr_rf_placebo*.csv (opt_frontier.jl on placebo rewards: class types shuffled
       within schools, opt_rewards.py --placebo --seed N) and the real frontier file(s)
Prints, per share q: real OPT - random, and the placebo mean and 95th percentile of
OPT - random. Then the best share of each real run against the best share of every
placebo (picking the most favourable budget is part of the procedure), as a one-sided
p-value: (1 + #placebos with max >= real max) / (1 + #placebos).
Each real run is compared with the placebos fitted on the same features: a run whose name
ends in __drop-<features> uses frontier_dr_rf_placebo*__drop-<features>.csv; every other
run uses the placebos with all features (the files without __drop).

Placebo OPT - random measures how far apart policies land by chance through the whole
pipeline: reward estimation, tree fitting and held-out evaluation.
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def opt_minus_random(path):
    r = pd.read_csv(path)
    tot = r.groupby(['q', 'policy']).sum_value.sum().unstack()
    n = r[r.policy == 'opt'].groupby('q').n.sum()
    return (tot.opt - tot.random) / n


def drop_suffix(stem):
    return stem[stem.index('__drop-'):] if '__drop-' in stem else ''


def label(suffix):
    return suffix.removeprefix('__') or 'all features'


nulls = {}  # drop suffix ('' = all features) -> DataFrame (q x placebo shuffle)
for f in sorted(HERE.glob('frontier_dr_rf_placebo*.csv')):
    nulls.setdefault(drop_suffix(f.stem), {})[f.stem.removeprefix('frontier_')] = opt_minus_random(f)
nulls = {k: pd.DataFrame(v) for k, v in nulls.items()}
for k, null in nulls.items():
    print(f'Placebo null ({label(k)}): {null.shape[1]} shuffles')

real_files = sys.argv[1:] or ['frontier_dr_rf.csv']
real = pd.DataFrame({f.removeprefix('frontier_').removesuffix('.csv'): opt_minus_random(HERE / f)
                     for f in real_files})
table = real.copy()
for k, null in nulls.items():
    table[f'placebo mean ({label(k)})'] = null.mean(axis=1)
    table[f'placebo p95 ({label(k)})'] = null.quantile(0.95, axis=1)
print('\nOPT - random by share of students in small classes (SD units):')
print(table.round(3).T.to_string())

for k, null in nulls.items():
    null_max = null.max()
    print(f'\nBest share per placebo ({label(k)}): mean {null_max.mean():.3f}, '
          f'95th percentile {null_max.quantile(0.95):.3f}, max {null_max.max():.3f}')
print()
for name, col in real.items():
    k = drop_suffix(name)
    if k not in nulls:
        print(f'{name}: no placebo null with the same features')
        continue
    null_max = nulls[k].max()
    p = (1 + (null_max >= col.max()).sum()) / (1 + len(null_max))
    print(f'{name}: best share q={col.idxmax():.1f}, OPT - random {col.max():+.3f}, '
          f'one-sided p = {p:.3f} (vs {len(null_max)} placebos, {label(k)})')
