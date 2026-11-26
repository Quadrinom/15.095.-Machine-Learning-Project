"""Compare real frontier results with the placebo null (task 7).

Run:   python placebo_summary.py [frontier_<tag>.csv ...]   (default: frontier_dr_rf.csv)
Input: frontier_dr_rf_placebo*.csv (opt_frontier.jl on placebo rewards: class types shuffled
       within schools, opt_rewards.py --placebo --seed N) and the real frontier file(s)
Prints, per share q: real OPT - random, and the placebo mean, sd and 95th percentile of
OPT - random. Then the best share of each real run against the best share of every
placebo (picking the most favourable budget is part of the procedure), as a one-sided
p-value: (1 + #placebos with max >= real max) / (1 + #placebos).

Placebo OPT - random measures how far apart policies land by chance through the whole
pipeline: reward estimation, tree fitting and held-out evaluation.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent


def opt_minus_random(path):
    r = pd.read_csv(path)
    tot = r.groupby(['q', 'policy']).sum_value.sum().unstack()
    n = r[r.policy == 'opt'].groupby('q').n.sum()
    return (tot.opt - tot.random) / n


placebo_files = sorted(HERE.glob('frontier_dr_rf_placebo*.csv'))
null = pd.DataFrame({f.stem.removeprefix('frontier_'): opt_minus_random(f) for f in placebo_files})
print(f'{null.shape[1]} placebo shuffles: {", ".join(null.columns)}')

real_files = sys.argv[1:] or ['frontier_dr_rf.csv']
real = pd.DataFrame({f.removeprefix('frontier_').removesuffix('.csv'): opt_minus_random(HERE / f)
                     for f in real_files})
table = real.assign(placebo_mean=null.mean(axis=1), placebo_sd=null.std(axis=1),
                    placebo_p95=null.quantile(0.95, axis=1))
print('\nOPT - random by share of students in small classes (SD units):')
print(table.round(3).to_string())

null_max = null.max()
print(f'\nBest share per placebo: mean {null_max.mean():.3f}, '
      f'95th percentile {null_max.quantile(0.95):.3f}, max {null_max.max():.3f}')
for name, col in real.items():
    p = (1 + (null_max >= col.max()).sum()) / (1 + len(null_max))
    print(f'{name}: best share q={col.idxmax():.1f}, OPT - random {col.max():+.3f}, '
          f'one-sided p = {p:.3f}')
