"""Summarize per-school capacity scenarios (task 10) with school-level bootstrap intervals.

Run:   python capacity_summary.py [capacity_<name>.csv ...]   (default: capacity_dr_rf.csv)
Input: capacity_*.csv from opt_capacity.jl (one row per held-out school x scale x policy)
       and, for the placebo yardstick, capacity_dr_rf_placebo*.csv if present
Prints, per capacity scale c (seats = c x observed STAR small and aide seats per school):
  - share of students in small and aide classes (the same for opt, random and
    free-lunch-first, by construction);
  - held-out value of OPT, random, free-lunch-first, and the tree's own prescription
    ignoring capacities (tree_raw);
  - OPT - random and OPT - free-lunch-first with 95% intervals (whole schools resampled,
    B = 2000, seed 15; trees held fixed);
  - quota violations of tree_raw: schools where it uses more small or aide seats than the
    school has, and the excess seats. opt / random / free-lunch-first are checked to have
    zero violations (asserted).
  - placebo OPT - random (mean and 95th percentile over the placebo shuffles) when
    placebo files exist: how far OPT and random land apart by chance.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
B = 2000
SEED = 15
POLICIES = ['opt', 'random', 'free_lunch_first', 'tree_raw']


def load(path):
    res = pd.read_csv(path)
    assert res.groupby(['school_id', 'scale', 'policy']).size().eq(1).all(), 'school held out twice'
    feas = res[res.policy != 'tree_raw']
    assert (feas.n_small == feas.cap_small).all() and (feas.n_aide == feas.cap_aide).all(), \
        'a capacity-respecting policy does not use exactly its seats'
    return res


def opt_minus_random(res):
    tot = res.groupby(['scale', 'policy']).sum_value.sum().unstack()
    n = res[res.policy == 'opt'].groupby('scale').n.sum()
    return (tot.opt - tot.random) / n


placebo = sorted(HERE.glob('capacity_dr_rf_placebo*.csv'))
placebo = [f for f in placebo if '__' not in f.stem]
null = pd.DataFrame({f.stem: opt_minus_random(load(f)) for f in placebo}) if placebo else None

files = sys.argv[1:] or ['capacity_dr_rf.csv']
for name in files:
    res = load(HERE / name)
    s = res.pivot_table(index=['school_id', 'scale'], columns='policy', values='sum_value')[POLICIES]
    assert not s.isna().any().any()
    schools = s.index.get_level_values('school_id').unique().to_numpy()
    scales = s.index.get_level_values('scale').unique().to_numpy()
    S = np.stack([s.xs(c, level='scale').loc[schools].to_numpy() for c in scales], axis=1)
    N = res[(res.policy == 'opt') & (res.scale == scales[0])].set_index('school_id').n.loc[schools].to_numpy()

    def values(w):
        return np.einsum('s,scp->cp', w, S) / (w @ N)

    point = values(np.ones(len(schools)))
    rng = np.random.default_rng(SEED)
    boot = np.stack([values(np.bincount(rng.integers(len(schools), size=len(schools)),
                                        minlength=len(schools)).astype(float))
                     for _ in range(B)])

    def ci(d):
        lo, hi = np.quantile(d, [0.025, 0.975], axis=0)
        return [f'[{a:+.3f}, {b:+.3f}]' for a, b in zip(lo, hi)]

    opt_rows = res[res.policy == 'opt'].groupby('scale')[['n_small', 'n_aide', 'n']].sum()
    raw = res[res.policy == 'tree_raw'].copy()
    raw['excess'] = ((raw.n_small - raw.cap_small).clip(lower=0)
                     + (raw.n_aide - raw.cap_aide).clip(lower=0))
    viol = raw.groupby('scale').agg(raw_viol_schools=('excess', lambda e: int((e > 0).sum())),
                                    raw_excess_seats=('excess', 'sum'))
    out = pd.DataFrame({
        'small': (opt_rows.n_small / opt_rows.n).loc[scales].to_numpy(),
        'aide': (opt_rows.n_aide / opt_rows.n).loc[scales].to_numpy(),
        'OPT': point[:, 0], 'random': point[:, 1], 'FL_first': point[:, 2],
        'tree_raw': point[:, 3],
        'OPT-random': point[:, 0] - point[:, 1], '95% CI': ci(boot[:, :, 0] - boot[:, :, 1]),
        'OPT-FL': point[:, 0] - point[:, 2], '95% CI ': ci(boot[:, :, 0] - boot[:, :, 2]),
    }, index=pd.Index(scales, name='scale')).round(3)
    out = out.join(viol)
    if null is not None:
        out['placebo mean'] = null.mean(axis=1).round(3)
        out['placebo p95'] = null.quantile(0.95, axis=1).round(3)
    raw_share = raw.groupby('scale')[['n_small', 'n_aide', 'n']].sum().iloc[0]
    print(f'\n### {name}: {len(schools)} held-out schools, {int(N.sum())} students'
          + (f'; placebo: {null.shape[1]} shuffles' if null is not None else ''))
    print(f'    tree_raw uses {raw_share.n_small / raw_share.n:.1%} small, '
          f'{raw_share.n_aide / raw_share.n:.1%} aide (same at every scale)')
    print(out.to_string())
