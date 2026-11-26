"""Summarize a fixed-share budget frontier with school-level bootstrap intervals.

Run:   python frontier_summary.py [frontier_<tag>.csv ...]     (default: frontier_dr_rf.csv)
Input: frontier_*.csv from opt_frontier.jl (one row per held-out school x share q x policy)
Prints, per share q of students in small classes, pooled over all held-out schools (each
school is held out exactly once): held-out value of OPT / random / free-lunch-first, and
OPT - random and OPT - free-lunch-first with 95% intervals.

Intervals resample whole schools (B = 2000, seed 15). The trees are held fixed, so the
intervals cover evaluation noise on the held-out schools, not variation from refitting
the trees.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
B = 2000
SEED = 15
POLICIES = ['opt', 'random', 'free_lunch_first']

files = sys.argv[1:] or ['frontier_dr_rf.csv']
for name in files:
    res = pd.read_csv(HERE / name)
    # school x q x policy -> sum of rewards; n per school
    s = res.pivot_table(index=['school_id', 'q'], columns='policy', values='sum_value')[POLICIES]
    n = res[res.policy == 'opt'].set_index(['school_id', 'q']).n.reindex(s.index)
    assert not s.isna().any().any() and not n.isna().any()
    assert res.groupby(['school_id', 'q', 'policy']).size().eq(1).all(), 'school held out twice'
    schools = s.index.get_level_values('school_id').unique().to_numpy()
    shares = s.index.get_level_values('q').unique().to_numpy()

    # arrays: [school, q, policy] sums and [school] sizes
    S = np.stack([s.xs(q, level='q').loc[schools].to_numpy() for q in shares], axis=1)
    N = n.xs(shares[0], level='q').loc[schools].to_numpy()

    def values(w):
        """Policy values (q x policy) with school weights w (bootstrap counts)."""
        return np.einsum('s,sqp->qp', w, S) / (w @ N)

    point = values(np.ones(len(schools)))
    rng = np.random.default_rng(SEED)
    boot = np.stack([values(np.bincount(rng.integers(len(schools), size=len(schools)),
                                        minlength=len(schools)).astype(float))
                     for _ in range(B)])
    diff_r, diff_f = boot[:, :, 0] - boot[:, :, 1], boot[:, :, 0] - boot[:, :, 2]

    def ci(d):
        lo, hi = np.quantile(d, [0.025, 0.975], axis=0)
        return [f'[{a:+.3f}, {b:+.3f}]' for a, b in zip(lo, hi)]

    opt_rows = res[res.policy == 'opt'].groupby('q')[['n_small', 'n']].sum()
    out = pd.DataFrame({
        'share_small': (opt_rows.n_small / opt_rows.n).loc[shares].to_numpy(),
        'OPT': point[:, 0], 'random': point[:, 1], 'FL_first': point[:, 2],
        'OPT-random': point[:, 0] - point[:, 1], '95% CI': ci(diff_r),
        'OPT-FL': point[:, 0] - point[:, 2], '95% CI ': ci(diff_f),
    }, index=pd.Index(shares, name='q')).round(3)
    print(f'\n### {name}: {len(schools)} held-out schools, {int(N.sum())} students')
    print(out.to_string())
