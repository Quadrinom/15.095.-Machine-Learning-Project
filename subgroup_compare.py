"""Compare OPT with the subgroup-level allocation (task 11) on held-out schools.

Run:   python subgroup_compare.py [name ...]
         name: run name as in frontier_<name>.csv (default: dr_rf and
         dr_rf__drop-female-ethnicity_grp)
Input: frontier_<name>.csv, capacity_<name>.csv (OPT, from opt_frontier.jl / opt_capacity.jl),
       frontier_subgroup_<name>.csv, capacity_subgroup_<name>.csv (subgroup_policy.py),
       and the placebo subgroup files frontier_subgroup_dr_rf_placebo*.csv /
       capacity_subgroup_dr_rf_placebo*.csv with the same feature set (for the yardstick)
Prints, per budget (frontier share q; capacity scale c): held-out value of OPT, the
subgroup policy and random; OPT - subgroup and subgroup - random with 95% intervals
(whole schools resampled, B = 2000, seed 15; trees and group averages held fixed); and
the placebo 95th percentile of subgroup - random. Then the best budget of the subgroup
policy against the placebos' best budgets (one-sided p, as in placebo_summary.py).

Reading (CLAUDE.md, subgroup-level allocation): subgroup ~ OPT and both beat random: the
tree keeps the benefit of targeting in a simpler form; subgroup > OPT > random: the tree
gives some up for simplicity; OPT > subgroup: the fine table overfits small groups;
all ~ random: no useful targeting on these features.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
B = 2000
SEED = 15


def drop_suffix(stem):
    return stem[stem.index('__drop-'):] if '__drop-' in stem else ''


def school_sums(opt_file, sub_file, budget):
    """school x budget x policy (opt, subgroup, random) sums, and school sizes."""
    o = pd.read_csv(opt_file)
    s = pd.read_csv(sub_file)
    o = o[o.policy.isin(['opt', 'random'])]
    s = s[s.policy == 'subgroup']
    both = pd.concat([o, s])
    t = both.pivot_table(index=['school_id', budget], columns='policy', values='sum_value')
    t = t[['opt', 'subgroup', 'random']]
    assert not t.isna().any().any()
    rnd_o = o[o.policy == 'random'].set_index(['school_id', budget]).sum_value
    rnd_s = pd.read_csv(sub_file).query('policy == "random"').set_index(['school_id', budget]).sum_value
    assert np.allclose(rnd_o.loc[rnd_s.index], rnd_s), 'random differs: seat counts differ'
    schools = t.index.get_level_values('school_id').unique().to_numpy()
    budgets = t.index.get_level_values(budget).unique().to_numpy()
    S = np.stack([t.xs(b, level=budget).loc[schools].to_numpy() for b in budgets], axis=1)
    N = o[o.policy == 'opt'].groupby('school_id').n.first().loc[schools].to_numpy()
    return S, N, budgets


def sub_minus_random(sub_file, budget):
    r = pd.read_csv(sub_file)
    tot = r.groupby([budget, 'policy']).sum_value.sum().unstack()
    n = r[r.policy == 'random'].groupby(budget).n.sum()
    return (tot.subgroup - tot.random) / n


def ci(d):
    lo, hi = np.quantile(d, [0.025, 0.975], axis=0)
    return [f'[{a:+.3f}, {b:+.3f}]' for a, b in zip(lo, hi)]


names = sys.argv[1:] or ['dr_rf', 'dr_rf__drop-female-ethnicity_grp']
for kind, budget in [('frontier', 'q'), ('capacity', 'scale')]:
    nulls = {}
    for f in sorted(HERE.glob(f'{kind}_subgroup_dr_rf_placebo*.csv')):
        nulls.setdefault(drop_suffix(f.stem), {})[f.stem] = sub_minus_random(f, budget)
    nulls = {k: pd.DataFrame(v) for k, v in nulls.items()}
    for name in names:
        S, N, budgets = school_sums(HERE / f'{kind}_{name}.csv', HERE / f'{kind}_subgroup_{name}.csv', budget)
        rng = np.random.default_rng(SEED)
        W = np.stack([np.bincount(rng.integers(len(N), size=len(N)), minlength=len(N))
                      for _ in range(B)]).astype(float)
        point = np.einsum('sbp->bp', S) / N.sum()
        boot = np.einsum('ws,sbp->wbp', W, S) / (W @ N)[:, None, None]
        out = pd.DataFrame({
            'OPT': point[:, 0], 'subgroup': point[:, 1], 'random': point[:, 2],
            'OPT-subgroup': point[:, 0] - point[:, 1], '95% CI': ci(boot[:, :, 0] - boot[:, :, 1]),
            'subgroup-random': point[:, 1] - point[:, 2], '95% CI ': ci(boot[:, :, 1] - boot[:, :, 2]),
            'OPT-random': point[:, 0] - point[:, 2],
        }, index=pd.Index(budgets, name=budget))
        null = nulls.get(drop_suffix(name))
        if null is not None:
            out['placebo p95 (sub-rnd)'] = null.quantile(0.95, axis=1).loc[budgets].to_numpy()
        print(f'\n### {kind}: {name}')
        print(out.round(3).to_string())
        if null is not None:
            col = pd.Series(point[:, 1] - point[:, 2], index=budgets)
            null_max = null.max()
            p = (1 + (null_max >= col.max()).sum()) / (1 + len(null_max))
            print(f'    subgroup best {budget} = {col.idxmax():g}: subgroup - random '
                  f'{col.max():+.3f}, one-sided p = {p:.3f} vs {len(null_max)} placebos '
                  f'(placebo best: mean {null_max.mean():.3f}, max {null_max.max():.3f})')
