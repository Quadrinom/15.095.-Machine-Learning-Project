"""Summarize the exact capacity-constrained policy (task 19, opt_exact.jl).

Run:   python exact_summary.py
Input: exact_capacity_<name>.csv and exact_insample_<name>.csv for
         dr_rf                              (all features)
         dr_rf__drop-female-ethnicity_grp   (without gender and ethnicity)
       plus the matching placebo runs exact_capacity_dr_rf_placebo*[...].csv
Prints, per capacity scale c (seats = c x observed STAR seats per school):
  1. In-sample (training schools; mean over the 5 folds): gain over random per student of
     the exact optimum and of the IAI policy (opt_capacity.jl). Exact >= IAI by
     construction (asserted in opt_exact.jl); the gap is what IAI's heuristic leaves.
  2. Held-out: value of exact, IAI (opt_expected) and random; exact - random,
     exact - IAI and IAI - random with 95% intervals (whole schools resampled, B = 2000,
     seed 15, policies held fixed). Ties within a leaf are scored by their exact expected
     value for every policy, so these IAI numbers differ from capacity_dr_rf.csv, which
     used one random tie-break draw.
  3. Placebo (20 shuffles, same features): exact - random by chance at each scale (mean,
     95th percentile), the per-scale p-value, and p for the best scale
     (1 + #placebos whose best scale is at or above the real best) / 21.
  4. Which features the exact trees split on (folds of 5).
"""
import pathlib
import re

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
B = 2000
SEED = 15
FEATS = ['female', 'free_lunch', 'age_at_entry', 'ethnicity_grp', 'locale']


def load(path):
    res = pd.read_csv(path)
    assert res.groupby(['school_id', 'scale', 'policy']).size().eq(1).all(), 'school held out twice'
    assert (res.n_small == res.cap_small).all() and (res.n_aide == res.cap_aide).all()
    return res


def per_student(res):
    tot = res.groupby(['scale', 'policy']).sum_value.sum().unstack()
    n = res[res.policy == 'random'].groupby('scale').n.sum()
    return tot.div(n, axis=0)


def placebo_null(suffix, policy='exact'):
    files = [f for f in sorted(HERE.glob('exact_capacity_dr_rf_placebo*.csv'))
             if f.stem.endswith(suffix) and (suffix or '__' not in f.stem)]
    vals = {f.stem: per_student(load(f)) for f in files}
    return pd.DataFrame({k: v[policy] - v.random for k, v in vals.items() if policy in v})


def tree_features(name):
    counts = {}
    for f in sorted((HERE / f'trees_exact_{name}').glob('fold*.txt')):
        for block in f.read_text().split('\n== scale c = ')[1:]:
            scale = float(block.split(':')[0])
            if 'IAI tree' in block:
                block = block.split('== IAI tree')[0]
            used = {ft for ft in FEATS if re.search(rf'\b{ft}\b', block.split('\n', 1)[1])}
            for ft in used:
                counts[(scale, ft)] = counts.get((scale, ft), 0) + 1
    out = {}
    for (scale, ft), c in counts.items():
        out.setdefault(scale, []).append(f'{ft}:{c}')
    return {s: ', '.join(sorted(v, key=lambda x: -int(x.split(':')[1]))) for s, v in out.items()}


for name, suffix in [('dr_rf', ''), ('dr_rf__drop-female-ethnicity_grp', '__drop-female-ethnicity_grp')]:
    path = HERE / f'exact_capacity_{name}.csv'
    if not path.exists():
        print(f'\n### {name}: missing'); continue
    res = load(path)
    ins = pd.read_csv(HERE / f'exact_insample_{name}.csv')
    s = res.pivot_table(index=['school_id', 'scale'], columns='policy', values='sum_value')
    pols = ['exact', 'opt_expected', 'random']
    schools = s.index.get_level_values('school_id').unique().to_numpy()
    scales = np.sort(s.index.get_level_values('scale').unique().to_numpy())
    S = np.stack([s.xs(c, level='scale').loc[schools, pols].to_numpy() for c in scales], axis=1)
    N = res[(res.policy == 'random') & (res.scale == scales[0])].set_index('school_id').n.loc[schools].to_numpy()

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

    g = ins.groupby('scale')
    print(f'\n######## {name}: {len(schools)} held-out schools, {int(N.sum())} students')
    print('1) In-sample gain over random per training student (mean over folds); '
          f'trees checked per fold: {sorted(ins.trees_checked.unique())}')
    print(pd.DataFrame({'exact': g.exact_gain.mean() - g.random_gain.mean(),
                        'IAI policy': g.iai_gain.mean() - g.random_gain.mean(),
                        'IAI tree, best order': g.iai_best_order_gain.mean() - g.random_gain.mean(),
                        'IAI depth (mean)': g.iai_depth.mean()}).round(4).to_string())

    d_er = point[:, 0] - point[:, 2]
    out = pd.DataFrame({
        'exact': point[:, 0], 'IAI': point[:, 1], 'random': point[:, 2],
        'exact-random': d_er, 'CI': ci(boot[:, :, 0] - boot[:, :, 2]),
        'exact-IAI': point[:, 0] - point[:, 1], 'CI ': ci(boot[:, :, 0] - boot[:, :, 1]),
        'IAI-random': point[:, 1] - point[:, 2], 'CI  ': ci(boot[:, :, 1] - boot[:, :, 2]),
    }, index=pd.Index(scales, name='scale')).round(3)
    null = placebo_null(suffix)
    if not null.empty:
        null = null.loc[scales]
        out['placebo mean'] = null.mean(axis=1).round(3).to_numpy()
        out['placebo p95'] = null.quantile(0.95, axis=1).round(3).to_numpy()
        out['p'] = [(1 + (null.loc[c] >= v).sum()) / (1 + null.shape[1]) for c, v in zip(scales, d_er)]
        out['p'] = out['p'].round(3)
    print('2-3) Held-out value per student (school bootstrap CIs; placebo: '
          f'{null.shape[1]} shuffles, same features)')
    print(out.to_string())
    if not null.empty:
        best = d_er.max(); best_c = scales[d_er.argmax()]
        p_best = (1 + (null.max(axis=0) >= best).sum()) / (1 + null.shape[1])
        print(f'   best scale c = {best_c}: exact - random = {best:+.3f}; placebo best-scale '
              f'mean {null.max(axis=0).mean():+.3f}, max {null.max(axis=0).max():+.3f}; p = {p_best:.3f}')
    inull = placebo_null(suffix, 'opt_expected')
    if not inull.empty:
        inull = inull.loc[scales]
        d_ir = point[:, 1] - point[:, 2]
        p_c = [(1 + (inull.loc[c] >= v).sum()) / (1 + inull.shape[1]) for c, v in zip(scales, d_ir)]
        p_best = (1 + (inull.max(axis=0) >= d_ir.max()).sum()) / (1 + inull.shape[1])
        print(f'   IAI placebo ({inull.shape[1]} shuffles, expected tie-break): p95 per scale '
              f'{inull.quantile(0.95, axis=1).round(3).tolist()}; per-scale p {np.round(p_c, 3).tolist()}; '
              f'best scale c = {scales[d_ir.argmax()]}: IAI - random = {d_ir.max():+.3f}, p = {p_best:.3f}')
    print('4) Features in the exact trees (folds of 5):')
    for c, v in sorted(tree_features(name).items()):
        print(f'   c = {c}: {v}')
