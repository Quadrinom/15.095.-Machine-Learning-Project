"""Subgroup values and runs without sensitive features (task 14).

Run:   python subgroup_summary.py [full_name] [drop_name]
         full_name (default dr_rf): run with all features
         drop_name (default <full_name>__drop-female-ethnicity_grp): run without them
Input, for both names (from opt_frontier.jl and opt_capacity.jl):
  frontier_<name>.csv, frontier_ranks_<name>.csv,
  capacity_<name>.csv, capacity_assign_<name>.csv, capacity_students_<name>.csv
  and project_star_analysis.csv for the groups.
Rebuilds every held-out student's assignment (checked against the school totals in the
frontier and capacity files) and prints:
  A) per budget, OPT with all features vs OPT without the dropped features: held-out
     value, the difference with a 95% interval, OPT(no sens.) - random with an interval,
     and the share of students whose class type changes;
  B) per budget in SUBGROUP_BUDGETS and per group (gender, free lunch, ethnicity): share
     in small classes and held-out value under each policy, and OPT - random within the
     group with a 95% interval.
Budgets: frontier share q (small seats = round(q n_s) per school, no aide), and capacity
scale c (c x observed small and aide seats per school).
Intervals resample whole schools (B = 2000, seed 15), trees held fixed.
Group values are means of held-out DR rewards within the group: estimates, noisy for
small groups (ethnicity "other/unknown" has 28 students).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
B = 2000
SEED = 15
SUBGROUP_BUDGETS = ['q=0.3', 'q=0.5', 'c=1.0']
ARMS = ['regular', 'small', 'aide']

full = sys.argv[1] if len(sys.argv) > 1 else 'dr_rf'
drop = sys.argv[2] if len(sys.argv) > 2 else f'{full}__drop-female-ethnicity_grp'


def frontier_students(name):
    """Student x q x policy: p_small, p_aide, value (expected value for random)."""
    res = pd.read_csv(HERE / f'frontier_{name}.csv')
    rk = pd.read_csv(HERE / f'frontier_ranks_{name}.csv')
    seats = res[res.policy == 'opt'][['school_id', 'q', 'n', 'n_small']]
    d = rk.merge(seats, on='school_id')
    out = []
    for policy, p_small in [('opt', (d.rank_opt <= d.n_small).astype(float)),
                            ('free_lunch_first', (d.rank_fl <= d.n_small).astype(float)),
                            ('random', d.n_small / d.n)]:
        out.append(d[['student_id', 'school_id', 'q', 'G_regular', 'G_small', 'G_aide']]
                   .assign(policy=policy, p_small=p_small, p_aide=0.0))
    out = pd.concat(out, ignore_index=True)
    out['value'] = (1 - out.p_small) * out.G_regular + out.p_small * out.G_small
    check = out.groupby(['school_id', 'q', 'policy']).value.sum()
    ref = res.set_index(['school_id', 'q', 'policy']).sum_value.loc[check.index]
    assert np.allclose(check, ref, atol=1e-8), f'{name}: rebuilt frontier does not match'
    out['budget'] = 'q=' + out.q.map(lambda x: f'{x:.1f}')
    return out.drop(columns='q')


def capacity_students(name):
    res = pd.read_csv(HERE / f'capacity_{name}.csv')
    st = pd.read_csv(HERE / f'capacity_students_{name}.csv')
    asg = pd.read_csv(HERE / f'capacity_assign_{name}.csv')
    d = asg.merge(st[['student_id', 'G_regular', 'G_small', 'G_aide']], on='student_id')
    d['p_small'] = (d.arm == 'small').astype(float)
    d['p_aide'] = (d.arm == 'aide').astype(float)
    rnd = st.merge(res[res.policy == 'random'][['school_id', 'scale', 'n', 'cap_small', 'cap_aide']],
                   on='school_id')
    rnd = rnd.assign(policy='random', p_small=rnd.cap_small / rnd.n, p_aide=rnd.cap_aide / rnd.n)
    cols = ['student_id', 'school_id', 'scale', 'policy', 'p_small', 'p_aide',
            'G_regular', 'G_small', 'G_aide']
    out = pd.concat([d[cols], rnd[cols]], ignore_index=True)
    out['value'] = ((1 - out.p_small - out.p_aide) * out.G_regular
                    + out.p_small * out.G_small + out.p_aide * out.G_aide)
    check = out.groupby(['school_id', 'scale', 'policy']).value.sum()
    ref = res.set_index(['school_id', 'scale', 'policy']).sum_value.loc[check.index]
    assert np.allclose(check, ref, atol=1e-8), f'{name}: rebuilt capacity runs do not match'
    out['budget'] = 'c=' + out.scale.map(lambda x: f'{x:.2f}'.rstrip('0').rstrip('.') if x != 1 else '1.0')
    return out.drop(columns='scale')


def students(name):
    return pd.concat([frontier_students(name), capacity_students(name)], ignore_index=True)


a, b = students(full), students(drop)
# random and free-lunch-first don't use the tree: they must be identical in both runs
for pol in ['random', 'free_lunch_first']:
    x = a[a.policy == pol].set_index(['student_id', 'budget']).sort_index()
    y = b[b.policy == pol].set_index(['student_id', 'budget']).sort_index()
    assert np.allclose(x[['p_small', 'p_aide']], y[['p_small', 'p_aide']]), pol
d = pd.concat([a, b[b.policy == 'opt'].assign(policy='opt_nosens')], ignore_index=True)

grp = pd.read_csv(HERE / 'project_star_analysis.csv',
                  usecols=['student_id', 'female', 'free_lunch', 'ethnicity_grp'])
grp['gender'] = np.where(grp.female == 1, 'girls', 'boys')
grp['free lunch'] = np.where(grp.free_lunch == 1, 'free lunch', 'no free lunch')
grp['ethnicity'] = grp.ethnicity_grp.where(grp.ethnicity_grp.isin(['white', 'black']), 'other/unknown')
d = d.merge(grp[['student_id', 'gender', 'free lunch', 'ethnicity']], on='student_id')

POL = ['opt', 'opt_nosens', 'random', 'free_lunch_first']
rng_boot = np.random.default_rng(SEED)
schools = np.sort(d.school_id.unique())
W = np.stack([np.bincount(rng_boot.integers(len(schools), size=len(schools)),
                          minlength=len(schools)) for _ in range(B)]).astype(float)  # B x school


def group_values(sub, group_col=None):
    """Point and bootstrap values: dicts group -> (point[policy], boot[B, policy])."""
    keys = [None] if group_col is None else sorted(sub[group_col].unique())
    out = {}
    for g in keys:
        s = sub if g is None else sub[sub[group_col] == g]
        tot = s.pivot_table(index='school_id', columns='policy', values='value', aggfunc='sum')
        tot = tot.reindex(index=schools, columns=POL).fillna(0.0)
        n = s[s.policy == 'opt'].groupby('school_id').size().reindex(schools).fillna(0).to_numpy()
        T = tot.to_numpy()
        out[g] = (T.sum(0) / n.sum(), (W @ T) / (W @ n)[:, None], int(n.sum()))
    return out


def ci(x):
    lo, hi = np.quantile(x, [0.025, 0.975])
    return f'[{lo:+.3f}, {hi:+.3f}]'


def budget_key(bud):
    kind, x = bud.split('=')
    return (kind != 'q', float(x))


# A) all features vs without sensitive features
opt_a = d[d.policy == 'opt'].set_index(['student_id', 'budget'])
opt_b = d[d.policy == 'opt_nosens'].set_index(['student_id', 'budget']).loc[opt_a.index]
changed = ((opt_a.p_small != opt_b.p_small) | (opt_a.p_aide != opt_b.p_aide)).groupby('budget').mean()
rows = []
for bud in sorted(d.budget.unique(), key=budget_key):
    p, bt, n = group_values(d[d.budget == bud])[None]
    rows.append({'budget': bud, 'OPT': p[0], 'OPT no sens.': p[1], 'random': p[2],
                 'no sens. - all': p[1] - p[0], '95% CI': ci(bt[:, 1] - bt[:, 0]),
                 'no sens. - random': p[1] - p[2], '95% CI ': ci(bt[:, 1] - bt[:, 2]),
                 'all - random': p[0] - p[2],
                 'assignments changed': changed[bud]})
print(f'### A) OPT with all features ({full}) vs without ({drop})')
print('    held-out DR value, SD units; q = frontier share (small only), '
      'c = capacity scale (small + aide)')
print(pd.DataFrame(rows).set_index('budget').round(3).to_string())

# B) subgroup values
for bud in SUBGROUP_BUDGETS:
    sub = d[d.budget == bud]
    print(f'\n### B) Subgroups at {bud}')
    for col in ['gender', 'free lunch', 'ethnicity']:
        share = sub.groupby([col, 'policy']).p_small.mean().unstack()[POL]
        rows = []
        for g, (p, bt, n) in group_values(sub, col).items():
            rows.append({col: g, 'n': n,
                         'small: OPT': share.loc[g, 'opt'], 'no sens.': share.loc[g, 'opt_nosens'],
                         'FL first': share.loc[g, 'free_lunch_first'], 'random': share.loc[g, 'random'],
                         'value: OPT': p[0], 'no sens. ': p[1], 'random ': p[2], 'FL first ': p[3],
                         'OPT-random': p[0] - p[2], '95% CI': ci(bt[:, 0] - bt[:, 2]),
                         'no sens.-random': p[1] - p[2], '95% CI ': ci(bt[:, 1] - bt[:, 2])})
        print(pd.DataFrame(rows).set_index(col).round(3).to_string())
        print()
