"""Does the boys-vs-girls difference in the small-class gain hold up? (notes section 16)

Run:    python gender_robustness.py
Input:  project_star_analysis.csv (kindergarten entrants, K treatment = intent-to-treat)
        project_star_clean.csv    (z-scores for grades 1-3: z_read_g, z_math_g, z_total_g)
Output: gender_robustness.csv, and a printed table

For each outcome, the small-class gain (small vs regular) for boys and for girls, and the
gap (boys - girls), estimated two ways:
  raw: difference in means, [mean(small) - mean(regular)] within each gender
  fe:  regression with school fixed effects (within-school, as randomization was):
       y = school + b1 small + b2 aide + b3 female + b4 small*female + b5 aide*female
       boys' gain = b1, girls' gain = b1 + b4, gap = -b4
95% intervals: whole schools resampled (B = 1000, seed 15); each resampled copy of a
school is its own fixed-effect group.
Treatment is always the class type in kindergarten. Later-grade outcomes are missing for
students who left STAR, and attrition differs by arm and gender, so every later-grade
row is paired with a "K outcome, same students" row: if the gap there matches the full-
sample K gap, attrition did not change it.
"""
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
B = 1000
SEED = 15

a = pd.read_csv(HERE / 'project_star_analysis.csv')
c = pd.read_csv(HERE / 'project_star_clean.csv', low_memory=False)
zcols = [f'z_{s}_{g}' for s in ('read', 'math', 'total') for g in '123']
d = a.merge(c[['student_id'] + zcols], on='student_id', how='left', validate='one_to_one')
assert len(d) == len(a) and np.allclose(d.outcome, c.set_index('student_id').loc[d.student_id, 'z_total_k'])
d['small'] = (d.treatment == 'small').astype(float)
d['aide'] = (d.treatment == 'aide').astype(float)

OUTCOMES = [  # (label, column, sample restricted to students with this column)
    ('K total (main outcome)', 'outcome', None),
    ('K reading', 'outcome_read', None),
    ('K math', 'outcome_math', None),
    ('Grade 1 total', 'z_total_1', None),
    ('  K total, same students as grade 1', 'outcome', 'z_total_1'),
    ('Grade 1 reading', 'z_read_1', None),
    ('Grade 1 math', 'z_math_1', None),
    ('Grade 2 total', 'z_total_2', None),
    ('  K total, same students as grade 2', 'outcome', 'z_total_2'),
    ('Grade 3 total', 'z_total_3', None),
    ('  K total, same students as grade 3', 'outcome', 'z_total_3'),
]


def demean(v, g):
    return v - pd.Series(v).groupby(g).transform('mean').to_numpy()


def estimates(y, small, aide, female, group):
    """(raw boys gain, raw girls gain, fe boys gain, fe girls gain)"""
    raw = []
    for f in (0, 1):
        m = female == f
        raw.append(y[m & (small == 1)].mean() - y[m & (small == 0) & (aide == 0)].mean())
    X = np.column_stack([small, aide, female, small * female, aide * female])
    Xd = np.column_stack([demean(X[:, j], group) for j in range(X.shape[1])])
    b = np.linalg.lstsq(Xd, demean(y, group), rcond=None)[0]
    return raw[0], raw[1], b[0], b[0] + b[3]


rows = []
rng = np.random.default_rng(SEED)
for label, col, restrict in OUTCOMES:
    sub = d[d[col].notna() & (d[restrict].notna() if restrict else True)]
    y, s, al, f = (sub[col].to_numpy(), sub.small.to_numpy(), sub.aide.to_numpy(),
                   sub.female.to_numpy())
    sch = sub.school_id.to_numpy()
    point = estimates(y, s, al, f, sch)
    schools = np.unique(sch)
    by_school = {k: np.flatnonzero(sch == k) for k in schools}
    boot = []
    for _ in range(B):
        pick = rng.choice(schools, size=len(schools), replace=True)
        idx = np.concatenate([by_school[k] for k in pick])
        grp = np.concatenate([np.full(len(by_school[k]), j) for j, k in enumerate(pick)])
        boot.append(estimates(y[idx], s[idx], al[idx], f[idx], grp))
    boot = np.array(boot)
    gap_raw, gap_fe = boot[:, 0] - boot[:, 1], boot[:, 2] - boot[:, 3]
    lo_r, hi_r = np.quantile(gap_raw, [0.025, 0.975])
    lo_f, hi_f = np.quantile(gap_fe, [0.025, 0.975])
    rows.append(dict(outcome=label, n=len(sub), share_present=len(sub) / len(d),
                     raw_boys=point[0], raw_girls=point[1], raw_gap=point[0] - point[1],
                     raw_lo=lo_r, raw_hi=hi_r,
                     fe_boys=point[2], fe_girls=point[3], fe_gap=point[2] - point[3],
                     fe_lo=lo_f, fe_hi=hi_f,
                     fe_gap_share_positive=(gap_fe > 0).mean()))

out = pd.DataFrame(rows)
out.to_csv(HERE / 'gender_robustness.csv', index=False)
pd.set_option('display.width', 220)
show = out[['outcome', 'n']].copy()
show['raw boys / girls'] = [f'{b:+.3f} / {g:+.3f}' for b, g in zip(out.raw_boys, out.raw_girls)]
show['raw gap [95% CI]'] = [f'{v:+.3f} [{l:+.3f}, {h:+.3f}]' for v, l, h in zip(out.raw_gap, out.raw_lo, out.raw_hi)]
show['FE boys / girls'] = [f'{b:+.3f} / {g:+.3f}' for b, g in zip(out.fe_boys, out.fe_girls)]
show['FE gap [95% CI]'] = [f'{v:+.3f} [{l:+.3f}, {h:+.3f}]' for v, l, h in zip(out.fe_gap, out.fe_lo, out.fe_hi)]
print('Small-class gain (small vs regular, SD), boys vs girls; treatment = K assignment')
print(show.to_string(index=False))
