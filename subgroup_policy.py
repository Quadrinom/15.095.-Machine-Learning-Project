"""Subgroup-level allocation baseline (task 11, teaching-staff feedback).

Run:   python subgroup_policy.py [tag] [eval_tag] [--drop]
         tag (default dr_rf): rewards the group averages are computed from (training rows)
         eval_tag (default = tag): rewards that score the held-out schools
         --drop: leave out female and ethnicity_grp (as opt_*.jl --drop=female,ethnicity_grp)
Input: project_star_analysis.csv, project_star_school_capacity.csv, rewards_<tag>.csv
       [, rewards_<eval_tag>.csv]
Output (<name> = tag [__eval_<eval_tag>] [__drop-female-ethnicity_grp], as for the trees):
  frontier_subgroup_<name>.csv   same layout as frontier_<name>.csv; policies subgroup
                                 and random
  capacity_subgroup_<name>.csv   same layout as capacity_<name>.csv; policies subgroup
                                 and random
  subgroup_cells_<name>.csv      per fold and group: training size and mean gains
Comparison with OPT: python subgroup_compare.py

Groups: every combination of female x free_lunch x ethnicity (white / black /
other-or-unknown) x locale x age band (< 5.165, 5.165-5.415, >= 5.415 years: the
cut-offs the trees use, between the quarter-year age values). 2 x 2 x 3 x 4 x 3 = 144
groups; with --drop, free_lunch x locale x age band = 24 groups. Imputed values are used
as they are (4 ages and 17 free-lunch values were missing; notes section 3).
Per held-out fold, as in opt_frontier.jl / opt_capacity.jl:
  1. In the training schools, each group's mean (G_small - G_regular) and
     (G_aide - G_regular). A held-out student whose group has no training students gets
     the overall training mean (counted in the output).
  2. The same seat-filling rules as OPT, with group gains in place of leaf gains:
     frontier: round(q n_s) small seats per school, highest group small gain first;
     capacity: B_s^small = round(c x observed small), B_s^aide = min(round(c x observed
     aide), n_s - B_s^small); small seats first by group small gain (aide gain breaks
     ties), then aide seats by group aide gain. Random tie-break (seed 15).
  With one resource per school, filling by group gain solves the staff's "small linear
  optimization" exactly for the group estimates; with small and aide seats the
  small-first rule is the same heuristic OPT uses, so the comparison is like for like.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
SEED = 15
SHARES = np.round(np.arange(0, 1.01, 0.1), 1)
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
AGE_CUTS = [5.165, 5.415]

args = [a for a in sys.argv[1:] if not a.startswith('--')]
drop = '--drop' in sys.argv[1:]
assert all(a in ('--drop',) for a in sys.argv[1:] if a.startswith('--')), sys.argv
tag = args[0] if args else 'dr_rf'
eval_tag = args[1] if len(args) > 1 else tag
name = (tag if eval_tag == tag else f'{tag}__eval_{eval_tag}') + ('__drop-female-ethnicity_grp' if drop else '')

df = pd.read_csv(HERE / 'project_star_analysis.csv')
df['eth3'] = df.ethnicity_grp.where(df.ethnicity_grp.isin(['white', 'black']), 'other/unknown')
df['age_band'] = np.digitize(df.age_at_entry, AGE_CUTS)
cell_cols = (['free_lunch', 'locale', 'age_band'] if drop
             else ['female', 'free_lunch', 'eth3', 'locale', 'age_band'])
df['cell'] = df[cell_cols].astype(str).agg('|'.join, axis=1)
cap = pd.read_csv(HERE / 'project_star_school_capacity.csv').set_index('school_id')
rw = pd.read_csv(HERE / f'rewards_{tag}.csv')
rw_eval = rw if eval_tag == tag else pd.read_csv(HERE / f'rewards_{eval_tag}.csv')
rng = np.random.default_rng(SEED)
feat = df[['student_id', 'school_id', 'free_lunch', 'cell']]


def order(key, key2, tiebreak):
    """Positions in priority order: higher key, then higher key2, then lower tiebreak."""
    return np.lexsort((tiebreak, -key2, -key))


front, capres, cells = [], [], []
n_unseen = 0
for k in sorted(rw.test_fold.unique()):
    tr = rw[(rw.test_fold == k) & (rw.role == 'train')].merge(feat, on='student_id')
    te = rw_eval[(rw_eval.test_fold == k) & (rw_eval.role == 'test')].merge(feat, on='student_id')
    assert not set(tr.school_id) & set(te.school_id), 'test school leaked into training'
    tr['gs'], tr['ga'] = tr.G_small - tr.G_regular, tr.G_aide - tr.G_regular
    g = tr.groupby('cell').agg(n_train=('gs', 'size'), gain_small=('gs', 'mean'),
                               gain_aide=('ga', 'mean'))
    cells.append(g.reset_index().assign(fold=k))
    te['sc_s'] = te.cell.map(g.gain_small)
    te['sc_a'] = te.cell.map(g.gain_aide)
    unseen = te.sc_s.isna()
    n_unseen += int(unseen.sum())
    te.loc[unseen, 'sc_s'] = tr.gs.mean()
    te.loc[unseen, 'sc_a'] = tr.ga.mean()
    te['tb'] = rng.random(len(te))

    for s, d in te.groupby('school_id', sort=True):
        n = len(d)
        Gr, Gs, Ga = d.G_regular.to_numpy(), d.G_small.to_numpy(), d.G_aide.to_numpy()
        ord_s = order(d.sc_s.to_numpy(), d.sc_a.to_numpy(), d.tb.to_numpy())
        ord_a = order(d.sc_a.to_numpy(), d.sc_s.to_numpy(), d.tb.to_numpy())
        for q in SHARES:
            m = round(q * n)
            small = np.zeros(n, bool)
            small[ord_s[:m]] = True
            row = dict(fold=k, school_id=s, q=q, n=n, n_small=m)
            front.append({**row, 'policy': 'subgroup', 'sum_value': np.where(small, Gs, Gr).sum()})
            front.append({**row, 'policy': 'random',
                          'sum_value': (1 - m / n) * Gr.sum() + m / n * Gs.sum()})
        c = cap.loc[s]
        assert c.n_total == n, f'school {s}: capacity file has {c.n_total} students, data {n}'
        for scale in SCALES:
            bs = min(round(scale * c.n_small), n)
            ba = min(round(scale * c.n_aide), n - bs)
            arm = np.zeros(n, int)          # 0 regular, 1 small, 2 aide
            arm[ord_s[:bs]] = 1
            rest = [i for i in ord_a if arm[i] == 0]
            arm[rest[:ba]] = 2
            assert (arm == 1).sum() == bs and (arm == 2).sum() == ba
            row = dict(fold=k, school_id=s, scale=scale, n=n, cap_small=bs, cap_aide=ba,
                       n_small=bs, n_aide=ba)
            capres.append({**row, 'policy': 'subgroup',
                           'sum_value': np.choose(arm, [Gr, Gs, Ga]).sum()})
            ps, pa = bs / n, ba / n
            capres.append({**row, 'policy': 'random',
                           'sum_value': ((1 - ps - pa) * Gr + ps * Gs + pa * Ga).sum()})

cols_f = ['fold', 'school_id', 'q', 'policy', 'n', 'n_small', 'sum_value']
cols_c = ['fold', 'school_id', 'scale', 'policy', 'n', 'cap_small', 'cap_aide', 'n_small',
          'n_aide', 'sum_value']
pd.DataFrame(front)[cols_f].to_csv(HERE / f'frontier_subgroup_{name}.csv', index=False)
pd.DataFrame(capres)[cols_c].to_csv(HERE / f'capacity_subgroup_{name}.csv', index=False)
cells = pd.concat(cells)[['fold', 'cell', 'n_train', 'gain_small', 'gain_aide']]
cells.to_csv(HERE / f'subgroup_cells_{name}.csv', index=False)
print(f'{name}: groups {", ".join(cell_cols)}; {cells.groupby("fold").size().min()}-'
      f'{cells.groupby("fold").size().max()} groups with training students per fold, median '
      f'{int(cells.n_train.median())} students; {n_unseen} held-out students in groups '
      f'unseen in training')
