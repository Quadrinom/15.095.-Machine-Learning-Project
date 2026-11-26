"""Summarize OPT vs baselines across runs, and which features the trees split on.

Run:   python summarize_runs.py [REPO_DIR]     (default: this script's folder)
Input: results_*.csv and trees_*/tree_fold<k>_lam<lam>.txt from opt_trees.jl
Prints, per run and cost penalty lambda (means over the 5 held-out school folds):
  share of small/aide places, chosen depth, held-out value of OPT / random / free-lunch
  first, OPT - random (mean and sd over folds), and how many folds' trees split on
  each feature.
Runs labelled SYNTH use planted effects (opt_rewards.py --plant): not STAR results.
Runs labelled PLACEBO shuffle class types within schools: a yardstick for noise.
"""
import pathlib
import sys

import pandas as pd

repo = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent
runs = [  # (label, results file, tree dir)
    ('DR-RF (original)', 'results_dr_rf.csv', 'trees_dr_rf'),
    ('DM-RF, scored with DR-RF', 'results_dm_rf__eval_dr_rf.csv', 'trees_dm_rf'),
    ('DR-linear, scored with DR-RF', 'results_dr_linear__eval_dr_rf.csv', 'trees_dr_linear'),
    ('DM-linear, scored with DR-RF', 'results_dm_linear__eval_dr_rf.csv', 'trees_dm_linear'),
    ('SYNTH +0.1 free lunch', 'results_dr_rf_plant_free_lunch0.1.csv', 'trees_dr_rf_plant_free_lunch0.1'),
    ('SYNTH +0.2 free lunch', 'results_dr_rf_plant_free_lunch0.2.csv', 'trees_dr_rf_plant_free_lunch0.2'),
    ('SYNTH +0.3 free lunch', 'results_dr_rf_plant_free_lunch0.3.csv', 'trees_dr_rf_plant_free_lunch0.3'),
    ('SYNTH +0.3 young', 'results_dr_rf_plant_young0.3.csv', 'trees_dr_rf_plant_young0.3'),
    # Placebo: class types shuffled within schools, so OPT - random should be ~0 (task 7)
    ('PLACEBO DR-linear', 'results_dr_linear_placebo.csv', 'trees_dr_linear_placebo'),
] + [(f'PLACEBO DR-RF seed {s}', f'results_dr_rf_placebo{x}.csv', f'trees_dr_rf_placebo{x}')
     for s, x in [(15, ''), (1, '_s1'), (2, '_s2'), (3, '_s3'), (4, '_s4')]]
# Longest names first, so free_lunch_missing isn't counted as free_lunch
FEATS = ['free_lunch_missing', 'age_at_entry_missing', 'free_lunch', 'age_at_entry',
         'female', 'ethnicity_grp', 'locale']


def split_features(txt):
    found = []
    for line in txt.splitlines():
        if 'Split:' in line:
            body = line.split('Split:')[1]
            for f in FEATS:
                if f in body:
                    found.append(f)
                    break
    return found


for label, res_file, tree_dir in runs:
    path = repo / res_file
    if not path.exists():
        print(f'\n### {label}: {res_file} missing')
        continue
    res = pd.read_csv(path)
    pol = res[res.lam.notna()]
    wide = pol.pivot_table(index=['fold', 'lam'], columns='policy', values='value')
    wide['opt_minus_random'] = wide.opt - wide.random_same_budget
    wide['opt_minus_fl'] = wide.opt - wide.free_lunch_first
    share = pol[pol.policy == 'opt'].set_index(['fold', 'lam'])[['share_small', 'share_aide', 'depth']]
    wide = wide.join(share)
    g = wide.groupby('lam')
    out = pd.DataFrame({
        'share_small': g.share_small.mean(),
        'share_aide': g.share_aide.mean(),
        'depth': g.depth.mean(),
        'OPT': g.opt.mean(),
        'random': g.random_same_budget.mean(),
        'FL_first': g.free_lunch_first.mean(),
        'OPT-random': g.opt_minus_random.mean(),
        'sd(OPT-random)': g.opt_minus_random.std(),
        'OPT-FL': g.opt_minus_fl.mean(),
    }).round(3)
    feats = {}
    for lam in out.index:
        counts = {}
        for k in sorted(res.fold.unique()):
            f = repo / tree_dir / f'tree_fold{k}_lam{lam:g}.txt'
            if not f.exists():
                f = repo / tree_dir / f'tree_fold{k}_lam{lam}.txt'
            if not f.exists():
                continue
            for name in set(split_features(f.read_text())):
                counts[name] = counts.get(name, 0) + 1
        feats[lam] = ', '.join(f'{n}:{c}' for n, c in sorted(counts.items(), key=lambda x: -x[1])) or 'no split'
    out['splits (folds of 5)'] = pd.Series(feats)
    uni = res[res.lam.isna()].groupby('policy').value.mean().round(3).to_dict()
    print(f'\n### {label}')
    print('    single-arm policies:', uni)
    print(out.to_string())
