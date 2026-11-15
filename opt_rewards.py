"""Stage 1 of the OPT pipeline: cross-fitted counterfactual rewards (Gamma, eq. 1).

Run: python opt_rewards.py [--method dr|dm|ipw] [--model rf|linear] [--placebo]
Input:  project_star_analysis.csv (from clean_project_star.py), same folder
Output: rewards_<method>_<model>[_placebo].csv, long format:
        test_fold, role (train/test), student_id, G_regular, G_small, G_aide
Stage 2 (trees + evaluation) is opt_trees.jl, run with the IAI Julia system image.

For each held-out school fold k:
  - test rewards:  outcome models fit on all training schools (folds != k), applied to fold k
  - train rewards: cross-fitted inside the training schools, so a student's outcome model
                   never saw their own fold (nor fold k)
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

HERE = pathlib.Path(__file__).resolve().parent
ARMS = ['regular', 'small', 'aide']
FEATURES = ['female', 'age_at_entry', 'free_lunch', 'ethnicity_grp', 'locale',
            'age_at_entry_missing', 'free_lunch_missing']
CATEGORICAL = ['ethnicity_grp', 'locale']
SEED = 15

parser = argparse.ArgumentParser()
parser.add_argument('--method', choices=['dr', 'dm', 'ipw'], default='dr')
parser.add_argument('--model', choices=['rf', 'linear'], default='rf')
parser.add_argument('--placebo', action='store_true',
                    help='shuffle treatment within school; OPT gains should vanish')
args = parser.parse_args()
tag = f'{args.method}_{args.model}' + ('_placebo' if args.placebo else '')
rng = np.random.default_rng(SEED)

df = pd.read_csv(HERE / 'project_star_analysis.csv')
if args.placebo:
    df['treatment'] = df.groupby('school_id').treatment.transform(lambda s: rng.permutation(s.values))
X = pd.get_dummies(df[FEATURES], columns=CATEGORICAL, dtype=float)
y = df.outcome.to_numpy()
T = df.treatment.to_numpy()
P = df[['p_' + t for t in ARMS]].to_numpy()
fold = df.school_fold.to_numpy()
folds = np.sort(np.unique(fold))


def outcome_model():
    if args.model == 'linear':
        return LinearRegression()
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, random_state=SEED, n_jobs=-1)


def rewards(fit_idx, pred_idx):
    """Gamma for students pred_idx, with outcome models fit on students fit_idx."""
    G = np.zeros((len(pred_idx), len(ARMS)))
    for j, t in enumerate(ARMS):
        on_t = fit_idx[T[fit_idx] == t]
        f = outcome_model().fit(X.iloc[on_t], y[on_t]).predict(X.iloc[pred_idx])
        w = (T[pred_idx] == t) / P[pred_idx, j]
        G[:, j] = {'dm': f, 'ipw': w * y[pred_idx], 'dr': f + w * (y[pred_idx] - f)}[args.method]
    return G


out = []
for k in folds:
    train, test = np.where(fold != k)[0], np.where(fold == k)[0]
    G_train = np.zeros((len(train), len(ARMS)))
    for j in folds[folds != k]:
        inner = fold[train] == j
        G_train[inner] = rewards(train[~inner], train[inner])
    for role, idx, G in (('train', train, G_train), ('test', test, rewards(train, test))):
        out.append(pd.DataFrame(G, columns=['G_' + t for t in ARMS])
                   .assign(test_fold=k, role=role, student_id=df.student_id.to_numpy()[idx]))
    print(f'fold {k} done')

out = pd.concat(out)[['test_fold', 'role', 'student_id'] + ['G_' + t for t in ARMS]]
out.to_csv(HERE / f'rewards_{tag}.csv', index=False)
test = out[out.role == 'test']
print(f'Wrote rewards_{tag}.csv')
print('Held-out mean reward per arm (SD units):',
      test[['G_' + t for t in ARMS]].mean().round(3).to_dict())
