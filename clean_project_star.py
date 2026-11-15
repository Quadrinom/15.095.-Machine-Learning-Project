"""Clean project_star_students.csv into an analysis-ready file for the OPT project.

Run: python clean_project_star.py [--cohort k|all]
Inputs:  project_star_students.csv (from prepare_project_star.py), same folder
Outputs: project_star_clean.csv            every student, derived columns + flags, nothing dropped
         project_star_analysis.csv         modeling sample (default: kindergarten cohort)
         project_star_school_capacity.csv  observed small/aide counts per school (for B_s)

Design choices:
- Treatment = class type in the student's FIRST STAR year (the randomized assignment, ITT).
  Later-year class types are kept for reference but are not randomized (some students switched).
- Outcome = mean of reading and math z-scores in the entry grade, standardized against
  regular + regular-with-aide students in that grade (Krueger-style; effects are in SD units).
- Features are pre-treatment only. Teacher variables depend on the assigned class and are
  excluded from the feature set.
- Propensity = share of each class type within the student's school x entry grade
  (assignment was randomized within schools).
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
GRADES = ['k', '1', '2', '3']
TREAT = {'regular': 'regular', 'small': 'small', 'regular+aide': 'aide'}
N_FOLDS = 5
SEED = 15

parser = argparse.ArgumentParser()
parser.add_argument('--cohort', choices=['k', 'all'], default='k',
                    help="'k' = kindergarten entrants only (default); 'all' = every entry grade")
args = parser.parse_args()

df = pd.read_csv(HERE / 'project_star_students.csv')
assert df.student_id.is_unique

# Entry grade and the values observed in that grade
in_star = df[['star' + g for g in GRADES]].notna().to_numpy()
assert in_star.any(axis=1).all(), 'student with no STAR year'
entry_idx = in_star.argmax(axis=1)
df['entry_grade'] = np.array(GRADES)[entry_idx]
rows = np.arange(len(df))

def at_entry(prefix):
    return df[[prefix + g for g in GRADES]].to_numpy()[rows, entry_idx]

df['treatment'] = pd.Series(at_entry('star')).map(TREAT)
df['school_id'] = at_entry('schoolid').astype(int)
df['system_id'] = at_entry('system').astype(int)
df['locale'] = at_entry('school')
df['free_lunch'] = pd.Series(at_entry('lunch')).map({'free': 1.0, 'non-free': 0.0})
df['years_in_star'] = in_star.sum(axis=1)
star_cols = df[['star' + g for g in GRADES]]
df['stayed_initial_type'] = (star_cols.isna() | star_cols.eq(at_entry('star'), axis=0)).all(axis=1)

# Demographics
df['female'] = df.gender.map({'female': 1.0, 'male': 0.0})
# Asian, Hispanic, American Indian and other together are <1% of students; too small to split on
df['ethnicity_grp'] = df.ethnicity.map({'cauc': 'white', 'afam': 'black'}).fillna(
    df.ethnicity.where(df.ethnicity.isna(), 'other'))
# Age on Sept 1 of the entry school year (K = fall 1985), using the birth quarter midpoint
year = pd.to_numeric(df.birth.str[:4], errors='coerce')
qtr = pd.to_numeric(df.birth.str[-1], errors='coerce')
birth_frac = year + (qtr - 1) * 0.25 + 0.125
entry_year = 1985 + df.entry_grade.map({'k': 0, '1': 1, '2': 2, '3': 3})
df['age_at_entry'] = (entry_year + 8 / 12 - birth_frac).round(2)
# Flag ages far from grade norm (likely retained, early, or miscoded birth)
norm_age = 5.5 + entry_year - 1985
df['age_outlier'] = (df.age_at_entry - norm_age).abs() > 1.5

# Outcomes: z-scores per grade/subject vs regular-class students, then averaged
for g in GRADES:
    reg = df['star' + g].isin(['regular', 'regular+aide'])
    for subj in ('read', 'math'):
        col = df[subj + g]
        df[f'z_{subj}_{g}'] = (col - col[reg].mean()) / col[reg].std()
    df['z_total_' + g] = (df['z_read_' + g] + df['z_math_' + g]) / 2  # NaN if either missing
df['outcome'] = df[['z_total_' + g for g in GRADES]].to_numpy()[rows, entry_idx]
df['outcome_read'] = df[['z_read_' + g for g in GRADES]].to_numpy()[rows, entry_idx]
df['outcome_math'] = df[['z_math_' + g for g in GRADES]].to_numpy()[rows, entry_idx]
df['outcome_g3'] = df.z_total_3  # secondary, long-run outcome (heavy attrition)

# Propensities: arm shares within school x entry grade
cell = df.groupby(['entry_grade', 'school_id']).treatment
for t in TREAT.values():
    df['p_' + t] = cell.transform(lambda s, t=t: (s == t).mean())
df['cell_size'] = cell.transform('size')
df['overlap_ok'] = df[['p_' + t for t in TREAT.values()]].gt(0).all(axis=1)

# Grouped folds for held-out-school evaluation (by entry school)
schools = np.sort(df.school_id.unique())
rng = np.random.default_rng(SEED)
fold_of = dict(zip(rng.permutation(schools), np.arange(len(schools)) % N_FOLDS))
df['school_fold'] = df.school_id.map(fold_of)

# Missing-value indicators, then simple imputation for features
FEATURES = ['female', 'age_at_entry', 'free_lunch']
for c in FEATURES:
    df[c + '_missing'] = df[c].isna().astype(int)
df['ethnicity_grp'] = df.ethnicity_grp.fillna('unknown')

df.to_csv(HERE / 'project_star_clean.csv', index=False)

# Analysis sample
sample = df if args.cohort == 'all' else df[df.entry_grade == 'k']
drops = {
    'missing entry outcome': sample.outcome.isna(),
    'no overlap (school lacks an arm)': ~sample.overlap_ok,
}
keep = ~pd.concat(drops, axis=1).any(axis=1)
ana = sample[keep].copy()
for c in FEATURES:
    ana[c] = ana[c].fillna(ana[c].median())

cols = ['student_id', 'school_id', 'system_id', 'school_fold', 'entry_grade', 'treatment',
        'outcome', 'outcome_read', 'outcome_math', 'outcome_g3',
        'female', 'ethnicity_grp', 'age_at_entry', 'free_lunch', 'locale',
        'female_missing', 'age_at_entry_missing', 'free_lunch_missing', 'age_outlier',
        'p_regular', 'p_small', 'p_aide', 'cell_size', 'years_in_star', 'stayed_initial_type']
ana[cols].to_csv(HERE / 'project_star_analysis.csv', index=False)

cap = (ana.groupby('school_id').treatment.value_counts().unstack(fill_value=0)
       .reindex(columns=list(TREAT.values()), fill_value=0)
       .rename(columns=lambda t: 'n_' + t))
cap['n_total'] = cap.sum(axis=1)
cap['school_fold'] = cap.index.map(fold_of)
cap.to_csv(HERE / 'project_star_school_capacity.csv')

print(f'All students: {len(df):,}  ->  cohort={args.cohort}: {len(sample):,}')
for name, mask in drops.items():
    print(f'  dropped {mask.sum():>5,}  {name}')
print(f'Analysis sample: {len(ana):,} students, {ana.school_id.nunique()} schools')
print(ana.treatment.value_counts().to_string())
print('Mean outcome by arm (SD units):', ana.groupby('treatment').outcome.mean().round(3).to_dict())
