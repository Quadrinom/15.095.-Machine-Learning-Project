"""Figures and tables for the report (task 16).

Run:   python make_figures.py
Input: project_star_analysis.csv, rewards_dr_rf.csv, and the result files of
       opt_frontier.jl, opt_capacity.jl, subgroup_policy.py (DR-RF runs, with and without
       female/ethnicity_grp) and their placebo runs
Output (folder figures/):
  table1_data_summary.csv        students by class type: n, features, outcomes
  fig1_gain_by_group.png/.csv    held-out small-class gain (G_small - G_regular) by group
  fig2_frontier.png/.csv         gain over random vs share of small seats (fixed-share frontier)
  fig3_capacity.png/.csv         gain over random vs capacity scale (per-school seats)
  fig4_gender_shares.png/.csv    share of boys and girls in small classes, value by gender
  fig5_trees.png                 example trees (fold 0): all features vs no gender/ethnicity
All values are held-out DR-RF rewards in SD units; intervals resample whole schools
(B = 2000, seed 15; trees held fixed). Placebo bands: 5th-95th percentile of the 20 DR-RF
placebo shuffles. Colours: one per policy, the same in every figure (reference
categorical palette, slots 1-4 in order); random is the grey zero line.
"""
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
B, SEED = 2000, 15
DROP = '__drop-female-ethnicity_grp'

# Palette: reference categorical slots 1-4 (light mode), text and surface tokens
C = {'opt': '#2a78d6', 'opt_nosens': '#eb6834', 'subgroup': '#1baf7a', 'fl': '#eda100'}
LABEL = {'opt': 'OPT, all features', 'opt_nosens': 'OPT, no gender/ethnicity',
         'subgroup': 'Subgroup table (staff baseline)', 'fl': 'Free-lunch-first'}
INK, INK2, MUTED, SURFACE = '#0b0b0b', '#52514e', '#898781', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False, 'lines.linewidth': 2,
})


def weights(n_schools):
    rng = np.random.default_rng(SEED)
    return np.stack([np.bincount(rng.integers(n_schools, size=n_schools), minlength=n_schools)
                     for _ in range(B)]).astype(float)


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'figures/{name}.png')


# ---------------------------------------------------------------- data
df = pd.read_csv(HERE / 'project_star_analysis.csv')
df['gender'] = np.where(df.female == 1, 'Girls', 'Boys')
df['lunch'] = np.where(df.free_lunch == 1, 'Free lunch', 'No free lunch')
df['eth'] = df.ethnicity_grp.map({'white': 'White', 'black': 'Black'}).fillna('Other/unknown')
df['age'] = pd.cut(df.age_at_entry, [0, 5.165, 5.415, 99],
                   labels=['Age < 5.17', 'Age 5.17-5.41', 'Age >= 5.42'], right=False).astype(str)
df['loc'] = df.locale.str.capitalize()

# ---------------------------------------------------------------- Table 1
arms = ['regular', 'small', 'aide']
rows = {}
for t in arms + ['all']:
    d = df if t == 'all' else df[df.treatment == t]
    rows[t] = {
        'students': len(d), 'schools': d.school_id.nunique(),
        'female %': 100 * d.female.mean(), 'free lunch %': 100 * d.free_lunch.mean(),
        'Black %': 100 * (d.ethnicity_grp == 'black').mean(),
        'white %': 100 * (d.ethnicity_grp == 'white').mean(),
        'age at entry (mean)': d.age_at_entry.mean(),
        **{f'{loc} %': 100 * (d.locale == loc).mean() for loc in ['rural', 'suburban', 'urban', 'inner-city']},
        'outcome (SD, mean)': d.outcome.mean(), 'reading (SD)': d.outcome_read.mean(),
        'math (SD)': d.outcome_math.mean(),
    }
t1 = pd.DataFrame(rows).round(2)
t1.to_csv(OUT / 'table1_data_summary.csv')
print('wrote figures/table1_data_summary.csv')

# ---------------------------------------------------------------- Fig 1: gain by group
rw = pd.read_csv(HERE / 'rewards_dr_rf.csv')
te = rw[rw.role == 'test'].merge(df, on='student_id')
te['gain'] = te.G_small - te.G_regular
schools = np.sort(te.school_id.unique())
W = weights(len(schools))
groups = [('All students', None, None), ('Boys', 'gender', 'Boys'), ('Girls', 'gender', 'Girls'),
          ('No free lunch', 'lunch', 'No free lunch'), ('Free lunch', 'lunch', 'Free lunch'),
          ('White', 'eth', 'White'), ('Black', 'eth', 'Black'),
          ('Rural', 'loc', 'Rural'), ('Suburban', 'loc', 'Suburban'), ('Urban', 'loc', 'Urban'),
          ('Inner-city', 'loc', 'Inner-city'),
          ('Age < 5.17', 'age', 'Age < 5.17'), ('Age 5.17-5.41', 'age', 'Age 5.17-5.41'),
          ('Age >= 5.42', 'age', 'Age >= 5.42')]
rows = []
for lab, col, val in groups:
    d = te if col is None else te[te[col] == val]
    s = d.groupby('school_id').gain.sum().reindex(schools).fillna(0).to_numpy()
    n = d.groupby('school_id').size().reindex(schools).fillna(0).to_numpy()
    bt = (W @ s) / (W @ n)
    rows.append({'group': lab, 'n': int(n.sum()), 'gain': s.sum() / n.sum(),
                 'lo': np.quantile(bt, 0.025), 'hi': np.quantile(bt, 0.975)})
g1 = pd.DataFrame(rows)
g1.round(4).to_csv(OUT / 'fig1_gain_by_group.csv', index=False)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
y = np.arange(len(g1))[::-1].astype(float)
ax.axvline(g1.gain[0], color=MUTED, lw=1, ls='--', zorder=1)
ax.hlines(y, g1.lo, g1.hi, color=C['opt'], lw=2, zorder=2)
ax.scatter(g1.gain, y, s=36, color=C['opt'], edgecolor=SURFACE, linewidth=2, zorder=3)
ax.set_yticks(y, [f'{g}  (n={n:,})' for g, n in zip(g1.group, g1.n)])
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel('Held-out gain from a small class (SD units); dashed line: all students')
ax.set_title('Small-class gain by group (95% school-bootstrap intervals)')
ax.grid(axis='y', visible=False)
save(fig, 'fig1_gain_by_group')


# ---------------------------------------------------------------- per-school sums
def per_school(path, budget, policy, label):
    r = pd.read_csv(path)
    r = r[r.policy == policy]
    return r.assign(label=label)[['school_id', budget, 'label', 'n', 'sum_value']]


def curves(parts, budget, placebo_glob, labels):
    """Gain over random (point, 95% CI) per budget for each label, and the placebo band."""
    long = pd.concat(parts)
    t = long.pivot_table(index=['school_id', budget], columns='label', values='sum_value')
    sch = t.index.get_level_values(0).unique().to_numpy()
    buds = np.sort(t.index.get_level_values(1).unique().to_numpy())
    n = long[long.label == 'random'].groupby('school_id').n.first().loc[sch].to_numpy()
    Wb = weights(len(sch))
    out = []
    for lab in labels:
        for b in buds:
            x = t.xs(b, level=1).loc[sch]
            d = (x[lab] - x['random']).to_numpy()
            bt = (Wb @ d) / (Wb @ n)
            out.append({budget: b, 'policy': lab, 'gain_over_random': d.sum() / n.sum(),
                        'lo': np.quantile(bt, 0.025), 'hi': np.quantile(bt, 0.975)})
    out = pd.DataFrame(out)
    null = []
    for f in sorted(HERE.glob(placebo_glob)):
        if '__drop' in f.stem:
            continue
        r = pd.read_csv(f)
        tot = r.groupby([budget, 'policy']).sum_value.sum().unstack()
        null.append((tot.opt - tot.random) / r[r.policy == 'opt'].groupby(budget).n.sum())
    null = pd.concat(null, axis=1)
    band = pd.DataFrame({'p05': null.quantile(0.05, axis=1), 'p95': null.quantile(0.95, axis=1)})
    return out, band, len(null.columns)


def gain_plot(ax, cv, band, budget, labels, xlab, xfmt, n_placebo):
    ax.fill_between(band.index, band.p05, band.p95, color=MUTED, alpha=0.22, lw=0,
                    label=f'Placebo 5-95% ({n_placebo} shuffles)')
    ax.axhline(0, color=INK2, lw=1, ls='--', label='Random allocation (same seats)')
    for lab in labels:
        d = cv[cv.policy == lab]
        if lab == 'opt':
            ax.fill_between(d[budget], d.lo, d.hi, color=C[lab], alpha=0.15, lw=0)
        ax.plot(d[budget], d.gain_over_random, color=C[lab], marker='o', ms=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=LABEL[lab] + (' (band: 95% CI)' if lab == 'opt' else ''))
    ax.set_xlabel(xlab)
    ax.set_ylabel('Gain over random allocation\n(held-out SD units)')
    ax.xaxis.set_major_formatter(xfmt)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)


# ---------------------------------------------------------------- Fig 2: frontier
labels = ['opt', 'opt_nosens', 'subgroup', 'fl']
parts = [per_school(HERE / 'frontier_dr_rf.csv', 'q', 'opt', 'opt'),
         per_school(HERE / 'frontier_dr_rf.csv', 'q', 'random', 'random'),
         per_school(HERE / 'frontier_dr_rf.csv', 'q', 'free_lunch_first', 'fl'),
         per_school(HERE / f'frontier_dr_rf{DROP}.csv', 'q', 'opt', 'opt_nosens'),
         per_school(HERE / 'frontier_subgroup_dr_rf.csv', 'q', 'subgroup', 'subgroup')]
cv, band, npl = curves(parts, 'q', 'frontier_dr_rf_placebo*.csv', labels)
cv.round(4).to_csv(OUT / 'fig2_frontier.csv', index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
gain_plot(ax, cv, band, 'q', labels, 'Share of each school\'s students given a small class',
          matplotlib.ticker.PercentFormatter(1.0), npl)
ax.set_title('Fixed-share frontier: targeting gains little, and only with gender')
save(fig, 'fig2_frontier')

# ---------------------------------------------------------------- Fig 3: capacity
parts = [per_school(HERE / 'capacity_dr_rf.csv', 'scale', 'opt', 'opt'),
         per_school(HERE / 'capacity_dr_rf.csv', 'scale', 'random', 'random'),
         per_school(HERE / 'capacity_dr_rf.csv', 'scale', 'free_lunch_first', 'fl'),
         per_school(HERE / f'capacity_dr_rf{DROP}.csv', 'scale', 'opt', 'opt_nosens'),
         per_school(HERE / 'capacity_subgroup_dr_rf.csv', 'scale', 'subgroup', 'subgroup')]
cv3, band3, npl3 = curves(parts, 'scale', 'capacity_dr_rf_placebo*.csv', labels)
cv3.round(4).to_csv(OUT / 'fig3_capacity.csv', index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
gain_plot(ax, cv3, band3, 'scale', labels,
          'Seats per school, as a multiple of its actual small and aide seats (assumed)',
          matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}x'), npl3)
ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
ax.set_title('Per-school capacities: no policy beats random beyond chance')
save(fig, 'fig3_capacity')


# ---------------------------------------------------------------- Fig 4: gender
def shares_by_gender(rank_file, frontier_file, q, rank_col):
    rk = pd.read_csv(rank_file)
    seats = pd.read_csv(frontier_file).query('policy == "opt" and q == @q')[['school_id', 'n_small']]
    d = rk.merge(seats, on='school_id').merge(df[['student_id', 'gender']], on='student_id')
    d['small'] = d[rank_col] <= d.n_small
    return d


Q = 0.3
a = shares_by_gender(HERE / 'frontier_ranks_dr_rf.csv', HERE / 'frontier_dr_rf.csv', Q, 'rank_opt')
b = shares_by_gender(HERE / f'frontier_ranks_dr_rf{DROP}.csv', HERE / f'frontier_dr_rf{DROP}.csv', Q, 'rank_opt')
f = shares_by_gender(HERE / 'frontier_ranks_dr_rf.csv', HERE / 'frontier_dr_rf.csv', Q, 'rank_fl')
sh = pd.DataFrame({'opt': a.groupby('gender').small.mean(), 'opt_nosens': b.groupby('gender').small.mean(),
                   'fl': f.groupby('gender').small.mean()})
sh['random'] = Q
# value by gender: OPT and OPT no-sens minus random, with school bootstrap
vals = []
for lab, d in [('opt', a), ('opt_nosens', b)]:
    d = d.copy()
    d['v'] = np.where(d.small, d.G_small, d.G_regular)
    d['vr'] = d.n_small / d.groupby('school_id').student_id.transform('size') * (d.G_small - d.G_regular) + d.G_regular
    for gname in ['Boys', 'Girls']:
        e = d[d.gender == gname]
        s = (e.v - e.vr).groupby(e.school_id).sum().reindex(schools).fillna(0).to_numpy()
        n = e.groupby('school_id').size().reindex(schools).fillna(0).to_numpy()
        bt = (W @ s) / (W @ n)
        vals.append({'policy': lab, 'gender': gname, 'share_small': sh.loc[gname, lab],
                     'gain_over_random': s.sum() / n.sum(),
                     'lo': np.quantile(bt, 0.025), 'hi': np.quantile(bt, 0.975)})
vals = pd.DataFrame(vals)
vals.round(4).to_csv(OUT / 'fig4_gender_shares.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2), gridspec_kw={'width_ratios': [1.25, 1]})
pols = ['opt', 'opt_nosens', 'fl']
x = np.arange(2)
w = 0.24
for i, p in enumerate(pols):
    v = sh.loc[['Boys', 'Girls'], p].to_numpy()
    ax1.bar(x + (i - 1) * w, v, w - 0.03, color=C[p], label=LABEL[p])
    for xi, vi in zip(x + (i - 1) * w, v):
        ax1.text(xi, vi + 0.01, f'{vi:.0%}', ha='center', va='bottom', fontsize=7.5, color=INK2)
ax1.axhline(Q, color=INK2, lw=1, ls='--')
ax1.text(1.5, Q, ' random', fontsize=8, color=INK2, va='center')
ax1.set_xticks(x, ['Boys', 'Girls'])
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax1.set_ylim(0, 0.65)
ax1.set_ylabel('Share in a small class')
ax1.set_title(f'Who gets the {Q:.0%} small seats')
ax1.grid(axis='x', visible=False)
ax1.legend(loc='upper right', fontsize=7.5)
for i, p in enumerate(['opt', 'opt_nosens']):
    d = vals[vals.policy == p]
    yy = np.arange(2) + (i - 0.5) * 0.3
    ax2.hlines(yy, d.lo, d.hi, color=C[p], lw=2)
    ax2.scatter(d.gain_over_random, yy, color=C[p], s=36, edgecolor=SURFACE, linewidth=2, zorder=3)
ax2.axvline(0, color=INK2, lw=1, ls='--')
ax2.set_yticks([0, 1], ['Boys', 'Girls'])
ax2.set_ylim(1.6, -0.6)
ax2.set_xlabel('Gain over random (SD)')
ax2.set_title('Who gains and who loses')
ax2.grid(axis='y', visible=False)
fig.tight_layout()
save(fig, 'fig4_gender_shares')


# ---------------------------------------------------------------- Fig 5: trees
def parse_tree(path):
    """Nodes from IAI's text tree: id -> dict(depth, text); children in order; leaf gains."""
    txt = pathlib.Path(path).read_text()
    nodes, stack, gains = {}, [], {}
    for line in txt.splitlines():
        m = re.match(r'^( *)(\d+)\) (Split|Prescribe): (.*)$', line)
        if m:
            depth = (len(m.group(1)) - 2) // 2
            nid = int(m.group(2))
            nodes[nid] = {'depth': depth, 'kind': m.group(3), 'text': m.group(4), 'kids': []}
            while stack and nodes[stack[-1]]['depth'] >= depth:
                stack.pop()
            if stack:
                nodes[stack[-1]]['kids'].append(nid)
            stack.append(nid)
        m = re.match(r'^leaf (\d+): n_train = (\d+), mean training gain = ([-\d.]+)', line)
        if m:
            gains[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
    return nodes, gains


def split_label(text):
    t = text.replace('female < 0.5', 'boy (female = 0)?')
    t = re.sub(r'locale in \[(.*?)\] or is missing', lambda m: 'locale: ' + m.group(1).replace(',', ', ') + '?', t)
    t = re.sub(r'age_at_entry < ([\d.]+)', r'age < \1?', t)
    return t


def draw_tree(ax, path, title):
    nodes, gains = parse_tree(path)
    leaves = [n for n in nodes if not nodes[n]['kids']]
    pos, order = {}, iter(range(len(leaves)))

    def place(n):
        if not nodes[n]['kids']:
            pos[n] = (next(order), -nodes[n]['depth'])
        else:
            for k in nodes[n]['kids']:
                place(k)
            xs = [pos[k][0] for k in nodes[n]['kids']]
            pos[n] = (np.mean(xs), -nodes[n]['depth'])
    place(1)
    for n, d in nodes.items():
        for j, k in enumerate(d['kids']):
            ax.plot([pos[n][0], pos[k][0]], [pos[n][1], pos[k][1]], color=MUTED, lw=1, zorder=1)
            ax.text((pos[n][0] + pos[k][0]) / 2, (pos[n][1] + pos[k][1]) / 2,
                    'yes' if j == 0 else 'no', fontsize=7.5, color=INK2, ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc=SURFACE, ec='none'))
    for n, d in nodes.items():
        x0, y0 = pos[n]
        if d['kids']:
            ax.text(x0, y0, split_label(d['text']), ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.4', fc='#eef3fb', ec=C['opt'], lw=1))
        else:
            n_tr, g = gains[n]
            ax.text(x0, y0, f'gain {g:+.2f} SD\n{n_tr:,} students', ha='center', va='center',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.4', fc=SURFACE, ec=MUTED, lw=1))
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.5, 0.5)
    ax.axis('off')
    ax.set_title(title, fontsize=9)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
draw_tree(ax1, HERE / 'trees_frontier_dr_rf' / 'tree_fold0.txt', 'All features (fold 0)')
draw_tree(ax2, HERE / f'trees_frontier_dr_rf{DROP}' / 'tree_fold0.txt',
          'No gender/ethnicity (fold 0)')
fig.suptitle('Policy trees: leaves rank students for small seats by estimated gain',
             x=0.01, ha='left', fontsize=10, fontweight='bold')
fig.tight_layout()
save(fig, 'fig5_trees')
