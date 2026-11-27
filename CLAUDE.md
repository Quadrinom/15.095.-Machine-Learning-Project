# CLAUDE.md — Project STAR Classroom Allocation Using Optimal Policy Trees

## Purpose of this file

Context for Claude Code on an MIT 15.095 (*Machine Learning Under a Modern Optimization Lens*) final project by **Daniel Krupp** and **Adi Palaniappan**. Either of them may be the user in a session. This file holds the plan, the decisions already made, the current status, the task list and the open questions. Detailed implementation notes and results are in `project_star_notes.txt`; read it before changing code.

Inspect the actual files before modifying anything. Never present an assumption as an observed fact. Last updated **2026-11-27**.

## Project in one paragraph

We learn an **interpretable, resource-constrained policy** that assigns elementary-school students to one of three classroom types using the Tennessee **Project STAR** randomized experiment: **regular class**, **small class**, and **regular class with a full-time teacher's aide**. The core method is an **Optimal Policy Tree (OPT)** trained on estimated counterfactual rewards for each student and class type. The main extension is **school-level capacity constraints** (small-class seats and aide-supported seats), possibly with class-type costs. Policies are evaluated on **held-out schools** against simple allocation rules, with subgroup/equity checks.

## Research question

> Can observable student characteristics guide assignments among regular classes, small classes, and regular classes with an aide to improve overall expected academic achievement under school-level resource constraints, while yielding a policy that administrators can understand?

Do not presuppose that small classes are best for every student, or that the aide arm is useful. "Targeting does not beat random allocation" is a valid finding if the data show it.

## Current status (as of 2026-11-26)

A first end-to-end pipeline exists (commit `07283cd`, 15 Nov 2026): data download → cleaning → cross-fitted rewards → policy trees → held-out evaluation. See `project_star_notes.txt` sections 1–9, sections 10–11 for the work of 18 Nov, section 12 for the placebo, the fixed-share frontier and the gender finding (26 Nov), section 13 for the runs without sensitive features, subgroup values and per-school capacities, and section 14 for the subgroup-level allocation and the figures (26 Nov). Section 12i lists what the 26 Nov commits changed in earlier work (including a regenerated `rewards_dr_linear_placebo.csv` and a corrected reading of the gender splits).

- **Runs done:**
  - Main configuration: DR rewards with random-forest outcome models (`rewards_dr_rf.csv`, `results_dr_rf.csv`, trees in `trees_dr_rf/`).
  - Trees trained on DM-RF, DR-linear and DM-linear rewards, each scored with the DR-RF test rewards (`results_<tag>__eval_dr_rf.csv`).
  - Semi-synthetic runs (file names contain `_plant_`; SYNTHETIC).
  - Placebo λ sweeps: DR-linear (seed 15) and DR-RF (seeds 15, 1–4) (`results_*placebo*.csv`).
  - Without sensitive features (`--drop=female,ethnicity_grp`, file names contain `__drop-`): frontier and capacity runs for all four reward types, plus `--drop=female` for DR-RF and 20 placebo frontiers.
  - Per-school capacities (`opt_capacity.jl`): DR-RF, DM-RF, DR-linear, DM-linear, with and without sensitive features (`capacity_*.csv`, `trees_capacity_*/`), plus 20 DR-RF placebos.
  - Fixed-share budget frontier (`opt_frontier.jl`): DR-RF, DM-RF, DR-linear, DM-linear (`frontier_*.csv`, `trees_frontier_*/`), plus 20 DR-RF placebo shuffles (`frontier_dr_rf_placebo*.csv`) and one SYNTHETIC smoke test.
- **Fixed-share frontier (notes section 12).** Each held-out school gives small places to a set share q of its students; OPT ranks students by its leaf's estimated gain, and random and free-lunch-first get the same seat count in every school. OPT beats random by +0.03 SD (DR-RF) to +0.05 SD (DR/DM-linear) at q = 0.3–0.4. Against 20 placebo shuffles that is p ≈ 0.05–0.19: at the edge of chance.
- **Boys gain more from small classes than girls** (notes 12f). Held-out DR gain: boys 0.306, girls 0.117 (gap +0.19, school-bootstrap 95% CI [+0.07, +0.31]); raw means and a school-fixed-effects regression agree (+0.21, +0.135). Every frontier tree splits on gender (5/5 folds for all four reward types), against 29 of 100 placebo folds. A policy that targets small classes by gender raises a fairness problem: see the next bullet.
- **Without gender and ethnicity, OPT is no better than random** (notes 13c). Frontier best share: DR-RF +0.005 (p 0.86 against placebos with the same features), DR/DM-linear +0.02 (p 0.33–0.38), DM-RF never positive. The trees then split on locale, which is the same for every student in a school, so within-school filling becomes random. All of the frontier's targeting value came from gender. Subgroups (DR-RF, q = 0.3): the all-feature policy puts 53% of boys and 5% of girls in small classes; girls do worse than under random allocation (−0.047 [−0.085, −0.007]). Without the sensitive features no group gains or loses.
- **Per-school capacities (notes 13d–e).** Seats = c × each school's observed STAR small and aide seats, c = 0.5–1.5 (assumed scenarios). OPT − random (DR-RF) is +0.028 at c = 0.5 down to +0.004 at c = 1.5, and +0.006 at the actual seats (c = 1.0); inside the placebo's 95th percentile at every scale. The tree's own prescriptions ignore capacities and break a quota in 53–71 of 78 schools.
- **Main result, λ sweep** (held-out DR value in SD units, mean over 5 folds): OPT does **not** beat random allocation or free-lunch-first at any cost level. Differences range from −0.02 to +0.01 SD, while results vary by 0.03–0.07 SD between folds. All-small 0.199, all-aide −0.002, all-regular −0.016.
- **Less noisy rewards don't change this.** Trees trained on DM or linear rewards beat random by at most +0.01 to +0.04 SD, at a single cost level and within fold noise. DM-trained trees send 5–15% of students to aide classes, which loses value. Notes section 11.
- **The method can find real differences (semi-synthetic test, SYNTHETIC).** With an extra +0.3 SD small-class benefit planted for free-lunch students, the trees split on free lunch in 5 of 5 folds and beat random by +0.07 ± 0.02 SD at λ=0.3. At +0.2 SD the effect is found in 3 of 5 folds; at +0.1 SD, in 1 of 5. In the real data the small-class gain differs by only 0.004 SD between free-lunch and other students (0.181 vs 0.177), far below that threshold. Notes section 11.
- **The real trees look like noise.** The split feature changes with the cost level and between folds.
  - Fold 0: age ≥ 6.165 → regular at λ=0.05; suburban → regular at λ=0.1 and 0.15; girls → regular, boys → small at λ=0.2. No fold-0 tree uses free lunch.
  - Across all five folds, free lunch appears only at λ=0.3, in 2 of 5.
  - Even with a strong planted effect the trees also split on gender (4 of 5 folds). This was first read as noise, but the planted runs sit on top of the real data, which has a real gender gap (notes 12f), so these splits most likely reflect it.
  - The λ-sweep placebos look the same as the real λ sweep (OPT − random −0.032 to +0.015), so the λ sweep finds nothing beyond noise.
- **Our own kindergarten data:** small classes +0.18 SD, aides +0.01 SD. The small-class gain is the same with and without free lunch.
- **Working conclusion so far:** in kindergarten, *how many* small seats there are matters far more than *who* gets them. The only feature that ranks students within a school is gender; targeting on it is worth about +0.03 to +0.05 SD over random (borderline against placebos) and nearly shuts girls out of small classes. Without it, and under per-school capacities, OPT does not beat random. Proposed framing for the report: the no-sensitive-features policy as the main policy; the gender result as a finding about heterogeneity, with its fairness cost.
- **Per-school capacity constraints (the main extension):** option (b) is done twice: the frontier (B_s = q · n_s, small only) and `opt_capacity.jl` (observed small and aide seats × c). Option (c) is done as an exact enumeration (`opt_exact.jl`, notes 15): every depth ≤ 2 tree and every leaf order, with equation 3 inside the fit, is a certified optimum of the training objective. In-sample it beats random by +0.040 to +0.054 SD per student, 0.02–0.04 SD more than IAI's policy (+0.005 to +0.029), but held-out exact − IAI covers 0 at every scale, exact − random reaches +0.031 (c = 1.25, placebo p = 0.095), and without gender/ethnicity exact is *worse* than random. More optimization overfits the noisy rewards; the optimizer is not the bottleneck.
- **Subgroup-level allocation (staff baseline, notes 14).** Done with `subgroup_policy.py` (144 groups, same seat filling as OPT). With DR rewards the fine table overfits (≈ random) and OPT beats it by +0.03 to +0.06 SD at q = 0.3–0.4, so the tree gives up nothing for interpretability. With DM-linear rewards subgroup ≈ OPT, both +0.05 over random (p 0.048). Without gender/ethnicity, and under capacities, nothing beats random.
- **Figures and Table 1** (`make_figures.py` → `figures/`, notes 14c): data summary by class type, gain by group, frontier, capacity, gender shares, example trees. Each figure has a CSV with its numbers.

## Decisions made

Details and counts are in `project_star_notes.txt` sections 2–6.

| Topic | Decision |
|---|---|
| Data | AER STAR from Rdatasets: 11,598 students × 52 columns, one row per student, separate columns per grade. `student_id` is the Rdatasets row index, not the original STAR ID |
| Sample | Kindergarten entrants: 6,325 → **5,752 students in 78 schools** after dropping 539 with no outcome and school 14 (no regular class). Regular 2,005 / small 1,725 / aide 2,022. `--cohort all` is a robustness check only |
| Treatment | Class type in the student's first STAR year (intent-to-treat). Column order everywhere: **regular, small, aide** |
| Outcome | Mean of reading and math z-scores in kindergarten, standardized against regular + aide students (SD units). `outcome_g3` is 51% missing and needs attrition weighting before use |
| Features | `female`, `age_at_entry`, `free_lunch`, `ethnicity_grp` (white/black/other/unknown), `locale` (rural/suburban/urban/inner-city), plus `*_missing` flags. Teacher variables are excluded (they depend on the assigned class) |
| Propensities | Each class type's share within school × entry grade (`p_regular`, `p_small`, `p_aide`). Minimum 0.161, no clipping |
| Splits | 5 folds of whole schools (`school_fold`, seed 15). Training rewards are cross-fitted within training schools; test rewards use outcome models fit on all training schools |
| Rewards code | `opt_rewards.py` (Python, scikit-learn). Options: `--method` dr / dm / ipw, `--model` rf / linear, `--placebo [--seed N]`, and `--plant DELTA --plant-group` free_lunch / young (semi-synthetic). The linear model drops one reference level per category |
| Trees code | `opt_trees.jl <tag> [eval_tag]` (Julia + IAI 3.2.2 `OptimalTreePolicyMaximizer`): GridSearch over `max_depth` 1–3, `minbucket` 50. The optional `eval_tag` picks the rewards that score held-out schools. Every fold's tree is saved as text |
| Budget (λ sweep) | Global cost penalty λ ∈ {0, .05, .1, .15, .2, .3} subtracted from small and aide training rewards (`opt_trees.jl`) |
| Budget (frontier) | `opt_frontier.jl`: one regular-vs-small tree per fold, small reward lowered by the mean training gain; each held-out school fills round(q · n_s) small places by leaf gain, q = 0, 0.1, …, 1. Random and free-lunch-first get the same per-school counts. Aide is left out |
| Capacities | `opt_capacity.jl`: one three-arm tree per fold (small and aide rewards each lowered by their mean training gain). B_s^small = round(c · observed small), B_s^aide = min(round(c · observed aide), n_s − B_s^small), c ∈ {0.5, 0.75, 1, 1.25, 1.5} (assumed). Fill small seats first by leaf small gain, then aide by leaf aide gain; all seats used; free-lunch-first fills both by free lunch; random = exact expectation |
| Sensitive features | `--drop=f1,f2` in `opt_frontier.jl` and `opt_capacity.jl`; the main comparison drops `female` and `ethnicity_grp`. Placebos are matched on features |
| Placebo | `opt_rewards.py --placebo --seed N` shuffles class types within schools; only the shuffle uses N, outcome models keep seed 15. 20 DR-RF shuffles (seeds 1–20) form the null for the frontier (`placebo_summary.py`) |

## Task list

Not split by person. Status as of 2026-11-27. When you start a task, add "(in progress: Adi)" or "(in progress: Daniel)" to it and push, so you don't both start the same one.

**Done:**
1. Data preparation and cleaning.
2. Counterfactual rewards: DR with random forests (main). DM and linear models were run for task 8; IPW is coded but not run.
3. Unconstrained OPT (IAI, depth 1–3).
4. Baselines: all-regular, all-small, all-aide, random, free-lunch-first, with gaps (see Baselines below).
5. Held-out DR evaluation for the DR-RF configuration.
6. Pipeline re-run on Adi's Mac (2026-11-18): cleaning outputs byte-for-byte identical, rewards identical up to rounding, and all 105 result rows and six fold-0 trees match. Details in notes section 10f.
8. Semi-synthetic check (2026-11-18): planted effects of +0.3 SD are found in 5 of 5 folds, +0.2 SD in 3 of 5, +0.1 SD in 1 of 5. Trees were also trained on DM-RF, DR-linear and DM-linear rewards (part of task 15), and a bug in the linear-model rewards was fixed. Details in notes section 11.
7. Placebo check (2026-11-26, Daniel): `--seed` added to `opt_rewards.py`. λ sweep on 6 placebos (no difference from the real λ sweep) and the frontier on 20 DR-RF placebos (the null for task 9). `rewards_dr_linear_placebo.csv` regenerated: 5 rows had been hit by the old linear-model bug. Notes 12a, b, e.
9. Fixed-share budget frontier (2026-11-26, Daniel): `opt_frontier.jl`, `frontier_summary.py`, `placebo_summary.py`. OPT +0.03 to +0.05 SD over random at q = 0.3–0.4, p ≈ 0.05–0.19 against the placebos. Notes 12c–e.
14. Subgroup values and runs without sensitive features (2026-11-26, Daniel): `--drop` option, `subgroup_summary.py`. Without gender and ethnicity OPT ≈ random; the gender-targeting policy makes girls worse off than random. Notes 13a–c.
10. Per-school capacities, option (b) (2026-11-26, Daniel): `opt_capacity.jl`, `capacity_summary.py`; observed small and aide seats × 0.5–1.5, three-arm tree, small-first seat filling. OPT − random +0.004 to +0.028, inside the placebo range. Notes 13d–e. The exact MIP is task 19.
11. Subgroup-level allocation baseline (2026-11-26, Daniel): `subgroup_policy.py`, `subgroup_compare.py`. DR rewards: OPT > subgroup table ≈ random; DM-linear: subgroup ≈ OPT > random; without sensitive features all ≈ random. Notes 14a–b.
16. Figures and data summary table (2026-11-26, Daniel): `make_figures.py` → `figures/` (Table 1, Figs 1–5 with CSVs). Notes 14c. Add more as the report needs them.
19. Exact capacity optimization (option c), done 2026-11-27 (Adi): `opt_exact.jl`, `exact_summary.py`. Exhaustive enumeration of every depth ≤ 2 tree (leaves ≥ 50) and every leaf order, with equation 3 inside the fit; certified optimum of the training objective, no MIP solver needed. Main run, without gender/ethnicity, and 20 placebos each. Exact beats IAI in-sample, not held-out; see notes 15. Also: scoring ties by their expected value instead of one random draw changes `opt_capacity.jl`'s OPT − random noticeably (c = 1.0: +0.006 → +0.022), still within the placebo range (notes 15d). Not done: allocating seats *between* schools.

**To do (rough priority order; all optional before the deadline except 17):**
17. Report and presentation (in progress: Adi). Outline: `report/report_outline.tex`; the report itself is in Overleaf.
12. Fix the baseline gaps in the λ sweep (`opt_trees.jl`): aide seats in the heuristic, per-school matching. Both are done in `opt_capacity.jl`; only needed if the λ sweep stays in the report.
13. Uncertainty: school-level bootstrap for policy values and for differences between policies. Done for the frontier, capacities and subgroups (`frontier_summary.py`, `capacity_summary.py`, `subgroup_summary.py`; trees held fixed); not for the λ sweep. A bootstrap that refits trees is not done.
18. Like-for-like placebos for the DM and linear runs (the current null uses DR-RF placebo trees), and capacity placebos without sensitive features. Placebo means are slightly positive at q = 0.6–0.9 (about +0.01); not investigated.
15. Robustness: propensity choice, depth, tree stability, other outcomes and cohort. (DM vs DR and linear vs RF are done for the λ sweep; see task 8.)

## Method specification

### Counterfactual rewards

Only one outcome is observed per student, so we build an `n × 3` reward matrix. DR is the primary estimator; DM and IPW are comparisons.

```text
Gamma_hat[i,t] = f_hat_t(x_i) + 1{T_i == t} * (y_i - f_hat_t(x_i)) / p_hat_t(x_i)
```

DR rewards are noisy pseudo-rewards for policy learning and evaluation. They are not individual treatment effects, which are unobservable.

### Optimal Policy Tree

```text
maximize over pi in Pi_D:  sum_i Gamma_hat[i, pi(x_i)]
```

Reference: Amram, Dunn & Zhuo (2022). IAI finds trees by heuristic local search, so a tree is not provably optimal and has no optimality gap. Say so where it matters.

### Capacity constraints (main extension)

For each school `s` with students `S_s`:

```text
sum_{i in S_s} 1{ pi(x_i) == small } <= B_s^small
sum_{i in S_s} 1{ pi(x_i) == aide  } <= B_s^aide
```

- **Current implementation:** (a) the global cost penalty λ in `opt_trees.jl`, the *optional* per-student cost model, not equation 3; it controls the budget on average, not per school. (b) `opt_frontier.jl` fills each held-out school's small places by leaf gain with B_s^small = round(q · n_s) and no aide seats. (b) again in `opt_capacity.jl`: observed small and aide seats × c, small seats filled first, then aide; zero quota violations (asserted). The tree's own prescriptions break quotas in 53–71 of 78 schools (notes 13e).
- **Options** (notes section 6): (a) cost penalty (built); (b) tree, then fill by school: rank students by their leaf's estimated gain and fill each school's `B_s` in that order; (c) a custom MIP (e.g. a depth-2 tree with constraint 3). JuMP and Gurobi are available for (c).
- **Starting capacities:** `project_star_school_capacity.csv` holds each school's observed counts. Scenarios such as 50/100/150% of observed seats are **assumed**, not real budgets.
- **Critical design issue:** a fixed tree mapping `x → class type` does not guarantee per-school quotas on new schools. Document the seat-filling rule used, apply the same rule to every compared policy, and check feasibility in every evaluated school. Quota violations must be zero for anything called feasible.
- Real costs (salaries, classrooms, class-size thresholds) are not sourced. Label any cost scenario as illustrative.

### Baselines

1. **All regular:** low-resource reference.
2. **All small:** unconstrained reference, generally infeasible.
3. **All aide:** third-arm reference.
4. **Random allocation** at the same resource level.
5. **Free-lunch-first heuristic:** free-lunch students get limited small seats first.
6. **Subgroup-level allocation** (staff feedback, below).

The universal policies are **unconstrained references**. Constrained comparisons are mainly against the random, heuristic and subgroup policies at the same resource levels.

Gaps in the current versions (`opt_trees.jl`):
- `free_lunch_first` never assigns aide seats. When OPT uses aide places the comparison is not like for like (fold 1, λ=0: OPT put 40% in aide, the heuristic 0%).
- Random and heuristic are matched to OPT's shares over the whole test fold, not per school.
- `random_same_budget` is the exact expected value of random allocation: fine for the mean, but there is no spread across seeds.

A random-forest or causal-forest **policy benchmark** was dropped from scope. Using a causal forest only as a diagnostic for heterogeneity is an open decision.

### Subgroup-level allocation (teaching staff feedback)

> "Very interesting idea! One thing to consider is that, if you only have a few discrete pre-treatment features, students will fall into a limited number of subgroups. You could compute the best class type for each subgroup directly by brute force, averaging the estimated rewards within the subgroup and picking the highest (capacity limits could be added as a small linear optimization problem). Given that a shallow tree won't separate every subgroup, comparing your tree against this subgroup-level allocation on held-out schools would show whether the tree gives up any achievement for more interpretability and less risk of overfitting."

How to build it:
- Each combination of features is a group: gender × free lunch × ethnicity × locale = 64 combinations, with `age_at_entry` cut into bands (age is continuous; the bands are an open decision).
- In the training schools, average `G_regular`, `G_small`, `G_aide` within each group and give the group the best class type.
- With capacities, decide how many students of each group get small/aide places with a small LP over groups.
- Score it on held-out schools with equation 4, exactly like OPT.

How to read it:
- On training data the subgroup policy can never score below a tree on the same discrete features, because each leaf is a merge of groups. The test is on held-out schools. Age is the exception: the tree picks its own cut-off.

| Held-out result | Meaning |
|---|---|
| Subgroup ≈ OPT, both beat random | The tree keeps the benefit of targeting in a simpler form |
| Subgroup > OPT > random | Targeting helps; the tree gives some of it up for simplicity |
| OPT > subgroup | The fine-grained table overfits small groups |
| Subgroup ≈ OPT ≈ random | No useful targeting on these features |

Caveats: "≈" needs an uncertainty interval on the difference, not just similar point estimates. Under capacities, both policies must use the same seat-filling rule. Small groups have noisy averages.

### Evaluation

```text
V_hat(pi) = (1 / n_test) * sum_{i in test} Gamma_hat_test[i, pi(x_i)]
```

- **Current protocol:** 5-fold cross-validation over schools; every school is a test school once. Tuning inside each fold is clean, but there is **no locked test set**. Any design choice made after looking at `results_dr_rf.csv` has seen every school (open decision).
- **One yardstick:** policies trained on other rewards are scored with the DR-RF test rewards (`opt_trees.jl <tag> dr_rf`), so every comparison uses the same evaluation.
- **Report:** mean value, difference from baselines, resource use, quota violations, and uncertainty that respects school clustering (school-level bootstrap).
- **Subgroups:** report values by free lunch, ethnicity and gender. Optimizing the overall mean can hide harm to disadvantaged groups.
- **Sensitive features:** run without ethnicity and gender; report the change in value and how many assignments change.
- **Stability:** compare splits across folds and bootstrap samples.

### Experiments and figures

Daniel's notes section 8 lists experiments 1–12; the staff's subgroup comparison is experiment 13. Planned figures and tables:
- data summary by class type
- learned tree diagrams
- policy-value comparison table
- budget frontier: value vs share of small places for OPT, random, heuristic and subgroup
- capacity scenario results
- subgroup values
- uncertainty intervals

### Equity objective (optional, not core)

A bottom-quartile objective must define the group **before treatment**. STAR has no pre-kindergarten test score, so the notes propose weighting students whose *predicted* regular-class score (`f_regular`) is in the bottom quartile. Keep this separate from core mean-value OPT.

## Environment

**Adi's Mac (checked 2026-11-18):**
- **Julia + IAI:** `julia +1.11.6/IAI/3.2.2` loads IAI 3.2.2 (academic license) and fits a policy tree. IAI is preloaded, so use `IAI.` without `using IAI`.
  - Its environment (`Julia_v1_11_6__IAI_v3_2_2`) has CSV, DataFrames and CategoricalArrays. CategoricalArrays v0.10.8 was added on 2026-11-18 because `opt_trees.jl` imports it.
  - There is no `@iai` environment here, so run `julia +1.11.6/IAI/3.2.2 opt_trees.jl dr_rf` without `--project=@iai`.
  - The full pipeline was re-run here on 2026-11-18 and reproduces Daniel's outputs (task 6).
- **Gurobi + JuMP:** installed for plain `julia +1.11.6` (default v1.11 environment). Academic license valid to 2027-08-24; a test MIP solved.
- **Python:** the system `python3` has no pandas. Use `~/opt/anaconda3/bin/python3` (pandas 1.4.4, scikit-learn 1.0.2, numpy 1.21.5).

**Daniel's machine (from the notes):** Windows, IAI license at `C:\Users\drkru\iai.lic`, run with `--project=@iai`.

## Guardrails

- The scientific question is **policy value and resource allocation**, not prediction accuracy.
- Keep **all three arms** in every pipeline and table, in the order regular, small, aide. Use explicit label-to-column mappings and alignment assertions.
- Keep school IDs and school-level splits; never leak a held-out school's outcomes into nuisance models, tuning or policy selection.
- Use only pre-assignment features. No teacher variables, no later-year class types.
- Distinguish **estimated reward**, **observed score**, **predicted outcome** and **unobservable individual effect**.
- Be cautious about attrition, class switching after kindergarten, and the grade-3 outcome.
- Keep literature claims separate from our findings. Krueger reported larger small-class gains for Black and free-lunch students; in our kindergarten data the gain is the same by free lunch.
- Single-fold results are noisy (fold spread 0.03–0.07 SD). Don't interpret a difference without an uncertainty estimate.
- Examine splits on gender or ethnicity carefully before reporting them. The frontier trees' gender split is backed by a real gap in the data (notes 12f), but a policy that targets small classes by gender is a fairness problem; report it alongside the run without sensitive features. Don't cite literature on gender differences in class-size effects without checking it.
- Only describe results that exist in results files (Current status lists which runs exist). Files with `_plant_` in the name are SYNTHETIC; never report them as STAR results.
- Do not silently substitute a non-OPT method for OPT. For exact MIPs, report solver status, time and gap; don't call an infeasible or approximate result optimal.
- Persist seeds, splits, model parameters, propensity assumptions, capacities, costs and results.
- Don't overwrite existing data, results or code without checking them first. Record changes in `project_star_notes.txt`.
- Develop incrementally with small synthetic tests. Label synthetic results clearly; never report them as STAR results.

## Deadlines

From the course's project information (supersedes the dates in Lecture 1, which said 12/08 and 12/14):
- Presentation slides on Gradescope: **Mon 1 Dec 2026, 11:59 AM (noon)**.
- 8-page report on Gradescope: **Mon 1 Dec 2026, 11:59 PM**.
- Presentations in class: **Tue 2 Dec 2026**, 8 minutes per team; attendance compulsory.

## Still open

- Report framing, to confirm together: the policy without gender/ethnicity as the main policy, and the gender result as a finding about who benefits, with its fairness cost.
- Whether the λ sweep stays in the report (currently one sentence).

## Settled (previously open)

- Capacity scenarios: the fixed-share frontier (q = 0–1) and c × observed STAR seats, c = 0.5–1.5. Both are assumed scenarios.
- Seat filling: option (b), small seats first by leaf priority (`opt_frontier.jl`, `opt_capacity.jl`); option (c), exact enumeration (`opt_exact.jl`).
- Locked test set: none. Five-fold school cross-validation only, stated as a limitation.
- Depth: IAI 1–3 by GridSearch; the exact enumeration covers depth ≤ 2.
- Age bands for the subgroup allocation: < 5.165, 5.165–5.415, ≥ 5.415 years.
- Not pursued: cost ratios beyond λ, a fairness objective beyond subgroup reporting, a causal forest diagnostic.

## Report organization

Target **8 pages** (course project page; the exemplar report put references and contributions after the 8 pages). The team has an Overleaf outline.
1. **Introduction:** STAR motivation, research question, contribution.
2. **Data:** randomization, class types, outcome/feature choices, cleaning, EDA.
3. **Methods:** DR/DM rewards, OPT, capacity constraints, subgroup-level allocation, optional cost/equity variants.
4. **Experiments and Results:** held-out-school setup, baselines, budget/capacity trade-offs, depth, tree illustrations, subgroup analysis.
5. **Discussion:** interpretation and limitations (attrition, class switching, noisy heterogeneity, uncertainty, costs, class-level feasibility, fairness).
6. **Conclusion.**
7. **References** and **Contributions**.

## References (from the proposal)

1. Amram, M., Dunn, J., & Zhuo, Y. D. (2022). "Optimal Policy Trees." *Machine Learning*, 111(7), 2741–2768. https://doi.org/10.1007/s10994-022-06128-5
2. Krueger, A. B. (1999). "Experimental Estimates of Education Production Functions." *The Quarterly Journal of Economics*, 114(2), 497–532.
3. Nye, B., Hedges, L. V., & Konstantopoulos, S. (2000). "The Effects of Small Classes on Academic Achievement: The Results of the Tennessee Class Size Experiment." *American Educational Research Journal*, 37(1), 123–151.
4. Whitehurst, G. J., & Chingos, M. M. (2011). *Class Size: What Research Says and What It Means for State Policy.* Brown Center on Education Policy, Brookings Institution.

Nye et al. studied early grades; it does not by itself show that effects are stronger in younger than older children.

## How Claude Code should collaborate

1. **Inspect the repo first** and state what exists. Don't assume the project is further along than it is.
2. **Before substantial edits, state a short plan** and flag any open question that affects validity.
3. **Write readable, modular, reproducible code** following the existing conventions (Python for data and rewards, Julia + IAI for trees), with documented inputs/outputs, assertions and small smoke tests.
4. **Never invent findings, fields, fitted results, costs or estimates.**
5. **Show meaningful diagnostics:** sample counts, reward-matrix dimensions, treatment balance, propensities, feasibility, solver status, training and held-out values.
6. **Optimize for a defensible course project:** an interpretable OPT, sensible constrained allocations, a fair held-out comparison against baselines, and honest limitations.
