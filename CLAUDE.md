# CLAUDE.md — Project STAR Classroom Allocation Using Optimal Policy Trees

## Purpose of this file

Context for Claude Code on an MIT 15.095 (*Machine Learning Under a Modern Optimization Lens*) final project by **Daniel Krupp** and **Adi Palaniappan**. Either of them may be the user in a session. This file holds the plan, the decisions already made, the current status, the task list and the open questions. Detailed implementation notes and results are in `project_star_notes.txt`; read it before changing code.

Inspect the actual files before modifying anything. Never present an assumption as an observed fact. Last updated **2026-11-18**.

## Project in one paragraph

We learn an **interpretable, resource-constrained policy** that assigns elementary-school students to one of three classroom types using the Tennessee **Project STAR** randomized experiment: **regular class**, **small class**, and **regular class with a full-time teacher's aide**. The core method is an **Optimal Policy Tree (OPT)** trained on estimated counterfactual rewards for each student and class type. The main extension is **school-level capacity constraints** (small-class seats and aide-supported seats), possibly with class-type costs. Policies are evaluated on **held-out schools** against simple allocation rules, with subgroup/equity checks.

## Research question

> Can observable student characteristics guide assignments among regular classes, small classes, and regular classes with an aide to improve overall expected academic achievement under school-level resource constraints, while yielding a policy that administrators can understand?

Do not presuppose that small classes are best for every student, or that the aide arm is useful. "Targeting does not beat random allocation" is a valid finding if the data show it.

## Current status (as of 2026-11-18)

A first end-to-end pipeline exists (commit `07283cd`, 15 Nov 2026): data download → cleaning → cross-fitted rewards → policy trees → held-out evaluation. See `project_star_notes.txt` sections 1–9.

- **Only one configuration has been run in full:** DR rewards with random-forest outcome models (`rewards_dr_rf.csv`, `results_dr_rf.csv`, fold-0 trees in `trees_dr_rf/`). Placebo rewards (`rewards_dr_linear_placebo.csv`) exist, but the tree stage has not been run on them.
- **First results** (held-out DR value in SD units, mean over 5 folds): OPT does **not** beat random allocation or free-lunch-first at any cost level. Differences range from −0.02 to +0.01 SD, while results vary by 0.03–0.07 SD between folds. All-small 0.199, all-aide −0.002, all-regular −0.016.
- **The fold-0 trees look like noise.** The split feature changes at every cost level (age ≥ 6.165 → regular at λ=0.05; suburban → regular at λ=0.1 and 0.15; girls → regular, boys → small at λ=0.2), and no tree uses free lunch. Decoded in notes section 10b.
- **Our own kindergarten data:** small classes +0.18 SD, aides +0.01 SD. The small-class gain is the same with and without free lunch.
- **Per-school capacity constraints (the main extension) are not implemented.** Only a global cost penalty exists.
- **The teaching staff asked for a subgroup-level allocation comparison** (see below). Not started.

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
| Rewards code | `opt_rewards.py` (Python, scikit-learn): `--method dr|dm|ipw`, `--model rf|linear`, `--placebo` |
| Trees code | `opt_trees.jl` (Julia + IAI 3.2.2 `OptimalTreePolicyMaximizer`): GridSearch over `max_depth` 1–3, `minbucket` 50 |
| Budget (current) | Global cost penalty λ ∈ {0, .05, .1, .15, .2, .3} subtracted from small and aide training rewards |

## Task list

Not split by person. Status as of 2026-11-18.

**Done (first versions):**
1. Data preparation and cleaning.
2. Counterfactual rewards: DR with random forests. DM, IPW and linear models are coded but not run.
3. Unconstrained OPT (IAI, depth 1–3).
4. Baselines: all-regular, all-small, all-aide, random, free-lunch-first, with gaps (see Baselines below).
5. Held-out DR evaluation for the DR-RF configuration.

**To do (rough priority order):**
6. Re-run the existing pipeline on Adi's Mac and confirm the results match (see Environment).
7. Placebo check: run the trees on placebo rewards. OPT's gain over random should be about zero.
8. Semi-synthetic check: plant a known effect (e.g. small classes help free-lunch students +0.3 SD more) and check that the tree finds it.
9. Budget frontier with a fixed share of small places instead of the λ sweep (the λ sweep does not control the budget smoothly).
10. Per-school capacity constraints (main extension). See Capacity constraints.
11. Subgroup-level allocation baseline (staff feedback).
12. Fix the baseline gaps: aide seats in the heuristic, per-school matching.
13. Uncertainty: school-level bootstrap for policy values and for differences between policies.
14. Subgroup values (free lunch, ethnicity, gender) and a run without sensitive features.
15. Robustness: DM vs DR, linear vs RF, propensity choice, depth, tree stability, other outcomes and cohort.
16. EDA figures and data summary table.
17. Report and presentation.

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

- **Current implementation:** only the global cost penalty λ. That is the *optional* per-student cost model, not equation 3. It controls the budget on average, not per school, and nothing checks per-school feasibility.
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
  - Its environment (`Julia_v1_11_6__IAI_v3_2_2`) has CSV and DataFrames but **not CategoricalArrays**, which `opt_trees.jl` imports. Fix once with `julia +1.11.6/IAI/3.2.2 -e 'import Pkg; Pkg.add("CategoricalArrays")'`.
  - There is no `@iai` environment here, so run `julia +1.11.6/IAI/3.2.2 opt_trees.jl dr_rf` without `--project=@iai`.
  - The pipeline has **not** yet been re-run on this machine.
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
- Examine splits on gender or ethnicity carefully before reporting them.
- Only describe results that exist in results files. Results exist only for DR-RF; don't describe other configurations as run.
- Do not silently substitute a non-OPT method for OPT. For exact MIPs, report solver status, time and gap; don't call an infeasible or approximate result optimal.
- Persist seeds, splits, model parameters, propensity assumptions, capacities, costs and results.
- Don't overwrite existing data, results or code without checking them first. Record changes in `project_star_notes.txt`.
- Develop incrementally with small synthetic tests. Label synthetic results clearly; never report them as STAR results.

## Still open

- Capacity scenarios (e.g. 50/100/150% of observed seats).
- The seat-filling rule that combines a tree with per-school quotas on new schools (option b vs c).
- Locked test set, or 5-fold cross-validation only.
- Depth range: the code uses 1–3, notes experiment 3 says 1–4, the original plan said 2–4.
- Age bands for the subgroup allocation, and a minimum group size.
- Cost ratios, if a cost model is kept beyond λ.
- Fairness objective beyond subgroup reporting.
- Whether to use a causal forest as a heterogeneity diagnostic.
- Deadlines and report length: verify against the current course announcement.

## Report organization

Target about **8 pages** (verify current instructions). The team has an Overleaf outline.
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
