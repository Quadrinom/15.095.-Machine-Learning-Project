# Per-school capacity constraints (task 10; equation 3, option (b) of notes section 6).
#
# Run:    julia +1.11.6/IAI/3.2.2 --project=@iai opt_capacity.jl [tag] [eval_tag] [--drop=f1,f2]
#         tag (default dr_rf): rewards the tree is trained on
#         eval_tag (default = tag): rewards used to score the held-out schools
#         --drop: features the tree may not use (task 14); adds __drop-f1-f2 to output names
# Input:  project_star_analysis.csv, project_star_school_capacity.csv, rewards_<tag>.csv
#         [, rewards_<eval_tag>.csv]
# Output: capacity_<name>.csv           one row per held-out school x scale x policy: n,
#                                       seat caps, seats used, sum of held-out rewards
#         capacity_assign_<name>.csv    one row per held-out student x scale x policy
#                                       (opt, free_lunch_first): the assigned arm
#         capacity_students_<name>.csv  one row per held-out student: leaf, leaf gains, the
#                                       tree's own prescription, test rewards
#         trees_capacity_<tag>[__drop-..]/tree_fold<k>.txt   tree and leaf gains
#         (<name> = tag, or <tag>__eval_<eval_tag>, plus the __drop suffix)
# Summary with school-level bootstrap intervals: python capacity_summary.py
#
# Capacities (ASSUMED scenarios, not real budgets): school s has B_s^small =
# round(c * observed small seats) and B_s^aide = min(round(c * observed aide seats),
# n_s - B_s^small), for c in SCALES. Observed seats are the STAR kindergarten counts in
# project_star_school_capacity.csv (the same students, so at c = 1 the school keeps the
# classes it actually had and only who sits in them changes). Everyone else is regular.
#
# Per held-out fold k:
#   1. Fit one three-arm OPT on the training rewards. The small and aide rewards are each
#      lowered by their mean training gain over regular, so splits must separate students
#      who gain more than average from those who gain less (as in opt_frontier.jl).
#   2. Leaf gains = mean training (G_small - G_regular) and (G_aide - G_regular) per leaf.
#   3. Seat-filling rule, the same for every policy, in every held-out school:
#        small seats first, to the B_s^small students with the highest priority;
#        then aide seats, to the B_s^aide remaining students with the highest priority.
#      OPT priority: leaf small gain, then leaf aide gain (random tie-break).
#      Free-lunch-first: free-lunch students first for both (random tie-break).
#      Random: exact expected value (each student small with prob B^small/n_s, aide with
#      prob B^aide/n_s).
#      Every policy fills all its seats, so all use exactly the same seats per school.
#   4. tree_raw: the tree's own prescription, ignoring capacities. Rows record how many
#      small/aide seats it would use, so quota violations can be counted.

using CSV, DataFrames, CategoricalArrays, Random, Statistics

const HERE = @__DIR__
const ALL_FEATURES = [:female, :age_at_entry, :free_lunch, :ethnicity_grp, :locale,
                      :age_at_entry_missing, :free_lunch_missing]
const SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
const SEED = 15
drop_args = filter(a -> startswith(a, "--drop="), ARGS)
pos = filter(a -> !startswith(a, "--"), ARGS)
@assert length(drop_args) + length(pos) == length(ARGS) "unknown option in $(ARGS)"
tag = isempty(pos) ? "dr_rf" : pos[1]
eval_tag = length(pos) >= 2 ? pos[2] : tag
drop = isempty(drop_args) ? Symbol[] :
       Symbol.(split(split(drop_args[end], "=")[2], ","))
@assert all(in(ALL_FEATURES), drop) "unknown feature in --drop: $(drop)"
const FEATURES = setdiff(ALL_FEATURES, drop)
suffix = isempty(drop) ? "" : "__drop-" * join(string.(drop), "-")
name = (eval_tag == tag ? tag : "$(tag)__eval_$(eval_tag)") * suffix
println("features: ", join(string.(FEATURES), ", "))

df = CSV.read(joinpath(HERE, "project_star_analysis.csv"), DataFrame)
for c in (:ethnicity_grp, :locale)
    df[!, c] = categorical(df[!, c])
end
cap = CSV.read(joinpath(HERE, "project_star_school_capacity.csv"), DataFrame)
cap_of = Dict(r.school_id => r for r in eachrow(cap))
rw = CSV.read(joinpath(HERE, "rewards_$tag.csv"), DataFrame)
rw_eval = eval_tag == tag ? rw : CSV.read(joinpath(HERE, "rewards_$eval_tag.csv"), DataFrame)
treedir = mkpath(joinpath(HERE, "trees_capacity_$tag$suffix"))
rng = MersenneTwister(SEED)

# Students in priority order: higher key first, then higher key2, then random tie-break
priority(key, key2, tiebreak) = sortperm(collect(zip(-key, -key2, tiebreak)))

"Arms (\"regular\"/\"small\"/\"aide\") for one school: small seats along ord_small, then
aide seats along ord_aide among students not already small."
function fill_seats(ord_small, ord_aide, n_small, n_aide)
    arm = fill("regular", length(ord_small))
    arm[ord_small[1:n_small]] .= "small"
    rest = filter(i -> arm[i] == "regular", ord_aide)
    arm[rest[1:n_aide]] .= "aide"
    return arm
end

value(arm, Gr, Gs, Ga) = sum(ifelse.(arm .== "small", Gs, ifelse.(arm .== "aide", Ga, Gr)))

results, assigns, students = [], [], []
for k in sort(unique(rw.test_fold))
    sub = rw[rw.test_fold .== k, :]
    tr = innerjoin(sub[sub.role .== "train", :], df, on=:student_id)
    te = innerjoin(rw_eval[(rw_eval.test_fold .== k) .& (rw_eval.role .== "test"), :], df,
                   on=:student_id)
    @assert isempty(intersect(tr.school_id, te.school_id)) "test school leaked into training"

    gs_tr, ga_tr = tr.G_small .- tr.G_regular, tr.G_aide .- tr.G_regular
    lam_s, lam_a = mean(gs_tr), mean(ga_tr)
    rewards = DataFrame(regular=tr.G_regular, small=tr.G_small .- lam_s,
                        aide=tr.G_aide .- lam_a)
    grid = IAI.GridSearch(IAI.OptimalTreePolicyMaximizer(random_seed=SEED, minbucket=50),
                          max_depth=1:3)
    IAI.fit!(grid, tr[:, FEATURES], rewards)
    lnr = IAI.get_learner(grid)

    leaf_tr = IAI.apply(lnr, tr[:, FEATURES])
    gain_s = Dict(l => mean(gs_tr[leaf_tr .== l]) for l in unique(leaf_tr))
    gain_a = Dict(l => mean(ga_tr[leaf_tr .== l]) for l in unique(leaf_tr))
    leaf_te = IAI.apply(lnr, te[:, FEATURES])
    sc_s = [gain_s[l] for l in leaf_te]
    sc_a = [gain_a[l] for l in leaf_te]
    raw = string.(IAI.predict(lnr, te[:, FEATURES]))
    open(joinpath(treedir, "tree_fold$(k).txt"), "w") do io
        println(io, lnr)
        println(io, "cost lambda: small = ", round(lam_s, digits=4),
                ", aide = ", round(lam_a, digits=4), " (mean training gains over regular)")
        for l in sort(collect(keys(gain_s)))
            println(io, "leaf $l: n_train = $(count(==(l), leaf_tr)), ",
                    "mean training gain small = $(round(gain_s[l], digits=4)), ",
                    "aide = $(round(gain_a[l], digits=4))")
        end
    end

    tiebreak = rand(rng, nrow(te))
    for s in sort(unique(te.school_id))
        idx = findall(==(s), te.school_id)
        n = length(idx)
        c = cap_of[s]
        @assert c.n_total == n "school $s: capacity file has $(c.n_total) students, data $n"
        Gr, Gs, Ga = te.G_regular[idx], te.G_small[idx], te.G_aide[idx]
        fl, tb = te.free_lunch[idx], tiebreak[idx]
        orders = Dict(
            "opt" => (priority(sc_s[idx], sc_a[idx], tb), priority(sc_a[idx], sc_s[idx], tb)),
            "free_lunch_first" => (priority(fl, fl, tb), priority(fl, fl, tb)))
        for (j, i) in enumerate(idx)
            push!(students, (fold=k, school_id=s, student_id=te.student_id[i], leaf=leaf_te[i],
                             gain_small=sc_s[i], gain_aide=sc_a[i], tree_arm=raw[i],
                             G_regular=Gr[j], G_small=Gs[j], G_aide=Ga[j]))
        end
        raw_s = raw[idx]
        for scale in SCALES
            b_small = min(round(Int, scale * c.n_small), n)
            b_aide = min(round(Int, scale * c.n_aide), n - b_small)
            row(policy, n_small, n_aide, v) = (fold=k, school_id=s, scale=scale, policy=policy,
                n=n, cap_small=b_small, cap_aide=b_aide, n_small=n_small, n_aide=n_aide,
                sum_value=v)
            for (policy, (ord_s, ord_a)) in orders
                arm = fill_seats(ord_s, ord_a, b_small, b_aide)
                @assert count(==("small"), arm) == b_small && count(==("aide"), arm) == b_aide
                push!(results, row(policy, b_small, b_aide, value(arm, Gr, Gs, Ga)))
                for (j, i) in enumerate(idx)
                    push!(assigns, (fold=k, school_id=s, student_id=te.student_id[i],
                                    scale=scale, policy=policy, arm=arm[j]))
                end
            end
            ps, pa = b_small / n, b_aide / n
            push!(results, row("random", b_small, b_aide,
                               sum((1 - ps - pa) .* Gr .+ ps .* Gs .+ pa .* Ga)))
            push!(results, row("tree_raw", count(==("small"), raw_s), count(==("aide"), raw_s),
                               value(raw_s, Gr, Gs, Ga)))
        end
    end
    println("fold $k done: depth $(IAI.get_best_params(grid)[:max_depth]), ",
            "$(length(gain_s)) leaves, lambda small $(round(lam_s, digits=3)), ",
            "aide $(round(lam_a, digits=3))")
end

CSV.write(joinpath(HERE, "capacity_$name.csv"), DataFrame(results))
CSV.write(joinpath(HERE, "capacity_assign_$name.csv"), DataFrame(assigns))
CSV.write(joinpath(HERE, "capacity_students_$name.csv"), DataFrame(students))
println("Wrote capacity_$name.csv, capacity_assign_$name.csv, capacity_students_$name.csv")
