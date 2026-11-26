# Budget frontier with a fixed share of small places (task 9), filled school by school.
#
# Run:    julia +1.11.6/IAI/3.2.2 --project=@iai opt_frontier.jl [tag] [eval_tag]
#         tag (default dr_rf): rewards the tree is trained on
#         eval_tag (default = tag): rewards used to score the held-out schools
# Input:  project_star_analysis.csv, rewards_<tag>.csv [, rewards_<eval_tag>.csv]
# Output: frontier_<tag>[__eval_<eval_tag>].csv   one row per held-out school x share q x
#                                                 policy: n, n_small, sum of rewards
#         trees_frontier_<tag>/tree_fold<k>.txt   each fold's tree and its leaf gains
# Summary and school-level bootstrap intervals: python frontier_summary.py <file>
#
# Per held-out fold k:
#   1. Fit one OPT on the training rewards for regular vs small only (aide's estimated
#      effect is ~0, and the frontier is about small seats). The small reward is lowered by
#      the mean training gain, so every split must separate students who gain more than
#      average from those who gain less. With no cost the tree usually doesn't split.
#   2. Each leaf's estimated gain = mean training (G_small - G_regular) in the leaf.
#   3. For each share q and each test school s, give small places to round(q * n_s)
#      students, highest leaf gain first (random tie-break); everyone else regular.
#   Free-lunch-first and random get exactly the same seat count in every school, so all
#   three policies use the same per-school budget. Random is scored by its exact
#   expected value.

using CSV, DataFrames, CategoricalArrays, Random, Statistics

const HERE = @__DIR__
const FEATURES = [:female, :age_at_entry, :free_lunch, :ethnicity_grp, :locale,
                  :age_at_entry_missing, :free_lunch_missing]
const SHARES = 0.0:0.1:1.0
const SEED = 15
tag = isempty(ARGS) ? "dr_rf" : ARGS[1]
eval_tag = length(ARGS) >= 2 ? ARGS[2] : tag

df = CSV.read(joinpath(HERE, "project_star_analysis.csv"), DataFrame)
for c in (:ethnicity_grp, :locale)
    df[!, c] = categorical(df[!, c])
end
rw = CSV.read(joinpath(HERE, "rewards_$tag.csv"), DataFrame)
rw_eval = eval_tag == tag ? rw : CSV.read(joinpath(HERE, "rewards_$eval_tag.csv"), DataFrame)
treedir = mkpath(joinpath(HERE, "trees_frontier_$tag"))
rng = MersenneTwister(SEED)

# Students of one school in priority order (higher key first, random tie-break)
priority(key, tiebreak) = sortperm(collect(zip(-key, tiebreak)))

results = []
for k in sort(unique(rw.test_fold))
    sub = rw[rw.test_fold .== k, :]
    tr = innerjoin(sub[sub.role .== "train", :], df, on=:student_id)
    te = innerjoin(rw_eval[(rw_eval.test_fold .== k) .& (rw_eval.role .== "test"), :], df,
                   on=:student_id)
    @assert isempty(intersect(tr.school_id, te.school_id)) "test school leaked into training"

    gain_tr = tr.G_small .- tr.G_regular
    lam = mean(gain_tr)
    rewards = DataFrame(regular=tr.G_regular, small=tr.G_small .- lam)
    grid = IAI.GridSearch(IAI.OptimalTreePolicyMaximizer(random_seed=SEED, minbucket=50),
                          max_depth=1:3)
    IAI.fit!(grid, tr[:, FEATURES], rewards)
    lnr = IAI.get_learner(grid)

    leaf_tr = IAI.apply(lnr, tr[:, FEATURES])
    leaf_gain = Dict(l => mean(gain_tr[leaf_tr .== l]) for l in unique(leaf_tr))
    score = [leaf_gain[l] for l in IAI.apply(lnr, te[:, FEATURES])]
    open(joinpath(treedir, "tree_fold$(k).txt"), "w") do io
        println(io, lnr)
        println(io, "cost lambda = mean training gain = ", round(lam, digits=4))
        for l in sort(collect(keys(leaf_gain)))
            println(io, "leaf $l: n_train = $(count(==(l), leaf_tr)), ",
                    "mean training gain = $(round(leaf_gain[l], digits=4))")
        end
    end

    tiebreak = rand(rng, nrow(te))
    for s in sort(unique(te.school_id))
        idx = findall(==(s), te.school_id)
        n = length(idx)
        Gr, Gs = te.G_regular[idx], te.G_small[idx]
        order = Dict("opt" => priority(score[idx], tiebreak[idx]),
                     "free_lunch_first" => priority(te.free_lunch[idx], tiebreak[idx]))
        for q in SHARES
            n_small = round(Int, q * n)
            for (policy, ord) in order
                small = falses(n)
                small[ord[1:n_small]] .= true
                push!(results, (fold=k, school_id=s, q=q, policy=policy, n=n, n_small=n_small,
                                sum_value=sum(ifelse.(small, Gs, Gr))))
            end
            push!(results, (fold=k, school_id=s, q=q, policy="random", n=n, n_small=n_small,
                            sum_value=(1 - n_small / n) * sum(Gr) + n_small / n * sum(Gs)))
        end
    end
    println("fold $k done: depth $(IAI.get_best_params(grid)[:max_depth]), ",
            "$(length(leaf_gain)) leaves, lambda $(round(lam, digits=3))")
end

res = DataFrame(results)
out = eval_tag == tag ? "frontier_$tag.csv" : "frontier_$(tag)__eval_$(eval_tag).csv"
CSV.write(joinpath(HERE, out), res)
println("Wrote $out ($(nrow(res)) rows)")
