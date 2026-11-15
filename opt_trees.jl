# Stage 2 of the OPT pipeline: Optimal Policy Trees (IAI) + held-out school evaluation.
#
# Run:    julia +1.11.6/IAI/3.2.2 --project=@iai opt_trees.jl [tag]     (tag default: dr_rf)
# Input:  project_star_analysis.csv, rewards_<tag>.csv (from opt_rewards.py), same folder
# Output: results_<tag>.csv    held-out value of every policy, per fold and cost penalty
#         trees_<tag>/         fold-0 trees as HTML, one per cost penalty
#
# IAI is preloaded by the system image, so it is used as `IAI.` without `using IAI`.
# Budget: a cost lambda (SD units per student) is subtracted from the small and aide rewards
# before fitting. Sweeping lambda traces the budget frontier; each OPT policy is compared with
# random allocation and a free-lunch-first heuristic that use the same number of small seats.

using CSV, DataFrames, CategoricalArrays, Random, Statistics

const HERE = @__DIR__
const ARMS = ["regular", "small", "aide"]
const FEATURES = [:female, :age_at_entry, :free_lunch, :ethnicity_grp, :locale,
                  :age_at_entry_missing, :free_lunch_missing]
const LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
const SEED = 15
tag = isempty(ARGS) ? "dr_rf" : ARGS[1]

df = CSV.read(joinpath(HERE, "project_star_analysis.csv"), DataFrame)
for c in (:ethnicity_grp, :locale)
    df[!, c] = categorical(df[!, c])
end
rw = CSV.read(joinpath(HERE, "rewards_$tag.csv"), DataFrame)
gcols = ["G_" * a for a in ARMS]
treedir = mkpath(joinpath(HERE, "trees_$tag"))
rng = MersenneTwister(SEED)

value(G, assign) = mean(G[i, assign[i]] for i in eachindex(assign))
random_value(G, share) = sum(vec(mean(G, dims=1)) .* share)
row(fold, policy, lam, depth, share, v) =
    (fold=fold, policy=policy, lam=lam, depth=depth,
     share_small=share[2], share_aide=share[3], value=v)

# Small seats to free-lunch students first (random tie-break), everyone else regular
function free_lunch_first(free_lunch, n_small)
    order = sortperm(collect(zip(-free_lunch, rand(rng, length(free_lunch)))))
    assign = fill(1, length(free_lunch))
    assign[order[1:n_small]] .= 2
    return assign
end

results = []
for k in sort(unique(rw.test_fold))
    sub = rw[rw.test_fold .== k, :]
    tr = innerjoin(sub[sub.role .== "train", :], df, on=:student_id)
    te = innerjoin(sub[sub.role .== "test", :], df, on=:student_id)
    Gtr, Gte, n = Matrix(tr[:, gcols]), Matrix(te[:, gcols]), nrow(te)

    for (j, a) in enumerate(ARMS)
        push!(results, row(k, "all_$a", NaN, 0, Float64.(1:3 .== j), value(Gte, fill(j, n))))
    end
    for lam in LAMBDAS
        rewards = DataFrame(Gtr .- [0.0 lam lam], ARMS)
        grid = IAI.GridSearch(IAI.OptimalTreePolicyMaximizer(random_seed=SEED, minbucket=50),
                              max_depth=1:3)
        IAI.fit!(grid, tr[:, FEATURES], rewards)
        lnr = IAI.get_learner(grid)
        assign = [findfirst(==(a), ARMS) for a in IAI.predict(lnr, te[:, FEATURES])]
        share = [mean(assign .== j) for j in 1:3]
        depth = IAI.get_best_params(grid)[:max_depth]
        n_small = count(==(2), assign)
        push!(results, row(k, "opt", lam, depth, share, value(Gte, assign)))
        push!(results, row(k, "random_same_budget", lam, depth, share, random_value(Gte, share)))
        push!(results, row(k, "free_lunch_first", lam, depth, [1 - n_small / n, n_small / n, 0.0],
                           value(Gte, free_lunch_first(te.free_lunch, n_small))))
        k == 0 && IAI.write_html(joinpath(treedir, "tree_fold0_lam$(lam).html"), lnr)
    end
    println("fold $k done")
end

res = DataFrame(results)
CSV.write(joinpath(HERE, "results_$tag.csv"), res)
summary = combine(groupby(res, [:policy, :lam]),
                  [:share_small, :share_aide, :value, :depth] .=> mean .=> [:share_small, :share_aide, :value, :depth])
sort!(summary, [:lam, :policy])
show(transform(summary, names(summary, Real) .=> (x -> round.(x, digits=3)), renamecols=false),
     allrows=true)
println()
