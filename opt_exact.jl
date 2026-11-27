# Exact optimal capacity-constrained policy (task 19; equation 3 inside the fit).
#
# Run:    julia +1.11.6/IAI/3.2.2 opt_exact.jl [tag] [--drop=f1,f2] [--no-iai]
#         (on Windows add --project=@iai, as for the other Julia scripts). Run it
#         single-threaded: with -t 10 under the IAI system image it crashed in the garbage
#         collector (2026-11-27); single-threaded it takes about 40 s.
#         tag (default dr_rf): rewards used to fit AND to score held-out schools
#         --drop: features the tree may not use; adds __drop-f1-f2 to output names
#         --no-iai: skip refitting Daniel's IAI tree (used for the placebo runs)
# Input:  project_star_analysis.csv, project_star_school_capacity.csv, rewards_<tag>.csv
# Output: exact_capacity_<name>.csv  one row per held-out school x scale x policy, same
#                                    columns as capacity_*.csv (policies: exact, random,
#                                    and opt_expected = the IAI policy of opt_capacity.jl)
#         exact_insample_<name>.csv  per fold x scale: training objective of each policy
#         trees_exact_<name>/fold<k>.txt   chosen tree, leaf orders, training values
# Summary: python exact_summary.py
#
# Policy class. A policy is (i) a tree of depth <= 2 on the features, every leaf holding at
# least MINBUCKET training students (as IAI's minbucket), and (ii) two leaf orders: small
# seats go to leaves in order sigma, then aide seats to the remaining students in order
# tau. In every school the seats are filled exactly as in opt_capacity.jl (small first,
# then aide, every seat used, capacities B_s = c x observed STAR seats); within a leaf the
# tie-break is random, so we use the exact EXPECTED value (as for random allocation).
# Candidate splits are every split IAI could make on these features: female, free_lunch,
# age_at_entry < t for every midpoint between consecutive observed ages, and every
# two-way split of the levels of ethnicity_grp and of locale. The missing flags are left
# out: a split on them would leave a leaf below MINBUCKET.
#
# Exactness. For each held-out fold we enumerate EVERY tree in the class and EVERY pair of
# leaf orders, and score each on the training schools with their own capacities. The best
# is the certified optimum of the training objective
#     sum_s sum_{i in S_s} Gamma_hat[i, pi_s(i)]  subject to equation (3) in every school,
# over this policy class. No solver is needed: the class is small enough to enumerate.
# IAI's policy (opt_capacity.jl: IAI tree fit to mean-shifted rewards, leaves ordered by
# mean training gain) belongs to the same class whenever its tree has depth <= 2 (it does
# in all five folds), so exact >= IAI on the training objective by construction (asserted).

using CSV, DataFrames, CategoricalArrays, Random, Statistics

permutations(v) = length(v) <= 1 ? [collect(v)] :
    [vcat(v[i], p) for i in eachindex(v) for p in permutations(v[[1:i-1; i+1:end]])]

const HERE = @__DIR__
const ALL_FEATURES = [:female, :age_at_entry, :free_lunch, :ethnicity_grp, :locale,
                      :age_at_entry_missing, :free_lunch_missing]
const SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]
const MINBUCKET = 50
const SEED = 15
const TOL = 1e-9

drop_args = filter(a -> startswith(a, "--drop="), ARGS)
use_iai = !("--no-iai" in ARGS)
pos = filter(a -> !startswith(a, "--"), ARGS)
tag = isempty(pos) ? "dr_rf" : pos[1]
drop = isempty(drop_args) ? Symbol[] :
       Symbol.(split(split(drop_args[end], "=")[2], ","))
@assert all(in(ALL_FEATURES), drop) "unknown feature in --drop: $(drop)"
const FEATURES = setdiff(ALL_FEATURES, drop)
suffix = isempty(drop) ? "" : "__drop-" * join(string.(drop), "-")
name = tag * suffix
println("tag $tag, features: ", join(string.(FEATURES), ", "), ", threads: ", Threads.nthreads())

df = CSV.read(joinpath(HERE, "project_star_analysis.csv"), DataFrame)
for c in (:ethnicity_grp, :locale)
    df[!, c] = categorical(df[!, c])
end
cap = CSV.read(joinpath(HERE, "project_star_school_capacity.csv"), DataFrame)
cap_of = Dict(r.school_id => r for r in eachrow(cap))
rw = CSV.read(joinpath(HERE, "rewards_$tag.csv"), DataFrame)
treedir = mkpath(joinpath(HERE, "trees_exact_$name"))

# ---------------------------------------------------------------- candidate splits
struct Split
    label::String
    test::Function   # DataFrame -> Vector{Bool} ("goes to the first child")
end

function candidate_splits()
    sp = Split[]
    :female in FEATURES && push!(sp, Split("female = 1", d -> d.female .== 1))
    :free_lunch in FEATURES && push!(sp, Split("free_lunch = 1", d -> d.free_lunch .== 1))
    if :age_at_entry in FEATURES
        ages = sort(unique(df.age_at_entry))
        for t in (ages[1:end-1] .+ ages[2:end]) ./ 2
            push!(sp, Split("age_at_entry < $(round(t, digits=3))", d -> d.age_at_entry .< t))
        end
    end
    for col in (:ethnicity_grp, :locale)
        col in FEATURES || continue
        lv = sort(unique(string.(df[!, col])))
        n = length(lv)
        for mask in 1:(2^(n - 1) - 1)          # subsets of the first n-1 levels: every
            sub = lv[1:n-1][[(mask >> (i - 1)) & 1 == 1 for i in 1:n-1]]  # each split once
            push!(sp, Split("$col in [$(join(sub, ","))]", d -> in.(string.(d[!, col]), Ref(sub))))
        end
    end
    return sp
end
const SPLITS = candidate_splits()
println("$(length(SPLITS)) candidate splits")

# A tree is (r, a, b): root split r; first child split a (0 = leaf); second child b.
# Raw cell ids: 1 = r & a, 2 = r & !a, 3 = !r & b, 4 = !r & !b (a = 0: cell 1; b = 0: cell 3)
cells(X, (r, a, b)) = [X[i, r] ? (a == 0 || X[i, a] ? 1 : 2) : (b == 0 || X[i, b] ? 3 : 4)
                       for i in 1:size(X, 1)]
cells(X, ::Nothing) = ones(Int, size(X, 1))
describe_tree(::Nothing) = "no split (one leaf)"
function describe_tree((r, a, b))
    s = "1) $(SPLITS[r].label)?\n   yes: " * (a == 0 ? "leaf 1" : "$(SPLITS[a].label)? yes: leaf 1, no: leaf 2")
    s * "\n   no:  " * (b == 0 ? "leaf 3" : "$(SPLITS[b].label)? yes: leaf 3, no: leaf 4")
end
bits(d) = reduce(hcat, [Bool.(s.test(d)) for s in SPLITS])

const PERMS = Dict(L => collect(permutations(1:L)) for L in 1:4)

# ---------------------------------------------------------------- expected seat filling
"Per-school, per-leaf tables: n[s,l], mean gains over regular gs[s,l], ga[s,l]."
function tables(leaf, school, Gd_s, Gd_a, S, L)
    n = zeros(Int, S, L); gs = zeros(S, L); ga = zeros(S, L)
    for i in eachindex(leaf)
        s, l = school[i], leaf[i]
        n[s, l] += 1; gs[s, l] += Gd_s[i]; ga[s, l] += Gd_a[i]
    end
    for j in eachindex(n)
        n[j] > 0 && (gs[j] /= n[j]; ga[j] /= n[j])
    end
    return n, gs, ga
end

"Expected gain over all-regular when small seats go to leaves in order sig, then aide
seats in order tau, with capacities Bs, Ba per school. Also returns seats used."
function fill_value(n, gs, ga, Bs, Ba, sig, tau, small)
    v = 0.0
    for s in axes(n, 1)
        rem = Bs[s]
        for l in sig
            k = min(rem, n[s, l]); small[s, l] = k; rem -= k; v += k * gs[s, l]
        end
        rem = Ba[s]
        for l in tau
            k = min(rem, n[s, l] - small[s, l]); rem -= k; v += k * ga[s, l]
        end
    end
    return v
end

"Best (value, sigma, tau) for one tree on the training schools, for each scale."
function best_orders(n, gs, ga, Bs_all, Ba_all, L)
    small = zeros(Int, size(n))
    best = [(-Inf, Int[], Int[]) for _ in SCALES]
    for (ci, _) in enumerate(SCALES)
        Bs, Ba = view(Bs_all, :, ci), view(Ba_all, :, ci)
        for sig in PERMS[L], tau in PERMS[L]
            v = fill_value(n, gs, ga, Bs, Ba, sig, tau, small)
            v > best[ci][1] + TOL && (best[ci] = (v, sig, tau))
        end
    end
    return best
end

caps(school_ids, scale) = begin
    bs = [min(round(Int, scale * cap_of[s].n_small), cap_of[s].n_total) for s in school_ids]
    ba = [min(round(Int, scale * cap_of[s].n_aide), cap_of[s].n_total - b)
          for (s, b) in zip(school_ids, bs)]
    bs, ba
end

# ---------------------------------------------------------------- main loop
results, insample = [], []
for k in sort(unique(rw.test_fold))
    t0 = time()
    sub = rw[rw.test_fold .== k, :]
    tr = innerjoin(sub[sub.role .== "train", :], df, on=:student_id)
    te = innerjoin(sub[sub.role .== "test", :], df, on=:student_id)
    @assert isempty(intersect(tr.school_id, te.school_id)) "test school leaked into training"

    schools_tr = sort(unique(tr.school_id)); S = length(schools_tr)
    sidx = Dict(s => j for (j, s) in enumerate(schools_tr))
    school = [sidx[s] for s in tr.school_id]
    for s in schools_tr
        @assert cap_of[s].n_total == count(==(s), tr.school_id) "school $s size mismatch"
    end
    Gd_s, Gd_a = tr.G_small .- tr.G_regular, tr.G_aide .- tr.G_regular
    Bs_all = reduce(hcat, [caps(schools_tr, c)[1] for c in SCALES])
    Ba_all = reduce(hcat, [caps(schools_tr, c)[2] for c in SCALES])
    Xtr = bits(tr)
    m, ntr = length(SPLITS), nrow(tr)

    # all trees: depth 0, and (r, a, b) with a, b in {0} U splits other than r
    trees = Any[nothing]
    for r in 1:m, a in 0:m, b in 0:m
        (a == r || b == r) && continue
        push!(trees, (r, a, b))
    end

    # enumerate (threaded); per-thread best per scale, tie-break on tree index
    nt = Threads.nthreads()
    tbest = [[(-Inf, 0, Int[], Int[], 0) for _ in SCALES] for _ in 1:nt]
    nvalid = Threads.Atomic{Int}(0)
    Threads.@threads :static for ti in eachindex(trees)
        tid = Threads.threadid()
        raw = cells(Xtr, trees[ti])
        cnt = [count(==(c), raw) for c in 1:4]
        used = findall(>(0), cnt)
        all(cnt[used] .>= MINBUCKET) || continue
        # skip trees whose leaves duplicate a smaller tree (a child split that does not
        # split): those leaves are empty and already excluded by MINBUCKET above
        Threads.atomic_add!(nvalid, 1)
        L = length(used)
        relabel = zeros(Int, 4); relabel[used] = 1:L
        leaf = relabel[raw]
        n, gs, ga = tables(leaf, school, Gd_s, Gd_a, S, L)
        for (ci, (v, sig, tau)) in enumerate(best_orders(n, gs, ga, Bs_all, Ba_all, L))
            bv, bti = tbest[tid][ci][1], tbest[tid][ci][2]
            if v > bv + TOL || (abs(v - bv) <= TOL && ti < bti)
                tbest[tid][ci] = (v, ti, sig, tau, L)
            end
        end
    end
    best = map(eachindex(SCALES)) do ci
        cands = [tbest[t][ci] for t in 1:nt if tbest[t][ci][2] > 0]
        sort(cands, by=c -> (-round(c[1] / TOL), c[2]))[1]
    end

    # random allocation (exact expected value) on the training schools, per scale
    sum_s = [sum(Gd_s[school .== s]) for s in 1:S]
    sum_a = [sum(Gd_a[school .== s]) for s in 1:S]
    nsch = [count(==(s), school) for s in 1:S]
    rand_tr = [sum(Bs_all[:, ci] ./ nsch .* sum_s .+ Ba_all[:, ci] ./ nsch .* sum_a)
               for ci in eachindex(SCALES)]

    # Daniel's IAI policy (opt_capacity.jl), refit identically, on the same objective
    iai = nothing
    if use_iai
        lam_s, lam_a = mean(Gd_s), mean(Gd_a)
        rewards = DataFrame(regular=tr.G_regular, small=tr.G_small .- lam_s,
                            aide=tr.G_aide .- lam_a)
        grid = IAI.GridSearch(IAI.OptimalTreePolicyMaximizer(random_seed=SEED, minbucket=50),
                              max_depth=1:3)
        IAI.fit!(grid, tr[:, FEATURES], rewards)
        lnr = IAI.get_learner(grid)
        depth = IAI.get_best_params(grid)[:max_depth]
        lt = IAI.apply(lnr, tr[:, FEATURES]); le = IAI.apply(lnr, te[:, FEATURES])
        lv = sort(unique(lt)); L = length(lv); lid = Dict(l => j for (j, l) in enumerate(lv))
        leaf_tr = [lid[l] for l in lt]
        g_s = [mean(Gd_s[leaf_tr .== j]) for j in 1:L]
        g_a = [mean(Gd_a[leaf_tr .== j]) for j in 1:L]
        sig = sortperm(collect(zip(-g_s, -g_a)))     # opt_capacity.jl priorities
        tau = sortperm(collect(zip(-g_a, -g_s)))
        n, gs, ga = tables(leaf_tr, school, Gd_s, Gd_a, S, L)
        small = zeros(Int, size(n))
        iai_tr = [fill_value(n, gs, ga, Bs_all[:, ci], Ba_all[:, ci], sig, tau, small)
                  for ci in eachindex(SCALES)]
        # IAI's partition with its best leaf orders (diagnostic). Only for <= 4 leaves:
        # a depth-3 IAI tree can have up to 8 leaves (8!^2 order pairs).
        iai_best = L <= 4 ? [b[1] for b in best_orders(n, gs, ga, Bs_all, Ba_all, L)] :
                   fill(NaN, length(SCALES))
        if depth <= 2 && L >= 1
            for ci in eachindex(SCALES)
                @assert best[ci][1] >= iai_best[ci] - 1e-6 "exact below IAI at scale $(SCALES[ci])"
            end
        end
        # cross-check against opt_capacity.jl's saved test leaves, when available
        f = joinpath(HERE, "capacity_students_$name.csv")
        if isfile(f)
            saved = CSV.read(f, DataFrame)
            saved = Dict(zip(saved.student_id[saved.fold .== k], saved.leaf[saved.fold .== k]))
            @assert all(saved[i] == l for (i, l) in zip(te.student_id, le)) "IAI refit differs from opt_capacity.jl"
        end
        iai = (depth=depth, L=L, lid=lid, le=le, g_s=g_s, g_a=g_a, sig=sig, tau=tau,
               tr=iai_tr, best=iai_best, txt=string(lnr))
    end

    # held-out schools: apply each policy, fill seats, score with test rewards
    Xte = bits(te)
    for (ci, scale) in enumerate(SCALES)
        v, ti, sig, tau, L = best[ci]
        raw = cells(Xte, trees[ti])
        used = sort(unique(cells(Xtr, trees[ti])))
        relabel = zeros(Int, 4); relabel[used] = 1:L
        leaf_ex = relabel[raw]
        @assert all(leaf_ex .> 0)
        pols = Any[("exact", leaf_ex, L, sig, tau)]
        iai !== nothing && push!(pols, ("opt_expected", [iai.lid[l] for l in iai.le],
                                        iai.L, iai.sig, iai.tau))
        for s in sort(unique(te.school_id))
            idx = findall(==(s), te.school_id); nn = length(idx)
            @assert cap_of[s].n_total == nn
            bs, ba = caps([s], scale); bs, ba = bs[1], ba[1]
            Gr, Gs, Ga = te.G_regular[idx], te.G_small[idx], te.G_aide[idx]
            row(p, vv) = (fold=k, school_id=s, scale=scale, policy=p, n=nn, cap_small=bs,
                          cap_aide=ba, n_small=bs, n_aide=ba, sum_value=vv)
            for (p, lf, LL, sg, ta) in pols
                n, gs, ga = tables(lf[idx], ones(Int, nn), Gs .- Gr, Ga .- Gr, 1, LL)
                small = zeros(Int, 1, LL)
                gain = fill_value(n, gs, ga, [bs], [ba], sg, ta, small)
                @assert sum(small) == bs
                push!(results, row(p, sum(Gr) + gain))
            end
            push!(results, row("random", sum((1 - bs / nn - ba / nn) .* Gr .+
                                             bs / nn .* Gs .+ ba / nn .* Ga)))
        end
        push!(insample, (fold=k, scale=scale, n_train=ntr, trees_checked=nvalid[],
                         exact_gain=v / ntr, random_gain=rand_tr[ci] / ntr,
                         iai_gain=iai === nothing ? NaN : iai.tr[ci] / ntr,
                         iai_best_order_gain=iai === nothing ? NaN : iai.best[ci] / ntr,
                         iai_depth=iai === nothing ? 0 : iai.depth, exact_leaves=L))
    end

    open(joinpath(treedir, "fold$(k).txt"), "w") do io
        println(io, "fold $k: $(nvalid[]) valid trees (depth <= 2, leaves >= $MINBUCKET) ",
                "x all leaf orders checked; $(length(SPLITS)) candidate splits")
        for (ci, scale) in enumerate(SCALES)
            v, ti, sig, tau, L = best[ci]
            println(io, "\n== scale c = $scale: exact training gain over all-regular ",
                    round(v / ntr, digits=4), " per student (random ",
                    round(rand_tr[ci] / ntr, digits=4),
                    iai === nothing ? "" : ", IAI policy $(round(iai.tr[ci] / ntr, digits=4))", ")")
            println(io, describe_tree(trees[ti]))
            println(io, "leaves renumbered 1..$L in that order; small-seat order: $sig; ",
                    "aide-seat order: $tau")
        end
        iai === nothing || println(io, "\n== IAI tree (opt_capacity.jl, depth $(iai.depth)):\n", iai.txt)
    end
    println("fold $k: $(nvalid[]) valid trees checked in $(round(time() - t0, digits=1)) s")
end

CSV.write(joinpath(HERE, "exact_capacity_$name.csv"), DataFrame(results))
CSV.write(joinpath(HERE, "exact_insample_$name.csv"), DataFrame(insample))
println("Wrote exact_capacity_$name.csv, exact_insample_$name.csv, trees_exact_$name/")
