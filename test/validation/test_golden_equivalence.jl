# test/validation/test_golden_equivalence.jl
#
# Refactor-equivalence gate for the R6/R7 refactor (Margins 3.0; R6_R7_DESIGN.md, step 0 and
# decision 1): rerun the fixed set of Margins calls in test/support/golden_cases.jl and
# require every estimate and SE to match test/support/golden/margins_golden.csv to a
# relative difference ≤ 1e-12 (automatic differentiation and backend-free rows) or ≤ 1e-9
# (finite-difference rows, case ids ending in "/fd").
#
# Why 1e-12: a refactor that only reorders sums over n ≤ 5000 rows moves AD results by about
# √n·eps ≈ 1e-14…1e-13; real algebra errors show at ≥ 1e-8. Why 1e-9 for FD: the
# finite-difference quotient divides by a step of order eps^(1/3), amplifying the same
# reordering to about 1e-11 (observed when building the snapshot), still far below any real
# error. (An absolute floor of 1e-15 keeps values that are ≈ 0 from producing meaningless
# relative differences.)
#
# Rows are matched by (case_id, key), where the key is built from the variable, the
# contrast and normalized context values (see golden_cases.jl), so the planned 3.0 schema
# changes (`t_stat` → `statistic` + `df`, `over_` group prefixes, `at_` scenario columns)
# do not break the match. A deliberate change of results (a bug fix) requires regenerating
# the snapshot with test/support/generate_golden_references.jl in the same PR, with the
# reason stated there.
#
# The snapshot depends on the fitted coefficients and on summation order, so it is only
# comparable when GLM, StatsModels and CategoricalArrays have the versions recorded in
# margins_golden_versions.txt AND the bounds-checking mode matches (the snapshot is made
# with --check-bounds=yes, as `Pkg.test` uses; run standalone with the same flag). Otherwise
# the value comparison is skipped (`@test_skip`) with a warning, but the case and key sets
# are still checked.

using Test
using CSV, DataFrames, Printf
using Margins

include(joinpath(@__DIR__, "..", "support", "golden_cases.jl"))

@testset "golden snapshot equivalence (rel. diff ≤ 1e-12 AD, ≤ 1e-9 FD)" begin
    golden_dir = joinpath(@__DIR__, "..", "support", "golden")
    golden = CSV.read(joinpath(golden_dir, "margins_golden.csv"), DataFrame;
                      types=Dict(:case_id => String, :operation => String, :key => String,
                                 :estimate => Float64, :se => Float64))
    recorded = only(filter(startswith("fingerprint: "),
                           readlines(joinpath(golden_dir, "margins_golden_versions.txt"))))[14:end]
    comparable = recorded == MarginsGolden.fingerprint()
    comparable || @warn "golden snapshot was generated with $recorded, running with " *
                        "$(MarginsGolden.fingerprint()); value comparison skipped"

    t0 = time()
    current = MarginsGolden.run_all()
    elapsed = time() - t0

    # Same cases, same rows
    gkeys = Set(zip(golden.case_id, golden.key))
    ckeys = Set(zip(current.case_id, current.key))
    missing_rows = setdiff(gkeys, ckeys)
    extra_rows = setdiff(ckeys, gkeys)
    isempty(missing_rows) || @error "rows in the golden snapshot not produced now" n=length(missing_rows) first=first(sort(collect(missing_rows)), 10)
    isempty(extra_rows) || @error "rows produced now but not in the golden snapshot" n=length(extra_rows) first=first(sort(collect(extra_rows)), 10)
    @test isempty(missing_rows)
    @test isempty(extra_rows)
    @test nrow(current) == length(ckeys)   # keys unique

    j = innerjoin(rename(golden, :estimate => :estimate_golden, :se => :se_golden),
                  select(current, :case_id, :key, :estimate => :estimate_now, :se => :se_now);
                  on=[:case_id, :key])
    reldiff(a, b) = (d = abs(a - b); d <= 1e-15 ? 0.0 : d / max(abs(a), abs(b)))
    j.rel_est = reldiff.(j.estimate_golden, j.estimate_now)
    j.rel_se = reldiff.(j.se_golden, j.se_now)
    j.rel = max.(j.rel_est, j.rel_se)
    j.tol = [endswith(c, "/fd") ? 1e-9 : 1e-12 for c in j.case_id]
    bad = sort(j[j.rel .> j.tol, :], :rel; rev=true)

    if !isempty(bad) && comparable
        io = IOBuffer()
        @printf(io, "%d of %d rows exceed their tolerance (AD 1e-12, FD 1e-9); worst rows:\n", nrow(bad), nrow(j))
        for r in eachrow(first(bad, 15))
            @printf(io, "  %-55s %-60s est %.17g vs %.17g (rel %.2e)  se %.17g vs %.17g (rel %.2e)\n",
                    r.case_id, r.key, r.estimate_golden, r.estimate_now, r.rel_est,
                    r.se_golden, r.se_now, r.rel_se)
        end
        ops = combine(groupby(bad, :operation), nrow => :rows, :rel => maximum => :max_rel)
        println(io, "by operation:\n", ops)
        @error String(take!(io))
    end
    if comparable
        @test nrow(bad) == 0   # details in the error log above
        @test nrow(j) == nrow(golden)
    else
        @test_skip nrow(bad) == 0
    end
    @info "golden equivalence" rows=nrow(j) cases=length(unique(j.case_id)) max_rel_est=maximum(j.rel_est) max_rel_se=maximum(j.rel_se) bitwise_equal=count(r -> r.estimate_golden == r.estimate_now && r.se_golden == r.se_now, eachrow(j)) seconds=round(elapsed; digits=1)
end
