# test/support/generate_golden_references.jl
#
# Golden snapshot for the R6/R7 refactor (Margins 3.0, step 0 of R6_R7_DESIGN.md).
#
# Runs the fixed, deterministic set of Margins calls defined in golden_cases.jl — the
# feature matrix {population, groups (categorical, binned), scenarios (continuous,
# categorical), profile (means_grid, cartesian_grid)} × {:response, :link} × {no weights,
# positive weights, weights with zeros} × {4 measures} × {:ad, :fd} × 7 links, plus the six
# benchmark operations on a mid-size synthetic logit model — and writes
#
#   test/support/golden/margins_golden.csv           case_id, operation, key, estimate, se, df
#   test/support/golden/margins_golden_versions.txt  versions the snapshot was made with
#
# test/validation/test_golden_equivalence.jl reruns the calls and requires a relative
# difference ≤ 1e-12 on every estimate and SE.
#
# Regenerate ONLY from code whose results are trusted (before a refactor step, never to make
# a failing equivalence test pass), with an environment in which Margins and its test
# dependencies are available, e.g.
#
#   julia --check-bounds=yes \
#         --project=<env with Margins dev'd, CSV, GLM, DataFrames, CategoricalArrays> \
#         test/support/generate_golden_references.jl
#
# `--check-bounds=yes` matches `Pkg.test` (which forces it); bounds checking changes SIMD
# summation order, so a snapshot made in the other mode differs at the ulp level and the
# equivalence test would skip its value comparison (see MarginsGolden.fingerprint).
#
# (This replaces an older JLD2-based script written for the FormulaCompiler-primitives
# migration, which activated a hard-coded project path.)

using CSV, DataFrames, Printf
using Margins

include(joinpath(@__DIR__, "golden_cases.jl"))
using .MarginsGolden

const GOLDEN_DIR = joinpath(@__DIR__, "golden")
mkpath(GOLDEN_DIR)

t0 = time()
Margins.clear_engine_cache!()
golden = MarginsGolden.run_all()
elapsed = time() - t0

@assert all(isfinite, golden.estimate) && all(isfinite, golden.se)
CSV.write(joinpath(GOLDEN_DIR, "margins_golden.csv"), golden)

open(joinpath(GOLDEN_DIR, "margins_golden_versions.txt"), "w") do io
    println(io, "fingerprint: ", MarginsGolden.fingerprint())
    println(io, "julia: ", VERSION)
    println(io, "Margins: ", pkgversion(Margins))
    println(io, "FormulaCompiler: ", pkgversion(Margins.FormulaCompiler))
    fd = Base.root_module(Base.PkgId(Base.UUID("f6369f11-7733-5829-9624-2563aa707210"), "ForwardDiff"))
    println(io, "ForwardDiff: ", pkgversion(fd))
end

@printf("wrote %d rows from %d cases in %.1f s to %s\n", nrow(golden),
        length(unique(golden.case_id)), elapsed, GOLDEN_DIR)
