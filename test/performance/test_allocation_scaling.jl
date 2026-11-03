# STRICT ZERO-ALLOCATION SCALING VALIDATION
#
# This file enforces the CORE PERFORMANCE GUARANTEE: computational hot paths
# must achieve ≤0.01 allocations per row/variable in steady state.
#
# ALLOCATION CEILING: ≤0.01 allocations per row/variable (effectively zero)
#
# PURPOSE: Ensures O(1) allocation behavior independent of dataset size.
# SCOPE: Tests COMPUTATIONAL PRIMITIVES with simple linear models.
# FOCUS: Scaling behavior - allocations must NOT grow with dataset size.
#
# NOTE: This is NOT the same as test_zero_allocations.jl, which tests
# high-level APIs where infrastructure overhead (DataFrame creation, grid
# construction) is acceptable. This file tests the CORE GUARANTEE.
#
# julia --project="." test/performance/test_allocation_scaling.jl > test/performance/test_allocation_scaling.txt 2>&1

using Test
using Random
using DataFrames, Tables, GLM
using StatsModels: Term
using BenchmarkTools
using Margins

const ROW_SIZES = (100, 1_000, 5_000, 10_000)
const VAR_SIZES = (1, 2, 4)
const MEASURES = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
const BACKENDS = (:ad, :fd)

function build_test_case(n_rows::Int, n_vars::Int)
    rng = MersenneTwister(hash((n_rows, n_vars)))
    df = DataFrame(y = randn(rng, n_rows))
    vars = Symbol.(string.('x', 1:n_vars))
    for var in vars
        df[!, var] = randn(rng, n_rows)
    end

    rhs = reduce(+, Term.(vars))
    formula = Term(:y) ~ rhs
    model = lm(formula, df)
    return model, df, vars
end

min_allocations(f) = float(minimum(@benchmark $f samples=10 evals=1).allocs)

@testset "Allocation Scaling Verification" begin
    for backend in BACKENDS
        @testset "Backend: $backend" begin
            @testset "Row scaling (≤0.01 allocs/row)" begin
                println("\n=== Row Scaling ($backend) ===")

                for n in ROW_SIZES
                    model, df, vars = build_test_case(n, 2)
                    # Warmup
                    population_margins(model, df; backend=backend, vars=vars)
                    # Measure
                    allocs = min_allocations(() -> population_margins(model, df; backend=backend, vars=vars))
                    allocs_per_row = allocs / n

                    println("  n=$n: $allocs total, $(round(allocs_per_row, digits=6)) per row")

                    # Direct assertion: ≤0.01 allocations per row
                    @test allocs_per_row <= 0.01
                end
            end

            @testset "Variable scaling (≤0.01 allocs/var)" begin
                println("\n=== Variable Scaling ($backend) ===")
                n_rows = 1_000

                for n_vars in VAR_SIZES
                    model, df, vars = build_test_case(n_rows, n_vars)
                    # Warmup
                    population_margins(model, df; backend=backend, vars=vars)
                    # Measure
                    allocs = min_allocations(() -> population_margins(model, df; backend=backend, vars=vars))
                    allocs_per_var = allocs / n_vars

                    println("  nvars=$n_vars: $allocs total, $(round(allocs_per_var, digits=6)) per var")

                    # Direct assertion: ≤0.01 allocations per variable
                    @test allocs_per_var <= 0.01
                end
            end

            @testset "Measure selection (consistent allocations)" begin
                println("\n=== Measure Selection ($backend) ===")
                model, df, vars = build_test_case(1_000, 2)

                alloc_results = Dict{Symbol, Float64}()
                for measure in MEASURES
                    # Warmup
                    population_margins(model, df; backend=backend, vars=vars, measure=measure)
                    # Measure
                    allocs = min_allocations(() -> population_margins(model, df; backend=backend, vars=vars, measure=measure))
                    alloc_results[measure] = allocs

                    println("  $measure: $allocs")

                    # Direct assertion: ≤0.01 allocations per row
                    @test allocs / 1_000 <= 0.01
                end

                # All measures should have similar allocation behavior
                min_allocs = minimum(values(alloc_results))
                max_allocs = maximum(values(alloc_results))
                @test max_allocs - min_allocs <= 5.0
            end

            @testset "Fixed overhead (O(1) independent of data)" begin
                println("\n=== Fixed Overhead ($backend) ===")
                model, df, vars = build_test_case(1_000, 2)
                # Warmup
                population_margins(model, df; backend=backend, vars=vars)
                # Measure
                total_allocs = min_allocations(() -> population_margins(model, df; backend=backend, vars=vars))

                println("  Total allocations: $total_allocs")

                # Direct assertion: total allocations should be O(1)
                # With ≤0.01 per row, 1000 rows → ≤10 total allocations
                @test total_allocs <= 10.0
            end
        end
    end
end

@testset "Categorical effects: per-call bytes do not grow with n (step 10)" begin
    # The observed levels come from the reference codes, not from a row-by-row Set of strings
    # (which cost ~300 KB per categorical variable at 20K rows)
    function cat_bytes(n)
        rng = MersenneTwister(3)
        d = DataFrame(x = randn(rng, n), g = categorical(rand(rng, ["a", "b", "c", "d"], n)),
                      t = rand(rng, Bool, n))
        d.y = d.x .+ (d.g .== "b") .+ randn(rng, n)
        m = lm(@formula(y ~ x + g + t), d)
        V = vcov(m)
        population_margins(m, d; vars = [:g, :t], vcov = V)
        nt = Margins.Tables.columntable(d)
        Margins._model_contrast_levels(nt.g, m, :g, nt)
        (minimum(@allocated(population_margins(m, d; vars = [:g, :t], vcov = V)) for _ in 1:3),
         minimum(@allocated(Margins._model_contrast_levels(nt.g, m, :g, nt)) for _ in 1:3))
    end
    (a_small, l_small) = cat_bytes(2_000)
    (a_large, l_large) = cat_bytes(40_000)
    @test l_large == l_small
    @test a_large <= a_small + 4096
    # The observed levels are those present (unused pool levels and missing are skipped)
    c = categorical(["b", "b", missing, "d"]; levels = ["a", "b", "c", "d"])
    @test Margins._observed_level_strings(c) == Set(["b", "d"])
    @test Margins._observed_level_strings(["x", "y", "x"]) == Set(["x", "y"])
end
