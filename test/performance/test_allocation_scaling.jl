# ALLOCATION SCALING VALIDATION
#
# population_margins has a fixed per-call overhead (result DataFrame, vcov products,
# per-variable buffers) but must not allocate per row: the count of allocations is
# O(1) in the number of rows, and each added variable costs a constant number of
# allocations, independent of the number of rows.
#
# julia --project="." test/performance/test_allocation_scaling.jl > test/performance/test_allocation_scaling.txt 2>&1

using Test
using Random
using DataFrames, Tables, GLM, CategoricalArrays
using StatsModels: Term
using BenchmarkTools
using Margins

const ROW_SIZES = (100, 1_000, 10_000)
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

# Calls f (the earlier `$f` interpolation benchmarked the closure object and always gave 0)
min_allocations(f) = float(minimum(@benchmark $f() samples=10 evals=1).allocs)

function call_allocations(n_rows, n_vars; kw...)
    model, df, vars = build_test_case(n_rows, n_vars)
    population_margins(model, df; vars = vars, kw...)   # warmup
    return min_allocations(() -> population_margins(model, df; vars = vars, kw...))
end

@testset "Allocation Scaling Verification" begin
    for backend in BACKENDS
        @testset "Backend: $backend" begin
            @testset "Row scaling (≤0.01 extra allocs/row)" begin
                allocs = [call_allocations(n, 2; backend = backend) for n in ROW_SIZES]
                # 100x more rows may add ≤0.01 allocations per row of the largest n
                # (fixed overhead varies by a few allocations; one per row would add ~9,900)
                @test maximum(allocs) - allocs[1] <= 0.01 * ROW_SIZES[end]
            end

            @testset "Variable scaling (constant cost per variable, no per-row work)" begin
                n_rows = 1_000
                allocs = [call_allocations(n_rows, k; backend = backend) for k in VAR_SIZES]
                for i in 2:length(VAR_SIZES)
                    per_var = (allocs[i] - allocs[1]) / (VAR_SIZES[i] - VAR_SIZES[1])
                    @test per_var <= 0.05 * n_rows   # one allocation per row per variable would give 1000
                end
            end

            @testset "Measure selection (same allocations for every measure)" begin
                allocs = [call_allocations(1_000, 2; backend = backend, measure = m) for m in MEASURES]
                @test maximum(allocs) - minimum(allocs) <= 5.0
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
