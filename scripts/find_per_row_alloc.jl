#!/usr/bin/env julia
# Find the source of ~1 alloc/row in population_margins

using Margins, GLM, DataFrames

# Small dataset for easier debugging
n = 100
data = DataFrame(x1=randn(n), x2=randn(n), y=randn(n))
data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
model = lm(@formula(y ~ x1 + x2), data)

# Prepare components
data_nt = Tables.columntable(data)
data_nt = Margins._convert_numeric_to_float64(data_nt)
engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)

# Warmup
Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, nothing)

println("=== Isolating allocation sources ===\n")

# Test just the loop computation
function test_just_loop(engine, data_nt)
    rows = 1:length(first(values(data_nt)))
    continuous_requested = engine.continuous_vars
    weights = nothing
    total_weight = 1.0

    # Use engine's pre-allocated buffers
    fill!(engine.batch_ame_values, 0.0)
    fill!(engine.batch_gradients, 0.0)

    # Just the loop, no DataFrame construction
    Margins._compute_continuous_marginal_effects!(
        engine, continuous_requested, rows, :response, weights, total_weight
    )
end

test_just_loop(engine, data_nt)  # Warmup

alloc_loop = @allocated test_just_loop(engine, data_nt)
println("Just the computation loop: $alloc_loop bytes, $(alloc_loop/n) bytes/row")

# Test DataFrame construction separately
function test_df_construction(engine, n_obs)
    total_rows = 2  # 2 continuous vars
    n_params = length(engine.β)
    Margins.build_results_dataframe(total_rows, n_obs, n_params)
end

alloc_df = @allocated test_df_construction(engine, n)
println("DataFrame construction: $alloc_df bytes")

# Test the full _ame_calculate
function test_full_ame(engine, data_nt)
    Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, nothing)
end

test_full_ame(engine, data_nt)  # Warmup

alloc_full = @allocated test_full_ame(engine, data_nt)
println("Full _ame_calculate: $alloc_full bytes, $(alloc_full/n) bytes/row")

println("\nAnalysis:")
println("  If loop allocates, problem is in computation")
println("  If loop doesn't allocate, problem is in DataFrame handling")
