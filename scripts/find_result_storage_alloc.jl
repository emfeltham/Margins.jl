#!/usr/bin/env julia
# Find allocation in result storage phase

using Margins, GLM, DataFrames

n = 100
data = DataFrame(x1=randn(n), x2=randn(n), y=randn(n))
data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
model = lm(@formula(y ~ x1 + x2), data)

data_nt = Tables.columntable(data)
data_nt = Margins._convert_numeric_to_float64(data_nt)
engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)

println("=== Testing result storage phase ===\n")

# Prepare pre-computed results
rows = 1:n
continuous_requested = engine.continuous_vars
weights = nothing
total_weight = 1.0

fill!(engine.batch_ame_values, 0.0)
fill!(engine.batch_gradients, 0.0)
Margins._compute_continuous_marginal_effects!(
    engine, continuous_requested, rows, :response, weights, total_weight
)

# Create DataFrame and matrix
results, G = Margins.build_results_dataframe(2, n, length(engine.β))

# Test storing results
function test_store_results(results, G, engine, continuous_requested, data_nt, rows, weights)
    ȳ = 0.0  # Not needed for :effect measure
    total_weight = 1.0

    cont_idx = 1
    for (var_idx, var) in enumerate(continuous_requested)
        ame_val = engine.batch_ame_values[var_idx]
        gradient = view(engine.batch_gradients, var_idx, :)

        # Apply measure transformations
        (final_val, transform_factor) = Margins.apply_measure_transformations(
            ame_val, :effect, var,
            data_nt, rows, weights, ȳ, total_weight
        )

        # Store result
        Margins.store_continuous_result!(results, G, cont_idx, var, final_val, gradient, engine.Σ, length(rows))
        cont_idx += 1
    end
end

test_store_results(results, G, engine, continuous_requested, data_nt, rows, weights)  # Warmup

alloc_store = @allocated test_store_results(results, G, engine, continuous_requested, data_nt, rows, weights)
println("Storing results: $alloc_store bytes")

# Test just apply_measure_transformations
function test_measure_transform(engine, continuous_requested, data_nt, rows, weights)
    ȳ = 0.0
    total_weight = 1.0

    for (var_idx, var) in enumerate(continuous_requested)
        ame_val = engine.batch_ame_values[var_idx]
        Margins.apply_measure_transformations(
            ame_val, :effect, var,
            data_nt, rows, weights, ȳ, total_weight
        )
    end
end

test_measure_transform(engine, continuous_requested, data_nt, rows, weights)  # Warmup

alloc_transform = @allocated test_measure_transform(engine, continuous_requested, data_nt, rows, weights)
println("Measure transformations: $alloc_transform bytes")

# Test just store_continuous_result!
function test_store_only(results, G, engine, continuous_requested)
    cont_idx = 1
    for (var_idx, var) in enumerate(continuous_requested)
        ame_val = engine.batch_ame_values[var_idx]
        gradient = view(engine.batch_gradients, var_idx, :)

        Margins.store_continuous_result!(results, G, cont_idx, var, ame_val, gradient, engine.Σ, n)
        cont_idx += 1
    end
end

test_store_only(results, G, engine, continuous_requested)  # Warmup

alloc_store_only = @allocated test_store_only(results, G, engine, continuous_requested)
println("store_continuous_result! only: $alloc_store_only bytes")
