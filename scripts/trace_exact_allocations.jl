#!/usr/bin/env julia
# Trace exact allocation location

using Margins, GLM, DataFrames, BenchmarkTools

# Use n where we see allocations (n=1000)
n = 1000
data = DataFrame(x1=randn(n), x2=randn(n), y=randn(n))
data.y = 0.5 * data.x1 + 0.3 * data.x2 + randn(n) * 0.1
model = lm(@formula(y ~ x1 + x2), data)

# Prepare everything
data_nt = Tables.columntable(data)
data_nt = Margins._convert_numeric_to_float64(data_nt)
engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)

# Warmup
Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, nothing)

println("=== Breaking down _ame_calculate allocations ===\n")

# Step 1: Build results DataFrame
alloc1 = @allocated Margins.build_results_dataframe(2, n, length(engine.β))
println("1. build_results_dataframe: $alloc1 bytes")

# Step 2: Just the computation
function just_compute(engine, data_nt)
    rows = 1:length(first(values(data_nt)))
    Margins._compute_continuous_marginal_effects!(
        engine, engine.continuous_vars, rows, :response, nothing, 1.0
    )
end

just_compute(engine, data_nt)  # Warmup
alloc2 = @allocated just_compute(engine, data_nt)
println("2. _compute_continuous_marginal_effects!: $alloc2 bytes")

# Step 3: Full _ame_calculate but wrapped in function
function wrapped_ame(engine, data_nt)
    Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, nothing)
end

wrapped_ame(engine, data_nt)  # Warmup
alloc3 = @allocated wrapped_ame(engine, data_nt)
println("3. _ame_calculate (in function): $alloc3 bytes")

# Step 4: Full population_margins wrapped in function
function wrapped_pm(model, data)
    population_margins(model, data; backend=:fd, vars=[:x1, :x2])
end

wrapped_pm(model, data)  # Warmup
alloc4 = @allocated wrapped_pm(model, data)
println("4. population_margins (in function): $alloc4 bytes")

# Step 5: Use BenchmarkTools for most accurate measurement
bench = @benchmark wrapped_pm($model, $data) samples=50 evals=1
println("\n5. BenchmarkTools min: ", minimum(bench).memory, " bytes")
println("   BenchmarkTools min allocs: ", minimum(bench).allocs)
