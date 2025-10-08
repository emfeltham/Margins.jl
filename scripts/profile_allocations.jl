#!/usr/bin/env julia
# Profile allocations in population_margins to identify remaining sources

using Margins, GLM, DataFrames, CategoricalArrays
using Tables

# Create test data (matches test_zero_allocations.jl)
n = 1000
data = DataFrame(
    x1 = randn(n),
    x2 = randn(n),
    x3 = rand([0, 1], n),
    cat_var = categorical(rand(["A", "B", "C"], n))
)
data.y = 0.5 * data.x1 + 0.3 * data.x2 + 0.2 * data.x3 + randn(n) * 0.1

# Fit model
model = lm(@formula(y ~ x1 + x2 + x3), data)

println("=== Step-by-step allocation profiling ===\n")

# Warmup
result = population_margins(model, data; backend=:fd, vars=[:x1, :x2])
println("Warmup complete\n")

# Step 1: Tables.columntable
alloc1 = @allocated Tables.columntable(data)
println("1. Tables.columntable: $alloc1 bytes")

# Step 2: Extract data_nt and check conversion
data_nt_raw = Tables.columntable(data)
alloc2 = @allocated Margins._convert_numeric_to_float64(data_nt_raw)
println("2. _convert_numeric_to_float64: $alloc2 bytes")

# Step 3: Check engine build/cache
data_nt = Margins._convert_numeric_to_float64(data_nt_raw)
alloc3 = @allocated Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)
println("3. get_or_build_engine (first call): $alloc3 bytes")

# Second call should hit cache
alloc3b = @allocated Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)
println("4. get_or_build_engine (cached): $alloc3b bytes")

# Step 4: Check _ame_calculate
engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)
weights_vec = nothing
alloc4 = @allocated Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, weights_vec)
println("5. _ame_calculate (first call): $alloc4 bytes")

# Second call
alloc4b = @allocated Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, weights_vec)
println("6. _ame_calculate (second call): $alloc4b bytes")

# Step 5: Full population_margins call
println("\n=== Full population_margins call ===")
alloc5 = @allocated population_margins(model, data; backend=:fd, vars=[:x1, :x2])
println("7. Full population_margins (cached engine): $alloc5 bytes")
println("   Allocs/row: $(alloc5 / n)")

# Step 6: Try to isolate the remaining allocations
println("\n=== Isolating specific allocation sources ===")

# Test if conversion is the issue
alloc_convert = @allocated begin
    data_nt_raw2 = Tables.columntable(data)
    data_nt2 = Margins._convert_numeric_to_float64(data_nt_raw2)
end
println("8. Combined columntable + convert: $alloc_convert bytes")

# Test parameter extraction
alloc_params = @allocated begin
    β = coef(model)
    Σ = vcov(model)
end
println("9. Parameter extraction (coef + vcov): $alloc_params bytes")
