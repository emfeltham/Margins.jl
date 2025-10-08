#!/usr/bin/env julia
# Test allocations in the hot loop

using Margins, GLM, DataFrames, CategoricalArrays, FormulaCompiler

# Create test data
n = 1000
data = DataFrame(
    x1 = randn(n),
    x2 = randn(n),
    x3 = rand([0, 1], n),
    cat_var = categorical(rand(["A", "B", "C"], n))
)
data.y = 0.5 * data.x1 + 0.3 * data.x2 + 0.2 * data.x3 + randn(n) * 0.1
model = lm(@formula(y ~ x1 + x2 + x3), data)

# Prepare engine
data_nt = Margins._convert_numeric_to_float64(Tables.columntable(data))
engine = Margins.get_or_build_engine(Margins.PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov, :fd)

# Setup for loop test
β = engine.β
rows = 1:n
g_all = engine.g_all_buf
Gβ_all = engine.Gβ_all_buf

println("=== Testing loop allocations ===\n")

# Warmup
for row in 1:10
    marginal_effects_eta!(g_all, Gβ_all, engine.de, β, row)
end

# Test just the FormulaCompiler call
alloc1 = @allocated begin
    for row in 1:n
        marginal_effects_eta!(g_all, Gβ_all, engine.de, β, row)
    end
end
println("1. FormulaCompiler loop (marginal_effects_eta!): $alloc1 bytes")
println("   Per row: $(alloc1/n) bytes")

# Test with range object
alloc2 = @allocated begin
    for row in rows
        marginal_effects_eta!(g_all, Gβ_all, engine.de, β, row)
    end
end
println("2. FormulaCompiler loop (range object): $alloc2 bytes")
println("   Per row: $(alloc2/n) bytes")

# Test the full computation path
continuous_requested = [:x1, :x2]
weights = nothing
total_weight = 1.0

alloc3 = @allocated Margins._compute_continuous_marginal_effects!(
    engine, continuous_requested, rows, :response, weights, total_weight
)
println("3. Full _compute_continuous_marginal_effects!: $alloc3 bytes")
println("   Per row: $(alloc3/n) bytes")

# Test _ame_calculate
alloc4 = @allocated Margins._ame_calculate(engine, data_nt, :response, :fd, :effect, :baseline, weights)
println("4. Full _ame_calculate: $alloc4 bytes")
println("   Per row: $(alloc4/n) bytes")
