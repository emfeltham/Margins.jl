#!/usr/bin/env julia

# Test the exact approach we're using in Margins.jl

using Margins, DataFrames, GLM, CategoricalArrays, Tables
using FormulaCompiler

# Create test data matching our actual test case
n = 100
df = DataFrame(
    y = randn(n),
    x1 = randn(n), 
    x2 = randn(n),
    group = categorical(rand(["A", "B", "C"], n)),
    treated = rand(Bool, n)
)

# Fit model
model = lm(@formula(y ~ x1 * x2 + group + treated), df)
data_nt = Tables.columntable(df)

println("=== Testing Margins.jl Scenario Approach ===")

# Step 1: Build engine (like Margins.jl does)
println("\\nStep 1: Building engine...")
try
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    vars_for_de = [:x1, :x2]  # continuous variables we want derivatives for
    
    println("  Continuous variables in data: $continuous_vars")
    println("  Variables for derivatives: $vars_for_de")
    
    # Build derivative evaluator
    de = FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=vars_for_de)
    println("✅ Engine built successfully")
    
    # Step 2: Create scenario
    profile = Dict(:x1 => 1.0, :x2 => 0.5, :group => "A", :treated => true)
    scenario = FormulaCompiler.create_scenario("test_profile", data_nt; profile...)
    println("✅ Scenario created successfully")
    
    # Step 3: Try to use derivative evaluator with scenario (THIS IS WHERE ERROR MIGHT OCCUR)
    g_buf = Vector{Float64}(undef, length(vars_for_de))
    
    println("\\nStep 3: Testing derivative evaluation with scenario...")
    println("  Original de.base_data keys: $(keys(de.base_data))")
    println("  Scenario data keys: $(keys(scenario.data))")
    println("  typeof(de.base_data.group): $(typeof(de.base_data.group))")
    println("  typeof(scenario.data.group): $(typeof(scenario.data.group))")
    
    # The key test: can we evaluate derivatives?
    FormulaCompiler.marginal_effects_mu!(g_buf, de, coef(model), 1; link=GLM.IdentityLink())
    println("✅ Marginal effects computed: $g_buf")
    
catch e
    println("❌ Error during test:")
    println("  $e")
    
    # Print detailed stacktrace
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        if i <= 5  # First 5 frames
            println("    $i. $frame")
        end
    end
end

println("\\n=== Test Complete ===")