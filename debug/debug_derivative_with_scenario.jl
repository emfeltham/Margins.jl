#!/usr/bin/env julia

# Debug derivative evaluator with scenario data

using DataFrames, GLM, CategoricalArrays, Tables
using FormulaCompiler

println("=== Debugging Derivative Evaluator with Scenario Data ===")

# Create test data
df = DataFrame(
    y = randn(10),
    x = randn(10), 
    group = categorical(["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"])
)

model = lm(@formula(y ~ x + group), df)
data_nt = Tables.columntable(df)

# Test 1: Build derivative evaluator with original data
println("\\nTest 1: Building derivative evaluator with original data...")
try
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    println("  Continuous variables: $continuous_vars")
    
    if !isempty(continuous_vars)
        de = FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=continuous_vars)
        println(" Derivative evaluator built successfully")
        println("  typeof(de.base_data.group): $(typeof(de.base_data.group))")
        
        # Test evaluation with original data
        g_buf = Vector{Float64}(undef, length(continuous_vars))
        FormulaCompiler.marginal_effects_mu!(g_buf, de, coef(model), 1; link=GLM.IdentityLink())
        println(" Marginal effects with original data: $g_buf")
    else
        println("  No continuous variables found")
    end
catch e
    println(" Error: $e")
end

# Test 2: Create scenario and try to use derivative evaluator
println("\\nTest 2: Using derivative evaluator with scenario data...")
try
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    
    if !isempty(continuous_vars)
        de = FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=continuous_vars)
        
        # Create scenario
        scenario = FormulaCompiler.create_scenario("test", data_nt; group = "B")
        println("  Scenario created successfully")
        println("  typeof(scenario.data.group): $(typeof(scenario.data.group))")
        
        # Try to use derivative evaluator with scenario data - THIS IS WHERE ERROR OCCURS
        g_buf = Vector{Float64}(undef, length(continuous_vars))
        
        # This should fail because de was built with original data, not scenario data
        FormulaCompiler.marginal_effects_mu!(g_buf, de, coef(model), 1; link=GLM.IdentityLink())
        println(" Marginal effects with scenario data: $g_buf")
    else
        println("  No continuous variables found")
    end
catch e
    println(" Error during scenario evaluation:")
    println("  Error: $e")
    println("  This error is EXPECTED - derivative evaluator built with original data can't use scenario data")
end

# Test 3: Build derivative evaluator WITH scenario data
println("\\nTest 3: Building derivative evaluator WITH scenario data...")
try
    scenario = FormulaCompiler.create_scenario("test", data_nt; group = "B")
    compiled_scenario = FormulaCompiler.compile_formula(model, scenario.data)
    continuous_vars = FormulaCompiler.continuous_variables(compiled_scenario, scenario.data)
    
    if !isempty(continuous_vars)
        de_scenario = FormulaCompiler.build_derivative_evaluator(compiled_scenario, scenario.data; vars=continuous_vars)
        println(" Derivative evaluator built with scenario data")
        
        g_buf = Vector{Float64}(undef, length(continuous_vars))
        FormulaCompiler.marginal_effects_mu!(g_buf, de_scenario, coef(model), 1; link=model.model.rr.d.link)
        println(" Marginal effects with scenario evaluator: $g_buf")
    else
        println("  No continuous variables found")
    end
catch e
    println(" Error: $e")
end

println("\\n=== Test Complete ===")