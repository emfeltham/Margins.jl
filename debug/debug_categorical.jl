#!/usr/bin/env julia

# Debug categorical scenario creation

using DataFrames, GLM, CategoricalArrays, Tables
using FormulaCompiler

println("=== Debugging Categorical Scenario Creation ===")

# Create simple test data
df = DataFrame(
    y = randn(10),
    x = randn(10), 
    group = categorical(["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"])
)

println("Original data types:")
println("  typeof(df.group): $(typeof(df.group))")
println("  levels(df.group): $(levels(df.group))")
println("  eltype(df.group): $(eltype(df.group))")

# Convert to column table
data_nt = Tables.columntable(df)
println("\\nColumn table types:")
println("  typeof(data_nt.group): $(typeof(data_nt.group))")

# Test scenario creation with string override
println("\\nTesting scenario creation with string override...")
try
    scenario = FormulaCompiler.create_scenario("test", data_nt; group = "B")
    println("Scenario created successfully")
    println("  typeof(scenario.data.group): $(typeof(scenario.data.group))")
    
    # Try to fit model and compile
    model = lm(@formula(y ~ x + group), df)
    println("Model fitted successfully")
    
    compiled = FormulaCompiler.compile_formula(model, scenario.data)
    println("Formula compiled successfully")
    
    # Try evaluation
    output = Vector{Float64}(undef, length(compiled))
    compiled(output, scenario.data, 1)
    println("Evaluation successful")
    println("  Output: $(output)")
    
catch e
    println("Error during scenario creation/evaluation:")
    println("  Error type: $(typeof(e))")
    println("  Error message: $e")
    
    # Print stacktrace
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        if i <= 10  # Limit to first 10 frames
            println("    $i. $frame")
        end
    end
end

println("\\n=== Test Complete ===")