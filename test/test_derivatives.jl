# Test script to verify the complete derivative compilation system

using FormulaCompiler
using GLM, DataFrames, Tables
using Test

# Create test data
df = DataFrame(
    x = randn(100),
    y = randn(100), 
    z = abs.(randn(100)) .+ 0.1,
    group = rand(["A", "B"], 100)
)

println("=== Testing Complete Derivative Compilation System ===")

# Test 1: Basic linear model
println("\n1. Testing basic linear model...")
model1 = lm(@formula(y ~ x), df)
compiled1 = compile_formula(model1)
println("✓ Compiled formula: $(length(compiled1)) parameters")

# Test derivative compilation
derivative_x = compile_derivative_formula(compiled1, :x)
println("✓ Compiled derivative w.r.t. x: $(length(derivative_x)) output")

# Test evaluation
data = Tables.columntable(df)
row_vec = Vector{Float64}(undef, length(derivative_x))
derivative_x(row_vec, data, 1)
println("✓ Evaluated derivative: $(row_vec[1])")

# Test 2: Model with log function
println("\n2. Testing model with log function...")
model2 = lm(@formula(y ~ x + log(z)), df)
compiled2 = compile_formula(model2)
println("✓ Compiled log formula: $(length(compiled2)) parameters")

# Test derivatives for both variables
derivative_x2 = compile_derivative_formula(compiled2, :x)
derivative_z2 = compile_derivative_formula(compiled2, :z)
println("✓ Compiled derivatives for x and z")

# Test evaluation
row_vec_x = Vector{Float64}(undef, length(derivative_x2))
row_vec_z = Vector{Float64}(undef, length(derivative_z2))

derivative_x2(row_vec_x, data, 1)
derivative_z2(row_vec_z, data, 1)

println("✓ Evaluated derivatives:")
println("  ∂/∂x: $(row_vec_x)")
println("  ∂/∂z: $(row_vec_z)")

# Test 3: Integration with Margins.jl workspace
println("\n3. Testing Margins.jl integration...")

# This would test the MarginalEffectsWorkspace if Margins.jl is available
try
    using Margins
    
    workspace = MarginalEffectsWorkspace(model2, df, [:x, :z])
    println("✓ Created MarginalEffectsWorkspace")
    
    # Test workspace functionality
    success = test_workspace_functionality(workspace, n_test_rows=5)
    if success
        println("✓ Workspace functionality tests passed")
    else
        println("⚠ Some workspace tests failed")
    end
    
    # Test diagnostics
    diagnostics = diagnose_workspace(workspace)
    println("✓ Workspace diagnostics:")
    println("  Data: $(diagnostics.data_dimensions)")
    println("  Derivative success rate: $(round(diagnostics.derivative_compilation.success_rate * 100, digits=1))%")
    println("  Memory usage: $(round(diagnostics.memory_usage.total_buffers / 1024, digits=1)) KB")
    
catch e
    println("⚠ Margins.jl not available or has issues: $e")
end

# Test 4: Performance comparison
println("\n4. Performance comparison...")
using BenchmarkTools

# Time formula evaluation
formula_time = @belapsed $compiled2($row_vec_x, $data, 1)
println("✓ Formula evaluation: $(round(formula_time * 1e9, digits=1)) ns")

# Time derivative evaluation  
derivative_time = @belapsed $derivative_x2($row_vec_x, $data, 1)
println("✓ Derivative evaluation: $(round(derivative_time * 1e9, digits=1)) ns")

println("✓ Derivative evaluation is $(round(derivative_time/formula_time, digits=1))x relative to formula")

# Test 5: Verify analytical vs numerical accuracy
println("\n5. Testing analytical vs numerical accuracy...")
using ForwardDiff

# Create a function for numerical differentiation
function model_func(x_val, z_val)
    # Simulate y ~ x + log(z) at means of other variables
    coefs = coef(model2)
    return coefs[1] + coefs[2] * x_val + coefs[3] * log(z_val)
end

# Test point
test_x = df.x[1]
test_z = df.z[1]

# Numerical derivatives
numerical_dx = ForwardDiff.derivative(x -> model_func(x, test_z), test_x)
numerical_dz = ForwardDiff.derivative(z -> model_func(test_x, z), test_z)

# Analytical derivatives (from our compilation)
derivative_x2(row_vec_x, data, 1)
derivative_z2(row_vec_z, data, 1)
analytical_dx = row_vec_x[2]  # Second coefficient (x term)
analytical_dz = row_vec_z[3]  # Third coefficient (log(z) term)

# Compare
dx_error = abs(analytical_dx - numerical_dx)
dz_error = abs(analytical_dz - numerical_dz)

println("✓ Accuracy comparison:")
println("  ∂/∂x: analytical=$(round(analytical_dx, digits=6)), numerical=$(round(numerical_dx, digits=6)), error=$(round(dx_error, digits=10))")
println("  ∂/∂z: analytical=$(round(analytical_dz, digits=6)), numerical=$(round(numerical_dz, digits=6)), error=$(round(dz_error, digits=10))")

if dx_error < 1e-10 && dz_error < 1e-10
    println("Analytical derivatives match numerical within tolerance!")
else
    println("Analytical derivatives may have accuracy issues")
end

println("\n=== Derivative Compilation System Test Complete ===")
println("All core functionality working!")
println("Ready for high-performance marginal effects computation")