# test_formulacompiler_integration.jl

using Test
using Margins
using DataFrames, GLM
using FormulaCompiler

@testset "FormulaCompiler Integration Tests" begin
    
    # Create test data
    Random.seed!(42)
    df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100))
    )
    
    # Fit test model
    model = lm(@formula(y ~ x + log(z) + group), df)
    
    @testset "Workspace Creation with FormulaCompiler" begin
        # Test workspace creation
        workspace = MarginalEffectsWorkspace(model, df, [:x, :z])
        
        @test workspace.compiled_formula isa CompiledFormula
        @test length(workspace.derivative_formulas) == 2
        @test haskey(workspace.derivative_formulas, :x)
        @test haskey(workspace.derivative_formulas, :z)
        @test get_observation_count(workspace) == 100
        @test get_parameter_count(workspace) > 0
    end
    
    @testset "Model Row Evaluation" begin
        workspace = MarginalEffectsWorkspace(model, df, [:x])
        
        # Test standard evaluation
        row_result = evaluate_model_row!(workspace, 1)
        @test length(row_result) == length(coef(model))
        @test all(isfinite.(row_result))
        
        # Test with overrides using FormulaCompiler scenarios
        row_result_override = evaluate_model_row!(workspace, 1; variable_overrides = Dict(:x => 2.0))
        @test length(row_result_override) == length(coef(model))
        @test all(isfinite.(row_result_override))
        
        # Results should be different due to override
        @test row_result != row_result_override
    end
    
    @testset "Analytical Derivative Evaluation" begin
        workspace = MarginalEffectsWorkspace(model, df, [:x, :z])
        
        # Test derivative evaluation for x
        deriv_result_x = evaluate_model_derivative!(workspace, 1, :x)
        @test length(deriv_result_x) == length(coef(model))
        @test all(isfinite.(deriv_result_x))
        
        # Test derivative evaluation for z (should use log derivative)
        deriv_result_z = evaluate_model_derivative!(workspace, 1, :z)
        @test length(deriv_result_z) == length(coef(model))
        @test all(isfinite.(deriv_result_z))
        
        # Derivatives should be different for different variables
        @test deriv_result_x != deriv_result_z
    end
    
    @testset "Zero Allocation Verification" begin
        workspace = MarginalEffectsWorkspace(model, df, [:x])
        
        # Warm up
        for i in 1:10
            evaluate_model_row!(workspace, i)
            evaluate_model_derivative!(workspace, i, :x)
        end
        
        # Test model row evaluation allocations
        allocs_model = @allocated evaluate_model_row!(workspace, 1)
        @test allocs_model == 0
        
        # Test derivative evaluation allocations  
        allocs_deriv = @allocated evaluate_model_derivative!(workspace, 1, :x)
        @test allocs_deriv == 0
    end
    
    @testset "Scenario Integration" begin
        workspace = MarginalEffectsWorkspace(model, df, [:x])
        
        # Test scenario creation and usage
        scenario_data = Dict(:x => 1.5, :z => 2.0)
        
        # Should not error with scenario overrides
        @test_nowarn evaluate_model_row!(workspace, 1; variable_overrides = scenario_data)
        @test_nowarn evaluate_model_derivative!(workspace, 1, :x; variable_overrides = scenario_data)
        
        # Results should be consistent with scenarios
        result1 = evaluate_model_row!(workspace, 1; variable_overrides = scenario_data)
        result2 = evaluate_model_row!(workspace, 1; variable_overrides = scenario_data)
        @test result1 == result2
    end
end
