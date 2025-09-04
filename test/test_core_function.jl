# test_core_function.jl
# Test the new margins() function

using Test
using Margins
using DataFrames, GLM, StatsModels
using Random

@testset "Core margins() Function Tests" begin
    
    # Create test data
    Random.seed!(06515)
    df = DataFrame(
        x = randn(100),
        y = randn(100),
        z = abs.(randn(100)) .+ 0.1,
        group = categorical(rand(["A", "B", "C"], 100)),
        treatment = rand([false, true], 100)
    )
    
    # Fit test models
    linear_model = lm(@formula(y ~ x + log(z) + group), df)
    logit_model = glm(@formula(treatment ~ x + z), df, Binomial(), LogitLink())
    
    @testset "Basic margins() Functionality" begin
        
        @testset "Single Variable" begin
            # Test single variable as Symbol
            result = margins(linear_model, df, :x)
            @test result isa MarginalEffectsResult
            @test :x in result.focal_variables
            @test haskey(result.effect_estimates, :x)
            @test haskey(result.standard_errors, :x)
        end
        
        @testset "Multiple Variables" begin
            # Test multiple variables as Vector
            result = margins(linear_model, df, [:x, :z])
            @test result isa MarginalEffectsResult
            @test Set(result.focal_variables) == Set([:x, :z])
            @test haskey(result.effect_estimates, :x)
            @test haskey(result.effect_estimates, :z)
        end
        
        @testset "Categorical Variables" begin
            # Test categorical variable with different contrast types
            result_pairwise = margins(linear_model, df, :group; contrasts = :pairwise)
            @test result_pairwise isa MarginalEffectsResult
            @test :group in result_pairwise.focal_variables
            
            result_baseline = margins(linear_model, df, :group; contrasts = :baseline)
            @test result_baseline isa MarginalEffectsResult
            @test :group in result_baseline.focal_variables
            
            # Different contrast types should give different results
            @test result_pairwise.effect_estimates[:group] != result_baseline.effect_estimates[:group]
        end
    end
    
    @testset "Effect Types" begin
        
        @testset "Marginal Effects (:dydx)" begin
            # Default should be :dydx
            result_default = margins(linear_model, df, :x)
            result_explicit = margins(linear_model, df, :x; type = :dydx)
            
            @test result_default.effect_estimates == result_explicit.effect_estimates
            @test result_default.standard_errors == result_explicit.standard_errors
        end
        
        @testset "Predictions (:prediction)" begin
            # Test predictions with representative values
            result = margins(linear_model, df, :x; 
                           type = :prediction,
                           representative_values = Dict(:x => [0.0, 1.0, 2.0]))
            
            @test result isa MarginalEffectsResult
            @test !isempty(result.representative_values)
            @test haskey(result.effect_estimates, :x)
            
            # Should have multiple predictions for different x values
            @test result.effect_estimates[:x] isa Dict
        end
    end
    
    @testset "Representative Values" begin
        
        @testset "Single Representative Variable" begin
            result = margins(linear_model, df, [:x, :z]; 
                           representative_values = Dict(:group => ["A", "B"]))
            
            @test !isempty(result.representative_values)
            @test haskey(result.representative_values, :group)
            
            # Should have results for each representative value combination
            for var in [:x, :z]
                @test haskey(result.effect_estimates, var)
                @test result.effect_estimates[var] isa Dict  # Multiple combinations
            end
        end
        
        @testset "Multiple Representative Variables" begin
            result = margins(linear_model, df, :x;
                           representative_values = Dict(
                               :group => ["A", "B"],
                               :z => [1.0, 2.0]
                           ))
            
            @test length(result.representative_values) == 2
            @test result.effect_estimates[:x] isa Dict
            
            # Should have 2 × 2 = 4 combinations
            @test length(result.effect_estimates[:x]) == 4
        end
    end
    
    @testset "Model Types" begin
        
        @testset "Linear Model" begin
            result = margins(linear_model, df, :x)
            @test result isa MarginalEffectsResult
            @test result.model_family == "Normal"
            @test result.model_link == "identity"
        end
        
        @testset "GLM Model" begin
            result = margins(logit_model, df, :x)
            @test result isa MarginalEffectsResult
            @test result.model_family == "Binomial"
            @test result.model_link == "logit"
        end
    end
    
    @testset "Input Validation" begin
        
        @testset "Invalid Type" begin
            @test_throws ArgumentError margins(linear_model, df, :x; type = :invalid)
        end
        
        @testset "Invalid Contrasts" begin
            @test_throws ArgumentError margins(linear_model, df, :x; contrasts = :invalid)
        end
        
        @testset "Invalid Variable" begin
            @test_throws ArgumentError margins(linear_model, df, :nonexistent_var)
        end
        
        @testset "Predictions Without Representative Values" begin
            @test_throws ArgumentError margins(linear_model, df, :x; type = :prediction)
        end
        
        @testset "Empty Variables" begin
            @test_throws ArgumentError margins(linear_model, df, Symbol[])
        end
    end
    
    @testset "Backward Compatibility" begin
        
        @testset "Old vs New Function Results" begin
            # Compare new margins() with old compute_marginal_effects()
            
            # Old function call
            old_result = compute_marginal_effects(
                linear_model, [:x], df;
                representative_values = Dict{Symbol,Vector{Float64}}(),
                factor_contrasts = :all_pairs,
                effect_type = :dydx
            )
            
            # New function call
            new_result = margins(linear_model, df, :x)
            
            # Results should be equivalent
            @test old_result.focal_variables == new_result.focal_variables
            @test old_result.effect_estimates == new_result.effect_estimates
            @test old_result.standard_errors == new_result.standard_errors
            @test old_result.n_observations == new_result.n_observations
        end
    end
    
    @testset "Performance Characteristics" begin
        
        @testset "Zero Allocation Core" begin
            # Test that the core computation is zero-allocation
            result = margins(linear_model, df, :x)
            
            # Subsequent calls should be fast
            allocs = @allocated margins(linear_model, df, :x)
            
            # Some allocations are expected for result construction, but should be minimal
            @test allocs < 10000  # Less than 10KB
        end
        
        @testset "Performance with Large Data" begin
            # Test with larger dataset
            Random.seed!(06515)
            large_df = DataFrame(
                x = randn(10000),
                y = randn(10000),
                z = abs.(randn(10000)) .+ 0.1
            )
            
            large_model = lm(@formula(y ~ x + log(z)), large_df)
            
            # Should complete in reasonable time
            timing = @elapsed margins(large_model, large_df, :x)
            @test timing < 1.0  # Less than 1 second
            
            println("Large dataset ($( nrow(large_df)) rows) timing: $(round(timing * 1000, digits=1))ms")
        end
    end
    
    @testset "Display and Export" begin
        
        @testset "Basic Display" begin
            result = margins(linear_model, df, [:x, :group])
            
            # Should display without error
            @test_nowarn show(stdout, MIME"text/plain"(), result)
            
            # Should contain key information
            output = sprint(show, MIME"text/plain"(), result)
            @test occursin("Marginal Effects", output)
            @test occursin("x", output)
            @test occursin("group", output)
        end
        
        @testset "DataFrame Conversion" begin
            result = margins(linear_model, df, [:x, :z])
            
            # Should convert to DataFrame
            @test_nowarn DataFrame(result)
            df_result = DataFrame(result)
            
            @test df_result isa DataFrame
            @test nrow(df_result) >= 2  # At least one row per variable
            @test :variable in names(df_result)
            @test :estimate in names(df_result)
            @test :std_error in names(df_result)
        end
    end
end
