# test_weights.jl - Tests for observation weights functionality

@testset "Observation Weights" begin
    
    @testset "Weight Parameter Validation" begin
        n = 100
        data = DataFrame(
            y = randn(n),
            x1 = randn(n),
            x2 = randn(n),
            sampling_weight = rand(0.5:0.01:2.0, n),
            zero_weights = zeros(n),
            negative_weights = [-1.0; ones(n-1)]
        )
        
        # Create separate vectors for length/type testing
        wrong_length = ones(n-10)
        zero_weights = zeros(n)
        negative_weights = [-1.0; ones(n-1)]
        
        model = lm(@formula(y ~ x1 + x2), data)
        
        # Valid weight specifications should work
        @test_nowarn population_margins(model, data; type=:effects, vars=[:x1], weights=data.sampling_weight)
        @test_nowarn population_margins(model, data; type=:effects, vars=[:x1], weights=:sampling_weight)
        @test_nowarn population_margins(model, data; type=:effects, vars=[:x1], weights=nothing)
        
        # Invalid weight specifications should error
        @test_throws ArgumentError population_margins(model, data; type=:effects, vars=[:x1], weights=:nonexistent_column)
        @test_throws ArgumentError population_margins(model, data; type=:effects, vars=[:x1], weights=wrong_length)
        @test_throws ArgumentError population_margins(model, data; type=:effects, vars=[:x1], weights=negative_weights)
        @test_throws ArgumentError population_margins(model, data; type=:effects, vars=[:x1], weights=zero_weights)
        @test_throws ArgumentError population_margins(model, data; type=:effects, vars=[:x1], weights="invalid_type")
    end
    
    @testset "Weighted vs Unweighted Results - Linear Model" begin
        # Linear model: marginal effects are coefficients (weights shouldn't affect estimates much)
        Random.seed!(42)
        n = 200
        data = DataFrame(
            y = randn(n),
            x1 = randn(n),
            x2 = randn(n)
        )
        
        # Create weights correlated with x1
        data.weights_uniform = ones(n)
        data.weights_nonuniform = 1.0 .+ 0.3 .* data.x1
        data.weights_nonuniform = max.(data.weights_nonuniform, 0.1)  # Ensure positive
        
        model = lm(@formula(y ~ x1 + x2), data)
        
        # Compare results
        result_unweighted = population_margins(model, data; type=:effects, vars=[:x1, :x2])
        result_weighted = population_margins(model, data; type=:effects, vars=[:x1, :x2], 
                                           weights=data.weights_nonuniform)
        result_uniform_weights = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                                  weights=data.weights_uniform)
        
        # Extract DataFrames for comparison
        df_unwt = DataFrame(result_unweighted)
        df_wt = DataFrame(result_weighted)
        df_uniform = DataFrame(result_uniform_weights)
        
        # For linear models, estimates should be identical (coefficients don't change)
        # But standard errors might differ slightly due to weighted gradient computation
        @test all(abs.(df_wt.estimate .- df_unwt.estimate) .< 1e-10)
        
        # Uniform weights should exactly match unweighted
        @test all(abs.(df_uniform.estimate .- df_unwt.estimate) .< 1e-12)
        @test all(abs.(df_uniform.se .- df_unwt.se) .< 1e-12)
        
        # Check that metadata correctly indicates weighting
        @test result_unweighted.metadata[:weighted] == false
        @test result_weighted.metadata[:weighted] == true
        @test result_uniform_weights.metadata[:weighted] == true
    end
    
    @testset "Weighted vs Unweighted Results - GLM Model" begin
        # GLM model: marginal effects depend on data points (weights should affect estimates)
        Random.seed!(123)
        n = 300
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n)
        )
        
        # Create binary outcome for logistic regression  
        data.y_binary = rand(n) .< (0.5 .+ 0.1 .* data.x1)
        
        # Create weights that favor certain observations
        data.weights_uniform = ones(n)
        data.weights_skewed = 1.0 .+ 0.5 .* data.x1  # Weights correlated with x1
        data.weights_skewed = max.(data.weights_skewed, 0.1)  # Ensure positive
        
        model = glm(@formula(y_binary ~ x1 + x2), data, Binomial(), LogitLink())
        
        # Compare results
        result_unweighted = population_margins(model, data; type=:effects, vars=[:x1, :x2])
        result_weighted = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                           weights=data.weights_skewed)
        result_uniform_weights = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                                  weights=data.weights_uniform)
        
        # Extract DataFrames for comparison
        df_unwt = DataFrame(result_unweighted)
        df_wt = DataFrame(result_weighted)
        df_uniform = DataFrame(result_uniform_weights)
        
        # For GLM, weighted results should differ from unweighted when weights are non-uniform
        @test any(abs.(df_wt.estimate .- df_unwt.estimate) .> 1e-6)
        @test any(abs.(df_wt.se .- df_unwt.se) .> 1e-6)
        
        # Uniform weights should exactly match unweighted
        @test all(abs.(df_uniform.estimate .- df_unwt.estimate) .< 1e-12)
        @test all(abs.(df_uniform.se .- df_unwt.se) .< 1e-12)
        
        # Check that results have expected structure
        @test nrow(df_wt) == 2
        @test all(df_wt.se .> 0)
        @test all(in.(df_wt.term, Ref(["x1", "x2"])))
    end
    
    @testset "Weighted Results - Extreme Weights" begin
        Random.seed!(456)
        n = 100
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n)
        )
        data.y_binary = rand(n) .< (0.5 .+ 0.1 .* data.x1)
        
        # Create extreme weights (most weight on first few observations)
        extreme_weights = fill(0.01, n)
        extreme_weights[1:5] .= 20.0  # First 5 observations get most weight
        
        model = glm(@formula(y_binary ~ x1 + x2), data, Binomial(), LogitLink())
        
        # Should not error with extreme weights
        @test_nowarn begin
            result_extreme = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                              weights=extreme_weights)
        end
        
        result_extreme = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                          weights=extreme_weights)
        result_unweighted = population_margins(model, data; type=:effects, vars=[:x1, :x2])
        
        # Results should be different
        df_extreme = DataFrame(result_extreme)
        df_unwt = DataFrame(result_unweighted)
        @test any(abs.(df_extreme.estimate .- df_unwt.estimate) .> 1e-4)
        
        # Standard errors should be sensible (positive, finite)
        @test all(df_extreme.se .> 0)
        @test all(isfinite.(df_extreme.se))
    end
    
    @testset "Weighted Results - Both Effects and Predictions" begin
        Random.seed!(789)
        n = 150
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n)
        )
        data.y_binary = rand(n) .< (0.5 .+ 0.1 .* data.x1)
        data.survey_weights = rand(0.5:0.01:3.0, n)
        
        model = glm(@formula(y_binary ~ x1 + x2), data, Binomial(), LogitLink())
        
        # Test both effects and predictions with weights
        effects_weighted = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                            weights=data.survey_weights)
        predictions_weighted = population_margins(model, data; type=:predictions,
                                                weights=data.survey_weights)
        
        # Compare to unweighted versions
        effects_unweighted = population_margins(model, data; type=:effects, vars=[:x1, :x2])
        predictions_unweighted = population_margins(model, data; type=:predictions)
        
        # Results should differ
        @test DataFrame(effects_weighted).estimate != DataFrame(effects_unweighted).estimate
        @test DataFrame(predictions_weighted).estimate != DataFrame(predictions_unweighted).estimate
        
        # All results should be finite and sensible
        @test all(isfinite.(DataFrame(effects_weighted).estimate))
        @test all(isfinite.(DataFrame(effects_weighted).se))
        @test all(DataFrame(effects_weighted).se .> 0)
        
        @test all(isfinite.(DataFrame(predictions_weighted).estimate))
        @test all(isfinite.(DataFrame(predictions_weighted).se))
        @test all(DataFrame(predictions_weighted).se .> 0)
    end
    
    @testset "Weighted Gradients - Statistical Correctness" begin
        # Test that our weighted gradient implementation is statistically sound
        Random.seed!(999)
        n = 200
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n)
        )
        data.y = rand(n) .< (0.5 .+ 0.15 .* data.x1)  # Strong x1 effect
        
        # Create informative weights
        data.weights = 1.0 .+ 0.4 .* data.x1
        data.weights = max.(data.weights, 0.1)
        
        model = glm(@formula(y ~ x1 + x2), data, Binomial(), LogitLink())
        
        # Test different backends produce consistent weighted results
        result_ad = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                     weights=data.weights, backend=:ad)
        result_fd = population_margins(model, data; type=:effects, vars=[:x1, :x2], 
                                     weights=data.weights, backend=:fd)
        
        df_ad = DataFrame(result_ad)
        df_fd = DataFrame(result_fd)
        
        # AD and FD should give very similar results for weighted computation
        @test all(abs.(df_ad.estimate .- df_fd.estimate) .< 1e-4)
        @test all(abs.(df_ad.se .- df_fd.se) .< 1e-3)  # Standard errors might differ slightly
        
        # Test response vs link scale with weights
        result_response = population_margins(model, data; type=:effects, vars=[:x1],
                                           weights=data.weights, scale=:response)
        result_link = population_margins(model, data; type=:effects, vars=[:x1],
                                       weights=data.weights, scale=:link)
        
        # Should produce different but sensible results
        @test DataFrame(result_response).estimate != DataFrame(result_link).estimate
        @test all(DataFrame(result_response).se .> 0)
        @test all(DataFrame(result_link).se .> 0)
    end
    
    @testset "Weight Column Reference" begin
        # Test using column names vs direct vectors
        n = 80
        data = DataFrame(
            y = randn(n),
            x1 = randn(n),
            sampling_wt = rand(0.5:0.01:2.0, n)
        )
        data.y_binary = data.y .> 0
        
        model = glm(@formula(y_binary ~ x1), data, Binomial(), LogitLink())
        
        # Both approaches should give identical results
        result_symbol = population_margins(model, data; type=:effects, vars=[:x1], 
                                         weights=:sampling_wt)
        result_vector = population_margins(model, data; type=:effects, vars=[:x1],
                                         weights=data.sampling_wt)
        
        @test DataFrame(result_symbol).estimate ≈ DataFrame(result_vector).estimate
        @test DataFrame(result_symbol).se ≈ DataFrame(result_vector).se
    end

end