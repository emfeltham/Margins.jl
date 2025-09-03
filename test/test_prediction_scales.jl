using Test
using Random
using DataFrames, GLM, StatsModels
using Margins

@testset "Prediction Scale Computation (:link vs :response)" begin
    Random.seed!(54321)
    
    # Test data setup
    n = 800
    df = DataFrame(
        x = randn(n),
        z = randn(n)
    )
    
    @testset "LogitLink Predictions" begin
        # Create binary outcome for logistic regression
        df.y_binom = rand(n) .< (1 ./ (1 .+ exp.(-(0.3 .+ 0.8 .* df.x .- 0.2 .* df.z))))
        m_logit = glm(@formula(y_binom ~ x + z), df, Binomial(), LogitLink())
        
        # Test population predictions
        pred_link = population_margins(m_logit, df; type=:predictions, scale=:link)
        pred_response = population_margins(m_logit, df; type=:predictions, scale=:response)
        
        @test nrow(DataFrame(pred_link)) == 1
        @test nrow(DataFrame(pred_response)) == 1
        @test all(isfinite, DataFrame(pred_link).estimate)
        @test all(isfinite, DataFrame(pred_response).estimate)
        
        # Link scale should be log-odds, response scale should be probabilities
        link_val = DataFrame(pred_link).estimate[1]
        response_val = DataFrame(pred_response).estimate[1]
        
        # Response scale should be bounded [0,1] for logistic
        @test 0.0 <= response_val <= 1.0
        # Link scale is unbounded (log-odds)
        @test abs(link_val) < 10.0  # Reasonable range
        
        # Should be different values (link vs response scale)
        @test abs(link_val - response_val) > 0.05
        
        # Test profile predictions at specific scenarios
        scenarios = Dict(:x => [-1.0, 0.0, 1.0])
        prof_link = profile_margins(m_logit, df; at=scenarios, type=:predictions, scale=:link)
        prof_response = profile_margins(m_logit, df; at=scenarios, type=:predictions, scale=:response)
        
        @test nrow(DataFrame(prof_link)) == 3
        @test nrow(DataFrame(prof_response)) == 3
        @test all(isfinite, DataFrame(prof_link).estimate)
        @test all(isfinite, DataFrame(prof_response).estimate)
        
        # All response predictions should be in [0,1]
        @test all(0.0 .<= DataFrame(prof_response).estimate .<= 1.0)
        
        # Link predictions should vary more widely
        @test maximum(DataFrame(prof_link).estimate) - minimum(DataFrame(prof_link).estimate) > 1.0
    end
    
    @testset "ProbitLink Predictions" begin
        m_probit = glm(@formula(y_binom ~ x + z), df, Binomial(), ProbitLink())
        
        pred_link = population_margins(m_probit, df; type=:predictions, scale=:link)  
        pred_response = population_margins(m_probit, df; type=:predictions, scale=:response)
        
        @test nrow(DataFrame(pred_link)) == 1
        @test nrow(DataFrame(pred_response)) == 1
        @test all(isfinite, DataFrame(pred_link).estimate)
        @test all(isfinite, DataFrame(pred_response).estimate)
        
        link_val = DataFrame(pred_link).estimate[1]
        response_val = DataFrame(pred_response).estimate[1]
        
        # Response scale should be bounded [0,1] for probit too
        @test 0.0 <= response_val <= 1.0
        @test abs(link_val - response_val) > 0.05
    end
    
    @testset "LogLink Predictions (Poisson)" begin
        # Create count outcome
        df.y_count = rand(1:15, n)
        m_log = glm(@formula(y_count ~ x + z), df, Poisson(), LogLink())
        
        pred_link = population_margins(m_log, df; type=:predictions, scale=:link)
        pred_response = population_margins(m_log, df; type=:predictions, scale=:response)
        
        @test nrow(DataFrame(pred_link)) == 1
        @test nrow(DataFrame(pred_response)) == 1
        @test all(isfinite, DataFrame(pred_link).estimate)
        @test all(isfinite, DataFrame(pred_response).estimate)
        
        link_val = DataFrame(pred_link).estimate[1] 
        response_val = DataFrame(pred_response).estimate[1]
        
        # Response scale should be positive (counts)
        @test response_val > 0.0
        # Link scale is log(counts), so could be negative
        @test abs(link_val - response_val) > 0.1
        
        # For log link: exp(link) should ≈ response 
        @test abs(exp(link_val) - response_val) < 0.1
    end
    
    @testset "IdentityLink Predictions (Linear)" begin
        # Linear model - scales should be identical
        m_linear = lm(@formula(z ~ x), df)
        
        pred_link = population_margins(m_linear, df; type=:predictions, scale=:link)
        pred_response = population_margins(m_linear, df; type=:predictions, scale=:response) 
        
        @test nrow(DataFrame(pred_link)) == 1
        @test nrow(DataFrame(pred_response)) == 1
        @test all(isfinite, DataFrame(pred_link).estimate)
        @test all(isfinite, DataFrame(pred_response).estimate)
        
        # Should be identical for identity link
        link_val = DataFrame(pred_link).estimate[1]
        response_val = DataFrame(pred_response).estimate[1]
        @test abs(link_val - response_val) < 1e-10
        
        # Test profiles too
        prof_link = profile_margins(m_linear, df; at=Dict(:x=>[0.0]), type=:predictions, scale=:link)
        prof_response = profile_margins(m_linear, df; at=Dict(:x=>[0.0]), type=:predictions, scale=:response)
        @test abs(DataFrame(prof_link).estimate[1] - DataFrame(prof_response).estimate[1]) < 1e-10
    end
end

@testset "Prediction Standard Errors by Scale" begin
    Random.seed!(54321)
    n = 600
    df = DataFrame(
        x = randn(n),
        z = randn(n)
    )
    df.y = rand(n) .< (1 ./ (1 .+ exp.(-(df.x))))
    
    m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())
    
    # Test that prediction standard errors are computed for both scales
    pred_link = population_margins(m, df; type=:predictions, scale=:link)
    pred_response = population_margins(m, df; type=:predictions, scale=:response)
    
    @test all(DataFrame(pred_link).se .> 0)  
    @test all(DataFrame(pred_response).se .> 0)
    @test all(isfinite, DataFrame(pred_link).se)
    @test all(isfinite, DataFrame(pred_response).se)
    
    # Standard errors should generally be different due to transformation
    @test abs(DataFrame(pred_link).se[1] - DataFrame(pred_response).se[1]) > 1e-6
    
    # Test basic result structure
    @test all(names(DataFrame(pred_link)) .⊇ ["term", "estimate", "se"])
    @test all(names(DataFrame(pred_response)) .⊇ ["term", "estimate", "se"])
end

@testset "Profile Prediction Grids with Scales" begin
    Random.seed!(54321)
    n = 400
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n)
    )
    df.y = rand(n) .< (1 ./ (1 .+ exp.(-(0.1 .+ 0.4.*df.x1 + 0.3.*df.x2))))
    
    m = glm(@formula(y ~ x1 + x2), df, Binomial(), LogitLink())
    
    # Test grid of predictions across scenarios and scales
    scenarios = Dict(:x1 => [-1.0, 0.0, 1.0], :x2 => [-0.5, 0.5])
    
    prof_link = profile_margins(m, df; at=scenarios, type=:predictions, scale=:link)
    prof_response = profile_margins(m, df; at=scenarios, type=:predictions, scale=:response)
    
    # Should have 3 × 2 = 6 scenario combinations
    @test nrow(DataFrame(prof_link)) == 6
    @test nrow(DataFrame(prof_response)) == 6
    
    # Check that profile columns are present
    @test "at_x1" in names(DataFrame(prof_link))
    @test "at_x2" in names(DataFrame(prof_link))
    @test "at_x1" in names(DataFrame(prof_response))
    @test "at_x2" in names(DataFrame(prof_response))
    
    # All predictions should be finite
    @test all(isfinite, DataFrame(prof_link).estimate)
    @test all(isfinite, DataFrame(prof_response).estimate)
    
    # Response predictions should be in [0,1]
    @test all(0.0 .<= DataFrame(prof_response).estimate .<= 1.0)
    
    # Test averaged profiles
    prof_avg_link = profile_margins(m, df; at=scenarios, type=:predictions, scale=:link, average=true)
    prof_avg_response = profile_margins(m, df; at=scenarios, type=:predictions, scale=:response, average=true)
    
    @test nrow(DataFrame(prof_avg_link)) == 1
    @test nrow(DataFrame(prof_avg_response)) == 1
    @test all(isfinite, DataFrame(prof_avg_link).estimate)
    @test all(isfinite, DataFrame(prof_avg_response).estimate)
end

@testset "Edge Cases and Error Handling" begin
    Random.seed!(54321)
    n = 200
    df = DataFrame(
        x = randn(n),
        y = rand(Bool, n)
    )
    
    m = glm(@formula(y ~ x), df, Binomial(), LogitLink())
    
    @testset "Single Scenario Profile" begin
        # Test with single scenario (should work)
        single_scenario = Dict(:x => [0.0])
        
        pred_link = profile_margins(m, df; at=single_scenario, type=:predictions, scale=:link)
        pred_response = profile_margins(m, df; at=single_scenario, type=:predictions, scale=:response)
        
        @test nrow(DataFrame(pred_link)) == 1
        @test nrow(DataFrame(pred_response)) == 1
        @test all(isfinite, DataFrame(pred_link).estimate)
        @test all(isfinite, DataFrame(pred_response).estimate)
    end
    
    @testset "Extreme Scenarios" begin
        # Test with extreme values (should still work)
        extreme_scenarios = Dict(:x => [-5.0, 5.0])
        
        pred_link = profile_margins(m, df; at=extreme_scenarios, type=:predictions, scale=:link)
        pred_response = profile_margins(m, df; at=extreme_scenarios, type=:predictions, scale=:response)
        
        @test nrow(DataFrame(pred_link)) == 2
        @test nrow(DataFrame(pred_response)) == 2
        @test all(isfinite, DataFrame(pred_link).estimate)
        @test all(isfinite, DataFrame(pred_response).estimate)
        
        # Response should still be bounded [0,1] even for extreme inputs
        @test all(0.0 .<= DataFrame(pred_response).estimate .<= 1.0)
    end
end