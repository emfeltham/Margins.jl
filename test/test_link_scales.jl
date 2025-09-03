using Test
using Random
using DataFrames, GLM, StatsModels
using Margins

@testset "Link Scale Computation (:eta vs :mu)" begin
    Random.seed!(12345)
    
    # Test data setup
    n = 1000
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        w = randn(n)
    )
    
    @testset "LogitLink (Binomial GLM)" begin
        # Create binary outcome
        df.y_binom = rand(n) .< (1 ./ (1 .+ exp.(-(0.5 .+ 1.0 .* df.x .- 0.3 .* df.z))))
        m_logit = glm(@formula(y_binom ~ x + z), df, Binomial(), LogitLink())
        
        # Test population margins
        eta_pop = population_margins(m_logit, df; type=:effects, vars=[:x], target=:eta)
        mu_pop = population_margins(m_logit, df; type=:effects, vars=[:x], target=:mu)
        
        @test nrow(DataFrame(eta_pop)) == 1
        @test nrow(DataFrame(mu_pop)) == 1
        @test all(isfinite, DataFrame(eta_pop).estimate)
        @test all(isfinite, DataFrame(mu_pop).estimate)
        
        # Critical test: effects should be different for LogitLink
        eta_val = DataFrame(eta_pop).estimate[1]
        mu_val = DataFrame(mu_pop).estimate[1]
        @test abs(eta_val - mu_val) > 0.001  # Should be substantially different
        @test eta_val / mu_val > 2.0  # η effects should be larger (chain rule: dμ/dη = μ(1-μ) ≤ 0.25)
        
        # Test profile margins  
        eta_prof = profile_margins(m_logit, df; at=:means, type=:effects, vars=[:x], target=:eta)
        mu_prof = profile_margins(m_logit, df; at=:means, type=:effects, vars=[:x], target=:mu)
        
        @test nrow(DataFrame(eta_prof)) == 1
        @test nrow(DataFrame(mu_prof)) == 1
        @test abs(DataFrame(eta_prof).estimate[1] - DataFrame(mu_prof).estimate[1]) > 0.001
        
        # Population and profile should be similar for large n
        @test abs(DataFrame(eta_pop).estimate[1] - DataFrame(eta_prof).estimate[1]) < 0.1  
        @test abs(DataFrame(mu_pop).estimate[1] - DataFrame(mu_prof).estimate[1]) < 0.1
    end
    
    @testset "ProbitLink (Binomial GLM)" begin
        # Reuse binary outcome but fit with ProbitLink
        m_probit = glm(@formula(y_binom ~ x + z), df, Binomial(), ProbitLink())
        
        eta_pop = population_margins(m_probit, df; type=:effects, vars=[:x], target=:eta)
        mu_pop = population_margins(m_probit, df; type=:effects, vars=[:x], target=:mu)
        
        @test nrow(DataFrame(eta_pop)) == 1
        @test nrow(DataFrame(mu_pop)) == 1
        @test all(isfinite, DataFrame(eta_pop).estimate)
        @test all(isfinite, DataFrame(mu_pop).estimate)
        
        # Critical test: effects should be different for ProbitLink
        eta_val = DataFrame(eta_pop).estimate[1]
        mu_val = DataFrame(mu_pop).estimate[1]
        @test abs(eta_val - mu_val) > 0.001
        @test eta_val / mu_val > 1.5  # η effects should be larger
    end
    
    @testset "LogLink (Poisson GLM)" begin
        # Create count outcome
        df.y_count = rand(1:20, n)
        m_log = glm(@formula(y_count ~ x + z), df, Poisson(), LogLink())
        
        eta_pop = population_margins(m_log, df; type=:effects, vars=[:x], target=:eta)
        mu_pop = population_margins(m_log, df; type=:effects, vars=[:x], target=:mu)
        
        @test nrow(DataFrame(eta_pop)) == 1 
        @test nrow(DataFrame(mu_pop)) == 1
        @test all(isfinite, DataFrame(eta_pop).estimate)
        @test all(isfinite, DataFrame(mu_pop).estimate)
        
        # Critical test: effects should be different for LogLink
        eta_val = DataFrame(eta_pop).estimate[1]
        mu_val = DataFrame(mu_pop).estimate[1]
        @test abs(eta_val - mu_val) > 0.001
        # For log link: dμ/dη = exp(η) = μ, so μ effects should be larger
        @test abs(mu_val / eta_val) > 2.0  
    end
    
    @testset "IdentityLink (Linear Model)" begin
        # Linear regression - should have identical effects
        m_linear = lm(@formula(w ~ x + z), df)
        
        eta_pop = population_margins(m_linear, df; type=:effects, vars=[:x], target=:eta)
        mu_pop = population_margins(m_linear, df; type=:effects, vars=[:x], target=:mu)
        
        @test nrow(DataFrame(eta_pop)) == 1
        @test nrow(DataFrame(mu_pop)) == 1
        @test all(isfinite, DataFrame(eta_pop).estimate)
        @test all(isfinite, DataFrame(mu_pop).estimate)
        
        # Critical test: effects should be identical for IdentityLink
        eta_val = DataFrame(eta_pop).estimate[1]
        mu_val = DataFrame(mu_pop).estimate[1]
        @test abs(eta_val - mu_val) < 1e-10  # Should be numerically identical
        
        # Test profile margins too
        eta_prof = profile_margins(m_linear, df; at=:means, type=:effects, vars=[:x], target=:eta)  
        mu_prof = profile_margins(m_linear, df; at=:means, type=:effects, vars=[:x], target=:mu)
        @test abs(DataFrame(eta_prof).estimate[1] - DataFrame(mu_prof).estimate[1]) < 1e-10
    end
end

@testset "_auto_link Function" begin
    Random.seed!(12345)
    n = 100
    df = DataFrame(x = randn(n), y = rand(Bool, n))
    
    @testset "GLM Models" begin
        # Test different GLM families
        m_logit = glm(@formula(y ~ x), df, Binomial(), LogitLink())
        m_probit = glm(@formula(y ~ x), df, Binomial(), ProbitLink()) 
        m_linear = lm(@formula(x ~ y), df)
        
        # Test link extraction
        using Margins: _auto_link
        
        link_logit = _auto_link(m_logit)
        link_probit = _auto_link(m_probit)
        link_linear = _auto_link(m_linear)
        
        @test link_logit isa LogitLink
        @test link_probit isa ProbitLink  
        @test link_linear isa GLM.IdentityLink
        
        # Test that _auto_link works with both TableRegressionModel and bare models
        @test _auto_link(m_logit.model) isa LogitLink
        @test _auto_link(m_linear.model) isa GLM.IdentityLink
    end
    
    @testset "Error Handling" begin
        using Margins: _auto_link
        
        # Test with invalid input - should fall back to IdentityLink
        struct FakeModel end
        @test _auto_link(FakeModel()) isa GLM.IdentityLink
    end
end

@testset "Link Scale Standard Errors" begin
    Random.seed!(12345)
    n = 500
    df = DataFrame(
        x = randn(n),
        z = randn(n) 
    )
    df.y = rand(Bool, n)
    
    m = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())
    
    # Test that standard errors are computed correctly for both scales
    eta_res = population_margins(m, df; type=:effects, vars=[:x], target=:eta)
    mu_res = population_margins(m, df; type=:effects, vars=[:x], target=:mu)
    
    @test all(DataFrame(eta_res).se .> 0)  # Positive standard errors
    @test all(DataFrame(mu_res).se .> 0)
    @test all(isfinite, DataFrame(eta_res).se)  # Finite standard errors
    @test all(isfinite, DataFrame(mu_res).se)
    
    # Standard errors should generally be different (due to delta method)
    @test abs(DataFrame(eta_res).se[1] - DataFrame(mu_res).se[1]) > 1e-6
    
    # Test basic result structure
    @test all(names(DataFrame(eta_res)) .⊇ ["term", "estimate", "se"])
    @test all(names(DataFrame(mu_res)) .⊇ ["term", "estimate", "se"])
end

@testset "Multi-Variable Link Scale Effects" begin
    Random.seed!(12345)
    n = 800
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        x3 = randn(n)
    )
    df.y = rand(Bool, n)
    
    m = glm(@formula(y ~ x1 + x2 + x3), df, Binomial(), LogitLink())
    
    # Test multiple variables simultaneously
    eta_multi = population_margins(m, df; type=:effects, vars=[:x1, :x2, :x3], target=:eta)
    mu_multi = population_margins(m, df; type=:effects, vars=[:x1, :x2, :x3], target=:mu)
    
    @test nrow(DataFrame(eta_multi)) == 3
    @test nrow(DataFrame(mu_multi)) == 3
    @test Set(DataFrame(eta_multi).term) == Set([:x1, :x2, :x3])
    @test Set(DataFrame(mu_multi).term) == Set([:x1, :x2, :x3])
    
    # All effects should differ between scales
    for i in 1:3
        eta_val = DataFrame(eta_multi).estimate[i]
        mu_val = DataFrame(mu_multi).estimate[i] 
        @test abs(eta_val - mu_val) > 0.001
        @test abs(eta_val) > abs(mu_val)  # Link scale effects should be larger in magnitude
    end
end