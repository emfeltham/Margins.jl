using Test
using Random
using DataFrames, GLM, Statistics
using Margins

@testset "Elasticity Features" begin
    Random.seed!(123)
    n = 100
    df = DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n),
        g = rand(["A", "B"], n)
    )
    m = lm(@formula(y ~ x1 + x2), df)
    
    @testset "Population Margins - Measure Parameter" begin
        # Test default behavior (measure = :effect)
        result_default = population_margins(m, df; type=:effects, vars=[:x1])
        result_effect = population_margins(m, df; type=:effects, vars=[:x1], measure=:effect)
        @test result_default.table.dydx ≈ result_effect.table.dydx
        
        # Test elasticity measures
        result_elasticity = population_margins(m, df; type=:effects, vars=[:x1], measure=:elasticity)
        result_semi_x = population_margins(m, df; type=:effects, vars=[:x1], measure=:semielasticity_x)
        result_semi_y = population_margins(m, df; type=:effects, vars=[:x1], measure=:semielasticity_y)
        
        # All should return valid results
        @test nrow(result_elasticity.table) == 1
        @test nrow(result_semi_x.table) == 1
        @test nrow(result_semi_y.table) == 1
        
        # Results should be different from marginal effects
        @test result_elasticity.table.dydx[1] != result_effect.table.dydx[1]
        @test result_semi_x.table.dydx[1] != result_effect.table.dydx[1]
        @test result_semi_y.table.dydx[1] != result_effect.table.dydx[1]
        
        # Test multiple variables
        result_multi = population_margins(m, df; type=:effects, vars=[:x1, :x2], measure=:elasticity)
        @test nrow(result_multi.table) == 2
        @test Set(result_multi.table.term) == Set([:x1, :x2])
    end
    
    @testset "Profile Margins - Measure Parameter" begin
        # Test at means
        result_means_default = profile_margins(m, df; at=:means, type=:effects, vars=[:x1])
        result_means_effect = profile_margins(m, df; at=:means, type=:effects, vars=[:x1], measure=:effect)
        @test result_means_default.table.dydx ≈ result_means_effect.table.dydx
        
        # Test elasticity measures at means
        result_means_elasticity = profile_margins(m, df; at=:means, type=:effects, vars=[:x1], measure=:elasticity)
        result_means_semi_x = profile_margins(m, df; at=:means, type=:effects, vars=[:x1], measure=:semielasticity_x)
        result_means_semi_y = profile_margins(m, df; at=:means, type=:effects, vars=[:x1], measure=:semielasticity_y)
        
        @test nrow(result_means_elasticity.table) == 1
        @test nrow(result_means_semi_x.table) == 1
        @test nrow(result_means_semi_y.table) == 1
        
        # Test at specific profiles
        at_dict = Dict(:x1 => [-1.0, 0.0, 1.0], :x2 => [0.0])
        result_profiles = profile_margins(m, df; at=at_dict, type=:effects, vars=[:x1], measure=:elasticity)
        @test nrow(result_profiles.table) == 3  # 3 x1 values
        
        # Test table-based profiles
        reference_grid = DataFrame(x1 = [-1.0, 0.0, 1.0], x2 = [0.0, 0.0, 0.0])
        result_table = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1], measure=:elasticity)
        @test nrow(result_table.table) == 3
        @test result_table.table.dydx ≈ result_profiles.table.dydx  # Should be same as dict-based approach
    end
    
    @testset "Parameter Validation" begin
        # Test invalid measure values
        @test_throws ArgumentError population_margins(m, df; type=:effects, measure=:invalid)
        @test_throws ArgumentError profile_margins(m, df; at=:means, type=:effects, measure=:invalid)
        
        # Test measure parameter only applies to effects, not predictions  
        @test_throws ArgumentError population_margins(m, df; type=:predictions, measure=:elasticity)
        @test_throws ArgumentError profile_margins(m, df; at=:means, type=:predictions, measure=:elasticity)
        
        # Valid cases should work
        @test_nowarn population_margins(m, df; type=:effects, measure=:elasticity)
        @test_nowarn population_margins(m, df; type=:predictions, measure=:effect)  # default is fine
        @test_nowarn profile_margins(m, df; at=:means, type=:effects, measure=:semielasticity_x)
        @test_nowarn profile_margins(m, df; at=:means, type=:predictions, measure=:effect)
    end
    
    @testset "GLM Models" begin
        # Test with logistic regression
        df_binary = DataFrame(
            y = rand([0, 1], n),
            x = randn(n),
            z = randn(n)
        )
        m_glm = glm(@formula(y ~ x + z), df_binary, Binomial(), LogitLink())
        
        # Test population elasticities for GLM
        result_glm_pop = population_margins(m_glm, df_binary; type=:effects, vars=[:x], measure=:elasticity, target=:mu)
        @test nrow(result_glm_pop.table) == 1
        
        # Test profile elasticities for GLM  
        result_glm_prof = profile_margins(m_glm, df_binary; at=:means, type=:effects, vars=[:x], measure=:elasticity, target=:mu)
        @test nrow(result_glm_prof.table) == 1
        
        # Results should be reasonable (not NaN, not Inf)
        @test isfinite(result_glm_pop.table.dydx[1])
        @test isfinite(result_glm_prof.table.dydx[1])
    end
    
    @testset "Consistency Checks" begin
        # Compare population vs profile at means for elasticity
        pop_result = population_margins(m, df; type=:effects, vars=[:x1], measure=:elasticity)
        prof_result = profile_margins(m, df; at=:means, type=:effects, vars=[:x1], measure=:elasticity)
        
        # They should be different (population average vs at-means) but both finite
        @test isfinite(pop_result.table.dydx[1])
        @test isfinite(prof_result.table.dydx[1])
        
        # Standard errors should be positive
        @test pop_result.table.se[1] > 0
        @test prof_result.table.se[1] > 0
    end
end