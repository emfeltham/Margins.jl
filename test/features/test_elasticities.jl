using Test
using Random
using DataFrames, GLM, Statistics, CategoricalArrays
using Margins

@testset "Elasticity Features" begin
    Random.seed!(06515)
    n = 100
    df = DataFrame(
        y = randn(n),
        x1 = randn(n),
        x2 = randn(n),
        g = categorical(rand(["A", "B"], n))
    )
    m = lm(@formula(y ~ x1 + x2), df)
    
    @testset "Population Margins - Measure Parameter" begin
        # Test default behavior (measure = :effect)
        result_default = population_margins(m, df; type=:effects, vars=[:x1])
        result_effect = population_margins(m, df; type=:effects, vars=[:x1], measure=:effect)
        @test DataFrame(result_default).estimate ≈ DataFrame(result_effect).estimate
        
        # Test elasticity measures
        result_elasticity = population_margins(m, df; type=:effects, vars=[:x1], measure=:elasticity)
        result_semi_x = population_margins(m, df; type=:effects, vars=[:x1], measure=:semielasticity_eydx)
        result_semi_y = population_margins(m, df; type=:effects, vars=[:x1], measure=:semielasticity_dyex)
        
        # All should return valid results
        @test nrow(DataFrame(result_elasticity)) == 1
        @test nrow(DataFrame(result_semi_x)) == 1
        @test nrow(DataFrame(result_semi_y)) == 1
        
        # Results should be different from marginal effects
        @test DataFrame(result_elasticity).estimate[1] != DataFrame(result_effect).estimate[1]
        @test DataFrame(result_semi_x).estimate[1] != DataFrame(result_effect).estimate[1]
        @test DataFrame(result_semi_y).estimate[1] != DataFrame(result_effect).estimate[1]
        
        # Test multiple variables
        result_multi = population_margins(m, df; type=:effects, vars=[:x1, :x2], measure=:elasticity)
        @test nrow(DataFrame(result_multi)) == 2
        @test Set(DataFrame(result_multi).term) == Set(["x1", "x2"])
    end
    
    @testset "Profile Margins - Measure Parameter" begin
        # Test at means
        result_means_default = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x1])
        result_means_effect = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x1], measure=:effect)
        @test DataFrame(result_means_default).estimate ≈ DataFrame(result_means_effect).estimate
        
        # Test elasticity measures at means
        result_means_elasticity = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x1], measure=:elasticity)
        result_means_semi_x = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x1], measure=:semielasticity_eydx)
        result_means_semi_y = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x1], measure=:semielasticity_dyex)
        
        @test nrow(DataFrame(result_means_elasticity)) == 1
        @test nrow(DataFrame(result_means_semi_x)) == 1
        @test nrow(DataFrame(result_means_semi_y)) == 1
        
        # Test at specific profiles
        result_profiles = profile_margins(m, df, cartesian_grid(df; x1=[-1.0, 0.0, 1.0], x2=[0.0]); type=:effects, vars=[:x1], measure=:elasticity)
        @test nrow(DataFrame(result_profiles)) == 3  # 3 x1 values
        
        # Test table-based profiles
        reference_grid = DataFrame(x1 = [-1.0, 0.0, 1.0], x2 = [0.0, 0.0, 0.0])
        result_table = profile_margins(m, df, reference_grid; type=:effects, vars=[:x1], measure=:elasticity)
        @test nrow(DataFrame(result_table)) == 3
        @test DataFrame(result_table).estimate ≈ DataFrame(result_profiles).estimate  # Should be same as dict-based approach
    end
    
    @testset "Parameter Validation" begin
        # Test invalid measure values
        @test_throws ArgumentError population_margins(m, df; type=:effects, measure=:invalid)
        @test_throws ArgumentError profile_margins(m, df, means_grid(df); type=:effects, measure=:invalid)
        
        # Test measure parameter only applies to effects, not predictions  
        @test_throws ArgumentError population_margins(m, df; type=:predictions, measure=:elasticity)
        @test_throws ArgumentError profile_margins(m, df, means_grid(df); type=:predictions, measure=:elasticity)
        
        # Valid cases should work
        @test_nowarn population_margins(m, df; type=:effects, measure=:elasticity)
        @test_nowarn population_margins(m, df; type=:predictions, measure=:effect)  # default is fine
        @test_nowarn profile_margins(m, df, means_grid(df); type=:effects, measure=:semielasticity_eydx)
        @test_nowarn profile_margins(m, df, means_grid(df); type=:predictions, measure=:effect)
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
        result_glm_pop = population_margins(m_glm, df_binary; type=:effects, vars=[:x], measure=:elasticity, scale=:response)
        @test nrow(DataFrame(result_glm_pop)) == 1
        
        # Test profile elasticities for GLM  
        result_glm_prof = profile_margins(m_glm, df_binary, means_grid(df_binary); type=:effects, vars=[:x], measure=:elasticity, scale=:response)
        @test nrow(DataFrame(result_glm_prof)) == 1
        
        # Results should be reasonable (not NaN, not Inf)
        @test isfinite(DataFrame(result_glm_pop).estimate[1])
        @test isfinite(DataFrame(result_glm_prof).estimate[1])
    end
    
    @testset "Consistency Checks" begin
        # Compare population vs profile at means for elasticity
        pop_result = population_margins(m, df; type=:effects, vars=[:x1], measure=:elasticity)
        prof_result = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x1], measure=:elasticity)
        
        # They should be different (population average vs at-means) but both finite
        @test isfinite(DataFrame(pop_result).estimate[1])
        @test isfinite(DataFrame(prof_result).estimate[1])
        
        # Standard errors should be positive
        @test DataFrame(pop_result).se[1] > 0
        @test DataFrame(prof_result).se[1] > 0
    end
end