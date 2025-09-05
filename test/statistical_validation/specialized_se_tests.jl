# specialized_se_tests.jl - Tier 5: Specialized SE Cases Testing
#
# This file implements Phase 4, Tier 5 specialized standard error testing:
# - Integer Variable SE Testing - Verify integer→float conversion doesn't affect SEs
# - Elasticity SE Testing - Validate delta method for elasticity transformations  
# - Categorical Mixture SE Testing - Test SE computation for weighted categorical scenarios
#
# These tests complement the comprehensive analytical (Tier 1A/1B), bootstrap (Tier 7),
# and robust SE (Tier 8) validation already implemented.

using Test
using Random
using DataFrames
using CategoricalArrays
using GLM
using Statistics
using StatsModels
using Margins
using LinearAlgebra
using Distributions

# Import required functions for mixture testing
using Margins: mix

@testset "Tier 5: Specialized SE Cases - Phase 4 Implementation" begin
    Random.seed!(12345)  # Reproducible specialized testing
    
    # === INTEGER VARIABLE SE TESTING ===
    @testset "Integer Variable SE Testing - Float Conversion Validation" begin
        
        @testset "Integer vs Float SE Consistency" begin
            # Create identical data with integer and float versions of same variable
            n = 1000
            df_int = DataFrame(
                x_int = rand(1:50, n),  # Integer version
                z = randn(n)
            )
            df_int.x_float = Float64.(df_int.x_int)  # Exact float conversion
            df_int.y = 0.5 * df_int.x_int + 0.3 * df_int.z + 0.1 * randn(n)
            
            # Fit identical models with integer and float versions
            model_int = lm(@formula(y ~ x_int + z), df_int)
            model_float = lm(@formula(y ~ x_float + z), df_int)
            
            # Test population effects SEs should be identical
            pop_int = population_margins(model_int, df_int; type=:effects, vars=[:x_int])
            pop_float = population_margins(model_float, df_int; type=:effects, vars=[:x_float])
            
            se_int = DataFrame(pop_int).se[1]
            se_float = DataFrame(pop_float).se[1]
            
            @test se_int ≈ se_float atol=1e-14  # Should be identical to machine precision
            @test validate_all_finite_positive(DataFrame(pop_int)).all_valid
            @test validate_all_finite_positive(DataFrame(pop_float)).all_valid
            
            # Test profile effects SEs should be identical
            prof_int = profile_margins(model_int, df_int, means_grid(df_int); type=:effects, vars=[:x_int])
            prof_float = profile_margins(model_float, df_int, means_grid(df_int); type=:effects, vars=[:x_float])
            
            prof_se_int = DataFrame(prof_int).se[1]
            prof_se_float = DataFrame(prof_float).se[1]
            
            @test prof_se_int ≈ prof_se_float atol=1e-14  # Should be identical
            @test validate_all_finite_positive(DataFrame(prof_int)).all_valid
            @test validate_all_finite_positive(DataFrame(prof_float)).all_valid
        end
        
        @testset "Integer Variable GLM SE Validation" begin
            # Test integer variables in GLM context
            n = 800
            df = DataFrame(
                age_int = rand(18:65, n),
                edu_int = rand(0:16, n),
                z_float = randn(n)
            )
            
            # Create logistic outcome
            η = 0.05 * df.age_int - 0.1 * df.edu_int + 0.2 * df.z_float
            df.outcome = rand.(Bernoulli.(1 ./ (1 .+ exp.(-η))))
            
            model = glm(@formula(outcome ~ age_int + edu_int + z_float), df, Binomial(), LogitLink())
            
            # Test that integer variables produce valid SEs in GLM context
            # Link scale (should be exact coefficient SEs)
            eta_result = population_margins(model, df; type=:effects, vars=[:age_int, :edu_int], scale=:link)
            eta_df = DataFrame(eta_result)
            
            # Compare to analytical coefficient SEs
            age_coef_se = analytical_linear_se(model, df, :age_int)
            edu_coef_se = analytical_linear_se(model, df, :edu_int)
            
            @test eta_df.se[1] ≈ age_coef_se atol=1e-12  # age_int SE should equal coefficient SE
            @test eta_df.se[2] ≈ edu_coef_se atol=1e-12  # edu_int SE should equal coefficient SE
            @test validate_all_finite_positive(eta_df).all_valid
            
            # Response scale (chain rule SEs should be valid and finite)
            mu_result = population_margins(model, df; type=:effects, vars=[:age_int, :edu_int], scale=:response)
            mu_df = DataFrame(mu_result)
            
            @test all(isfinite.(mu_df.se))
            @test all(mu_df.se .> 0)
            @test validate_all_finite_positive(mu_df).all_valid
        end
        
        @testset "Integer Variable Profile SE Edge Cases" begin
            # Test integer variables in profile margins with specific integer values
            n = 500
            df = DataFrame(
                x_int = rand(1:10, n),
                z = randn(n)
            )
            df.y = 0.8 * df.x_int + 0.2 * df.z + 0.05 * randn(n)
            
            model = lm(@formula(y ~ x_int + z), df)
            
            # Test profile at specific integer values (not means)
            for test_val in [3, 7, 9]  # Different integer values
                prof_result = profile_margins(model, df, DataFrame(x_int=[test_val], z=[mean(df.z)]); type=:effects, vars=[:x_int])
                prof_df = DataFrame(prof_result)
                
                # SE should be coefficient SE (linear model)
                coef_se = analytical_linear_se(model, df, :x_int)
                @test prof_df.se[1] ≈ coef_se atol=1e-12
                @test validate_all_finite_positive(prof_df).all_valid
            end
        end
    end
    
    # === ELASTICITY SE TESTING ===
    @testset "Elasticity SE Testing - Delta Method Validation" begin
        
        @testset "Elasticity SE Mathematical Properties" begin
            # Create data with known elasticity properties
            n = 1000
            df = DataFrame(
                x = exp.(randn(n)) .* 2.0,  # Log-normal, ensures positive values
                z = randn(n)
            )
            df.y = 2.0 * df.x .^ 0.5 + 0.1 * df.z + 0.05 * randn(n)  # Square root relationship
            
            model = lm(@formula(y ~ x + z), df)
            
            # Test population elasticity SE
            elast_result = population_margins(model, df; type=:effects, vars=[:x], measure=:elasticity)
            elast_df = DataFrame(elast_result)
            
            @test validate_all_finite_positive(elast_df).all_valid
            @test length(elast_df.se) == 1
            @test elast_df.se[1] > 0
            @test isfinite(elast_df.se[1])
            
            # Test profile elasticity SE
            prof_elast = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x], measure=:elasticity)
            prof_elast_df = DataFrame(prof_elast)
            
            @test validate_all_finite_positive(prof_elast_df).all_valid
            @test length(prof_elast_df.se) == 1
            @test prof_elast_df.se[1] > 0
            @test isfinite(prof_elast_df.se[1])
            
            # Profile elasticity SE should generally be different from population
            # (unless very specific circumstances, which is unlikely with this data)
            @test prof_elast_df.se[1] != elast_df.se[1]  # Different SE methods
        end
        
        @testset "Semi-elasticity SE Testing" begin
            n = 800
            df = DataFrame(
                x = randn(n) .+ 3,  # Ensure reasonable range
                z = randn(n)
            )
            df.y = exp.(0.2 * df.x + 0.1 * df.z + 0.05 * randn(n))  # Exponential relationship
            
            model = lm(@formula(y ~ x + z), df)
            
            # Test x semi-elasticity (∂ln(y)/∂x)
            semi_x = population_margins(model, df; type=:effects, vars=[:x], measure=:semielasticity_dyex)
            semi_x_df = DataFrame(semi_x)
            
            @test validate_all_finite_positive(semi_x_df).all_valid
            @test semi_x_df.se[1] > 0
            @test isfinite(semi_x_df.se[1])
            
            # Test y semi-elasticity (∂y/∂ln(x))  
            semi_y = population_margins(model, df; type=:effects, vars=[:x], measure=:semielasticity_eydx)
            semi_y_df = DataFrame(semi_y)
            
            @test validate_all_finite_positive(semi_y_df).all_valid
            @test semi_y_df.se[1] > 0
            @test isfinite(semi_y_df.se[1])
            
            # Different measures should generally have different SEs
            @test semi_x_df.se[1] != semi_y_df.se[1]
        end
        
        @testset "Elasticity SE vs Regular Effect SE Comparison" begin
            # Compare elasticity SEs to regular effect SEs - should be different
            n = 600
            df = DataFrame(
                x = rand(n) .* 4 .+ 1,  # Range [1, 5]
                z = randn(n)
            )
            df.y = 1.5 * df.x + 0.3 * df.z + 0.1 * randn(n)
            
            model = lm(@formula(y ~ x + z), df)
            
            # Regular effect SE
            regular = population_margins(model, df; type=:effects, vars=[:x])
            regular_df = DataFrame(regular)
            
            # Elasticity SE  
            elastic = population_margins(model, df; type=:effects, vars=[:x], measure=:elasticity)
            elastic_df = DataFrame(elastic)
            
            # Both should be valid but different
            @test validate_all_finite_positive(regular_df).all_valid
            @test validate_all_finite_positive(elastic_df).all_valid
            @test regular_df.se[1] != elastic_df.se[1]  # Different measures → different SEs
            @test regular_df.se[1] > 0
            @test elastic_df.se[1] > 0
        end
        
        @testset "GLM Elasticity SE Validation" begin
            # Test elasticities in GLM context (more complex delta method)
            n = 700
            df = DataFrame(
                x = rand(n) .* 3 .+ 0.5,  # Range [0.5, 3.5]
                z = randn(n)
            )
            
            # Logistic model
            η = 0.8 * log.(df.x) - 0.5 * df.z  # Log(x) relationship
            df.outcome = rand.(Bernoulli.(1 ./ (1 .+ exp.(-η))))
            
            model = glm(@formula(outcome ~ x + z), df, Binomial(), LogitLink())
            
            # Test elasticity SEs on both scales
            eta_elast = population_margins(model, df; type=:effects, vars=[:x], 
                                         scale=:link, measure=:elasticity)
            mu_elast = population_margins(model, df; type=:effects, vars=[:x], 
                                        scale=:response, measure=:elasticity)
            
            eta_df = DataFrame(eta_elast)
            mu_df = DataFrame(mu_elast)
            
            @test validate_all_finite_positive(eta_df).all_valid
            @test validate_all_finite_positive(mu_df).all_valid
            
            # Different scales should have different elasticity SEs
            @test eta_df.se[1] != mu_df.se[1]
            @test eta_df.se[1] > 0
            @test mu_df.se[1] > 0
        end
    end
    
    # === CATEGORICAL MIXTURE SE TESTING ===
    @testset "Categorical Mixture SE Testing - Weighted Scenario SEs" begin
        
        @testset "Simple Categorical Mixture SE Validation" begin
            # Create data with categorical variable
            n = 1000
            Random.seed!(456)
            df = DataFrame(
                x = randn(n),
                education = categorical(rand(["High School", "College", "Graduate"], n))
            )
            
            # Create outcome dependent on both
            edu_effects = Dict("High School" => 0.0, "College" => 0.5, "Graduate" => 1.0)
            df.y = 2.0 * df.x + [edu_effects[level] for level in df.education] + 0.2 * randn(n)
            
            model = lm(@formula(y ~ x + education), df)
            
            # Test categorical mixture at profile
            mixture_spec = Dict(
                :x => mean(df.x),
                :education => mix("High School" => 0.4, "College" => 0.4, "Graduate" => 0.2)
            )
            
            mix_result = profile_margins(model, df, DataFrame(x=[mean(df.x)], education=[mix("High School" => 0.4, "College" => 0.4, "Graduate" => 0.2)]); type=:effects, vars=[:x])
            mix_df = DataFrame(mix_result)
            
            @test validate_all_finite_positive(mix_df).all_valid
            @test length(mix_df.se) == 1
            @test mix_df.se[1] > 0
            @test isfinite(mix_df.se[1])
            
            # Test categorical mixture for predictions
            mix_pred = profile_margins(model, df, DataFrame(x=[mean(df.x)], education=[mix("High School" => 0.4, "College" => 0.4, "Graduate" => 0.2)]); type=:predictions)
            mix_pred_df = DataFrame(mix_pred)
            
            @test validate_all_finite_positive(mix_pred_df).all_valid
            @test mix_pred_df.se[1] > 0
            @test isfinite(mix_pred_df.se[1])
        end
        
        @testset "Boolean Mixture SE Testing" begin
            # Test fractional Boolean values (probability mixtures)
            n = 800
            df = DataFrame(
                x = randn(n),
                treatment = rand([true, false], n),  # Boolean variable
                z = randn(n)
            )
            
            df.y = 1.2 * df.x + 0.8 * df.treatment + 0.3 * df.z + 0.15 * randn(n)
            
            model = lm(@formula(y ~ x + treatment + z), df)
            
            # Test Boolean mixture (fractional treatment probability)
            bool_mixture = Dict(
                :x => 0.0,  # Fixed value
                :treatment => 0.7,  # 70% treatment probability 
                :z => mean(df.z)
            )
            
            bool_result = profile_margins(model, df, DataFrame(x=[0.0], treatment=[0.7], z=[mean(df.z)]); type=:effects, vars=[:x])
            bool_df = DataFrame(bool_result)
            
            @test validate_all_finite_positive(bool_df).all_valid
            @test bool_df.se[1] > 0
            @test isfinite(bool_df.se[1])
            
            # Test Boolean mixture predictions
            bool_pred = profile_margins(model, df, DataFrame(x=[0.0], treatment=[0.7], z=[mean(df.z)]); type=:predictions)
            bool_pred_df = DataFrame(bool_pred)
            
            @test validate_all_finite_positive(bool_pred_df).all_valid
            @test bool_pred_df.se[1] > 0
            @test isfinite(bool_pred_df.se[1])
        end
        
        @testset "Complex Mixture SE Testing" begin
            # Test mixtures with multiple categorical variables
            n = 600
            Random.seed!(789)
            df = DataFrame(
                x = randn(n),
                region = categorical(rand(["North", "South", "East", "West"], n)),
                gender = categorical(rand(["Male", "Female"], n)),
                z = randn(n)
            )
            
            # Complex outcome with interactions
            region_effects = Dict("North" => 0.0, "South" => -0.3, "East" => 0.2, "West" => 0.1)
            gender_effects = Dict("Male" => 0.0, "Female" => 0.4)
            
            df.y = 1.5 * df.x + 
                   [region_effects[r] for r in df.region] + 
                   [gender_effects[g] for g in df.gender] +
                   0.1 * randn(n)
            
            model = lm(@formula(y ~ x + region + gender + z), df)
            
            # Complex mixture specification
            complex_mix = Dict(
                :x => mean(df.x),
                :region => mix("North" => 0.25, "South" => 0.25, "East" => 0.25, "West" => 0.25),
                :gender => mix("Male" => 0.48, "Female" => 0.52),  # Slight imbalance
                :z => 0.0
            )
            
            complex_result = profile_margins(model, df, DataFrame(
                x=[mean(df.x)], 
                region=[mix("North" => 0.25, "South" => 0.25, "East" => 0.25, "West" => 0.25)],
                gender=[mix("Male" => 0.48, "Female" => 0.52)],
                z=[0.0]
            ); type=:effects, vars=[:x])
            complex_df = DataFrame(complex_result)
            
            @test validate_all_finite_positive(complex_df).all_valid
            @test complex_df.se[1] > 0
            @test isfinite(complex_df.se[1])
            
            # Test multiple variables with mixture
            multi_vars = profile_margins(model, df, DataFrame(
                x=[mean(df.x)], 
                region=[mix("North" => 0.25, "South" => 0.25, "East" => 0.25, "West" => 0.25)],
                gender=[mix("Male" => 0.48, "Female" => 0.52)],
                z=[0.0]
            ); type=:effects, vars=[:x, :z])
            multi_df = DataFrame(multi_vars)
            
            @test validate_all_finite_positive(multi_df).all_valid
            @test all(multi_df.se .> 0)
            @test all(isfinite.(multi_df.se))
            @test length(multi_df.se) == 2  # Two variables requested
        end
        
        @testset "GLM Categorical Mixture SE Testing" begin
            # Test categorical mixtures in GLM context (complex delta method)
            n = 700
            Random.seed!(101112)
            df = DataFrame(
                x = randn(n),
                category = categorical(rand(["A", "B", "C"], n)),
                z = randn(n)
            )
            
            # Logistic model with categorical predictors
            cat_effects = Dict("A" => 0.0, "B" => 0.6, "C" => -0.4)
            η = 1.0 * df.x + [cat_effects[c] for c in df.category] + 0.2 * df.z
            df.outcome = rand.(Bernoulli.(1 ./ (1 .+ exp.(-η))))
            
            model = glm(@formula(outcome ~ x + category + z), df, Binomial(), LogitLink())
            
            # Test categorical mixture in GLM
            glm_mix = Dict(
                :x => 0.5,
                :category => mix("A" => 0.3, "B" => 0.5, "C" => 0.2),
                :z => -0.2
            )
            
            # Effects on both scales
            eta_mix = profile_margins(model, df, DataFrame(
                x=[0.5], 
                category=[mix("A" => 0.3, "B" => 0.5, "C" => 0.2)],
                z=[-0.2]
            ); type=:effects, vars=[:x], scale=:link)
            mu_mix = profile_margins(model, df, DataFrame(
                x=[0.5], 
                category=[mix("A" => 0.3, "B" => 0.5, "C" => 0.2)],
                z=[-0.2]
            ); type=:effects, vars=[:x], scale=:response)
            
            eta_df = DataFrame(eta_mix)
            mu_df = DataFrame(mu_mix)
            
            @test validate_all_finite_positive(eta_df).all_valid
            @test validate_all_finite_positive(mu_df).all_valid
            @test eta_df.se[1] > 0
            @test mu_df.se[1] > 0
            @test eta_df.se[1] != mu_df.se[1]  # Different scales → different SEs
            
            # Test predictions with mixture
            pred_mix = profile_margins(model, df, DataFrame(
                x=[0.5], 
                category=[mix("A" => 0.3, "B" => 0.5, "C" => 0.2)],
                z=[-0.2]
            ); type=:predictions)
            pred_df = DataFrame(pred_mix)
            
            @test validate_all_finite_positive(pred_df).all_valid
            @test pred_df.se[1] > 0
            @test isfinite(pred_df.se[1])
        end
    end
    
    # === ERROR PROPAGATION TESTING ===
    @testset "Error Propagation Testing - Edge Cases and Boundary Conditions" begin
        
        @testset "Near-Singular Matrix Error Propagation" begin
            # Create data with near-perfect collinearity - this should produce invalid SEs (NaN)
            # Testing that Margins.jl correctly identifies statistical invalidity
            n = 500
            df = DataFrame(
                x1 = randn(n),
                z = randn(n)
            )
            df.x2 = df.x1 + 1e-12 * randn(n)  # Nearly perfect collinearity
            df.y = 0.5 * df.x1 + 0.3 * df.z + 0.1 * randn(n)
            
            model = lm(@formula(y ~ x1 + x2 + z), df)
            
            # Per CLAUDE.md: "Error out rather than approximate when statistical correctness cannot be guaranteed"
            # With perfect collinearity, SE computation should fail (produce NaN)
            result = population_margins(model, df; type=:effects, vars=[:x1, :x2])
            result_df = DataFrame(result)
            
            # Verify that statistical invalidity is properly detected
            validation = validate_all_finite_positive(result_df)
            @test !validation.all_valid  # Should detect the invalid SEs
            @test !validation.finite_ses  # SEs should be non-finite (NaN/Inf)
            @test any(isnan.(result_df.se)) || any(isinf.(result_df.se))  # Should be NaN or Inf
            
            # This demonstrates ERROR-FIRST principle: invalid statistics are detected, not hidden
        end
        
        @testset "Extreme Value SE Robustness" begin
            # Test with extreme but valid values
            n = 300
            df = DataFrame(
                x = [randn(n÷2) .* 0.01; randn(n÷2) .* 100],  # Mix of very small and large values
                z = randn(n)
            )
            df.y = 0.001 * df.x + df.z + 0.1 * randn(n)  # Small coefficient
            
            model = lm(@formula(y ~ x + z), df)
            
            result = population_margins(model, df; type=:effects, vars=[:x, :z])
            result_df = DataFrame(result)
            
            @test validate_all_finite_positive(result_df).all_valid
            @test all(isfinite.(result_df.se))
            @test all(result_df.se .> 0)
            
            # Profile margins should also work
            prof_result = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x, :z])
            prof_df = DataFrame(prof_result)
            
            @test validate_all_finite_positive(prof_df).all_valid
            @test all(isfinite.(prof_df.se))
            @test all(prof_df.se .> 0)
        end
    end
end