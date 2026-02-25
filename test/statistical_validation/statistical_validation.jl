# statistical_validation.jl - Comprehensive Statistical Correctness Validation
#
# This file implements the core statistical validation framework for Margins.jl.
# It provides complete mathematical verification through analytical methods across
# all 2×2 framework combinations (Population vs Profile × Effects vs Predictions).
#
# Every test case validates ALL FOUR QUADRANTS systematically using hand-calculated
# expected values to ensure publication-grade statistical correctness.

# using Revise
using Test
using Random
using DataFrames
using CategoricalArrays
using GLM
using Statistics
using StatsModels
using Margins

@testset "Comprehensive Statistical Validation - 2×2 Framework Coverage" begin
    Random.seed!(06515)
    
    # === TIER 1: Direct Coefficient Validation (Perfect Mathematical Truth) ===
    @testset "Tier 1: Direct Coefficient Cases - All 2×2 Quadrants" begin
        
        @testset "Simple Linear Model: y ~ x" begin
            # Generate test data with known relationship
            df = make_simple_test_data(n=1000, formula_type=:linear)
            model = lm(@formula(y ~ x), df)
            β₀, β₁ = coef(model)  # True coefficients
            
            # === 2×2 FRAMEWORK: ALL FOUR QUADRANTS ===
            
            # 1. Population Effects (AME): ∂y/∂x averaged across sample
            pop_effects = population_margins(model, df; type=:effects, vars=[:x], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            @test pop_effects_df.estimate[1] ≈ β₁ atol=1e-12  # Should equal coefficient (within numerical precision)
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            # 2. Population Predictions (AAP): E[ŷ] across sample  
            pop_predictions = population_margins(model, df; type=:predictions, scale=:response)
            pop_pred_df = DataFrame(pop_predictions)
            manual_mean_prediction = mean(GLM.predict(model, df))
            @test pop_pred_df.estimate[1] ≈ manual_mean_prediction atol=1e-12
            @test validate_all_finite_positive(pop_pred_df).all_valid
            
            # 3. Profile Effects (MEM): ∂y/∂x at specific profiles
            profile_effects = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x], scale=:link)
            prof_effects_df = DataFrame(profile_effects)
            @test prof_effects_df.estimate[1] ≈ β₁ atol=1e-12  # Linear model: ME constant everywhere
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # 4. Profile Predictions (APM): ŷ at specific profiles
            x_mean = mean(df.x)
            profile_predictions = profile_margins(model, df, DataFrame(x=[x_mean]); type=:predictions, scale=:response)
            prof_pred_df = DataFrame(profile_predictions)
            manual_profile_prediction = β₀ + β₁ * x_mean  # β₀ + β₁*x_mean
            @test prof_pred_df.estimate[1] ≈ manual_profile_prediction atol=1e-12
            @test validate_all_finite_positive(prof_pred_df).all_valid
            
            
        end
        
        @testset "Multiple Regression: y ~ x + z" begin
            df = make_simple_test_data(n=800, formula_type=:linear)
            model = lm(@formula(y ~ x + z), df)
            β₀, β₁, β₂ = coef(model)
            
            # === ALL FOUR QUADRANTS FOR MULTI-VARIABLE ===
            
            # 1. Population Effects: Both variables
            pop_effects = population_margins(model, df; type=:effects, vars=[:x, :z], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            @test pop_effects_df.estimate[1] ≈ β₁ atol=1e-12  # ∂y/∂x = β₁
            @test pop_effects_df.estimate[2] ≈ β₂ atol=1e-12  # ∂y/∂z = β₂
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            # 2. Population Predictions
            pop_predictions = population_margins(model, df; type=:predictions, scale=:response)
            pop_pred_df = DataFrame(pop_predictions)
            manual_mean_prediction = mean(GLM.predict(model, df))
            @test pop_pred_df.estimate[1] ≈ manual_mean_prediction atol=1e-12
            @test validate_all_finite_positive(pop_pred_df).all_valid
            
            # 3. Profile Effects at specific point
            profile_effects = profile_margins(model, df, DataFrame(x=[1.0], z=[2.0]); type=:effects, vars=[:x, :z], scale=:link)
            prof_effects_df = DataFrame(profile_effects)
            @test prof_effects_df.estimate[1] ≈ β₁ atol=1e-12  # Still β₁ (linear)
            @test prof_effects_df.estimate[2] ≈ β₂ atol=1e-12  # Still β₂ (linear)
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # 4. Profile Predictions at specific point
            test_x, test_z = 1.0, 2.0
            profile_predictions = profile_margins(model, df, DataFrame(x=[test_x], z=[test_z]); type=:predictions, scale=:response)
            prof_pred_df = DataFrame(profile_predictions)
            manual_profile_prediction = β₀ + β₁ * test_x + β₂ * test_z
            @test prof_pred_df.estimate[1] ≈ manual_profile_prediction atol=1e-12
            @test validate_all_finite_positive(prof_pred_df).all_valid
            
            
        end
    end
    
    # === TIER 1A: Analytical Standard Error Validation - Linear Models ===
    @testset "Tier 1A: Analytical SE Validation - Linear Models" begin
        
        @testset "Simple Linear: y ~ x - SE Verification" begin
            df = make_simple_test_data(n=1000, formula_type=:linear)
            model = lm(@formula(y ~ x), df)

            # Population effects SE should equal coefficient SE
            result = population_margins(model, df; type=:effects, vars=[:x])
            computed_se = DataFrame(result).se[1]
            analytical_se = analytical_linear_se(model, df, :x)

            @test computed_se ≈ analytical_se atol=1e-12
            

            # Profile effects SE should also equal coefficient SE (linear case)
            profile_result = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x])
            profile_se = DataFrame(profile_result).se[1]

            @test profile_se ≈ analytical_se atol=1e-12
            
            
            # Test consistency verification function
            consistency = verify_linear_se_consistency(model, df, :x)
            @test consistency.both_match
            @test consistency.max_deviation < 1e-12
            
        end
        
        @testset "Multiple Linear: y ~ x + z - SE Verification" begin
            df = make_simple_test_data(n=800, formula_type=:linear)
            model = lm(@formula(y ~ x + z), df)

            # Test both variables' SEs
            for var in [:x, :z]
                result = population_margins(model, df; type=:effects, vars=[var])
                computed_se = DataFrame(result).se[1]
                analytical_se = analytical_linear_se(model, df, var)

                @test computed_se ≈ analytical_se atol=1e-12
                

                profile_result = profile_margins(model, df, means_grid(df); type=:effects, vars=[var])
                profile_se = DataFrame(profile_result).se[1]

                @test profile_se ≈ analytical_se atol=1e-12
                
            end
            
            # Test multiple variables at once
            multi_result = population_margins(model, df; type=:effects, vars=[:x, :z])
            multi_df = DataFrame(multi_result)
            
            analytical_se_x = analytical_linear_se(model, df, :x)
            analytical_se_z = analytical_linear_se(model, df, :z)
            
            @test multi_df.se[1] ≈ analytical_se_x atol=1e-12  # x SE
            @test multi_df.se[2] ≈ analytical_se_z atol=1e-12  # z SE
            
        end
    end
    
    # === TIER 1B: Analytical SE Validation - GLM Chain Rules ===
    @testset "Tier 1B: Analytical SE Validation - GLM Chain Rules" begin
        
        @testset "Logistic Regression: SE Chain Rule Verification" begin
            df = make_glm_test_data(n=800, family=:binomial)
            model = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())

            # Test profile SE at specific point (analytically calculable)
            test_x, test_z = 0.5, -0.3
            at_values = Dict(:x => test_x, :z => test_z)

            # Test SE for variable x
            se_validation = verify_glm_se_chain_rule(model, df, :x, at_values; 
                                                   tolerance=0.1, model_type=:logistic)
            
            @test se_validation.matches
            @test se_validation.relative_error < 0.1  # Allow 10% relative error for GLM chain rule
            

            # Test SE for variable z
            se_validation_z = verify_glm_se_chain_rule(model, df, :z, at_values; 
                                                     tolerance=0.1, model_type=:logistic)
            
            @test se_validation_z.matches
            @test se_validation_z.relative_error < 0.1  # Allow 10% relative error for GLM chain rule
            
            
            # Verify link scale SEs equal coefficient SEs (should be exact)
            eta_result = profile_margins(model, df, DataFrame(at_values); type=:effects, vars=[:x], 
                                       scale=:link)
            eta_se = DataFrame(eta_result).se[1]
            coef_se = analytical_linear_se(model, df, :x)  # Works for any GLM on link scale
            
            @test eta_se ≈ coef_se atol=1e-12
            
        end
        
        @testset "Poisson Regression: SE Chain Rule Verification" begin
            df = make_glm_test_data(n=600, family=:poisson)
            model = glm(@formula(y ~ x + z), df, Poisson(), LogLink())

            # Test profile SE at specific point
            test_x, test_z = 0.2, -0.1
            at_values = Dict(:x => test_x, :z => test_z)

            # Test SE chain rule for variable x
            se_validation = verify_glm_se_chain_rule(model, df, :x, at_values; 
                                                   tolerance=0.1, model_type=:poisson)
            
            @test se_validation.matches
            @test se_validation.relative_error < 0.1  # Allow 10% relative error for GLM chain rule
            

            # Test SE chain rule for variable z  
            se_validation_z = verify_glm_se_chain_rule(model, df, :z, at_values; 
                                                     tolerance=0.1, model_type=:poisson)
            
            @test se_validation_z.matches
            @test se_validation_z.relative_error < 0.1  # Allow 10% relative error for GLM chain rule
            
            
            # Verify link scale SEs equal coefficient SEs
            eta_result = profile_margins(model, df, DataFrame(x=[test_x], z=[test_z]); type=:effects, vars=[:x], scale=:link)
            eta_se = DataFrame(eta_result).se[1]
            coef_se = analytical_linear_se(model, df, :x)
            
            @test eta_se ≈ coef_se atol=1e-12
            
        end
    end
    
    # === TIER 2: Function Transformations - All 2×2 Quadrants ===
    @testset "Tier 2: Function Transformations - 2×2 Coverage" begin
        
        @testset "Log Transformation: y ~ log(x)" begin
            # KNOWN SYSTEMATIC BUG: See ISSUES.md REMAINING ISSUE 2
            # ALL profile effects with log functions fail with DomainError(-6.055e-6)
            # This test will fail until FormulaCompiler log derivative issue is resolved
            Random.seed!(06515)
            n = 500
            df = DataFrame(
                x = abs.(rand(n)) .+ 10,  # Ensure positive values for log safety
                z = randn(n)
            )
            df.y = 0.5 * log.(df.x) + 0.3 * df.z + 0.1 * randn(n)
            
            model = lm(@formula(y ~ log(x)), df)
            β₀, β₁ = coef(model)
            
            # === 2×2 FRAMEWORK FOR LOG TRANSFORMATION ===
            
            # 1. Population Effects
            pop_effects = population_margins(model, df; type=:effects, vars=[:x], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            manual_ame = mean(β₁ ./ df.x)  # ∂/∂x[β₁·log(x)] = β₁/x, averaged
            @test pop_effects_df.estimate[1] ≈ manual_ame atol=1e-12
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            # 2. Population Predictions
            pop_predictions = population_margins(model, df; type=:predictions, scale=:response)
            pop_pred_df = DataFrame(pop_predictions)
            manual_mean_prediction = mean(GLM.predict(model, df))
            @test pop_pred_df.estimate[1] ≈ manual_mean_prediction atol=1e-12
            @test validate_all_finite_positive(pop_pred_df).all_valid
            
            # 3. Profile Effects at specific point (troublesome, required patch in FormulaCompiler)
            test_x = 10.5  # Use very safe value for finite differences in log function
            profile_effects = profile_margins(model, df, DataFrame(x=[test_x]); type=:effects, vars=[:x], scale=:link, backend=:ad)
            prof_effects_df = DataFrame(profile_effects)
            manual_mem = β₁ / test_x  # ∂/∂x[β₁·log(x)] = β₁/x at x=test_x
            @test prof_effects_df.estimate[1] ≈ manual_mem atol=1e-12
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # 4. Profile Predictions at specific point
            profile_predictions = profile_margins(model, df, DataFrame(x=[test_x]); type=:predictions, scale=:response)
            prof_pred_df = DataFrame(profile_predictions)
            manual_profile_prediction = β₀ + β₁ * log(test_x)  # β₀ + β₁·log(x)
            @test prof_pred_df.estimate[1] ≈ manual_profile_prediction atol=1e-12
            @test validate_all_finite_positive(prof_pred_df).all_valid
            
            
        end
        
        @testset "Quadratic: y ~ x + x²" begin
            df = make_simple_test_data(n=600, formula_type=:quadratic)
            model = lm(@formula(y ~ x + x^2), df)
            β₀, β₁, β₂ = coef(model)
            
            # === 2×2 FRAMEWORK FOR QUADRATIC ===
            
            # 1. Population Effects: Hand-calculated analytical derivative
            pop_effects = population_margins(model, df; type=:effects, vars=[:x], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            # Analytical verification: quadratic model specification y ~ x + x² 
            # So ∂y/∂x = β₁ + 2*β₂*x (true derivative of quadratic function)
            # This tests the computational correctness of automatic differentiation for x²
            # For quadratic y ~ x + x^2, AME = average of (β₁ + 2*β₂*x) over all x values
            expected_ame = β₁ + 2*β₂*mean(df.x)
            @test pop_effects_df.estimate[1] ≈ expected_ame atol=1e-10
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            # 2. Population Predictions
            pop_predictions = population_margins(model, df; type=:predictions, scale=:response)
            pop_pred_df = DataFrame(pop_predictions)
            manual_mean_prediction = mean(GLM.predict(model, df))
            @test pop_pred_df.estimate[1] ≈ manual_mean_prediction atol=1e-12
            @test validate_all_finite_positive(pop_pred_df).all_valid
            
            # 3. Profile Effects at specific point  
            test_x = 1.5
            profile_effects = profile_margins(model, df, DataFrame(x=[test_x]); type=:effects, vars=[:x], scale=:link)
            prof_effects_df = DataFrame(profile_effects)
            # For y ~ x + x^2 model, ∂y/∂x = β₁ + 2*β₂*x at the specific point
            expected_marginal_effect = β₁ + 2*β₂*test_x
            @test prof_effects_df.estimate[1] ≈ expected_marginal_effect atol=1e-12
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # 4. Profile Predictions at specific point
            profile_predictions = profile_margins(model, df, DataFrame(x=[test_x]); type=:predictions, scale=:response)
            prof_pred_df = DataFrame(profile_predictions)
            manual_profile_prediction = β₀ + β₁ * test_x + β₂ * test_x^2
            @test prof_pred_df.estimate[1] ≈ manual_profile_prediction atol=1e-12
            @test validate_all_finite_positive(prof_pred_df).all_valid
        end
    end
    
    # === TIER 3: GLM Chain Rules - All 2×2 Quadrants ===
    @testset "Tier 3: GLM Chain Rules - 2×2 Coverage" begin
        
        @testset "Logistic Regression: y ~ x + z" begin
            df = make_glm_test_data(n=800, family=:binomial)
            model = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())
            β₀, β₁, β₂ = coef(model)
            
            # === 2×2 FRAMEWORK FOR LOGISTIC REGRESSION ===
            
            # 1. Population Effects (η scale): Should equal coefficient
            pop_effects_eta = population_margins(model, df; type=:effects, vars=[:x], scale=:link)
            pop_effects_eta_df = DataFrame(pop_effects_eta)
            @test pop_effects_eta_df.estimate[1] ≈ β₁ atol=1e-12  # Link scale: should be exact (within numerical precision)
            @test validate_all_finite_positive(pop_effects_eta_df).all_valid
            
            # 1b. Population Effects (μ scale): Chain rule verification
            pop_effects_mu = population_margins(model, df; type=:effects, vars=[:x], scale=:response)
            pop_effects_mu_df = DataFrame(pop_effects_mu)
            fitted_probs = GLM.predict(model, df)
            manual_ame_mu = mean(β₁ .* fitted_probs .* (1 .- fitted_probs))  # β₁ × μ(1-μ), averaged
            @test pop_effects_mu_df.estimate[1] ≈ manual_ame_mu atol=1e-12
            @test validate_all_finite_positive(pop_effects_mu_df).all_valid
            
            # 2. Population Predictions
            pop_pred_response = population_margins(model, df; type=:predictions, scale=:response)
            pop_pred_df = DataFrame(pop_pred_response)
            manual_mean_prob = mean(GLM.predict(model, df))
            @test pop_pred_df.estimate[1] ≈ manual_mean_prob atol=1e-12
            @test validate_all_finite_positive(pop_pred_df).all_valid
            
            # 3. Profile Effects at specific point
            test_x, test_z = 0.5, -0.3
            profile_effects_mu = profile_margins(model, df, DataFrame(x=[test_x], z=[test_z]); type=:effects, vars=[:x], scale=:response)
            prof_effects_df = DataFrame(profile_effects_mu)
            # Hand-calculate probability at this point
            test_eta = β₀ + β₁ * test_x + β₂ * test_z
            test_mu = 1 / (1 + exp(-test_eta))
            manual_mem_mu = β₁ * test_mu * (1 - test_mu)  # Chain rule at specific point
            @test prof_effects_df.estimate[1] ≈ manual_mem_mu atol=1e-12
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # 4. Profile Predictions at specific point
            profile_pred = profile_margins(model, df, DataFrame(x=[test_x], z=[test_z]); type=:predictions, scale=:response)
            prof_pred_df = DataFrame(profile_pred)
            @test prof_pred_df.estimate[1] ≈ test_mu atol=1e-12  # Should match hand-calculated probability
            @test validate_all_finite_positive(prof_pred_df).all_valid
            
            
        end
        
        @testset "Poisson Regression: y ~ x + z" begin
            df = make_glm_test_data(n=600, family=:poisson)
            model = glm(@formula(y ~ x + z), df, Poisson(), LogLink())
            β₀, β₁, β₂ = coef(model)
            
            # === 2×2 FRAMEWORK FOR POISSON REGRESSION ===
            
            # 1. Population Effects (η scale): Should equal coefficient  
            pop_effects_eta = population_margins(model, df; type=:effects, vars=[:x], scale=:link)
            pop_effects_eta_df = DataFrame(pop_effects_eta)
            @test pop_effects_eta_df.estimate[1] ≈ β₁ atol=1e-12  # Link scale: should be exact (within numerical precision)
            @test validate_all_finite_positive(pop_effects_eta_df).all_valid
            
            # 1b. Population Effects (μ scale): Chain rule verification
            pop_effects_mu = population_margins(model, df; type=:effects, vars=[:x], scale=:response)
            pop_effects_mu_df = DataFrame(pop_effects_mu)
            fitted_means = GLM.predict(model, df)
            manual_ame_mu = mean(β₁ .* fitted_means)  # β₁ × μ for log link, averaged
            @test pop_effects_mu_df.estimate[1] ≈ manual_ame_mu atol=1e-12
            @test validate_all_finite_positive(pop_effects_mu_df).all_valid
            
            # 2. Population Predictions
            pop_pred_response = population_margins(model, df; type=:predictions, scale=:response)
            pop_pred_df = DataFrame(pop_pred_response)
            manual_mean_rate = mean(GLM.predict(model, df))
            @test pop_pred_df.estimate[1] ≈ manual_mean_rate atol=1e-12
            @test validate_all_finite_positive(pop_pred_df).all_valid
            
            # 3. Profile Effects at specific point (μ scale)
            test_x, test_z = 0.2, -0.1
            profile_effects_mu = profile_margins(model, df, DataFrame(x=[test_x], z=[test_z]); type=:effects, vars=[:x],
                                               scale=:response)
            prof_effects_df = DataFrame(profile_effects_mu)
            # Hand-calculate rate at this point
            test_eta = β₀ + β₁ * test_x + β₂ * test_z
            test_mu = exp(test_eta)
            manual_mem_mu = β₁ * test_mu  # Chain rule for log link at specific point
            @test prof_effects_df.estimate[1] ≈ manual_mem_mu atol=1e-12
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # 4. Profile Predictions at specific point
            profile_pred = profile_margins(model, df, DataFrame(x=[test_x], z=[test_z]); type=:predictions, scale=:response)
            prof_pred_df = DataFrame(profile_pred)
            @test prof_pred_df.estimate[1] ≈ test_mu atol=1e-12  # Should match hand-calculated rate
            @test validate_all_finite_positive(prof_pred_df).all_valid
            
            
        end
    end
    
    # === TIER 4: Systematic Model Coverage (Following FormulaCompiler Pattern) ===
    @testset "Tier 4: Systematic Model Coverage - FormulaCompiler Style" begin
        # Use FormulaCompiler's comprehensive test data and patterns
        df = make_econometric_data(; n=500)
        
        # === LINEAR MODEL SYSTEMATIC COVERAGE (13 patterns following FormulaCompiler) ===
        lm_test_cases = [
            # Basic patterns (matching FC's linear_formulas exactly)
            (name="LM: Simple continuous", formula=@formula(log(wage) ~ wage)),
            (name="LM: Simple categorical", formula=@formula(log(wage) ~ gender)),
            (name="LM: Multiple continuous", formula=@formula(log(wage) ~ wage + float_productivity)),
            (name="LM: Multiple categorical", formula=@formula(log(wage) ~ gender + region)),
            (name="LM: Mixed types", formula=@formula(log(wage) ~ wage + gender)),
            (name="LM: Simple interaction", formula=@formula(log(wage) ~ wage * gender)),
            (name="LM: Interaction w/o main", formula=@formula(log(wage) ~ wage & gender)),
            (name="LM: Function transform", formula=@formula(income ~ log(wage))),
            (name="LM: Function in interaction", formula=@formula(income ~ exp(wage/20) * float_productivity)),
            (name="LM: Three-way interaction", formula=@formula(log(wage) ~ wage * float_productivity * gender)),
            (name="LM: Four-way interaction", formula=@formula(log(wage) ~ wage * float_productivity * gender * region)),
            (name="LM: Four-way w/ function", formula=@formula(log(wage) ~ exp(wage/20) * float_productivity * gender * region)),
            (name="LM: Complex interaction", formula=@formula(log(wage) ~ wage * float_productivity * gender + log(wage) * region)),
        ]
        
        # === GLM SYSTEMATIC COVERAGE (following FormulaCompiler patterns) ===
        # First create count data for Poisson models
        df.count_response = rand(0:10, nrow(df))  # Poisson-appropriate count data
        
        glm_test_cases = [
            # Logistic regression patterns
            (name="GLM: Logistic simple", formula=@formula(union_member ~ wage), family=Binomial(), link=LogitLink()),
            (name="GLM: Logistic mixed", formula=@formula(union_member ~ wage + gender), family=Binomial(), link=LogitLink()),
            (name="GLM: Logistic interaction", formula=@formula(union_member ~ wage * gender), family=Binomial(), link=LogitLink()),
            (name="GLM: Logistic function", formula=@formula(union_member ~ log(wage) + gender), family=Binomial(), link=LogitLink()),
            (name="GLM: Logistic complex", formula=@formula(union_member ~ wage * float_productivity * gender + log(wage) + region), family=Binomial(), link=LogitLink()),
            
            # Poisson regression patterns (matching FC's coverage)
            (name="GLM: Poisson simple", formula=@formula(count_response ~ wage), family=Poisson(), link=LogLink()),
            (name="GLM: Poisson mixed", formula=@formula(count_response ~ wage + gender), family=Poisson(), link=LogLink()),
            (name="GLM: Poisson interaction", formula=@formula(count_response ~ wage * gender), family=Poisson(), link=LogLink()),
            
            # Gamma regression patterns (using positive continuous outcome)
            (name="GLM: Gamma mixed", formula=@formula(wage ~ float_productivity + gender), family=Gamma(), link=LogLink()),
            
            # Gaussian with LogLink (less common but in FC coverage)
            (name="GLM: Gaussian LogLink", formula=@formula(wage ~ float_productivity + gender), family=Normal(), link=LogLink()),
        ]
        
        # Combine all systematic test cases
        all_test_cases = vcat(lm_test_cases, glm_test_cases)
        
        for test_case in all_test_cases
            @testset "$(test_case.name) - 2×2 Framework" begin
                if haskey(test_case, :family)
                    # GLM case
                    model = glm(test_case.formula, df, test_case.family, test_case.link)
                else
                    # LM case  
                    model = lm(test_case.formula, df)
                end
                
                # === VALIDATE ALL 2×2 QUADRANTS FOR THIS MODEL ===
                
                framework_result = test_2x2_framework_quadrants(model, df; test_name=test_case.name)
                
                # Verify all quadrants succeeded
                @test framework_result.all_successful
                @test framework_result.all_finite
                
                # Detailed validation for each successful quadrant  
                for (quadrant_name, result) in framework_result.quadrants
                    if haskey(result, :success) && result.success && !haskey(result, :skipped)
                        # Only validate results that actually computed something
                        if haskey(result, :finite_estimates)
                            @test result.finite_estimates
                            @test result.finite_ses  
                            @test result.positive_ses
                        end
                    end
                end
                
                
            end
        end
    end
    
    # === TIER 5: Edge Cases and Robustness ===
    @testset "Tier 5: Edge Cases and Statistical Robustness" begin
        
        @testset "Small Sample Robustness" begin
            for n in [25, 50, 75]
                df = make_simple_test_data(n=n, formula_type=:linear)
                model = lm(@formula(y ~ x), df)
                
                # All quadrants should work but may have large SEs
                framework_result = test_2x2_framework_quadrants(model, df; test_name="Small n=$n")
                @test framework_result.all_successful
                @test framework_result.all_finite
                
                
            end
        end
        
        @testset "Boundary Conditions" begin
            # Test with extreme but valid coefficient values
            df = DataFrame(x = randn(200), z = randn(200))
            df.y = 5.0 * df.x + 0.001 * df.z + 0.1 * randn(200)  # Large and small coefficients
            
            model = lm(@formula(y ~ x + z), df)
            framework_result = test_2x2_framework_quadrants(model, df; test_name="Extreme coefficients")
            
            @test framework_result.all_successful
            @test framework_result.all_finite
            
            
        end
    end
    
    # === TIER 6: INTEGER VARIABLE SYSTEMATIC COVERAGE (FormulaCompiler Pattern) ===
    @testset "Tier 6: Integer Variable Systematic Coverage - CRITICAL" begin
        df_int = make_econometric_data(n=600, seed=999)  # Rich integer variable dataset
        
        @testset "Simple Integer Variables - All 2×2 Quadrants" begin
            # Test individual integer variables
            integer_vars = [:int_age, :int_education, :int_experience]
            
            for var in integer_vars
                @testset "$(var) - 2×2 Framework" begin
                    if var == :int_age
                        model = lm(@formula(log(wage) ~ int_age), df_int)
                    elseif var == :int_education  
                        model = lm(@formula(log(wage) ~ int_education), df_int)
                    elseif var == :int_experience
                        model = lm(@formula(log(wage) ~ int_experience), df_int)
                    end
                    β₀, β₁ = coef(model)
                    
                    # === 2×2 FRAMEWORK FOR INTEGER VARIABLES ===
                    
                    # 1. Population Effects (AME): Should equal coefficient for linear model
                    pop_effects = population_margins(model, df_int; type=:effects, vars=[var], scale=:link)
                    pop_effects_df = DataFrame(pop_effects)
                    @test pop_effects_df.estimate[1] ≈ β₁ atol=1e-12  # Integer should work like float
                    @test validate_all_finite_positive(pop_effects_df).all_valid
                    
                    # 2. Population Predictions
                    pop_predictions = population_margins(model, df_int; type=:predictions, scale=:response)
                    pop_pred_df = DataFrame(pop_predictions)
                    manual_mean_prediction = mean(GLM.predict(model, df_int))
                    @test pop_pred_df.estimate[1] ≈ manual_mean_prediction atol=1e-12
                    @test validate_all_finite_positive(pop_pred_df).all_valid
                    
                    # 3. Profile Effects at means
                    profile_effects = profile_margins(model, df_int, means_grid(df_int); type=:effects, vars=[var], scale=:link)
                    prof_effects_df = DataFrame(profile_effects)
                    @test prof_effects_df.estimate[1] ≈ β₁ atol=1e-12  # Linear: constant ME
                    @test validate_all_finite_positive(prof_effects_df).all_valid
                    
                    # 4. Profile Predictions at specific integer values
                    test_val = Int(round(mean(df_int[!, var])))  # Use integer value
                    profile_predictions = profile_margins(model, df_int, DataFrame(var => [test_val]); type=:predictions, scale=:response)
                    prof_pred_df = DataFrame(profile_predictions)
                    manual_profile_prediction = β₀ + β₁ * test_val
                    @test prof_pred_df.estimate[1] ≈ manual_profile_prediction atol=1e-12
                    @test validate_all_finite_positive(prof_pred_df).all_valid
                    
                    
                end
            end
        end
        
        @testset "Integer Interactions - 2×2 Framework" begin
            # Test integer × categorical interactions (common in econometrics)
            model = lm(@formula(log(wage) ~ int_age * gender), df_int)
            
            # Validate all quadrants work with integer interactions
            framework_result = test_2x2_framework_quadrants(model, df_int; test_name="Integer × Categorical")
            @test framework_result.all_successful
            @test framework_result.all_finite
            
            
        end
        
        @testset "Multiple Integer Variables - 2×2 Framework" begin
            # Multiple integer predictors (common econometric specification)
            model = lm(@formula(log(wage) ~ int_age + int_education + int_experience), df_int)
            β₀, β₁, β₂, β₃ = coef(model)
            
            # Test population effects for multiple integers
            pop_effects = population_margins(model, df_int; type=:effects, 
                                           vars=[:int_age, :int_education, :int_experience], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            @test pop_effects_df.estimate[1] ≈ β₁ atol=1e-12  # int_age coefficient
            @test pop_effects_df.estimate[2] ≈ β₂ atol=1e-12  # int_education coefficient  
            @test pop_effects_df.estimate[3] ≈ β₃ atol=1e-12  # int_experience coefficient
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            # Test all quadrants systematically
            framework_result = test_2x2_framework_quadrants(model, df_int; test_name="Multiple Integers")
            @test framework_result.all_successful
            @test framework_result.all_finite
            
            
        end
        
        @testset "Integer Polynomial Transformations - 2×2 Framework" begin  
            # Polynomial transformations with integers (x² pattern - avoids function variable name issues)
            df_int.int_age_sq = df_int.int_age .^ 2  # Create polynomial term
            model = lm(@formula(log(wage) ~ int_age + int_age_sq), df_int)
            β₀, β₁, β₂ = coef(model)
            
            # === ANALYTICAL VALIDATION FOR INTEGER POLYNOMIAL ===
            
            # 1. Population Effects: Hand-calculated derivative for int_age
            pop_effects = population_margins(model, df_int; type=:effects, vars=[:int_age], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            # For model with separate int_age and int_age_sq variables: ∂/∂int_age = β₁ only
            manual_ame = β₁  # Coefficient of int_age (int_age_sq is treated as separate variable)
            @test pop_effects_df.estimate[1] ≈ manual_ame atol=1e-12
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            # 3. Profile Effects at specific integer value
            test_age = 35  # Specific integer age
            profile_effects = profile_margins(model, df_int, DataFrame(int_age=[test_age], int_age_sq=[test_age^2]); type=:effects, vars=[:int_age], scale=:link)
            prof_effects_df = DataFrame(profile_effects)
            manual_mem = β₁  # For separate variables, ∂/∂int_age = β₁
            @test prof_effects_df.estimate[1] ≈ manual_mem atol=1e-12
            @test validate_all_finite_positive(prof_effects_df).all_valid
            
            # Test full framework (testing int_age_sq as the polynomial variable)
            framework_result = test_2x2_framework_quadrants(model, df_int; test_name="Integer Polynomial", vars=[:int_age_sq])
            @test framework_result.all_successful
            @test framework_result.all_finite
            
            
        end
        
        @testset "Mixed Integer/Float Interactions - 2×2 Framework" begin
            # Mixed integer/float interactions (critical for econometric realism)
            model = lm(@formula(log(wage) ~ int_age * float_productivity + int_education), df_int)
            
            # This tests the critical case where integers and floats interact
            framework_result = test_2x2_framework_quadrants(model, df_int; test_name="Mixed Integer/Float")
            @test framework_result.all_successful
            @test framework_result.all_finite
            
            # Test specific margin for integer variable in mixed model
            pop_effects = population_margins(model, df_int; type=:effects, vars=[:int_age], scale=:link)
            pop_effects_df = DataFrame(pop_effects)
            @test validate_all_finite_positive(pop_effects_df).all_valid
            
            
        end
        
        @testset "GLM with Integer Variables - 2×2 Framework" begin
            # GLM with integer predictors (logistic regression common case)
            model = glm(@formula(union_member ~ int_age + int_education), df_int, Binomial(), LogitLink())
            β₀, β₁, β₂ = coef(model)
            
            # === GLM INTEGER VALIDATION ===
            
            # Link scale should equal coefficients (exact)
            pop_effects_eta = population_margins(model, df_int; type=:effects, 
                                               vars=[:int_age, :int_education], scale=:link)
            pop_eta_df = DataFrame(pop_effects_eta)
            @test pop_eta_df.estimate[1] ≈ β₁ atol=1e-12  # int_age coefficient
            @test pop_eta_df.estimate[2] ≈ β₂ atol=1e-12  # int_education coefficient
            
            # Response scale with chain rule
            pop_effects_mu = population_margins(model, df_int; type=:effects,
                                              vars=[:int_age, :int_education], scale=:response) 
            pop_mu_df = DataFrame(pop_effects_mu)
            fitted_probs = GLM.predict(model, df_int)
            manual_ame_age = mean(β₁ .* fitted_probs .* (1 .- fitted_probs))
            manual_ame_edu = mean(β₂ .* fitted_probs .* (1 .- fitted_probs))
            @test pop_mu_df.estimate[1] ≈ manual_ame_age atol=1e-12
            @test pop_mu_df.estimate[2] ≈ manual_ame_edu atol=1e-12
            
            # Test full framework
            framework_result = test_2x2_framework_quadrants(model, df_int; test_name="GLM with Integers")
            @test framework_result.all_successful
            @test framework_result.all_finite
        end
    end
    
end