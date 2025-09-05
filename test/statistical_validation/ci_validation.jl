# ci_validation.jl - Fast Statistical Validation for CI/CD
#
# This file provides a streamlined subset of statistical validation tests
# optimized for continuous integration. It focuses on the most critical
# statistical correctness tests while maintaining fast execution (< 30 seconds).
#
# For complete validation, use the full statistical_validation.jl suite.

using Test
using Random
using DataFrames
using CategoricalArrays
using GLM
using Statistics
using StatsModels
using Margins

# Load testing utilities
# Testing utilities loaded centrally in runtests.jl

@testset "CI Statistical Validation - Critical Subset" begin
    Random.seed!(06515)  
    
    # === TIER 1: CRITICAL MATHEMATICAL CORRECTNESS ===
    @testset "Tier 1: Core Mathematical Correctness (CI Critical)" begin
        
        @testset "Linear Model Coefficient Validation" begin
            # Essential test: Linear model coefficients must be exact
            df = make_simple_test_data(n=500, formula_type=:linear)
            model = lm(@formula(y ~ x + z), df)
            β₀, β₁, β₂ = coef(model)
            
            # Population effects must equal coefficients exactly
            pop_effects = population_margins(model, df; type=:effects, vars=[:x, :z], scale=:link)
            pop_df = DataFrame(pop_effects)
            @test pop_df.estimate[1] ≈ β₁ atol=1e-12
            @test pop_df.estimate[2] ≈ β₂ atol=1e-12
            @test validate_all_finite_positive(pop_df).all_valid
            
            # Profile effects must equal coefficients exactly
            profile_effects = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x, :z], scale=:link)
            prof_df = DataFrame(profile_effects)
            @test prof_df.estimate[1] ≈ β₁ atol=1e-12
            @test prof_df.estimate[2] ≈ β₂ atol=1e-12
            @test validate_all_finite_positive(prof_df).all_valid
            
            @info "✓ CI: Linear model coefficient validation passed"
        end
        
        @testset "GLM Chain Rule Validation" begin
            # Essential test: GLM chain rules must be mathematically correct
            df = make_glm_test_data(n=500, family=:binomial)
            model = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())
            β₁ = coef(model)[2]
            
            # Link scale must equal coefficient exactly
            effects_eta = population_margins(model, df; type=:effects, vars=[:x], scale=:link)
            eta_df = DataFrame(effects_eta)
            @test eta_df.estimate[1] ≈ β₁ atol=1e-12
            @test validate_all_finite_positive(eta_df).all_valid
            
            # Response scale must follow chain rule
            effects_mu = population_margins(model, df; type=:effects, vars=[:x], scale=:response)
            mu_df = DataFrame(effects_mu)
            fitted_probs = GLM.predict(model, df)
            manual_ame = mean(β₁ .* fitted_probs .* (1 .- fitted_probs))
            @test mu_df.estimate[1] ≈ manual_ame atol=1e-12
            @test validate_all_finite_positive(mu_df).all_valid
            
            @info "✓ CI: GLM chain rule validation passed"
        end
    end
    
    # === TIER 2: CRITICAL INTEGER VARIABLE SUPPORT ===
    @testset "Tier 2: Integer Variable Support (CI Critical)" begin
        df = make_econometric_data(n=400, seed=123)
        
        @testset "Simple Integer Variable" begin
            # Critical for econometric data
            model = lm(@formula(log_wage ~ int_age), df)
            β₁ = coef(model)[2]
            
            # Test all four quadrants work with integers
            pop_effects = population_margins(model, df; type=:effects, vars=[:int_age], scale=:link)
            @test DataFrame(pop_effects).estimate[1] ≈ β₁ atol=1e-12
            
            pop_pred = population_margins(model, df; type=:predictions)
            @test validate_all_finite_positive(DataFrame(pop_pred)).all_valid
            
            prof_effects = profile_margins(model, df, means_grid(df); type=:effects, vars=[:int_age], scale=:link)
            @test DataFrame(prof_effects).estimate[1] ≈ β₁ atol=1e-12
            
            prof_pred = profile_margins(model, df, means_grid(df); type=:predictions)
            @test validate_all_finite_positive(DataFrame(prof_pred)).all_valid
            
            @info "✓ CI: Integer variable support validated"
        end
        
        @testset "Integer × Categorical Interaction" begin
            # Critical mixed-type interaction
            model = lm(@formula(log_wage ~ int_age * gender), df)
            framework_result = test_2x2_framework_quadrants(model, df; test_name="Integer × Categorical")
            @test framework_result.all_successful
            @test framework_result.all_finite
            
            @info "✓ CI: Integer × categorical interaction validated"
        end
    end
    
    # === TIER 3: BACKEND CONSISTENCY (ESSENTIAL) ===
    @testset "Tier 3: Backend Consistency (CI Essential)" begin
        df = make_simple_test_data(n=300, formula_type=:linear, seed=456)
        model = lm(@formula(y ~ x + z), df)
        
        # Critical: AD and FD must produce identical results
        consistency_result = test_backend_consistency(model, df; 
                                                   rtol_estimate=1e-8, 
                                                   rtol_se=1e-6)
        @test consistency_result.all_consistent
        @test consistency_result.all_estimates_agree
        @test consistency_result.all_ses_agree
        
        @info "✓ CI: Backend consistency validated"
    end
    
    # === TIER 4: SYSTEMATIC MODEL SAMPLING ===
    @testset "Tier 4: Critical Model Types (CI Sampling)" begin
        df = make_econometric_data(n=300, seed=789)
        
        # Sample of most important model types for CI
        critical_models = [
            (name="Simple LM", model=lm(@formula(log_wage ~ float_wage), df)),
            (name="Multiple LM", model=lm(@formula(log_wage ~ float_wage + gender), df)),
            (name="Interaction LM", model=lm(@formula(log_wage ~ float_wage * gender), df)),
            (name="Simple Logistic", model=glm(@formula(union_member ~ float_wage), df, Binomial(), LogitLink())),
            (name="Logistic Interaction", model=glm(@formula(union_member ~ float_wage * gender), df, Binomial(), LogitLink())),
        ]
        
        for (name, model) in critical_models
            @testset "$(name) - Critical 2×2 Validation" begin
                framework_result = test_2x2_framework_quadrants(model, df; test_name=name)
                @test framework_result.all_successful
                @test framework_result.all_finite
                
                @info "✓ CI: $(name) validated"
            end
        end
    end
    
    @info "🚀 CI STATISTICAL VALIDATION: COMPLETE"
    @info "Critical statistical correctness verified for CI/CD pipeline"
    @info "✓ Mathematical correctness (coefficient validation)"
    @info "✓ Integer variable support (econometric data)"
    @info "✓ Backend consistency (computational reliability)"
    @info "✓ Core model types (systematic sampling)"
end