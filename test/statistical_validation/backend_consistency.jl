# backend_consistency.jl - AD vs FD Backend Consistency Validation
#
# This file validates computational consistency between Automatic Differentiation (AD)
# and Finite Differences (FD) backends across all 2×2 framework quadrants.
# Following FormulaCompiler's backend consistency testing patterns.
#
# CRITICAL: This validation ensures that different computational approaches produce
# statistically equivalent results, verifying implementation correctness.

using Test
using Random
using DataFrames
using GLM
using Statistics
using Margins

# Load testing utilities
include("testing_utilities.jl")

@testset "Backend Consistency Tests - AD vs FD Validation" begin
    Random.seed!(789)  # Different seed from main validation for independence
    
    # === ESSENTIAL BACKEND CONSISTENCY ACROSS ALL 2×2 QUADRANTS ===
    @testset "AD vs FD Computational Consistency" begin
        df = make_econometric_data(; n=500, seed=789)
        
        # Test across different model complexities (FormulaCompiler pattern)
        test_models = [
            (name="Simple LM", model=lm(@formula(log_wage ~ int_education), df)),
            (name="Multiple LM", model=lm(@formula(log_wage ~ int_education + int_experience), df)),
            (name="Quadratic LM", model=lm(@formula(log_wage ~ int_education + experience_sq), df)),
            (name="Simple Logistic", model=glm(@formula(union_member ~ int_education), df, Binomial(), LogitLink())),
            (name="Multiple Logistic", model=glm(@formula(union_member ~ int_education + int_experience), df, Binomial(), LogitLink())),
        ]
        
        for (name, model) in test_models
            @testset "$(name) - Full 2×2 Backend Consistency" begin
                # Use the comprehensive backend consistency test from utilities
                consistency_result = test_backend_consistency(model, df; 
                                                           rtol_estimate=1e-8,
                                                           rtol_se=1e-6)
                
                @test consistency_result.all_consistent
                @test consistency_result.all_estimates_agree
                @test consistency_result.all_ses_agree
                @test consistency_result.n_successful == consistency_result.n_total
                
                # Detailed validation for each quadrant
                for (quadrant_name, result) in consistency_result.quadrants
                    if haskey(result, :success) && result.success
                        @test result.estimates_agree
                        @test result.ses_agree
                        
                        # Log maximum differences for monitoring
                        if result.max_estimate_diff > 1e-12
                            @info "$(name) $(quadrant_name): Max estimate diff = $(result.max_estimate_diff)"
                        end
                        if result.max_se_diff > 1e-10  
                            @info "$(name) $(quadrant_name): Max SE diff = $(result.max_se_diff)"
                        end
                    else
                        @error "$(quadrant_name) failed for $name: $(get(result, :error, "Unknown error"))"
                    end
                end
            end
        end
    end
    
    # === SPECIFIC QUADRANT DETAILED TESTING ===
    @testset "Individual Quadrant Backend Consistency" begin
        df = make_simple_test_data(n=400, formula_type=:linear, seed=123)
        model = lm(@formula(y ~ x + z), df)
        
        @testset "Population Effects - AD vs FD" begin
            # Test with explicit backend specification
            ad_result = population_margins(model, df; type=:effects, vars=[:x, :z], backend=:ad)
            fd_result = population_margins(model, df; type=:effects, vars=[:x, :z], backend=:fd)
            
            ad_df = DataFrame(ad_result)
            fd_df = DataFrame(fd_result)
            
            @test all(isapprox.(ad_df.estimate, fd_df.estimate; rtol=1e-10))
            @test all(isapprox.(ad_df.se, fd_df.se; rtol=1e-8))
            
            # Test that both produce valid statistical results
            @test validate_all_finite_positive(ad_df).all_valid
            @test validate_all_finite_positive(fd_df).all_valid
        end
        
        @testset "Population Predictions - AD vs FD" begin
            ad_result = population_margins(model, df; type=:predictions, backend=:ad)
            fd_result = population_margins(model, df; type=:predictions, backend=:fd)
            
            ad_df = DataFrame(ad_result)
            fd_df = DataFrame(fd_result)
            
            @test all(isapprox.(ad_df.estimate, fd_df.estimate; rtol=1e-10))
            @test all(isapprox.(ad_df.se, fd_df.se; rtol=1e-8))
            
            @test validate_all_finite_positive(ad_df).all_valid
            @test validate_all_finite_positive(fd_df).all_valid
        end
        
        @testset "Profile Effects - AD vs FD" begin
            ad_result = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x, :z], backend=:ad)
            fd_result = profile_margins(model, df, means_grid(df); type=:effects, vars=[:x, :z], backend=:fd)
            
            ad_df = DataFrame(ad_result)
            fd_df = DataFrame(fd_result)
            
            @test all(isapprox.(ad_df.estimate, fd_df.estimate; rtol=1e-10))
            @test all(isapprox.(ad_df.se, fd_df.se; rtol=1e-8))
            
            @test validate_all_finite_positive(ad_df).all_valid
            @test validate_all_finite_positive(fd_df).all_valid
        end
        
        @testset "Profile Predictions - AD vs FD" begin
            ad_result = profile_margins(model, df, means_grid(df); type=:predictions, backend=:ad)
            fd_result = profile_margins(model, df, means_grid(df); type=:predictions, backend=:fd)
            
            ad_df = DataFrame(ad_result)
            fd_df = DataFrame(fd_result)
            
            @test all(isapprox.(ad_df.estimate, fd_df.estimate; rtol=1e-10))
            @test all(isapprox.(ad_df.se, fd_df.se; rtol=1e-8))
            
            @test validate_all_finite_positive(ad_df).all_valid
            @test validate_all_finite_positive(fd_df).all_valid
        end
    end
    
    # === GLM CHAIN RULE CONSISTENCY ===
    @testset "GLM Chain Rule Consistency - AD vs FD" begin
        df = make_glm_test_data(n=600, family=:binomial, seed=456)
        model = glm(@formula(y ~ x + z), df, Binomial(), LogitLink())
        
        @testset "Logistic Model - Both Target Scales" begin
            # Link scale (η) - should be identical (no chain rule)
            ad_eta = population_margins(model, df; type=:effects, vars=[:x], scale=:link, backend=:ad)
            fd_eta = population_margins(model, df; type=:effects, vars=[:x], scale=:link, backend=:fd)
            
            ad_eta_df = DataFrame(ad_eta)
            fd_eta_df = DataFrame(fd_eta)
            
            @test all(isapprox.(ad_eta_df.estimate, fd_eta_df.estimate; rtol=1e-12))
            @test all(isapprox.(ad_eta_df.se, fd_eta_df.se; rtol=1e-10))
            
            # Response scale (μ) - chain rule involved, still should be consistent
            ad_mu = population_margins(model, df; type=:effects, vars=[:x], scale=:response, backend=:ad)
            fd_mu = population_margins(model, df; type=:effects, vars=[:x], scale=:response, backend=:fd)
            
            ad_mu_df = DataFrame(ad_mu)
            fd_mu_df = DataFrame(fd_mu)
            
            @test all(isapprox.(ad_mu_df.estimate, fd_mu_df.estimate; rtol=1e-8))
            @test all(isapprox.(ad_mu_df.se, fd_mu_df.se; rtol=1e-6))
        end
    end
    
    # === PERFORMANCE CONSISTENCY (Following FormulaCompiler Pattern) ===
    @testset "Performance Characteristics - FD Zero Allocation" begin
        df = make_econometric_data(n=300, seed=999)
        model = lm(@formula(log_wage ~ int_education + int_experience), df)
        
        # Warm-up to ensure compilation (FormulaCompiler pattern)
        warmup_result = population_margins(model, df; backend=:fd, vars=[:int_education])
        @test validate_all_finite_positive(DataFrame(warmup_result)).all_valid
        
        # Test that FD backend achieves near-zero allocations for population margins
        # Note: We can't easily test allocations here without BenchmarkTools,
        # but we can verify the computational correctness holds across sample sizes
        
        @testset "Scaling Consistency" begin
            for n in [100, 200, 500]
                df_scaled = make_econometric_data(n=n, seed=n)
                model_scaled = lm(@formula(log_wage ~ int_education), df_scaled)
                
                # Both backends should produce consistent results regardless of scale
                ad_scaled = population_margins(model_scaled, df_scaled; type=:effects, vars=[:int_education], backend=:ad)
                fd_scaled = population_margins(model_scaled, df_scaled; type=:effects, vars=[:int_education], backend=:fd)
                
                ad_scaled_df = DataFrame(ad_scaled)
                fd_scaled_df = DataFrame(fd_scaled)
                
                @test all(isapprox.(ad_scaled_df.estimate, fd_scaled_df.estimate; rtol=1e-8))
                @test all(isapprox.(ad_scaled_df.se, fd_scaled_df.se; rtol=1e-6))
                
                @test validate_all_finite_positive(ad_scaled_df).all_valid
                @test validate_all_finite_positive(fd_scaled_df).all_valid
            end
        end
    end
    
    # === EDGE CASES AND ROBUSTNESS ===
    @testset "Backend Consistency Edge Cases" begin
        
        @testset "Small Sample Consistency" begin
            df_small = make_simple_test_data(n=30, formula_type=:linear, seed=111)
            model_small = lm(@formula(y ~ x), df_small)
            
            # Even with small samples, backends should be consistent
            consistency_result = test_backend_consistency(model_small, df_small; 
                                                       rtol_estimate=1e-8, rtol_se=1e-6)
            
            @test consistency_result.all_consistent
        end
        
        @testset "Extreme Coefficient Consistency" begin
            df_extreme = DataFrame(x = randn(100))
            df_extreme.y = 10.0 * df_extreme.x + 0.01 * randn(100)  # Large coefficient, small noise
            model_extreme = lm(@formula(y ~ x), df_extreme)
            
            consistency_result = test_backend_consistency(model_extreme, df_extreme; 
                                                       rtol_estimate=1e-8, rtol_se=1e-6)
            
            @test consistency_result.all_consistent
        end
    end
end