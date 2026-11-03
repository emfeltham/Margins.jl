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

@testset "Backend Consistency Tests - AD vs FD Validation" begin
    Random.seed!(789)  # Different seed from main validation for independence
    
    # === ESSENTIAL BACKEND CONSISTENCY ACROSS ALL 2×2 QUADRANTS ===
    # Linear models: AD and FD agree to 1e-10 / 1e-8 (estimates / SEs); logistic (response
    # scale, chain rule through the link): 1e-8 / 1e-6
    @testset "AD vs FD Computational Consistency" begin
        df = make_econometric_data(; n=500, seed=789)
        dfs = make_simple_test_data(n=400, formula_type=:linear, seed=123)
        lm_tol = (rtol_estimate=1e-10, rtol_se=1e-8)
        glm_tol = (rtol_estimate=1e-8, rtol_se=1e-6)
        test_models = [
            (name="Simple LM", model=lm(@formula(log(wage) ~ int_education), df), data=df, tol=lm_tol),
            (name="Multiple LM", model=lm(@formula(log(wage) ~ int_education + int_experience), df), data=df, tol=lm_tol),
            (name="Quadratic LM", model=lm(@formula(log(wage) ~ int_education + int_experience^2), df), data=df, tol=lm_tol),
            (name="LM y ~ x + z", model=lm(@formula(y ~ x + z), dfs), data=dfs, tol=lm_tol),
            (name="Simple Logistic", model=glm(@formula(union_member ~ int_education), df, Binomial(), LogitLink()), data=df, tol=glm_tol),
            (name="Multiple Logistic", model=glm(@formula(union_member ~ int_education + int_experience), df, Binomial(), LogitLink()), data=df, tol=glm_tol),
        ]

        for (name, model, data, tol) in test_models
            @testset "$(name) - Full 2×2 Backend Consistency" begin
                consistency_result = test_backend_consistency(model, data; tol...)
                @test consistency_result.n_successful == consistency_result.n_total
                for (quadrant_name, result) in consistency_result.quadrants
                    @test result.success
                    result.success || continue
                    @test result.estimates_agree
                    @test result.ses_agree
                end
            end
        end
        # population AME of an lm (moved from backend_reliability_tests.jl, same data): FD of a
        # linear term is exact up to rounding
        d999 = make_simple_test_data(n=500, formula_type=:linear, seed=999)
        m999 = lm(@formula(y ~ x + z), d999)
        ad = DataFrame(population_margins(m999, d999; vars=[:x, :z], backend=:ad))
        fd = DataFrame(population_margins(m999, d999; vars=[:x, :z], backend=:fd))
        @test fd.estimate ≈ ad.estimate rtol=1e-12
        @test fd.se ≈ ad.se rtol=1e-10
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
            
            @test all(isapprox.(ad_mu_df.estimate, fd_mu_df.estimate; rtol=1e-10))
            @test all(isapprox.(ad_mu_df.se, fd_mu_df.se; rtol=1e-8))
        end
    end
    
    # === DOMAIN BOUNDARIES: log and sqrt near 0 ===
    # FD steps x by ~eps^(1/3)·max(1, |x|); both backends must stay in the domain, and match
    # the hand derivative: AME = β·mean(f'(xᵢ)), SE = |mean(f'(xᵢ))|·SE(β)
    @testset "Domain-Sensitive Functions (log, sqrt near 0)" begin
        rng = MersenneTwister(12345)
        for (fname, lo, width, fp, mk) in (
                ("log",  0.1,  1.8, x -> 1 / x,            d -> lm(@formula(y ~ log(x)), d)),
                ("sqrt", 0.05, 0.9, x -> 1 / (2 * sqrt(x)), d -> lm(@formula(y ~ sqrt(x)), d)))
            @testset "$fname(x), x ∈ [$lo, $(lo + width)]" begin
                df = DataFrame(x = rand(rng, 200) .* width .+ lo, z = randn(rng, 200))
                df.y = 0.5 .* (fname == "log" ? log.(df.x) : sqrt.(df.x)) .+ 0.3 .* df.z .+ 0.1 .* randn(rng, 200)
                model = mk(df)
                β = coef(model)[2]; se_β = sqrt(vcov(model)[2, 2]); mfp = mean(fp.(df.x))
                res = Dict(bk => DataFrame(population_margins(model, df; type=:effects, backend=bk)) for bk in (:ad, :fd))
                @test res[:ad].estimate[1] ≈ β * mfp rtol=1e-10
                @test res[:ad].se[1] ≈ abs(mfp) * se_β rtol=1e-10
                @test res[:fd].estimate[1] ≈ β * mfp rtol=1e-6
                @test res[:fd].se[1] ≈ abs(mfp) * se_β rtol=1e-6
                @test res[:fd].estimate ≈ res[:ad].estimate rtol=1e-6
                @test res[:fd].se ≈ res[:ad].se rtol=1e-8
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