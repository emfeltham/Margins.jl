# test_contrast_invariance.jl - Contrast-coding invariance validation
#
# This test validates a fundamental mathematical requirement: AME (Average Marginal Effects)
# must be contrast-coding invariant. The marginal effect of changing from baseline to
# level should be identical regardless of whether the model uses dummy, effects, or
# helmert contrast coding.
#
# Mathematical Principle:
# For categorical variable C with levels {A, B, C} where A is baseline:
#   AME(A→B) should be identical across all contrast coding schemes
#   AME(A→C) should be identical across all contrast coding schemes
#
# This tests the statistical correctness of the override system and ensures that the
# baseline level is the same under every coding: the contrast labels Margins reports
# ("level - baseline") must agree across schemes, and a Bool-valued categorical's must be
# "true - false". (Baseline detection with DummyCoding(base = …) is checked in
# primitives/test_formulacompiler_public_api.jl.)

using Test
using Random
using DataFrames, CategoricalArrays, GLM, StatsModels
using Margins

# Test data for all three contrast coding schemes
const contrast_schemes = [
    (:dummy, StatsModels.DummyCoding()),
    (:effects, StatsModels.EffectsCoding()), 
    (:helmert, StatsModels.HelmertCoding())
]

@testset "Contrast-Coding Invariance Tests" begin
    Random.seed!(12345)
    
    @testset "Three-Level Categorical Variable" begin
        # Create test data with three-level categorical variable
        n = 500
        df = DataFrame(
            x = randn(n),                                    # Continuous covariate
            group = rand(["Low", "Medium", "High"], n),      # Three-level categorical
            y = randn(n)                                     # Response variable
        )
        
        # Make group categorical for contrast coding
        df.group = categorical(df.group)
        
        ame_results = Dict()

        @testset "Fit models with different contrast codings" begin
            for (scheme_name, contrast_type) in contrast_schemes
                # Set contrast coding for this scheme
                contrasts_dict = Dict(:group => contrast_type)

                # Fit model with specific contrast coding
                model = lm(@formula(y ~ x + group), df; contrasts=contrasts_dict)

                # Compute AME for the categorical variable
                ame = population_margins(model, df; type=:effects, vars=[:group], scale=:response)
                ame_df = DataFrame(ame)

                # Store results
                ame_results[scheme_name] = ame_df

                # every non-baseline level against the first level, whatever the coding
                @test sort(ame_df.contrast) == ["Low - High", "Medium - High"]

                # Debug output
                @debug "Contrast scheme results" scheme=scheme_name contrasts=ame_df.contrast estimates=ame_df.estimate
            end
        end

        @testset "AME contrast invariance" begin
            # Rows matched by contrast label (the same baseline in every scheme)
            byc(s) = [ame_results[s].estimate[findfirst(==(c), ame_results[s].contrast)]
                      for c in ("Low - High", "Medium - High")]
            dummy_estimates = byc(:dummy)
            effects_estimates = byc(:effects)
            helmert_estimates = byc(:helmert)

            # Debug detailed comparison
            @debug "AME comparison across schemes" dummy=dummy_estimates effects=effects_estimates helmert=helmert_estimates
            
            # Mathematical requirement: AME should be contrast-coding invariant
            # The marginal effect represents a real change in outcome, independent of coding
            @test dummy_estimates ≈ effects_estimates atol=1e-10 rtol=1e-8
            @test dummy_estimates ≈ helmert_estimates atol=1e-10 rtol=1e-8
            @test effects_estimates ≈ helmert_estimates atol=1e-10 rtol=1e-8
        end
    end
    
    @testset "Binary Categorical Variable (Boolean Convention)" begin
        # Test boolean false→true convention with different contrast codings
        n = 400
        df = DataFrame(
            x = randn(n),
            treatment = rand([false, true], n),    # Boolean variable
            y = randn(n) 
        )
        
        # Convert to categorical for explicit contrast coding control
        df.treatment_cat = categorical(df.treatment)
        
        ame_results_bool = Dict()

        @testset "Boolean variable with contrast schemes" begin
            for (scheme_name, contrast_type) in contrast_schemes
                contrasts_dict = Dict(:treatment_cat => contrast_type)

                model = lm(@formula(y ~ x + treatment_cat), df; contrasts=contrasts_dict)

                ame = population_margins(model, df; type=:effects, vars=[:treatment_cat], scale=:response)
                ame_df = DataFrame(ame)

                ame_results_bool[scheme_name] = ame_df

                # one contrast, with false as the baseline (the false → true convention)
                @test ame_df.contrast == ["true - false"]
            end
        end

        @testset "Boolean AME invariance" begin
            # AME should be identical across contrast schemes for boolean variables
            dummy_est = ame_results_bool[:dummy].estimate[1]
            effects_est = ame_results_bool[:effects].estimate[1]
            helmert_est = ame_results_bool[:helmert].estimate[1]
            
            @debug "Boolean AME comparison" dummy=dummy_est effects=effects_est helmert=helmert_est
            
            @test dummy_est ≈ effects_est atol=1e-12
            @test dummy_est ≈ helmert_est atol=1e-12  
            @test effects_est ≈ helmert_est atol=1e-12
        end
    end
    
    @testset "Mixed Model with Interactions" begin
        # Test that contrast invariance holds even with interaction terms
        n = 300
        df = DataFrame(
            x = randn(n),
            group = categorical(rand(["A", "B", "C"], n)),
            z = randn(n),
            y = randn(n)
        )
        
        ame_interaction_results = Dict()
        
        @testset "Models with interactions" begin
            for (scheme_name, contrast_type) in contrast_schemes
                contrasts_dict = Dict(:group => contrast_type)
                
                # Model with interaction between continuous and categorical
                model = lm(@formula(y ~ x * group + z), df; contrasts=contrasts_dict)
                
                # AME should still be contrast-invariant even with interactions
                ame = population_margins(model, df; type=:effects, vars=[:group], scale=:response)
                ame_df = DataFrame(ame)
                
                ame_interaction_results[scheme_name] = ame_df

                @test sort(ame_df.contrast) == ["B - A", "C - A"]
            end
        end
        
        @testset "Interaction model AME invariance" begin
            # Even with interactions, AME should be contrast-invariant
            byc(s) = [ame_interaction_results[s].estimate[findfirst(==(c), ame_interaction_results[s].contrast)]
                      for c in ("B - A", "C - A")]
            dummy_est = byc(:dummy)
            effects_est = byc(:effects)
            helmert_est = byc(:helmert)
            
            @debug "Interaction model AME comparison" dummy=dummy_est effects=effects_est helmert=helmert_est
            
            @test dummy_est ≈ effects_est atol=1e-10 rtol=1e-8
            @test dummy_est ≈ helmert_est atol=1e-10 rtol=1e-8
        end
    end
end