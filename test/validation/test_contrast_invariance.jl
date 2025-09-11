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
# This tests the statistical correctness of the override system and ensures
# that baseline detection (_get_baseline_level) works properly across contrast types.

using Test
using Random
using DataFrames, CategoricalArrays, GLM, StatsModels
using Margins
using FormulaCompiler

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
        baseline_levels = Dict()
        
        @testset "Fit models with different contrast codings" begin
            for (scheme_name, contrast_type) in contrast_schemes
                # Set contrast coding for this scheme
                contrasts_dict = Dict(:group => contrast_type)
                
                # Fit model with specific contrast coding
                model = lm(@formula(y ~ x + group), df; contrasts=contrasts_dict)
                
                # Store baseline level detection
                baseline_levels[scheme_name] = FormulaCompiler._get_baseline_level(model, :group)
                
                # Compute AME for the categorical variable
                ame = population_margins(model, df; type=:effects, vars=[:group], scale=:response)
                ame_df = DataFrame(ame)
                
                # Store results
                ame_results[scheme_name] = ame_df
                
                @test nrow(ame_df) >= 1  # Should have at least one contrast
                @test all(isfinite, ame_df.estimate)
                @test !isempty(ame_df.variable)
                
                # Debug output
                @debug "Contrast scheme results" scheme=scheme_name baseline=baseline_levels[scheme_name] n_contrasts=nrow(ame_df) estimates=ame_df.estimate
            end
        end
        
        @testset "Baseline level consistency" begin
            # All schemes should detect the same baseline level
            # (StatsModels.jl should handle this consistently)
            baseline_dummy = baseline_levels[:dummy]
            baseline_effects = baseline_levels[:effects] 
            baseline_helmert = baseline_levels[:helmert]
            
            @debug "Baseline level detection" dummy=baseline_dummy effects=baseline_effects helmert=baseline_helmert
            
            # For categorical variables, baseline should be consistent across contrast schemes
            # This is a mathematical requirement - the baseline is an inherent property of the data
            @test baseline_dummy == baseline_effects
            @test baseline_dummy == baseline_helmert
        end
        
        @testset "AME contrast invariance" begin
            # Extract AME estimates for comparison
            dummy_estimates = sort(ame_results[:dummy].estimate)
            effects_estimates = sort(ame_results[:effects].estimate)
            helmert_estimates = sort(ame_results[:helmert].estimate)
            
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
        baseline_levels_bool = Dict()
        
        @testset "Boolean variable with contrast schemes" begin
            for (scheme_name, contrast_type) in contrast_schemes
                contrasts_dict = Dict(:treatment_cat => contrast_type)
                
                model = lm(@formula(y ~ x + treatment_cat), df; contrasts=contrasts_dict)
                
                # Baseline detection for boolean
                baseline_levels_bool[scheme_name] = FormulaCompiler._get_baseline_level(model, :treatment_cat)
                
                ame = population_margins(model, df; type=:effects, vars=[:treatment_cat], scale=:response)
                ame_df = DataFrame(ame)
                
                ame_results_bool[scheme_name] = ame_df
                
                @test nrow(ame_df) == 1  # Binary variable should have exactly one contrast
                @test isfinite(ame_df.estimate[1])
            end
        end
        
        @testset "Boolean baseline convention" begin
            # For boolean variables, baseline should always be false (0)
            # This tests the false→true (0→1) convention from the plan
            for (scheme_name, baseline) in baseline_levels_bool
                @test baseline == false
                if baseline != false
                    @error "Boolean baseline should be false for $scheme_name scheme, got $baseline"
                end
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
                
                @test nrow(ame_df) >= 1
                @test all(isfinite, ame_df.estimate)
            end
        end
        
        @testset "Interaction model AME invariance" begin
            # Even with interactions, AME should be contrast-invariant
            dummy_est = sort(ame_interaction_results[:dummy].estimate)
            effects_est = sort(ame_interaction_results[:effects].estimate)
            helmert_est = sort(ame_interaction_results[:helmert].estimate)
            
            @debug "Interaction model AME comparison" dummy=dummy_est effects=effects_est helmert=helmert_est
            
            @test dummy_est ≈ effects_est atol=1e-10 rtol=1e-8
            @test dummy_est ≈ helmert_est atol=1e-10 rtol=1e-8
        end
    end
end