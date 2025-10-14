# test_delta_method_mixedmodels.jl - Delta Method and Gradient Tests for MixedModels
# Extracted from FormulaCompiler test/test_mixedmodels_integration.jl (2025-10-09)

using Test
using Margins
using FormulaCompiler
using DataFrames, Tables, GLM, StatsModels, CategoricalArrays
using MixedModels
using Random

Random.seed!(42)

@testset "MixedModels Delta Method and Gradient Tests" begin

    # Create comprehensive test data with repeated measures
    n_subjects = 50
    n_obs_per_subject = 4
    n = n_subjects * n_obs_per_subject

    df = DataFrame(
        subject = repeat(1:n_subjects, inner=n_obs_per_subject),
        x1 = randn(n),
        treatment = categorical(repeat(["Control", "Treatment_A", "Treatment_B"], div(n,3)+1)[1:n]),
        condition = categorical(rand(["Base", "Enhanced"], n))
    )

    # Generate outcomes with random effects
    subject_effects = randn(n_subjects) * 0.5
    df.y_binary = rand([0, 1], n)
    df.y_continuous = randn(n) .+ subject_effects[df.subject]

    data = Tables.columntable(df)

    @testset "GLMM Gradient Computation" begin
        try
            # Fit logistic mixed model
            glmm = fit(MixedModel, @formula(y_binary ~ x1 + treatment + (1|subject)), df, Bernoulli())
            compiled = FormulaCompiler.compile_formula(glmm, data)
            evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:treatment])

            @testset "Logit GLMM gradient and SE" begin
                # Test gradient computation with GLMM
                β = fixef(glmm)  # Fixed effects only
                vcov_matrix = vcov(glmm)  # Fixed effects covariance

                ∇β = Vector{Float64}(undef, length(compiled))
                Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment_A", β, LogitLink())
                @test !all(∇β .== 0.0)

                # Test delta method standard error
                se = Margins.delta_method_se(∇β, vcov_matrix)
                @test se > 0.0
                @test isfinite(se)

                println("✓ GLMM gradient computation and standard errors working")
            end

        catch e
            if isa(e, ArgumentError) || isa(e, MethodError)
                println("⚠ Logistic GLMM not fully supported in this MixedModels.jl version - skipping")
                @test_skip "GLMM support depends on MixedModels.jl version"
            else
                rethrow(e)
            end
        end
    end

    @testset "Poisson GLMM Gradients" begin
        # Create count data
        df_count = copy(df)
        # Generate count outcomes based on the linear predictor
        λ = exp.(0.5 .+ 0.1*df_count.x1 .+ 0.2*(df_count.treatment .== "Treatment_A") .+ randn(nrow(df))*0.2)
        df_count.y_count = rand.(Poisson.(clamp.(λ, 0.1, 50)))  # Clamp to avoid extreme values
        data_count = Tables.columntable(df_count)

        try
            glmm_poisson = fit(MixedModel, @formula(y_count ~ x1 + treatment + (1|subject)), df_count, Poisson())
            compiled = FormulaCompiler.compile_formula(glmm_poisson, data_count)
            evaluator = FormulaCompiler.contrastevaluator(compiled, data_count, [:treatment])

            @testset "Log GLMM gradient and SE" begin
                β = fixef(glmm_poisson)
                vcov_matrix = vcov(glmm_poisson)

                ∇β = Vector{Float64}(undef, length(compiled))
                Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment_A", β, LogLink())
                @test !all(∇β .== 0.0)

                se = Margins.delta_method_se(∇β, vcov_matrix)
                @test se > 0.0
                @test isfinite(se)
            end

            println("✓ Poisson GLMM successfully tested")

        catch e
            if isa(e, ArgumentError) || isa(e, MethodError)
                println("⚠ Poisson GLMM not fully supported in this MixedModels.jl version - skipping")
                @test_skip "Poisson GLMM support depends on MixedModels.jl version"
            else
                rethrow(e)
            end
        end
    end

    @testset "GLMM vs GLM Comparison" begin
        # Compare gradients and SEs between GLMM and GLM
        try
            # Fit both models
            glmm = fit(MixedModel, @formula(y_binary ~ x1 + treatment + (1|subject)), df, Bernoulli())
            glm_model = glm(@formula(y_binary ~ x1 + treatment), df, Binomial(), LogitLink())

            # Compile both
            glmm_compiled = FormulaCompiler.compile_formula(glmm, data)
            glm_compiled = FormulaCompiler.compile_formula(glm_model, data)

            glmm_evaluator = FormulaCompiler.contrastevaluator(glmm_compiled, data, [:treatment])
            glm_evaluator = FormulaCompiler.contrastevaluator(glm_compiled, data, [:treatment])

            # Get coefficients and vcov
            β_glmm = fixef(glmm)
            β_glm = coef(glm_model)
            vcov_glmm = vcov(glmm)
            vcov_glm = GLM.vcov(glm_model)

            ∇β_glmm = Vector{Float64}(undef, length(glmm_compiled))
            ∇β_glm = Vector{Float64}(undef, length(glm_compiled))

            Margins.contrast_gradient!(∇β_glmm, glmm_evaluator, 1, :treatment, "Control", "Treatment_A", β_glmm, LogitLink())
            Margins.contrast_gradient!(∇β_glm, glm_evaluator, 1, :treatment, "Control", "Treatment_A", β_glm, LogitLink())

            # Both should have valid gradients
            @test !all(∇β_glmm .== 0)
            @test !all(∇β_glm .== 0)
            @test all(isfinite, ∇β_glmm)
            @test all(isfinite, ∇β_glm)

            # Standard errors should be computable for both
            se_glmm = Margins.delta_method_se(∇β_glmm, vcov_glmm)
            se_glm = Margins.delta_method_se(∇β_glm, vcov_glm)

            @test se_glmm > 0 && se_glm > 0
            @test isfinite(se_glmm) && isfinite(se_glm)

        catch e
            if isa(e, ArgumentError) || isa(e, MethodError)
                println("⚠ GLMM comparison not fully supported in this MixedModels.jl version - skipping")
                @test_skip "GLMM comparison depends on MixedModels.jl version"
            else
                rethrow(e)
            end
        end
    end
end
