# test_delta_method_glm.jl - Delta Method and Gradient Tests for GLM Models
# Extracted from FormulaCompiler test/test_glm_integration.jl (2025-10-09)

using Test
using Margins
using FormulaCompiler
using DataFrames, Tables, GLM, StatsModels, CategoricalArrays
using Random

Random.seed!(123)

@testset "GLM Delta Method and Gradient Tests" begin

    # Create comprehensive test data
    n = 500
    df = DataFrame(
        x1 = randn(n),
        x2 = randn(n),
        treatment = categorical(repeat(["Control", "Treatment_A", "Treatment_B"], div(n,3)+1)[1:n]),
        female = rand([0, 1], n),
        region = categorical(rand(["North", "South", "East", "West"], n)),
        exposure = rand(0.5:0.1:2.0, n)
    )
    df.y_binary = rand([0, 1], n)
    df.y_count = rand(0:20, n)
    df.y_nb = rand(0:50, n)
    df.y_normal = randn(n)

    data = Tables.columntable(df)

    @testset "Logistic Regression Inference" begin
        model = glm(@formula(y_binary ~ x1 + treatment + female), df, Binomial(), LogitLink())
        compiled = FormulaCompiler.compile_formula(model, data)
        evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:treatment, :female])

        @testset "Logistic gradient computation" begin
            # Test gradient computation with logistic link
            β = coef(model)
            vcov_matrix = GLM.vcov(model)

            ∇β = Vector{Float64}(undef, length(compiled))
            Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment_A", β, LogitLink())
            @test !all(∇β .== 0.0)

            # Test delta method standard error
            se = Margins.delta_method_se(∇β, vcov_matrix)
            @test se > 0.0
            @test isfinite(se)
        end
    end

    @testset "Poisson Regression Inference" begin
        model = glm(@formula(y_count ~ x1 + x2 + treatment + female), df, Poisson(), LogLink())
        compiled = FormulaCompiler.compile_formula(model, data)
        evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:treatment, :female])

        @testset "Poisson gradient computation" begin
            # Test gradient computation with log link
            β = coef(model)
            vcov_matrix = GLM.vcov(model)

            ∇β = Vector{Float64}(undef, length(compiled))
            Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment_A", β, LogLink())
            @test !all(∇β .== 0.0)

            # Test delta method with log link
            se = Margins.delta_method_se(∇β, vcov_matrix)
            @test se > 0.0
            @test isfinite(se)

            # Compare with identity link (linear scale)
            ∇β_linear = Vector{Float64}(undef, length(compiled))
            Margins.contrast_gradient!(∇β_linear, evaluator, 1, :treatment, "Control", "Treatment_A", β)

            # Should be different due to link function
            @test !(∇β ≈ ∇β_linear)
        end
    end

    @testset "Poisson with Offset Inference" begin
        # Add exposure/offset variable
        df_offset = copy(df)
        data_offset = Tables.columntable(df_offset)

        model = glm(@formula(y_count ~ x1 + treatment + female), df_offset, Poisson(), LogLink(); offset=log.(df_offset.exposure))
        compiled = FormulaCompiler.compile_formula(model, data_offset)
        evaluator = FormulaCompiler.contrastevaluator(compiled, data_offset, [:treatment, :female])

        @testset "Gradient computation with offset" begin
            β = coef(model)
            vcov_matrix = GLM.vcov(model)

            ∇β = Vector{Float64}(undef, length(compiled))
            Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment_A", β, LogLink())
            @test !all(∇β .== 0.0)

            se = Margins.delta_method_se(∇β, vcov_matrix)
            @test se > 0.0
            @test isfinite(se)
        end
    end

    @testset "Negative Binomial Inference" begin
        # Note: Negative Binomial requires GLM.jl version that supports it
        @testset "NB gradient computation" begin
            try
                model = glm(@formula(y_nb ~ x1 + x2 + treatment + female), df, NegativeBinomial(), LogLink())
                compiled = FormulaCompiler.compile_formula(model, data)
                evaluator = FormulaCompiler.contrastevaluator(compiled, data, [:treatment, :female])

                β = coef(model)
                vcov_matrix = GLM.vcov(model)

                ∇β = Vector{Float64}(undef, length(compiled))
                Margins.contrast_gradient!(∇β, evaluator, 1, :treatment, "Control", "Treatment_A", β, LogLink())
                @test !all(∇β .== 0.0)

                # Test delta method
                se = Margins.delta_method_se(∇β, vcov_matrix)
                @test se > 0.0
                @test isfinite(se)

                println("✓ Negative Binomial models supported and working")

            catch e
                if e isa MethodError
                    println("⚠ Negative Binomial models not supported in this GLM.jl version - skipping")
                    @test_skip "Negative Binomial support depends on GLM.jl version"
                else
                    rethrow(e)
                end
            end
        end
    end

    @testset "Link Function Comparison" begin
        # Test that different link functions give different results
        logit_model = glm(@formula(y_binary ~ x1 + treatment), df, Binomial(), LogitLink())
        probit_model = glm(@formula(y_binary ~ x1 + treatment), df, Binomial(), ProbitLink())

        logit_compiled = FormulaCompiler.compile_formula(logit_model, data)
        probit_compiled = FormulaCompiler.compile_formula(probit_model, data)

        logit_evaluator = FormulaCompiler.contrastevaluator(logit_compiled, data, [:treatment])
        probit_evaluator = FormulaCompiler.contrastevaluator(probit_compiled, data, [:treatment])

        # Gradients should be different due to different link functions
        β_logit = coef(logit_model)
        β_probit = coef(probit_model)
        vcov_logit = GLM.vcov(logit_model)
        vcov_probit = GLM.vcov(probit_model)

        ∇β_logit = Vector{Float64}(undef, length(logit_compiled))
        ∇β_probit = Vector{Float64}(undef, length(probit_compiled))

        Margins.contrast_gradient!(∇β_logit, logit_evaluator, 1, :treatment, "Control", "Treatment_A", β_logit, LogitLink())
        Margins.contrast_gradient!(∇β_probit, probit_evaluator, 1, :treatment, "Control", "Treatment_A", β_probit, ProbitLink())

        # Should be different due to different link derivatives
        @test !(∇β_logit ≈ ∇β_probit)

        # Standard errors should also be different
        se_logit = Margins.delta_method_se(∇β_logit, vcov_logit)
        se_probit = Margins.delta_method_se(∇β_probit, vcov_probit)

        @test se_logit != se_probit
        @test se_logit > 0.0 && se_probit > 0.0
    end
end
