# Per-row allocation tests for ContrastEvaluator and derivative operations
#
# Validates zero-allocation performance for:
# - ContrastEvaluator with categorical contrasts
# - Continuous derivatives (marginal_effects_eta!/mu!)
# - Categorical mixtures
# - Interaction terms
# - Warm-up behavior (first call vs steady state)
#
# julia --project="." test/test_per_row_allocations.jl

using Test
using Margins, GLM, DataFrames, CategoricalArrays, Tables, BenchmarkTools
using FormulaCompiler:
    compile_formula, derivativeevaluator,
    contrastevaluator, contrast_modelrow!, contrast_gradient!
using LinearAlgebra: dot
using Margins: marginal_effects_eta!, marginal_effects_mu!, delta_method_se

@testset "Per-Row Allocation Coverage" begin
    # Test data setup
    n = 100
    df = DataFrame(
        y = randn(n),
        x = randn(n),
        z = randn(n),
        treatment = categorical(rand(["Control", "Drug_A", "Drug_B"], n)),
        binary_var = rand([false, true], n)
    )

    # Fit model with interactions
    model = lm(@formula(y ~ x * treatment + z * binary_var), df)
    data = Tables.columntable(df)
    compiled = compile_formula(model, data)
    β = coef(model)

    @testset "Continuous derivatives (marginal_effects_eta!)" begin
        println("\n=== Continuous Derivatives ===")

        # Build derivative evaluator
        vars = [:x, :z]
        de = derivativeevaluator(:fd, compiled, data, vars)
        g = Vector{Float64}(undef, length(vars))
        Gβ = Matrix{Float64}(undef, length(compiled), length(vars))

        println("1. Warmup call (may allocate for compilation):")
        @btime marginal_effects_eta!($g, $Gβ, $de, $β, 1)

        println("\n2. Steady-state call (should be 0 bytes):")
        result = @benchmark marginal_effects_eta!($g, $Gβ, $de, $β, 1) samples=100 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs)")

        println("\n3. Loop over rows (should be 0 bytes):")
        # Inline loop benchmark to avoid function scope issues
        result_loop = @benchmark begin
            for i in 1:50
                marginal_effects_eta!($g, $Gβ, $de, $β, i)
            end
        end samples=50 evals=1
        @test minimum(result_loop).allocs == 0
        println("   Allocations: $(minimum(result_loop).allocs)")
    end

    @testset "Continuous derivatives (marginal_effects_mu! with link)" begin
        # Test with GLM logit model
        df_logit = copy(df)
        df_logit.y_binary = rand([0, 1], n)
        model_logit = glm(@formula(y_binary ~ x + z), df_logit, Binomial(), LogitLink())
        data_logit = Tables.columntable(df_logit)
        compiled_logit = compile_formula(model_logit, data_logit)
        β_logit = coef(model_logit)

        println("\n=== Continuous Derivatives (Response Scale) ===")

        vars = [:x, :z]
        de = derivativeevaluator(:fd, compiled_logit, data_logit, vars)
        g = Vector{Float64}(undef, length(vars))
        Gβ = Matrix{Float64}(undef, length(compiled_logit), length(vars))
        link = LogitLink()

        println("1. Warmup call:")
        @btime marginal_effects_mu!($g, $Gβ, $de, $β_logit, $link, 1)

        println("\n2. Steady-state call (should be 0 bytes):")
        result = @benchmark marginal_effects_mu!($g, $Gβ, $de, $β_logit, $link, 1) samples=100 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs)")
    end

    @testset "Categorical contrasts (ContrastEvaluator)" begin
        println("\n=== Categorical Contrasts ===")

        # Build contrast evaluator
        evaluator = contrastevaluator(compiled, data, [:treatment, :binary_var])
        contrast_buf = Vector{Float64}(undef, length(compiled))

        println("1. Warmup call:")
        @btime contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Drug_A")

        println("\n2. Steady-state call (should be 0 bytes):")
        result = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Drug_A") samples=100 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs)")

        println("\n3. Binary variable contrast (should be 0 bytes):")
        result_binary = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :binary_var, false, true) samples=100 evals=1
        @test minimum(result_binary).allocs == 0
        println("   Allocations: $(minimum(result_binary).allocs)")

        println("\n4. Loop over multiple contrasts (should be 0 bytes):")
        result_loop = @benchmark begin
            for i in 1:30
                contrast_modelrow!($contrast_buf, $evaluator, i, :treatment, "Control", "Drug_A")
                contrast_modelrow!($contrast_buf, $evaluator, i, :treatment, "Control", "Drug_B")
                contrast_modelrow!($contrast_buf, $evaluator, i, :binary_var, false, true)
            end
        end samples=50 evals=1
        @test minimum(result_loop).allocs == 0
        println("   Allocations: $(minimum(result_loop).allocs)")
    end

    @testset "Categorical contrast gradients" begin
        println("\n=== Categorical Contrast Gradients ===")

        evaluator = contrastevaluator(compiled, data, [:treatment])
        ∇β = Vector{Float64}(undef, length(compiled))

        println("1. Linear scale gradient (warmup):")
        @btime contrast_gradient!($∇β, $evaluator, 1, :treatment, "Control", "Drug_A", $β)

        println("\n2. Linear scale gradient (should be 0 bytes):")
        result = @benchmark contrast_gradient!($∇β, $evaluator, 1, :treatment, "Control", "Drug_A", $β) samples=100 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs)")

        # Test with link function
        df_logit = copy(df)
        df_logit.y_binary = rand([0, 1], n)
        model_logit = glm(@formula(y_binary ~ treatment), df_logit, Binomial(), LogitLink())
        data_logit = Tables.columntable(df_logit)
        compiled_logit = compile_formula(model_logit, data_logit)
        β_logit = coef(model_logit)
        evaluator_logit = contrastevaluator(compiled_logit, data_logit, [:treatment])
        ∇β_logit = Vector{Float64}(undef, length(compiled_logit))
        link = LogitLink()

        println("\n3. Response scale gradient with link (warmup):")
        @btime contrast_gradient!($∇β_logit, $evaluator_logit, 1, :treatment, "Control", "Drug_A", $β_logit, $link)

        println("\n4. Response scale gradient with link (should be 0 bytes):")
        result_link = @benchmark contrast_gradient!($∇β_logit, $evaluator_logit, 1, :treatment, "Control", "Drug_A", $β_logit, $link) samples=100 evals=1
        @test minimum(result_link).allocs == 0
        println("   Allocations: $(minimum(result_link).allocs)")
    end

    @testset "Interaction terms" begin
        println("\n=== Interaction Terms ===")

        # Model with x * treatment interaction
        # Continuous marginal effect should work across interaction
        vars = [:x]
        de = derivativeevaluator(:fd, compiled, data, vars)
        g = Vector{Float64}(undef, length(vars))
        Gβ = Matrix{Float64}(undef, length(compiled), length(vars))

        println("1. Continuous ME with interaction (warmup):")
        @btime marginal_effects_eta!($g, $Gβ, $de, $β, 1)

        println("\n2. Continuous ME with interaction (should be 0 bytes):")
        result = @benchmark marginal_effects_eta!($g, $Gβ, $de, $β, 1) samples=100 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs)")

        # Categorical contrast with interaction
        evaluator = contrastevaluator(compiled, data, [:treatment])
        contrast_buf = Vector{Float64}(undef, length(compiled))

        println("\n3. Categorical contrast with interaction (should be 0 bytes):")
        result_contrast = @benchmark contrast_modelrow!($contrast_buf, $evaluator, 1, :treatment, "Control", "Drug_A") samples=100 evals=1
        @test minimum(result_contrast).allocs == 0
        println("   Allocations: $(minimum(result_contrast).allocs)")
    end

    @testset "Delta method standard errors" begin
        println("\n=== Delta Method SE ===")

        evaluator = contrastevaluator(compiled, data, [:treatment])
        vcov_matrix = vcov(model)

        println("1. Delta method SE (warmup):")
        @btime delta_method_se($evaluator, 1, :treatment, "Control", "Drug_A", $β, $vcov_matrix)

        println("\n2. Delta method SE (should be 0 bytes):")
        result = @benchmark delta_method_se($evaluator, 1, :treatment, "Control", "Drug_A", $β, $vcov_matrix) samples=100 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs)")
    end

    @testset "Evaluator reuse across rows" begin
        println("\n=== Evaluator Reuse ===")

        # Test that single evaluator can be reused for different rows without allocations
        evaluator = contrastevaluator(compiled, data, [:treatment, :binary_var])
        contrast_buf = Vector{Float64}(undef, length(compiled))

        println("1. Multiple rows with same evaluator (should be 0 bytes):")
        result = @benchmark begin
            for row in 1:50
                contrast_modelrow!($contrast_buf, $evaluator, row, :treatment, "Control", "Drug_A")
                contrast_modelrow!($contrast_buf, $evaluator, row, :binary_var, false, true)
            end
        end samples=50 evals=1
        @test minimum(result).allocs == 0
        println("   Allocations: $(minimum(result).allocs) (50 rows × 2 contrasts = 100 calls)")
    end

    @testset "Warm-up vs steady-state comparison" begin
        println("\n=== Warm-up Behavior Analysis ===")

        # Test that first call may allocate but subsequent calls don't
        fresh_data = DataFrame(
            y = randn(50),
            x = randn(50),
            group = categorical(rand(["A", "B"], 50))
        )
        fresh_model = lm(@formula(y ~ x * group), fresh_data)
        fresh_data_nt = Tables.columntable(fresh_data)
        fresh_compiled = compile_formula(fresh_model, fresh_data_nt)

        # Fresh evaluator
        fresh_evaluator = contrastevaluator(fresh_compiled, fresh_data_nt, [:group])
        fresh_buf = Vector{Float64}(undef, length(fresh_compiled))

        println("1. First call on fresh evaluator:")
        first_result = @benchmark contrast_modelrow!($fresh_buf, $fresh_evaluator, 1, :group, "A", "B") samples=1 evals=1
        println("   Allocations: $(first_result.allocs[1])")

        println("\n2. Second call (steady state):")
        second_result = @benchmark contrast_modelrow!($fresh_buf, $fresh_evaluator, 1, :group, "A", "B") samples=100 evals=1
        steady_allocs = minimum(second_result).allocs
        println("   Allocations: $steady_allocs")

        println("\n3. 10th call (fully warmed up):")
        for _ in 1:8
            contrast_modelrow!(fresh_buf, fresh_evaluator, 1, :group, "A", "B")
        end
        tenth_result = @benchmark contrast_modelrow!($fresh_buf, $fresh_evaluator, 1, :group, "A", "B") samples=100 evals=1
        @test minimum(tenth_result).allocs == 0
        println("   Allocations: $(minimum(tenth_result).allocs)")

        println("\nConclusion: Warm-up phase complete, steady-state achieves 0 allocations")
    end
end

println("\n" * "="^60)
println("SUMMARY: All per-row operations achieve zero allocations")
println("after warm-up, validating production-ready performance.")
println("="^60)
