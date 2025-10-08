# Test that categorical batch function maintains zero allocations
using Test, Margins, GLM, DataFrames, Tables, BenchmarkTools, CategoricalArrays
using Margins: build_engine, PopulationUsage, HasDerivatives, categorical_contrast_ame_batch!
using Margins: generate_contrast_pairs, ContrastPair

@testset "Categorical Batch Function Zero-Allocation" begin
    # Setup test data
    n = 100
    df = DataFrame(
        y = rand(Bool, n),
        x = randn(n),
        treatment = categorical(rand(["Control", "Drug_A", "Drug_B"], n))
    )

    model = glm(@formula(y ~ x + treatment), df, Binomial(), LogitLink())
    data_nt = Tables.columntable(df)
    n_coef = length(coef(model))

    @testset "Batch function: multi-level categorical zero-alloc" begin
        # Build engine
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        # Generate contrast pairs
        var_col = getproperty(data_nt, :treatment)
        rows = collect(1:50)
        contrast_pairs = generate_contrast_pairs(var_col, rows, :baseline, model, :treatment, data_nt)

        # Pre-allocate result arrays
        n_contrasts = length(contrast_pairs)
        results_ame = Vector{Float64}(undef, n_contrasts)
        results_se = Vector{Float64}(undef, n_contrasts)
        gradient_matrix = Matrix{Float64}(undef, n_contrasts, n_coef)

        # Warmup
        for _ in 1:10
            categorical_contrast_ame_batch!(
                results_ame, results_se, gradient_matrix,
                engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
                engine.contrast, :treatment, contrast_pairs,
                engine.β, engine.Σ, engine.link,
                rows, nothing
            )
        end

        # Benchmark for zero allocations
        b = @benchmark categorical_contrast_ame_batch!(
            $results_ame, $results_se, $gradient_matrix,
            $(engine.contrast_buf), $(engine.contrast_grad_buf), $(engine.contrast_grad_accum),
            $(engine.contrast), :treatment, $contrast_pairs,
            $(engine.β), $(engine.Σ), $(engine.link),
            $rows, nothing
        ) samples=100

        println("\nMulti-level categorical batch:")
        println("  Time (median): ", median(b.times) / 1000, " μs")
        println("  Allocations:   ", b.allocs)
        println("  Memory:        ", b.memory, " bytes")

        @test b.allocs == 0
        @test b.memory == 0

        # Verify results are reasonable
        @test all(!isnan, results_ame)
        @test all(!isnan, results_se)
        @test all(results_se .>= 0.0)
    end

    @testset "Batch function: weighted computation zero-alloc" begin
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        var_col = getproperty(data_nt, :treatment)
        rows = collect(1:50)
        contrast_pairs = generate_contrast_pairs(var_col, rows, :baseline, model, :treatment, data_nt)
        weights = ones(Float64, n)

        n_contrasts = length(contrast_pairs)
        results_ame = Vector{Float64}(undef, n_contrasts)
        results_se = Vector{Float64}(undef, n_contrasts)
        gradient_matrix = Matrix{Float64}(undef, n_contrasts, n_coef)

        # Warmup
        for _ in 1:10
            categorical_contrast_ame_batch!(
                results_ame, results_se, gradient_matrix,
                engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
                engine.contrast, :treatment, contrast_pairs,
                engine.β, engine.Σ, engine.link,
                rows, weights
            )
        end

        # Benchmark
        b = @benchmark categorical_contrast_ame_batch!(
            $results_ame, $results_se, $gradient_matrix,
            $(engine.contrast_buf), $(engine.contrast_grad_buf), $(engine.contrast_grad_accum),
            $(engine.contrast), :treatment, $contrast_pairs,
            $(engine.β), $(engine.Σ), $(engine.link),
            $rows, $weights
        ) samples=100

        println("\nWeighted computation:")
        println("  Time (median): ", median(b.times) / 1000, " μs")
        println("  Allocations:   ", b.allocs)
        println("  Memory:        ", b.memory, " bytes")

        @test b.allocs == 0
        @test b.memory == 0
    end

    @testset "Batch function: boolean categorical" begin
        # Test with boolean categorical (the problematic case)
        df_bool = DataFrame(
            y = randn(n),
            x = randn(n),
            treated = rand([false, true], n)
        )

        model_bool = lm(@formula(y ~ x + treated), df_bool)
        data_nt_bool = Tables.columntable(df_bool)

        engine = build_engine(PopulationUsage, HasDerivatives, model_bool, data_nt_bool, [:treated], GLM.vcov, :ad)

        var_col = getproperty(data_nt_bool, :treated)
        rows = collect(1:50)
        contrast_pairs = generate_contrast_pairs(var_col, rows, :baseline, model_bool, :treated, data_nt_bool)

        n_contrasts = length(contrast_pairs)
        results_ame = Vector{Float64}(undef, n_contrasts)
        results_se = Vector{Float64}(undef, n_contrasts)
        gradient_matrix = Matrix{Float64}(undef, n_contrasts, length(coef(model_bool)))

        # Warmup
        for _ in 1:10
            categorical_contrast_ame_batch!(
                results_ame, results_se, gradient_matrix,
                engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
                engine.contrast, :treated, contrast_pairs,
                engine.β, engine.Σ, engine.link,
                rows, nothing
            )
        end

        # Benchmark
        b = @benchmark categorical_contrast_ame_batch!(
            $results_ame, $results_se, $gradient_matrix,
            $(engine.contrast_buf), $(engine.contrast_grad_buf), $(engine.contrast_grad_accum),
            $(engine.contrast), :treated, $contrast_pairs,
            $(engine.β), $(engine.Σ), $(engine.link),
            $rows, nothing
        ) samples=100

        println("\nBoolean categorical batch:")
        println("  Time (median): ", median(b.times) / 1000, " μs")
        println("  Allocations:   ", b.allocs)
        println("  Memory:        ", b.memory, " bytes")

        # Boolean categoricals - tracking the allocation issue
        @test b.allocs == 0
        @test b.memory == 0

        @test all(!isnan, results_ame)
        @test all(!isnan, results_se)
    end

    @testset "Correctness: batch matches single contrast" begin
        # Verify batch function gives same results as calling single contrast multiple times
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        var_col = getproperty(data_nt, :treatment)
        rows = collect(1:50)
        contrast_pairs = generate_contrast_pairs(var_col, rows, :baseline, model, :treatment, data_nt)

        n_contrasts = length(contrast_pairs)

        # Call batch function
        results_ame_batch = Vector{Float64}(undef, n_contrasts)
        results_se_batch = Vector{Float64}(undef, n_contrasts)
        gradient_matrix_batch = Matrix{Float64}(undef, n_contrasts, n_coef)

        categorical_contrast_ame_batch!(
            results_ame_batch, results_se_batch, gradient_matrix_batch,
            engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
            engine.contrast, :treatment, contrast_pairs,
            engine.β, engine.Σ, engine.link,
            rows, nothing
        )

        # Call single contrast function for each pair
        for (i, pair) in enumerate(contrast_pairs)
            ame, se = Margins.categorical_contrast_ame!(
                engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
                engine.contrast, :treatment, pair.level1, pair.level2,
                engine.β, engine.Σ, engine.link, rows, nothing
            )

            # Check results match
            @test results_ame_batch[i] ≈ ame rtol=1e-10
            @test results_se_batch[i] ≈ se rtol=1e-10

            # Check gradient matches
            @test all(gradient_matrix_batch[i, :] .≈ engine.contrast_grad_accum)
        end
    end

    @testset "Result array dimensions" begin
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        var_col = getproperty(data_nt, :treatment)
        rows = collect(1:50)
        contrast_pairs = generate_contrast_pairs(var_col, rows, :baseline, model, :treatment, data_nt)

        n_contrasts = length(contrast_pairs)
        results_ame = Vector{Float64}(undef, n_contrasts)
        results_se = Vector{Float64}(undef, n_contrasts)
        gradient_matrix = Matrix{Float64}(undef, n_contrasts, n_coef)

        # Should not error with correctly sized arrays
        @test_nowarn categorical_contrast_ame_batch!(
            results_ame, results_se, gradient_matrix,
            engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
            engine.contrast, :treatment, contrast_pairs,
            engine.β, engine.Σ, engine.link,
            rows, nothing
        )

        # Check all elements were written
        @test all(!isnan, results_ame)
        @test all(!isnan, results_se)
        @test all(!isnan, gradient_matrix)
    end
end
