# Test that categorical kernel maintains zero allocations through FC primitives
using Test, Margins, GLM, DataFrames, Tables, BenchmarkTools, CategoricalArrays
using Margins: build_engine, PopulationUsage, HasDerivatives, categorical_contrast_ame!
using FormulaCompiler: contrast_modelrow!, contrast_gradient!
using LinearAlgebra: dot

@testset "Categorical Kernel Zero-Allocation Verification (Phase 2.6)" begin
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

    @testset "FC Primitives: contrast_modelrow! zero-alloc" begin
        # Build engine
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        # Verify ContrastEvaluator exists
        @test !isnothing(engine.contrast)

        # Warmup
        for i in 1:50
            contrast_modelrow!(engine.contrast_buf, engine.contrast, i, :treatment, "Control", "Drug_A")
        end

        # Benchmark for zero allocations
        b = @benchmark contrast_modelrow!(
            $(engine.contrast_buf),
            $(engine.contrast),
            1,
            :treatment,
            "Control",
            "Drug_A"
        ) samples=100 evals=10

        @test b.allocs == 0
        @test b.memory == 0
    end

    @testset "FC Primitives: contrast_gradient! zero-alloc" begin
        # Build engine
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        # Warmup
        for i in 1:50
            contrast_gradient!(engine.contrast_grad_buf, engine.contrast, i, :treatment, "Control", "Drug_A", engine.β, engine.link)
        end

        # Benchmark for zero allocations
        b = @benchmark contrast_gradient!(
            $(engine.contrast_grad_buf),
            $(engine.contrast),
            1,
            :treatment,
            "Control",
            "Drug_A",
            $(engine.β),
            $(engine.link)
        ) samples=100 evals=10

        @test b.allocs == 0
        @test b.memory == 0
    end

    @testset "Full kernel: categorical_contrast_ame! zero-alloc" begin
        # Build engine with categorical variables
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        rows = collect(1:50)

        # Warmup
        for _ in 1:10
            categorical_contrast_ame!(
                engine.contrast_buf,
                engine.contrast_grad_buf,
                engine.contrast_grad_accum,
                engine.contrast,
                :treatment,
                "Control",
                "Drug_A",
                engine.β,
                engine.Σ,
                engine.link,
                rows,
                nothing
            )
        end

        # Benchmark for zero allocations
        b = @benchmark categorical_contrast_ame!(
            $(engine.contrast_buf),
            $(engine.contrast_grad_buf),
            $(engine.contrast_grad_accum),
            $(engine.contrast),
            :treatment,
            "Control",
            "Drug_A",
            $(engine.β),
            $(engine.Σ),
            $(engine.link),
            $rows,
            nothing
        ) samples=100

        @test b.allocs == 0
        @test b.memory == 0

        # Verify results are reasonable
        ame, se = categorical_contrast_ame!(
            engine.contrast_buf,
            engine.contrast_grad_buf,
            engine.contrast_grad_accum,
            engine.contrast,
            :treatment,
            "Control",
            "Drug_A",
            engine.β,
            engine.Σ,
            engine.link,
            rows,
            nothing
        )
        @test !isnan(ame)
        @test !isnan(se)
        @test se >= 0.0
    end

    @testset "Kernel with weighted computation zero-alloc" begin
        # Build engine
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)

        rows = collect(1:50)
        weights = ones(Float64, n)  # Equal weights

        # Warmup
        for _ in 1:10
            categorical_contrast_ame!(
                engine.contrast_buf,
                engine.contrast_grad_buf,
                engine.contrast_grad_accum,
                engine.contrast,
                :treatment,
                "Control",
                "Drug_A",
                engine.β,
                engine.Σ,
                engine.link,
                rows,
                weights
            )
        end

        # Benchmark for zero allocations
        b = @benchmark categorical_contrast_ame!(
            $(engine.contrast_buf),
            $(engine.contrast_grad_buf),
            $(engine.contrast_grad_accum),
            $(engine.contrast),
            :treatment,
            "Control",
            "Drug_A",
            $(engine.β),
            $(engine.Σ),
            $(engine.link),
            $rows,
            $weights
        ) samples=100

        @test b.allocs == 0
        @test b.memory == 0

        # Verify results
        ame, se = categorical_contrast_ame!(
            engine.contrast_buf,
            engine.contrast_grad_buf,
            engine.contrast_grad_accum,
            engine.contrast,
            :treatment,
            "Control",
            "Drug_A",
            engine.β,
            engine.Σ,
            engine.link,
            rows,
            weights
        )
        @test !isnan(ame)
        @test !isnan(se)
    end

    @testset "Boolean categorical zero-alloc" begin
        # Test with boolean categorical (the problematic case from Phase 4)
        df_bool = DataFrame(
            y = randn(n),
            x = randn(n),
            treated = rand([false, true], n)
        )

        model_bool = lm(@formula(y ~ x + treated), df_bool)
        data_nt_bool = Tables.columntable(df_bool)

        # Build engine
        engine = build_engine(PopulationUsage, HasDerivatives, model_bool, data_nt_bool, [:treated], GLM.vcov, :ad)

        rows = collect(1:50)

        # Warmup
        for _ in 1:10
            categorical_contrast_ame!(
                engine.contrast_buf,
                engine.contrast_grad_buf,
                engine.contrast_grad_accum,
                engine.contrast,
                :treated,
                false,
                true,
                engine.β,
                engine.Σ,
                engine.link,
                rows,
                nothing
            )
        end

        # Benchmark for zero allocations
        b = @benchmark categorical_contrast_ame!(
            $(engine.contrast_buf),
            $(engine.contrast_grad_buf),
            $(engine.contrast_grad_accum),
            $(engine.contrast),
            :treated,
            false,
            true,
            $(engine.β),
            $(engine.Σ),
            $(engine.link),
            $rows,
            nothing
        ) samples=100

        # Boolean categoricals should also achieve zero allocations
        @test b.allocs == 0
        @test b.memory == 0

        ame, se = categorical_contrast_ame!(
            engine.contrast_buf,
            engine.contrast_grad_buf,
            engine.contrast_grad_accum,
            engine.contrast,
            :treated,
            false,
            true,
            engine.β,
            engine.Σ,
            engine.link,
            rows,
            nothing
        )
        @test !isnan(ame)
        @test !isnan(se)
    end

    @testset "Performance benchmarks" begin
        # Build engine
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)
        rows = collect(1:100)

        # Warmup
        for _ in 1:10
            categorical_contrast_ame!(
                engine.contrast_buf,
                engine.contrast_grad_buf,
                engine.contrast_grad_accum,
                engine.contrast,
                :treatment,
                "Control",
                "Drug_A",
                engine.β,
                engine.Σ,
                engine.link,
                rows,
                nothing
            )
        end

        # Benchmark
        b = @benchmark categorical_contrast_ame!(
            $(engine.contrast_buf),
            $(engine.contrast_grad_buf),
            $(engine.contrast_grad_accum),
            $(engine.contrast),
            :treatment,
            "Control",
            "Drug_A",
            $(engine.β),
            $(engine.Σ),
            $(engine.link),
            $rows,
            nothing
        ) samples=1000

        println("\nPerformance Results:")
        println("  Time (median): ", median(b.times) / 1000, " μs")
        println("  Time (min):    ", minimum(b.times) / 1000, " μs")
        println("  Allocations:   ", b.allocs)
        println("  Memory:        ", b.memory, " bytes")

        # Verify zero allocations
        @test b.allocs == 0
        @test b.memory == 0
    end

    @testset "Gradient correctness" begin
        # Verify that gradient is properly accumulated in gradient_accum
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:treatment], GLM.vcov, :ad)
        rows = collect(1:10)

        # Clear gradient accumulator
        fill!(engine.contrast_grad_accum, 0.0)

        # Compute AME
        ame, se = categorical_contrast_ame!(
            engine.contrast_buf,
            engine.contrast_grad_buf,
            engine.contrast_grad_accum,
            engine.contrast,
            :treatment,
            "Control",
            "Drug_A",
            engine.β,
            engine.Σ,
            engine.link,
            rows,
            nothing
        )

        # Verify gradient was populated
        @test !all(engine.contrast_grad_accum .== 0.0)
        @test all(!isnan, engine.contrast_grad_accum)
        @test length(engine.contrast_grad_accum) == length(engine.β)

        # Verify SE is consistent with gradient
        manual_se = sqrt(max(0.0, dot(engine.contrast_grad_accum, engine.Σ, engine.contrast_grad_accum)))
        @test se ≈ manual_se rtol=1e-10
    end
end
