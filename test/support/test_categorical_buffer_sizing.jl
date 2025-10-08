# Test MarginsEngine categorical buffer sizing (Phase 4 zero-allocation optimization)
using Test, Margins, GLM, DataFrames, Tables, BenchmarkTools, CategoricalArrays
using Margins: build_engine, PopulationUsage, ProfileUsage, HasDerivatives, NoDerivatives
using Margins: verify_and_repair_engine_buffers!

@testset "Categorical Buffer Sizing Tests (Phase 4)" begin
    # Setup test data with categorical variables
    n = 100
    df = DataFrame(
        y = rand(Bool, n),
        x1 = randn(n),
        x2 = randn(n),
        treatment = categorical(rand(["Control", "Drug_A", "Drug_B"], n)),
        group = categorical(rand(["A", "B", "C"], n))
    )

    model = glm(@formula(y ~ x1 + x2 + treatment + group), df, Binomial(), LogitLink())
    data_nt = Tables.columntable(df)
    n_coef = length(coef(model))

    @testset "Categorical-only Engine (NoDerivatives)" begin
        # Test engine with only categorical variables
        categorical_vars = [:treatment, :group]
        engine = build_engine(PopulationUsage, NoDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)

        # Test field existence and types
        @test hasfield(typeof(engine), :contrast_buf)
        @test hasfield(typeof(engine), :contrast_grad_buf)
        @test hasfield(typeof(engine), :contrast_grad_accum)

        @test engine.contrast_buf isa Vector{Float64}
        @test engine.contrast_grad_buf isa Vector{Float64}
        @test engine.contrast_grad_accum isa Vector{Float64}

        # Test correct sizing (all buffers should be sized to n_coef)
        @test length(engine.contrast_buf) == n_coef
        @test length(engine.contrast_grad_buf) == n_coef
        @test length(engine.contrast_grad_accum) == n_coef

        # Test that categorical variables were detected
        @test engine.categorical_vars == categorical_vars
        # Note: continuous_vars may include formula variables even if not requested
        # This is expected behavior - we only analyze requested vars

        # Test that ContrastEvaluator was built
        @test !isnothing(engine.contrast)
    end

    @testset "Mixed continuous + categorical Engine (HasDerivatives)" begin
        # Test engine with both continuous and categorical variables
        mixed_vars = [:x1, :treatment, :group]
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, mixed_vars, GLM.vcov, :ad)

        # Test categorical buffer existence
        @test hasfield(typeof(engine), :contrast_buf)
        @test hasfield(typeof(engine), :contrast_grad_buf)
        @test hasfield(typeof(engine), :contrast_grad_accum)

        @test engine.contrast_buf isa Vector{Float64}
        @test engine.contrast_grad_buf isa Vector{Float64}
        @test engine.contrast_grad_accum isa Vector{Float64}

        # Test correct sizing
        @test length(engine.contrast_buf) == n_coef
        @test length(engine.contrast_grad_buf) == n_coef
        @test length(engine.contrast_grad_accum) == n_coef

        # Test variable detection
        @test :x1 in engine.continuous_vars
        @test :treatment in engine.categorical_vars
        @test :group in engine.categorical_vars

        # Test that both evaluators exist
        @test !isnothing(engine.de)  # DerivativeEvaluator for continuous
        @test !isnothing(engine.contrast)  # ContrastEvaluator for categorical
    end

    @testset "Zero-allocation buffer access" begin
        categorical_vars = [:treatment]
        engine = build_engine(PopulationUsage, NoDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)

        # Warmup
        _ = engine.contrast_buf[1]
        _ = engine.contrast_grad_buf[1]
        _ = engine.contrast_grad_accum[1]

        # Test zero-allocation scalar access
        b1 = @benchmark $(engine.contrast_buf)[1] samples=50 evals=1
        @test minimum(b1.memory) == 0

        b2 = @benchmark $(engine.contrast_grad_buf)[1] samples=50 evals=1
        @test minimum(b2.memory) == 0

        b3 = @benchmark $(engine.contrast_grad_accum)[1] samples=50 evals=1
        @test minimum(b3.memory) == 0

        # Test zero-allocation view access (hot path pattern)
        b4 = @benchmark view($(engine.contrast_buf), 1:$n_coef) samples=50 evals=1
        @test minimum(b4.memory) == 0

        b5 = @benchmark view($(engine.contrast_grad_buf), 1:$n_coef) samples=50 evals=1
        @test minimum(b5.memory) == 0

        b6 = @benchmark view($(engine.contrast_grad_accum), 1:$n_coef) samples=50 evals=1
        @test minimum(b6.memory) == 0
    end

    @testset "Buffer reuse pattern (fill! operations)" begin
        categorical_vars = [:treatment, :group]
        engine = build_engine(PopulationUsage, NoDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)

        # Warmup
        fill!(engine.contrast_buf, 0.0)
        fill!(engine.contrast_grad_buf, 0.0)
        fill!(engine.contrast_grad_accum, 0.0)

        # Test zero-allocation fill! (used in hot loop)
        b1 = @benchmark fill!($(engine.contrast_buf), 0.0) samples=50 evals=1
        @test minimum(b1.memory) == 0

        b2 = @benchmark fill!($(engine.contrast_grad_buf), 0.0) samples=50 evals=1
        @test minimum(b2.memory) == 0

        b3 = @benchmark fill!($(engine.contrast_grad_accum), 0.0) samples=50 evals=1
        @test minimum(b3.memory) == 0

        # Test zero-allocation copyto! (gradient accumulation pattern)
        temp = zeros(n_coef)
        b4 = @benchmark copyto!($(engine.contrast_grad_accum), $temp) samples=50 evals=1
        @test minimum(b4.memory) == 0
    end

    @testset "All engine type combinations" begin
        usage_types = [PopulationUsage, ProfileUsage]
        deriv_types = [HasDerivatives, NoDerivatives]
        categorical_vars = [:treatment, :group]

        for usage in usage_types, deriv in deriv_types
            engine = build_engine(usage, deriv, model, data_nt, categorical_vars, GLM.vcov, :ad)

            # Basic existence checks
            @test hasfield(typeof(engine), :contrast_buf)
            @test hasfield(typeof(engine), :contrast_grad_buf)
            @test hasfield(typeof(engine), :contrast_grad_accum)

            # All buffers should be sized to n_coef
            @test length(engine.contrast_buf) == n_coef
            @test length(engine.contrast_grad_buf) == n_coef
            @test length(engine.contrast_grad_accum) == n_coef

            # Categorical variables should be detected
            @test Set(engine.categorical_vars) == Set(categorical_vars)
        end
    end

    @testset "verify_and_repair_engine_buffers! - HasDerivatives" begin
        categorical_vars = [:treatment]
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)

        # Create engine with incorrectly sized buffers (simulate corruption)
        bad_engine = Margins.MarginsEngine{typeof(engine.link), PopulationUsage, HasDerivatives}(
            engine.compiled, engine.de, engine.contrast,
            engine.g_buf, engine.gβ_accumulator, engine.η_buf, engine.row_buf,
            engine.batch_ame_values, engine.batch_gradients, engine.batch_var_indices,
            engine.deta_dx_buf, engine.cont_var_indices_buf,
            engine.g_all_buf, engine.Gβ_all_buf,
            Vector{Float64}(undef, 1),  # Wrong size: should be n_coef
            Vector{Float64}(undef, 1),  # Wrong size: should be n_coef
            Vector{Float64}(undef, 1),  # Wrong size: should be n_coef
            engine.effects_buffers,
            engine.continuous_vars, engine.categorical_vars,
            engine.model, engine.β, engine.Σ, engine.link, engine.vars, engine.data_nt
        )

        # Verify buffers are wrong
        @test length(bad_engine.contrast_buf) != n_coef
        @test length(bad_engine.contrast_grad_buf) != n_coef
        @test length(bad_engine.contrast_grad_accum) != n_coef

        # Repair should fix buffer sizes
        repaired = verify_and_repair_engine_buffers!(bad_engine)

        # Verify buffers are now correct
        @test length(repaired.contrast_buf) == n_coef
        @test length(repaired.contrast_grad_buf) == n_coef
        @test length(repaired.contrast_grad_accum) == n_coef
    end

    @testset "verify_and_repair_engine_buffers! - NoDerivatives" begin
        categorical_vars = [:treatment, :group]
        engine = build_engine(PopulationUsage, NoDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)

        # Create engine with incorrectly sized buffers
        bad_engine = Margins.MarginsEngine{typeof(engine.link), PopulationUsage, NoDerivatives}(
            engine.compiled, engine.de, engine.contrast,
            engine.g_buf, engine.gβ_accumulator, engine.η_buf, engine.row_buf,
            engine.batch_ame_values, engine.batch_gradients, engine.batch_var_indices,
            engine.deta_dx_buf, engine.cont_var_indices_buf,
            engine.g_all_buf, engine.Gβ_all_buf,
            Vector{Float64}(undef, 2),  # Wrong size: should be n_coef
            Vector{Float64}(undef, 2),  # Wrong size: should be n_coef
            Vector{Float64}(undef, 2),  # Wrong size: should be n_coef
            engine.effects_buffers,
            engine.continuous_vars, engine.categorical_vars,
            engine.model, engine.β, engine.Σ, engine.link, engine.vars, engine.data_nt
        )

        # Verify buffers are wrong
        @test length(bad_engine.contrast_buf) != n_coef
        @test length(bad_engine.contrast_grad_buf) != n_coef
        @test length(bad_engine.contrast_grad_accum) != n_coef

        # Repair should fix buffer sizes (NoDerivatives version)
        repaired = verify_and_repair_engine_buffers!(bad_engine)

        # Verify buffers are now correct
        @test length(repaired.contrast_buf) == n_coef
        @test length(repaired.contrast_grad_buf) == n_coef
        @test length(repaired.contrast_grad_accum) == n_coef
    end

    @testset "Boolean categorical buffers" begin
        # Test with boolean categorical (the problematic case from Phase 4)
        df_bool = DataFrame(
            y = randn(n),
            x = randn(n),
            treated = rand([false, true], n)
        )

        model_bool = lm(@formula(y ~ x + treated), df_bool)
        data_nt_bool = Tables.columntable(df_bool)
        n_coef_bool = length(coef(model_bool))

        engine = build_engine(PopulationUsage, NoDerivatives, model_bool, data_nt_bool, [:treated], GLM.vcov, :ad)

        # Boolean categoricals should have same buffer sizing as multi-level
        @test length(engine.contrast_buf) == n_coef_bool
        @test length(engine.contrast_grad_buf) == n_coef_bool
        @test length(engine.contrast_grad_accum) == n_coef_bool

        # Should detect boolean as categorical
        @test :treated in engine.categorical_vars
        @test !isnothing(engine.contrast)
    end

    @testset "Profile vs Population buffer sizing" begin
        categorical_vars = [:treatment]

        # Build both usage types
        pop_engine = build_engine(PopulationUsage, NoDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)
        prof_engine = build_engine(ProfileUsage, NoDerivatives, model, data_nt, categorical_vars, GLM.vcov, :ad)

        # Categorical buffers should be same size regardless of usage pattern
        # (sized to n_coef, not data-dependent)
        @test length(pop_engine.contrast_buf) == n_coef
        @test length(prof_engine.contrast_buf) == n_coef
        @test length(pop_engine.contrast_buf) == length(prof_engine.contrast_buf)

        @test length(pop_engine.contrast_grad_buf) == n_coef
        @test length(prof_engine.contrast_grad_buf) == n_coef
        @test length(pop_engine.contrast_grad_buf) == length(prof_engine.contrast_grad_buf)

        @test length(pop_engine.contrast_grad_accum) == n_coef
        @test length(prof_engine.contrast_grad_accum) == n_coef
        @test length(pop_engine.contrast_grad_accum) == length(prof_engine.contrast_grad_accum)
    end
end
