# Test MarginsEngine batch buffer allocation behavior
using Test, Margins, GLM, DataFrames, Tables, BenchmarkTools
using Margins: build_engine, PopulationUsage, ProfileUsage, HasDerivatives, NoDerivatives

@testset "Engine Batch Buffer Tests" begin
    # Setup test data
    n = 100
    df = DataFrame(
        y = rand(Bool, n),
        x1 = randn(n),
        x2 = randn(n),
        x3 = randn(n)
    )

    model = glm(@formula(y ~ x1 + x2 + x3), df, Binomial(), LogitLink())
    data_nt = Tables.columntable(df)
    vars = [:x1, :x2]

    @testset "PopulationUsage + HasDerivatives" begin
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)

        # Test field existence and types
        @test hasfield(typeof(engine), :batch_ame_values)
        @test hasfield(typeof(engine), :batch_gradients)
        @test hasfield(typeof(engine), :batch_var_indices)
        @test hasfield(typeof(engine), :effects_buffers)
        @test hasfield(typeof(engine), :continuous_vars)

        @test engine.batch_ame_values isa Vector{Float64}
        @test engine.batch_gradients isa Matrix{Float64}
        @test engine.batch_var_indices isa Vector{Int}
        @test engine.effects_buffers isa Margins.EffectsBuffers
        @test engine.continuous_vars isa Vector{Symbol}

        # Test sizes (should be at least as large as number of continuous variables)
        n_continuous = 3  # x1, x2, x3
        n_coef = length(coef(model))

        @test length(engine.batch_ame_values) >= n_continuous
        @test size(engine.batch_gradients, 1) >= n_continuous
        @test size(engine.batch_gradients, 2) == n_coef
        @test length(engine.batch_var_indices) >= n_continuous

        # Test zero-allocation access after warmup
        # First access for warmup
        _ = engine.batch_ame_values[1]
        _ = engine.batch_gradients[1, 1]
        _ = engine.batch_var_indices[1]

        # Test zero-allocation access
        b1 = @benchmark $(engine.batch_ame_values)[1] samples=50 evals=1
        @test minimum(b1.memory) == 0

        b2 = @benchmark $(engine.batch_gradients)[1, 1] samples=50 evals=1
        @test minimum(b2.memory) == 0

        b3 = @benchmark $(engine.batch_var_indices)[1] samples=50 evals=1
        @test minimum(b3.memory) == 0

        # Test zero-allocation view access
        b4 = @benchmark view($(engine.batch_ame_values), 1:2) samples=50 evals=1
        @test minimum(b4.memory) == 0

        b5 = @benchmark view($(engine.batch_gradients), 1:2, :) samples=50 evals=1
        @test minimum(b5.memory) == 0
    end

    @testset "ProfileUsage + HasDerivatives" begin
        engine = build_engine(ProfileUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)

        # Similar tests but may have different sizing strategy
        @test engine.batch_ame_values isa Vector{Float64}
        @test engine.batch_gradients isa Matrix{Float64}
        @test engine.batch_var_indices isa Vector{Int}
        @test engine.effects_buffers isa Margins.EffectsBuffers

        # ProfileUsage might have different buffer sizing but still adequate
        n_continuous = 3
        n_coef = length(coef(model))

        @test length(engine.batch_ame_values) >= n_continuous
        @test size(engine.batch_gradients, 1) >= n_continuous
        @test size(engine.batch_gradients, 2) == n_coef

        # Zero-allocation access tests
        _ = engine.batch_ame_values[1]  # warmup

        b1 = @benchmark $(engine.batch_ame_values)[1] samples=50 evals=1
        @test minimum(b1.memory) == 0
    end

    @testset "NoDerivatives Engine" begin
        engine = build_engine(PopulationUsage, NoDerivatives, model, data_nt, vars, GLM.vcov)

        # Even NoDerivatives engines should have batch buffers for consistency
        @test hasfield(typeof(engine), :batch_ame_values)
        @test hasfield(typeof(engine), :batch_gradients)
        @test hasfield(typeof(engine), :batch_var_indices)
        @test hasfield(typeof(engine), :effects_buffers)

        # Should have minimal but usable sizes
        @test length(engine.batch_ame_values) >= 1
        @test size(engine.batch_gradients, 1) >= 1
        @test length(engine.batch_var_indices) >= 1

        # Zero-allocation access
        _ = engine.batch_ame_values[1]  # warmup

        b1 = @benchmark $(engine.batch_ame_values)[1] samples=50 evals=1
        @test minimum(b1.memory) == 0
    end

    @testset "Buffer Reuse Pattern" begin
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)

        # Simulate realistic buffer usage pattern
        n_vars = 2

        # Warmup
        engine.batch_ame_values[1:n_vars] .= 0.0
        fill!(view(engine.batch_gradients, 1:n_vars, :), 0.0)
        engine.batch_var_indices[1:n_vars] .= [1, 2]

        # Test zero-allocation view access (most common pattern)
        b_view1 = @benchmark view($(engine.batch_ame_values), 1:$n_vars) samples=50 evals=1
        @test minimum(b_view1.memory) == 0

        b_view2 = @benchmark view($(engine.batch_gradients), 1:$n_vars, :) samples=50 evals=1
        @test minimum(b_view2.memory) == 0

        b_view3 = @benchmark view($(engine.batch_var_indices), 1:$n_vars) samples=50 evals=1
        @test minimum(b_view3.memory) == 0

        # Test direct access (should also be zero-allocation)
        b_access = @benchmark begin
            val = $(engine.batch_ame_values)[1]
            grad = $(engine.batch_gradients)[1, 1]
            idx = $(engine.batch_var_indices)[1]
        end samples=50 evals=1
        @test minimum(b_access.memory) == 0
    end

    @testset "Multiple Engine Types" begin
        # Test that all engine type combinations work
        usage_types = [PopulationUsage, ProfileUsage]
        deriv_types = [HasDerivatives, NoDerivatives]

        for usage in usage_types, deriv in deriv_types
            engine = build_engine(usage, deriv, model, data_nt, vars, GLM.vcov)

            # Basic existence checks
            @test hasfield(typeof(engine), :batch_ame_values)
            @test hasfield(typeof(engine), :batch_gradients)
            @test hasfield(typeof(engine), :batch_var_indices)
            @test hasfield(typeof(engine), :effects_buffers)

            # All should have positive sizes
            @test length(engine.batch_ame_values) > 0
            @test size(engine.batch_gradients, 1) > 0
            @test size(engine.batch_gradients, 2) > 0
            @test length(engine.batch_var_indices) > 0
        end
    end
end
