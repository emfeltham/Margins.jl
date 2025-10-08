# Comprehensive validation tests for population_margins_raw! API
using Test, Margins, FormulaCompiler, DataFrames, Tables, GLM, BenchmarkTools
using Margins: EffectsBuffers, population_margins_raw!, build_engine, PopulationUsage, HasDerivatives
using Distributions: Normal

@testset "Population Raw Buffers Validation" begin

    @testset "Binomial/Logit Model" begin
        # Test realistic binary outcome model
        n = 1000
        df = DataFrame(
            y = rand(Bool, n),
            x1 = randn(n),
            x2 = randn(n),
            x3 = randn(n),
            category = rand(["A", "B", "C"], n)
        )
        model = glm(@formula(y ~ x1 + x2 + x3), df, Binomial(), LogitLink())
        vars = [:x1, :x2, :x3]
        data_nt = Tables.columntable(df)

        # Build engine and buffers
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)
        buffers = EffectsBuffers(length(vars), length(coef(model)))

        @testset "Zero Allocation Validation - AD Backend" begin
            # Warmup calls
            population_margins_raw!(buffers, engine, data_nt; backend=:ad, scale=:response)
            population_margins_raw!(buffers, engine, data_nt; backend=:ad, scale=:response)

            # Test response scale (allow small allocations for now due to variable filtering)
            b_ad_resp = @benchmark population_margins_raw!($buffers, $engine, $data_nt;
                                                          backend=:ad, scale=:response) samples=100 evals=1
            @test minimum(b_ad_resp.memory) == 0

            # Test link scale
            b_ad_link = @benchmark population_margins_raw!($buffers, $engine, $data_nt;
                                                          backend=:ad, scale=:link) samples=100 evals=1
            @test minimum(b_ad_link.memory) == 0
        end

        @testset "Zero Allocation Validation - FD Backend" begin
            # Warmup calls
            population_margins_raw!(buffers, engine, data_nt; backend=:fd, scale=:response)
            population_margins_raw!(buffers, engine, data_nt; backend=:fd, scale=:response)

            # Test response scale (allow small allocations for now due to variable filtering)
            b_fd_resp = @benchmark population_margins_raw!($buffers, $engine, $data_nt;
                                                          backend=:fd, scale=:response) samples=100 evals=1
            @test minimum(b_fd_resp.memory) == 0

            # Test link scale
            b_fd_link = @benchmark population_margins_raw!($buffers, $engine, $data_nt;
                                                          backend=:fd, scale=:link) samples=100 evals=1
            @test minimum(b_fd_link.memory) == 0
        end

        @testset "Numerical Accuracy vs population_margins" begin
            # Compare against legacy implementation for multiple backends and scales
            test_configs = [
                (backend=:ad, scale=:response),
                (backend=:fd, scale=:response),
                (backend=:ad, scale=:link),
                (backend=:fd, scale=:link)
            ]

            for config in test_configs
                # Get reference result from population_margins
                reference = population_margins(model, df; vars=vars, type=:effects,
                                             backend=config.backend, scale=config.scale)

                # Compute with raw API
                n_obs = population_margins_raw!(buffers, engine, data_nt;
                                               backend=config.backend, scale=config.scale)

                # Compare results
                @test n_obs == size(df, 1)
                @test length(reference.estimates) == length(vars)

                # Extract results from buffers
                raw_estimates = buffers.estimates[1:length(vars)]
                raw_ses = buffers.standard_errors[1:length(vars)]

                # Numerical comparison with tight tolerances
                @test raw_estimates ≈ reference.estimates rtol=1e-12 atol=1e-15
                @test raw_ses ≈ reference.standard_errors rtol=1e-12 atol=1e-15

                println("✓ $(config.backend)/$(config.scale): estimates match within 1e-12 relative tolerance")
            end
        end
    end

    @testset "Gaussian/Identity Model" begin
        # Test continuous outcome model
        n = 800
        df = DataFrame(
            y = randn(n),
            x1 = randn(n),
            x2 = randn(n),
            x3 = 0.5 * randn(n) .+ 2.0  # Different scale to test robustness
        )
        model = lm(@formula(y ~ x1 + x2 + x3), df)  # Linear model with identity link
        vars = [:x1, :x2]  # Test subset of variables
        data_nt = Tables.columntable(df)

        # Build engine and buffers
        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)
        buffers = EffectsBuffers(length(vars), length(coef(model)))

        @testset "Zero Allocation Validation" begin
            # Warmup
            population_margins_raw!(buffers, engine, data_nt; backend=:fd, scale=:response)
            population_margins_raw!(buffers, engine, data_nt; backend=:fd, scale=:response)

            # For Identity link, response and linear scales should be identical (small fixed allocations)
            b_identity = @benchmark population_margins_raw!($buffers, $engine, $data_nt;
                                                           backend=:fd, scale=:response) samples=100 evals=1
            @test minimum(b_identity.memory) == 0
        end

        @testset "Numerical Accuracy for Linear Model" begin
            # Compare AD vs FD backends for consistency
            n_obs_ad = population_margins_raw!(buffers, engine, data_nt; backend=:ad, scale=:response)
            estimates_ad = copy(buffers.estimates[1:length(vars)])
            ses_ad = copy(buffers.standard_errors[1:length(vars)])

            n_obs_fd = population_margins_raw!(buffers, engine, data_nt; backend=:fd, scale=:response)
            estimates_fd = copy(buffers.estimates[1:length(vars)])
            ses_fd = copy(buffers.standard_errors[1:length(vars)])

            # AD and FD should give nearly identical results for linear models
            @test n_obs_ad == n_obs_fd
            @test estimates_ad ≈ estimates_fd rtol=1e-10
            @test ses_ad ≈ ses_fd rtol=1e-8  # SEs may have slightly more variation

            # Compare against population_margins
            reference = population_margins(model, df; vars=vars, type=:effects, backend=:ad)
            @test estimates_ad ≈ reference.estimates rtol=1e-12
            @test ses_ad ≈ reference.standard_errors rtol=1e-12
        end
    end

    @testset "Edge Cases and Error Handling" begin
        n = 100
        df = DataFrame(
            y = rand(Bool, n),
            x1 = randn(n),
            x2 = randn(n)
        )
        model = glm(@formula(y ~ x1 + x2), df, Binomial(), LogitLink())
        data_nt = Tables.columntable(df)
        vars = [:x1]

        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)
        buffers = EffectsBuffers(2, length(coef(model)))

        @testset "Weighted Analysis Guard" begin
            weights = ones(n)
            @test_throws ArgumentError population_margins_raw!(buffers, engine, data_nt;
                                                             weights=weights)
        end

        @testset "No Derivatives Engine Guard" begin
            engine_no_deriv = build_engine(PopulationUsage, Margins.NoDerivatives, model, data_nt, vars, GLM.vcov)
            @test_throws ArgumentError population_margins_raw!(buffers, engine_no_deriv, data_nt)
        end

        @testset "Buffer Capacity Management" begin
            # Test with small buffers that need expansion
            small_buffers = EffectsBuffers(1, 2)  # Too small

            # Should work due to ensure_capacity!
            n_obs = population_margins_raw!(small_buffers, engine, data_nt; backend=:fd)
            @test n_obs == n
            @test small_buffers.estimates[1] isa Float64  # Should have valid result
        end

        @testset "Empty Variables List" begin
            # Test behavior with no continuous variables requested
            engine_empty = build_engine(PopulationUsage, HasDerivatives, model, data_nt, Symbol[], GLM.vcov)
            buffers_empty = EffectsBuffers(1, length(coef(model)))

            n_obs = population_margins_raw!(buffers_empty, engine_empty, data_nt)
            @test n_obs == 0  # Should return 0 for no variables to process
        end
    end

    @testset "Performance Characteristics" begin
        # Test that the raw API is significantly faster than DataFrame creation
        n = 500
        df = DataFrame(y = rand(Bool, n), x1 = randn(n), x2 = randn(n))
        model = glm(@formula(y ~ x1 + x2), df, Binomial(), LogitLink())
        vars = [:x1, :x2]
        data_nt = Tables.columntable(df)

        engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, vars, GLM.vcov)
        buffers = EffectsBuffers(length(vars), length(coef(model)))

        # Warmup both approaches
        population_margins_raw!(buffers, engine, data_nt; backend=:fd)
        population_margins(model, df; vars=vars, backend=:fd)

        # Benchmark raw API
        b_raw = @benchmark population_margins_raw!($buffers, $engine, $data_nt; backend=:fd) samples=100

        # Benchmark legacy API
        b_legacy = @benchmark population_margins($model, $df; vars=$vars, backend=:fd) samples=20

        println("Raw API median time: $(median(b_raw.times) / 1e6) ms")
        println("Legacy API median time: $(median(b_legacy.times) / 1e6) ms")
        println("Raw API allocations: $(minimum(b_raw.memory)) bytes")
        println("Legacy API allocations: $(minimum(b_legacy.memory)) bytes")

        # Raw API should have small fixed allocations (much better than legacy)
        @test minimum(b_raw.memory) == 0
        @test minimum(b_raw.memory) < minimum(b_legacy.memory) / 5  # At least 5x better

        # Raw API should be faster (though legacy includes DataFrame construction)
        @test median(b_raw.times) < median(b_legacy.times)
    end
end
