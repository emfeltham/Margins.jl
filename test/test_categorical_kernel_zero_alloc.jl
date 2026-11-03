# test_categorical_kernel_zero_alloc.jl
#
# The categorical-contrast kernels (`kernels/categorical.jl`):
# - `categorical_contrast_ame!` (single pair, used by profile margins) allocates nothing, for a
#   multi-level categorical and a Bool variable, on both scales, unweighted and weighted;
# - `categorical_contrast_ame_batch!` allocates nothing for a Bool variable and for weights
#   without an offset (the multi-level cases with and without an offset are also in
#   `performance/test_ame_fd_loop_zero_alloc.jl`);
# - the batch reproduces the single-pair kernel exactly, and the single-pair SE is √(g'Σg).
#
# Allocations are measured after a warm-up, inside a function (on Julia 1.10, `@allocated` at
# testset scope counts boxing).
#
# (3.0.0: merges the former `test_categorical_batch_zero_alloc.jl` into this file.)

using Test, Random
using Margins, GLM, DataFrames, Tables, CategoricalArrays
using Margins: build_engine, PopulationUsage, categorical_contrast_ame!,
               categorical_contrast_ame_batch!, generate_contrast_pairs
using LinearAlgebra: dot

function _catk_single_allocs(engine, var, from, to, s, rows, weights)
    f() = categorical_contrast_ame!(engine.contrast_buf, engine.gβ_accumulator, engine.contrast, var,
                                    from, to, engine.β, engine.Σ, s, rows, weights)
    f()
    return @allocated f()
end

function _catk_batch_allocs(ame, se, G, engine, var, pairs, s, rows, weights)
    f() = categorical_contrast_ame_batch!(ame, se, G, engine.contrast_buf, engine.contrast, var,
                                          pairs, engine.β, engine.Σ, s, rows, weights)
    f()
    return @allocated f()
end

@testset "Categorical kernels: zero allocation and batch = single pair" begin
    Random.seed!(20251102)
    n = 100
    df = DataFrame(y = rand(Bool, n), x = randn(n),
                   treatment = categorical(rand(["Control", "Drug_A", "Drug_B"], n)))
    model = glm(@formula(y ~ x + treatment), df, Binomial(), LogitLink())
    data_nt = Tables.columntable(df)

    df_bool = DataFrame(y = randn(n), x = randn(n), treated = rand([false, true], n))
    model_bool = lm(@formula(y ~ x + treated), df_bool)
    data_bool = Tables.columntable(df_bool)

    w = rand(n) .+ 0.1
    cases = ((model, data_nt, :treatment, "Control", "Drug_A"),
             (model_bool, data_bool, :treated, false, true))

    @testset "Zero allocation: $var" for (m, d, var, from, to) in cases
        engine = build_engine(PopulationUsage, m, d, [var], GLM.vcov, :ad)
        @test !isnothing(engine.contrast)
        rows = collect(1:50)
        pairs = generate_contrast_pairs(getproperty(d, var), rows, :baseline, m, var, d)
        k = length(pairs); p = length(engine.β)
        ame = fill(NaN, k); se = fill(NaN, k); G = fill(NaN, k, p)
        for s in (Margins.LinkScale(), Margins.ResponseScale(engine.link)), weights in (nothing, w)
            @test _catk_single_allocs(engine, var, from, to, s, rows, weights) == 0
            est, sd = categorical_contrast_ame!(engine.contrast_buf, engine.gβ_accumulator,
                                                engine.contrast, var, from, to, engine.β, engine.Σ,
                                                s, rows, weights)
            @test isfinite(est) && isfinite(sd) && sd > 0

            @test _catk_batch_allocs(ame, se, G, engine, var, pairs, s, rows, weights) == 0
            @test all(isfinite, ame) && all(isfinite, se) && all(isfinite, G)
        end
    end

    @testset "Single-pair SE is the delta method on its gradient" begin
        engine = build_engine(PopulationUsage, model, data_nt, [:treatment], GLM.vcov, :ad)
        fill!(engine.gβ_accumulator, 0.0)
        ame, se = categorical_contrast_ame!(engine.contrast_buf, engine.gβ_accumulator, engine.contrast,
                                            :treatment, "Control", "Drug_A", engine.β, engine.Σ,
                                            Margins.ResponseScale(engine.link), 1:10, nothing)
        @test !all(iszero, engine.gβ_accumulator)
        @test se ≈ sqrt(max(0.0, dot(engine.gβ_accumulator, engine.Σ, engine.gβ_accumulator))) rtol = 1e-10
    end

    @testset "Batched pairs reproduce the single-pair kernel (pairwise, both scales, weights)" begin
        # The batch evaluates X₀ once per row and level; the arithmetic per pair is the same
        # as the single-pair kernel's, so estimates and gradients agree exactly.
        engine = build_engine(PopulationUsage, model, data_nt, [:treatment], GLM.vcov, :ad)
        n_coef = length(engine.β)
        var_col = data_nt.treatment
        rows = collect(1:n)
        wz = rand(n); wz[1:5:end] .= 0.0
        for contrasts in (:baseline, :pairwise), s in (Margins.LinkScale(), Margins.ResponseScale(engine.link)),
            weights in (nothing, wz)
            pairs = generate_contrast_pairs(var_col, rows, contrasts, model, :treatment, data_nt)
            k = length(pairs)
            ame_b = zeros(k); se_b = zeros(k); G_b = zeros(k, n_coef)
            categorical_contrast_ame_batch!(ame_b, se_b, G_b, engine.contrast_buf, engine.contrast,
                                            :treatment, pairs, engine.β, engine.Σ, s, rows, weights)
            for (i, pair) in enumerate(pairs)
                ame, se = categorical_contrast_ame!(
                    engine.contrast_buf, engine.gβ_accumulator, engine.contrast, :treatment,
                    pair.level1, pair.level2, engine.β, engine.Σ, s, rows, weights)
                @test ame_b[i] == ame
                @test G_b[i, :] == engine.gβ_accumulator
                @test se_b[i] ≈ se rtol = 1e-14
            end
        end
    end
end
