# test_links.jl
# Migrated from FormulaCompiler.jl to Margins.jl
# julia --project="." test/primitives/test_links.jl > test/primitives/test_links.txt 2>&1
# Tests link functions for marginal effects on μ scale (statistical interface)

using Test
using FormulaCompiler
using Margins
using DataFrames, Tables, GLM, CategoricalArrays
using LinearAlgebra: dot

# Import functions that are now in Margins
using Margins: marginal_effects_eta!, marginal_effects_mu!

@testset "Link functions: marginal effects μ" begin
    n = 200
    df = DataFrame(y = randn(n), x = randn(n), z = abs.(randn(n)) .+ 0.1,
                   group3 = categorical(rand(["A","B","C"], n)))
    data = Tables.columntable(df)
    model = lm(@formula(y ~ 1 + x + z + x & group3), df)
    compiled = FormulaCompiler.compile_formula(model, data)
    vars = [:x, :z]
    de = FormulaCompiler.derivativeevaluator(:ad, compiled, data, vars)  # Use AD for link function tests
    β = coef(model)

    gη = Vector{Float64}(undef, length(vars))
    Gβ = Matrix{Float64}(undef, length(β), length(vars))
    marginal_effects_eta!(gη, Gβ, de, β, 3)

    links = Any[
        IdentityLink(), LogLink(), LogitLink(),
        ProbitLink(), CloglogLink(), CauchitLink(),
        InverseLink(), SqrtLink()
    ]

    for L in links
        gμ = Vector{Float64}(undef, length(vars))
        # Warm path and correctness (allocations are validated in test_allocations.jl)
        marginal_effects_mu!(gμ, Gβ, de, β, L, 3)
        @test all(isfinite, gμ)
    end

    # Spot check scale agreement for a few links (Identity, Log, Logit)
    row = 5
    xrow = Vector{Float64}(undef, length(compiled))
    compiled(xrow, data, row)
    η = dot(β, xrow)
    # Recompute gη for this row
    gη = Vector{Float64}(undef, length(vars))
    marginal_effects_eta!(gη, Gβ, de, β, row)

    # Identity: scale = 1
    gμ = Vector{Float64}(undef, length(vars))
    marginal_effects_mu!(gμ, Gβ, de, β, IdentityLink(), row)
    @test isapprox(gμ, gη; rtol=1e-8, atol=1e-10)

    # Log: scale = exp(η)
    marginal_effects_mu!(gμ, Gβ, de, β, LogLink(), row)
    @test isapprox(gμ, exp(η) .* gη; rtol=1e-8, atol=1e-10)

    # Logit: scale = σ(η)(1-σ(η))
    σ(x) = inv(1 + exp(-x))
    marginal_effects_mu!(gμ, Gβ, de, β, LogitLink(), row)
    @test isapprox(gμ, (σ(η)*(1-σ(η))) .* gη; rtol=1e-8, atol=1e-10)
end
