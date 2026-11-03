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

    # The μ-scale effect is dμ/dη at the row's η times the η-scale effect, for every link
    # (reference: GLM.mueta; FormulaCompiler's _dmu_deta itself is checked against BigFloat in
    # its test_link_derivatives.jl)
    Gβ = Matrix{Float64}(undef, length(β), length(vars))
    xrow = Vector{Float64}(undef, length(compiled))
    gη = Vector{Float64}(undef, length(vars))
    gμ = Vector{Float64}(undef, length(vars))
    links = Any[
        IdentityLink(), LogLink(), LogitLink(),
        ProbitLink(), CloglogLink(), CauchitLink(),
        InverseLink(), SqrtLink()
    ]
    for row in (3, 5)
        compiled(xrow, data, row)
        η = dot(β, xrow)
        marginal_effects_eta!(gη, Gβ, de, β, row)
        for L in links
            marginal_effects_mu!(gμ, Gβ, de, β, L, row)
            @test isapprox(gμ, GLM.mueta(L, η) .* gη; rtol=1e-8, atol=1e-10)
        end
    end
end
