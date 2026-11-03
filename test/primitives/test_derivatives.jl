# test_derivatives.jl
# Margins' delta_method_se (both methods).
#
# FormulaCompiler's derivative_modelrow!, AD vs FD, integer columns, transforms, MixedModels and
# contrast_modelrow!, fd_jacobian_column! and FD at extreme scales are tested in FormulaCompiler's
# own suite; the internal test oracles
# marginal_effects_eta!/_mu! in validation/test_gradient_correctness.jl and primitives/test_links.jl.

using Test
using Random
using FormulaCompiler
using Margins
using DataFrames, Tables, GLM, CategoricalArrays, StatsModels
using LinearAlgebra: dot

using Margins: delta_method_se

@testset "delta_method_se" begin
    # gradient form: sqrt(g' Σ g)
    Σ = [1.0 0.1; 0.1 1.0]
    g = [0.5, -0.3]
    @test delta_method_se(g, Σ) ≈ sqrt(0.25 - 2 * 0.015 + 0.09) rtol = 1e-14
    # a unit gradient gives that parameter's SE
    @test delta_method_se([0.0, 1.0], Σ) == 1.0
    # a negative quadratic form (not a covariance matrix) warns and gives NaN
    neg = @test_logs (:warn, r"Negative variance detected") delta_method_se([1.0, 1.0], [1.0 -2.0; -2.0 1.0])
    @test isnan(neg)

    # contrast form: the gradient of the discrete change, on the link and response scales
    rng = MersenneTwister(06515)
    n = 200
    df = DataFrame(x = randn(rng, n), group = categorical(rand(rng, ["A", "B", "C"], n)))
    df.y = rand(rng, n) .< 1 ./ (1 .+ exp.(-(0.5 .* df.x .+ (df.group .== "B"))))
    m = glm(@formula(y ~ x * group), df, Binomial(), LogitLink())
    β, V = coef(m), vcov(m)
    data = Tables.columntable(df)
    ev = FormulaCompiler.contrastevaluator(FormulaCompiler.compile_formula(m, data), data, [:group])
    rhs = formula(m).rhs
    row = 7
    X(level) = vec(modelcols(rhs, DataFrame(x = [df.x[row]], group = categorical([level]; levels = levels(df.group)))))
    Xa, Xc = X("A"), X("C")
    μ′(η) = (p = 1 / (1 + exp(-η)); p * (1 - p))
    for (link, ∇) in ((nothing, Xc .- Xa), (LogitLink(), μ′(dot(Xc, β)) .* Xc .- μ′(dot(Xa, β)) .* Xa))
        @test delta_method_se(ev, row, :group, "A", "C", β, V, link) ≈ sqrt(∇' * V * ∇) rtol = 1e-10
    end
end

