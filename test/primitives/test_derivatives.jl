# test_derivatives.jl
# Margins' delta_method_se (both methods), and two FormulaCompiler derivative checks that no
# FormulaCompiler test makes yet (fd_jacobian_column!, FD at extreme variable scales).
#
# FormulaCompiler's derivative_modelrow!, AD vs FD, integer columns, transforms, MixedModels and
# contrast_modelrow! are tested in FormulaCompiler's own suite; the internal test oracles
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

@testset "FormulaCompiler FD checks not in its own suite" begin
    Random.seed!(06515)

    @testset "fd_jacobian_column! equals the Jacobian's column" begin
        n = 200
        df = DataFrame(y = randn(n), x = randn(n), z = abs.(randn(n)) .+ 0.1,
                       group3 = categorical(rand(["A", "B", "C"], n)))
        data = Tables.columntable(df)
        compiled = FormulaCompiler.compile_formula(lm(@formula(y ~ 1 + x + z + x & group3), df), data)
        vars = [:x, :z]
        de_ad = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)
        de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)
        J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        J_fd = similar(J_ad)
        FormulaCompiler.derivative_modelrow!(J_ad, de_ad, 5)
        FormulaCompiler.derivative_modelrow!(J_fd, de_fd, 5)
        for i in eachindex(vars)
            Jk = Vector{Float64}(undef, length(compiled))
            FormulaCompiler.fd_jacobian_column!(Jk, de_fd, 5, i)
            @test isapprox(Jk, J_ad[:, i]; rtol = 1e-6, atol = 1e-8)
            @test Jk == J_fd[:, i]
        end
    end

    @testset "FD at extreme variable scales (1e-6, 1e6) matches AD" begin
        n = 100
        df = DataFrame(y = randn(n), x_tiny = randn(n) * 1e-6, x_large = randn(n) * 1e6,
                       x_normal = randn(n),
                       group4 = categorical(rand(["A", "B", "C", "D"], n)),
                       group2 = categorical(rand(["X", "Y"], n)))
        data = Tables.columntable(df)
        model = lm(@formula(y ~ 1 + x_tiny + x_large + x_normal + x_tiny & group4 +
                            x_normal & group2 + x_large & group4), df)
        compiled = FormulaCompiler.compile_formula(model, data)
        vars = [:x_tiny, :x_large, :x_normal]
        de_ad = FormulaCompiler.derivativeevaluator_ad(compiled, data, vars)
        de_fd = FormulaCompiler.derivativeevaluator_fd(compiled, data, vars)
        J_ad = Matrix{Float64}(undef, length(compiled), length(vars))
        J_fd = similar(J_ad)
        for row in (1, 10, 15, 25, 50, 75, 90)
            FormulaCompiler.derivative_modelrow!(J_ad, de_ad, row)
            FormulaCompiler.derivative_modelrow!(J_fd, de_fd, row)
            @test isapprox(J_fd, J_ad; rtol = 1e-3, atol = 1e-6)
        end
    end
end
