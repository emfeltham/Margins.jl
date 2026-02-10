# test_derivatives_log_profile_regression.jl
# Migrated from FormulaCompiler.jl to Margins.jl
# Regression test for FD derivatives with log transformations and marginal effects

using Test, Random
using DataFrames, Tables, GLM
using FormulaCompiler
using Margins

# Import functions that are now in Margins
using Margins: marginal_effects_eta!

@testset "FD log profile regression" begin
    # Synthetic data with strictly positive x to keep log in-domain
    Random.seed!(123)
    n = 500
    df = DataFrame(
        x = abs.(rand(n)) .+ 10.0,
        z = randn(n),
    )
    df.y = 0.5 .* log.(df.x) .+ 0.3 .* df.z .+ 0.1 .* randn(n)

    # Fit simple log model and compile
    model = lm(@formula(y ~ log(x)), df)
    data = Tables.columntable(df)
    compiled = FormulaCompiler.compile_formula(model, data)

    # Build derivative evaluators for x
    de_fd = FormulaCompiler.derivativeevaluator(:fd, compiled, data, [:x])
    de_ad = FormulaCompiler.derivativeevaluator(:ad, compiled, data, [:x])

    # Target row = 1 (regression covers prior row-1 initialization bug)
    row = 1
    xval = df.x[row]

    # 1) Single-column FD Jacobian should be finite and match analytic [0, 1/x]
    Jk = Vector{Float64}(undef, length(compiled))
    x_idx = findfirst(==(:x), de_fd.vars)
    @test_nowarn FormulaCompiler.fd_jacobian_column!(Jk, de_fd, row, x_idx)
    @test Jk[1] ≈ 0.0 atol=1e-12
    @test Jk[2] ≈ 1.0 / xval rtol=1e-8

    # 2) Full FD Jacobian equals analytic derivative
    J = Matrix{Float64}(undef, length(compiled), length(de_fd.vars))
    @test_nowarn FormulaCompiler.derivative_modelrow!(J, de_fd, row)
    @test J[1, 1] ≈ 0.0 atol=1e-12
    @test J[2, 1] ≈ 1.0 / xval rtol=1e-8

    # 3) AD marginal effects on η should match β1/x (now from Margins)
    β = coef(model)
    g_ad = Vector{Float64}(undef, length(de_ad.vars))
    Gβ_ad = Matrix{Float64}(undef, length(de_ad), length(de_ad.vars))
    @test_nowarn marginal_effects_eta!(g_ad, Gβ_ad, de_ad, β, row)
    @test g_ad[1] ≈ β[2] / xval rtol=1e-8
end
