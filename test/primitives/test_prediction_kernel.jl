# test_prediction_kernel.jl
#
# The prediction kernel (src/computation/predictions.jl, R6 of the R6/R7 refactor):
# `predscale`, `yhat`/`yhat2`, `predict_row!`, `average_prediction!` and `predict_rows!`,
# against direct computation from the model matrix, for every link Margins supports; plus
# zero allocation of the per-row and averaging kernels.

using Test, Random, LinearAlgebra
using DataFrames, Tables, CategoricalArrays, GLM, StatsModels
using FormulaCompiler, Margins
using Margins: PredScale, LinkScale, ResponseScale, predscale, yhat, yhat2,
               predict_row!, average_prediction!, predict_rows!, rowweight

@testset "prediction kernel" begin
    @testset "predscale / yhat / yhat2" begin
        @test predscale(:link, LogitLink()) === LinkScale()
        @test predscale(:response, LogitLink()) === ResponseScale(LogitLink())
        @test_throws ArgumentError predscale(:mu, LogitLink())
        @test Margins.scale_symbol(LinkScale()) === :link
        @test Margins.scale_symbol(ResponseScale(LogLink())) === :response
        @test yhat(LinkScale(), 0.3) === (0.3, 1.0)
        @test yhat2(LinkScale(), 0.3) === (0.3, 1.0, 0.0)
        links = (IdentityLink(), LogitLink(), ProbitLink(), CloglogLink(), LogLink(),
                 InverseLink(), SqrtLink(), CauchitLink(), NegativeBinomialLink(2.0))
        for L in links, η in (0.2, 0.7, 1.9)
            s = ResponseScale(L)
            μ, d = yhat(s, η)
            @test μ === GLM.linkinv(L, η)
            @test d === FormulaCompiler._dmu_deta(L, η)
            @test yhat2(s, η) === (μ, d, FormulaCompiler._d2mu_deta2(L, η))
            # derivatives agree with central differences of linkinv
            h = 1e-5
            @test d ≈ (GLM.linkinv(L, η + h) - GLM.linkinv(L, η - h)) / 2h rtol = 1e-7
        end
        @test rowweight(nothing, 3) === 1.0
        @test rowweight([1.0, 2.5, 0.0], 2) === 2.5
    end

    Random.seed!(7)
    n = 400
    df = DataFrame(x = randn(n), z = rand(n) .+ 0.5, g = categorical(rand(["a", "b", "c"], n)),
                   b = rand(Bool, n))
    df.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.4 .* df.x .- 0.3 .* df.z .+ 0.2 .* df.b)))
    model = glm(@formula(y ~ x * z + x * g + b), df, Binomial(), LogitLink())
    data = Margins._prepare_margins_data(Tables.columntable(df), model)
    compiled = FormulaCompiler.compile_formula(model, data)
    X = modelmatrix(model)
    β = coef(model)
    η = X * β
    p = length(β)
    x = Vector{Float64}(undef, p)
    weights = rand(n) .+ 0.1
    weights[1:7:end] .= 0.0

    for (s, ŷ, dŷ) in ((LinkScale(), η, ones(n)),
                       (ResponseScale(LogitLink()), GLM.linkinv.(LogitLink(), η),
                        FormulaCompiler._dmu_deta.(LogitLink(), η)))
        @testset "$(Margins.scale_symbol(s))" begin
            # predict_row!
            for row in (1, 17, n)
                yv, d = predict_row!(x, compiled, data, β, s, row)
                @test x ≈ X[row, :] rtol = 1e-14
                @test yv ≈ ŷ[row] rtol = 1e-13
                @test d ≈ dŷ[row] rtol = 1e-13
            end
            # average_prediction!, unweighted and weighted (with zero weights)
            g = zeros(p)
            m = average_prediction!(g, x, compiled, data, β, s, 1:n, nothing, Float64(n))
            @test m ≈ sum(ŷ) / n rtol = 1e-13
            @test g ≈ vec(sum(dŷ .* X; dims=1)) ./ n rtol = 1e-12
            tw = sum(weights)
            m = average_prediction!(g, x, compiled, data, β, s, 1:n, weights, tw)
            @test m ≈ sum(weights .* ŷ) / tw rtol = 1e-13
            @test g ≈ vec(sum((weights .* dŷ) .* X; dims=1)) ./ tw rtol = 1e-12
            # a subset of rows
            rows = collect(5:3:200)
            m = average_prediction!(g, x, compiled, data, β, s, rows, nothing, Float64(length(rows)))
            @test m ≈ sum(ŷ[rows]) / length(rows) rtol = 1e-13
            # predict_rows!
            ŷk = zeros(length(rows)); G = zeros(length(rows), p)
            predict_rows!(ŷk, G, x, compiled, data, β, s, rows)
            @test ŷk ≈ ŷ[rows] rtol = 1e-13
            @test G ≈ dŷ[rows] .* X[rows, :] rtol = 1e-13

            # zero allocation (measured inside functions: on Julia 1.10, @allocated at testset
            # scope can count the boxing of the returned value)
            alloc_avg(w, tw) = @allocated average_prediction!(g, x, compiled, data, β, s, 1:n, w, tw)
            alloc_row(r) = @allocated predict_row!(x, compiled, data, β, s, r)
            alloc_rows(rs) = @allocated predict_rows!(ŷk, G, x, compiled, data, β, s, rs)
            alloc_avg(weights, tw); alloc_avg(nothing, Float64(n)); alloc_row(3); alloc_rows(rows)
            @test alloc_avg(weights, tw) == 0
            @test alloc_avg(nothing, Float64(n)) == 0
            @test alloc_row(3) == 0
            @test alloc_rows(rows) == 0
        end
    end
end
