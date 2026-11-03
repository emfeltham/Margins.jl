# test_second_differences_observed.jl
# at = :observed: the average derivative over the observed modifier values

using Test
using Margins
using DataFrames
using GLM
using CategoricalArrays
using StatsModels
using Tables
using Statistics
using Random

@testset "second differences: at = :observed" begin
    Random.seed!(41)
    n = 500
    df = DataFrame(x = randn(n), w = randn(n) .+ 0.3, z = randn(n), t = rand([0.0, 1.0], n), sw = 0.5 .+ rand(n))
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.2df.z .+ 0.5df.x .* df.w .+ randn(n)
    df.yb = (0.1 .+ 0.7df.x .+ 0.4df.w .+ 0.3df.x .* df.w .+ 0.3 .* df.x .^ 2 .+ randn(n)) .> 0
    lin = lm(@formula(y ~ x * w + z), df)
    lg = glm(@formula(yb ~ x * w + z + t + x^2), df, Binomial(), LogitLink())

    @testset "linear model: the interaction coefficient, like any evaluation point" begin
        k = findfirst(==("x & w"), coefnames(lin))
        ob = second_differences(lin, df, :x, :w; at = :observed)
        @test nrow(ob) == 1
        @test ob.second_diff[1] ≈ coef(lin)[k] rtol = 1e-6
        @test ob.se[1] ≈ sqrt(vcov(lin)[k, k]) rtol = 1e-6
        @test ob.df[1] == dof_residual(lin)
        @test second_differences(lin, df, :x, :w).second_diff[1] ≈ ob.second_diff[1] rtol = 1e-6
    end

    @testset "row layout" begin
        ob = second_differences(lg, df, :x, :w; at = :observed, warn = false)
        @test isnan(ob.eval_point[1]) && isnan(ob.modifier_level1[1]) && isnan(ob.modifier_level2[1])
        @test ob.delta_used[1] > 0 && ob.modifier_type == [:continuous]
        @test ob.variable == [:x] && ob.contrast == ["dy/dx"]
        # differs from the derivative at the mean of the modifier in a nonlinear model
        @test !isapprox(ob.second_diff[1], second_differences(lg, df, :x, :w; at = :mean, warn = false).second_diff[1]; rtol = 1e-3)
        # profile strata, several points, several variables, p_adjust, gradients, contrast selection
        pr = second_differences(lg, df, [:x, :z], :w; at = :observed, profile = (t = [0.0, 1.0],), p_adjust = :holm,
                                include_gradients = true, warn = false)
        @test nrow(pr) == 4 && sort(unique(pr.t)) == [0.0, 1.0] && all(isnan, pr.eval_point)
        @test hasproperty(pr, :p_adjusted) && hasproperty(pr, :gradient)
        @test second_differences(lg, df, :x, :w; at = :observed, contrast = "dy/dx", warn = false).second_diff == ob.second_diff
    end

    @testset "equals the average unit-level cross-derivative (closed-form oracle)" begin
        rhs = formula(lg).rhs
        X(dd) = modelcols(rhs, Tables.columntable(dd))
        function est(β, scale, w; δ = 0.01, h = 1e-5)
            function ame(shift)
                dd = copy(df); dd.w .+= shift
                dp = copy(dd); dp.x .+= h
                dm = copy(dd); dm.x .-= h
                pred(Xs) = scale === :response ? 1 ./ (1 .+ exp.(-(Xs * β))) : Xs * β
                g = (pred(X(dp)) .- pred(X(dm))) ./ (2h)
                isnothing(w) ? mean(g) : sum(w .* g) / sum(w)
            end
            (ame(δ) - ame(-δ)) / (2δ)
        end
        β = coef(lg)
        for (scale, w) in ((:response, nothing), (:link, nothing), (:response, df.sw))
            grad = map(eachindex(β)) do k
                st = 1e-5 * max(abs(β[k]), 1e-2)
                bp = copy(β); bp[k] += st; bm = copy(β); bm[k] -= st
                (est(bp, scale, w) - est(bm, scale, w)) / (2st)
            end
            sd = second_differences(lg, df, :x, :w; at = :observed, delta = 0.01, scale, weights = w, warn = false)
            @test sd.second_diff[1] ≈ est(β, scale, w) rtol = 1e-5
            @test sd.se[1] ≈ sqrt(grad' * vcov(lg) * grad) rtol = 2e-3
        end
    end

    @testset "a modifier the model does not use is a structural zero" begin
        m = lm(@formula(y ~ x + z), df)
        sd = @test_logs (:warn, r"zero by construction") second_differences(m, df, :x, :w; at = :observed)
        @test sd.second_diff[1] == 0.0 && isnan(sd.p_value[1])
    end

    @testset "errors" begin
        @test_throws ArgumentError second_differences(lg, df, :x, :t; at = :observed)   # 0/1 modifier is discrete
        @test_throws ArgumentError second_differences(lg, df, :x, :w; at = :nonsense)
    end
end
