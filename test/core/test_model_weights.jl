# test_model_weights.jl
# weights=:model (the default): averages use the fitted model's estimation weights. Checks that the
# default equals the explicit column, :none (and nothing) give the 2.x unweighted result, the
# canonical-link identity Σwᵢ(yᵢ − μ̂ᵢ) = 0 (weighted AAP == weighted mean of y), a length mismatch
# errors, metadata[:weights_source] in every case, second differences, MixedModels /
# FixedEffectModels fits, and invalid weights= values (wrong type or length, negative, all zero).

using Test
using Margins
using DataFrames
using GLM
using MixedModels
using FixedEffectModels
using StatsBase
using Random

@testset "weights=:model (model weights by default)" begin
    Random.seed!(20261019)
    n = 800
    df = DataFrame(x = randn(n), m = randn(n), t = rand([0.0, 1.0], n), g = rand(1:10, n))
    df.w = 0.1 .+ 4 .* rand(n) .^ 3                       # skewed, so weighted ≠ unweighted
    η = 0.2 .+ 0.8df.x .+ 0.5df.x .* df.m .+ 0.4df.t .+ 0.3df.x .* df.t
    df.yb = rand(n) .< 1 ./ (1 .+ exp.(-η))
    df.cnt = [rand() < 0.5 ? 0.0 : Float64(round(Int, 3 * exp(0.3 * v))) for v in df.x]
    df.y = 1.0 .+ df.x .+ 0.5df.x .* df.t .+ randn(n)
    wmean(v) = sum(df.w .* v) / sum(df.w)

    logit_w = glm(@formula(yb ~ x * m + x * t), df, Binomial(), LogitLink(); wts = df.w)
    logit_0 = glm(@formula(yb ~ x * m + x * t), df, Binomial(), LogitLink())

    @testset "GLM: default uses the model weights" begin
        d = population_margins(logit_w, df; vars = [:x, :t])
        u = population_margins(logit_w, df; vars = [:x, :t], weights = :w)
        v = population_margins(logit_w, df; vars = [:x, :t], weights = df.w)
        @test d.estimates == u.estimates && d.standard_errors == u.standard_errors
        @test d.estimates == v.estimates && d.standard_errors == v.standard_errors   # a vector is the column
        @test d.metadata[:weights_source] === :model && d.metadata[:weighted]
        @test u.metadata[:weights_source] === :user && v.metadata[:weights_source] === :user
    end

    @testset ":none and nothing give the unweighted (2.x) result" begin
        a = population_margins(logit_w, df; vars = :x, weights = :none)
        b = population_margins(logit_w, df; vars = :x, weights = nothing)
        c = population_margins(logit_w, df; vars = :x, weights = ones(n))
        @test a.estimates == b.estimates
        @test a.estimates ≈ c.estimates rtol = 1e-12
        @test a.metadata[:weights_source] === :none && !a.metadata[:weighted]
        @test b.metadata[:weights_source] === :none
        @test !isapprox(a.estimates, population_margins(logit_w, df; vars = :x).estimates; rtol = 1e-3)
    end

    @testset "unweighted model: :model means none; explicit weights allowed silently" begin
        r = population_margins(logit_0, df; vars = :x)
        @test r.metadata[:weights_source] === :none && !r.metadata[:weighted]
        @test r.estimates == population_margins(logit_0, df; vars = :x, weights = :none).estimates
        e = @test_logs population_margins(logit_0, df; vars = :x, weights = :w)   # no warning
        @test e.metadata[:weights_source] === :user
    end

    @testset "canonical link: weighted AAP == weighted mean(y)" begin
        aap = population_margins(logit_w, df; type = :predictions)
        @test aap.metadata[:weights_source] === :model
        @test only(aap.estimates) ≈ wmean(df.yb) rtol = 1e-7
        # the unweighted AAP matches neither mean
        aap0 = population_margins(logit_w, df; type = :predictions, weights = :none)
        @test !isapprox(only(aap0.estimates), wmean(df.yb); rtol = 1e-3)
        pois = glm(@formula(cnt ~ x + t), df, Poisson(), LogLink(); wts = df.w)
        @test only(population_margins(pois, df; type = :predictions).estimates) ≈ wmean(df.cnt) rtol = 1e-7
        lin = lm(@formula(y ~ x * t), df; wts = df.w)
        @test only(population_margins(lin, df; type = :predictions).estimates) ≈ wmean(df.y) rtol = 1e-10
    end

    @testset "contexts and other paths set weights_source" begin
        s = population_margins(logit_w, df; vars = :x, scenarios = (t = [0.0, 1.0],))
        @test s.metadata[:weights_source] === :model && s.metadata[:weighted]
        @test s.estimates == population_margins(logit_w, df; vars = :x, scenarios = (t = [0.0, 1.0],), weights = :w).estimates
        g = population_margins(logit_w, df; vars = :x, groups = (:m, 2), weights = :none)
        @test g.metadata[:weights_source] === :none
        p = profile_margins(logit_w, df, means_grid(df); vars = :x)
        @test p.metadata[:weights_source] === :none && !p.metadata[:weighted]   # profile margins are unweighted
    end

    @testset "length mismatch (fit dropped rows) is an error" begin
        dm = allowmissing(df)
        dm.x[5] = missing
        cc = completecases(dm, [:yb, :x, :m, :t])
        fit_dropped = glm(@formula(yb ~ x * m + x * t), dm, Binomial(), LogitLink(); wts = df.w[cc])
        dcc = dropmissing(dm, :x)
        err = try
            population_margins(fit_dropped, df; vars = :m)   # n rows vs n - 1 weights
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("weights=", sprint(showerror, err))
        @test_throws ArgumentError second_differences(fit_dropped, df, :m, :t)
        # the estimation sample works, and explicit weights for the full data are accepted
        @test population_margins(fit_dropped, dcc; vars = :m).metadata[:weights_source] === :model
        @test population_margins(fit_dropped, df; vars = :m, weights = :w).metadata[:weights_source] === :user
    end

    @testset "second differences use the model weights by default" begin
        # discrete modifier
        d = second_differences(logit_w, df, :x, :t)
        u = second_differences(logit_w, df, :x, :t; weights = :w)
        z = second_differences(logit_w, df, :x, :t; weights = :none)
        @test d.second_diff == u.second_diff && d.se == u.se
        @test !isapprox(d.second_diff, z.second_diff; rtol = 1e-3)
        @test metadata(d, "weights_source") === :model
        @test metadata(u, "weights_source") === :user
        @test metadata(z, "weights_source") === :none
        # the discrete second difference is the difference of the weighted scenario AMEs
        ames = population_margins(logit_w, df; vars = :x, scenarios = (t = [0.0, 1.0],), weights = :w)
        @test only(d.second_diff) ≈ ames.estimates[2] - ames.estimates[1] rtol = 1e-10
        # continuous modifier: default at=:observed, and at=:mean uses the weighted mean
        c = second_differences(logit_w, df, :x, :m)
        @test c.second_diff ≈ second_differences(logit_w, df, :x, :m; weights = :w).second_diff
        @test metadata(c, "weights_source") === :model
        cm = second_differences(logit_w, df, :x, :m; at = :mean)
        @test only(cm.eval_point) ≈ wmean(df.m)
        @test only(second_differences(logit_w, df, :x, :m; at = :mean, weights = :none).eval_point) ≈ sum(df.m) / n
        @test metadata(second_differences(logit_0, df, :x, :t), "weights_source") === :none
    end

    @testset "MixedModels" begin
        glmm_w = fit(MixedModel, @formula(yb ~ x * m + t + (1 | g)), df, Bernoulli(); wts = df.w, progress = false)
        glmm_0 = fit(MixedModel, @formula(yb ~ x * m + t + (1 | g)), df, Bernoulli(); progress = false)
        @test Margins._model_weights(glmm_w) ≈ df.w
        @test isnothing(Margins._model_weights(glmm_0))   # StatsAPI.weights would return ones
        r = population_margins(glmm_w, df; vars = :x)
        @test r.metadata[:weights_source] === :model
        @test r.estimates == population_margins(glmm_w, df; vars = :x, weights = :w).estimates
        @test !isapprox(r.estimates, population_margins(glmm_w, df; vars = :x, weights = :none).estimates; rtol = 1e-3)
        @test population_margins(glmm_0, df; vars = :x).metadata[:weights_source] === :none

        lmm_w = fit(MixedModel, @formula(y ~ x * t + (1 | g)), df; wts = df.w, progress = false)
        lmm_0 = fit(MixedModel, @formula(y ~ x * t + (1 | g)), df; progress = false)
        @test Margins._model_weights(lmm_w) ≈ df.w
        @test isnothing(Margins._model_weights(lmm_0))
        a = population_margins(lmm_w, df; vars = :x)
        @test a.metadata[:weights_source] === :model
        @test a.estimates ≈ population_margins(lmm_w, df; vars = :x, weights = :w).estimates
        sd = second_differences(lmm_w, df, :x, :t)
        @test metadata(sd, "weights_source") === :model
        @test sd.second_diff ≈ second_differences(lmm_w, df, :x, :t; weights = :w).second_diff
    end

    @testset "FixedEffectModels: weights are not stored, so :model means none (with a warning)" begin
        df.state = string.(df.g)
        fe_w = reg(df, @formula(y ~ x * t + fe(state)); weights = :w)
        @test Margins._model_weights(fe_w) isa Margins.UnknownWeights
        r = @test_logs (:warn, r"weights=:model") population_margins(fe_w, df; vars = :x)
        @test r.metadata[:weights_source] === :none && !r.metadata[:weighted]
        u = population_margins(fe_w, df; vars = :x, weights = :w)
        @test u.metadata[:weights_source] === :user && u.metadata[:weighted]
        # the contexts path records the weights too (it used to omit :weighted)
        c = population_margins(fe_w, df; vars = :x, scenarios = (t = [0.0, 1.0],), weights = :w)
        @test c.metadata[:weights_source] === :user && c.metadata[:weighted]
    end

    @testset "stored weight types (_weights_from_storage)" begin
        w = [1.0, 2.0, 3.0]
        @test isnothing(Margins._weights_from_storage(Float64[]))
        @test isnothing(Margins._weights_from_storage(StatsBase.uweights(3)))
        @test Margins._weights_from_storage(w) == w
        @test Margins._weights_from_storage(StatsBase.fweights(w)) == w
        @test Margins._weights_from_storage(StatsBase.pweights(w)) == w
        @test_throws ArgumentError Margins._weights_from_storage(StatsBase.aweights(w))
    end

    @testset "invalid weights keyword" begin
        @test_throws ArgumentError population_margins(logit_w, df; vars = :x, weights = "w")
        @test_throws ArgumentError population_margins(logit_w, df; vars = :x, weights = :not_a_column)
        @test_throws ArgumentError population_margins(logit_w, df; vars = :x, weights = ones(n - 10))   # wrong length
        @test_throws ArgumentError population_margins(logit_w, df; vars = :x, weights = [-1.0; ones(n - 1)])
        @test_throws ArgumentError population_margins(logit_w, df; vars = :x, weights = zeros(n))
    end
end
