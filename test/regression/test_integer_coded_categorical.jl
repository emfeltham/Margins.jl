# test/regression/test_integer_coded_categorical.jl
#
# Regression test: a numeric column the MODEL treats as categorical
# (`contrasts = Dict(:k => DummyCoding())` on an integer column).
#
# Margins decided "continuous vs categorical" from the column's element type, so an integer
# k was continuous to Margins but categorical to the model: `population_margins` failed with
# "Categorical variable k not found in ContrastEvaluator level maps", `groups=:k` was
# rejected as an unbinned numeric variable, and `profile_margins(m, df, means_grid(df))`
# evaluated the model at the mean code k = 1.94 (not a level). The model now decides:
# model-categorical columns are handled as categorical everywhere, contrasts enumerate the
# model's levels, and grid completion replaces a non-level value with the variable's
# frequency mixture.
#
# Oracle: design rows from `modelcols` on the model's schema with integer k values (mixtures
# as frequency-weighted averages of design rows); unit derivatives by central differences;
# SEs by central differences in β.

using Test
using DataFrames, GLM, StatsModels, Statistics, LinearAlgebra, Random, Tables
using Margins

@testset "integer-coded categorical declared via contrasts" begin
    Random.seed!(5); n = 1000
    d = DataFrame(x = randn(n), k = rand(1:3, n), z = randn(n))
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.4 .* d.x .+ [0, 0.8, -0.6][d.k] .+ 0.3 .* d.z .+ 0.2 .* d.x .* (d.k .== 2))))
    m = glm(@formula(y ~ x * k + z), d, Binomial(), LogitLink(); contrasts = Dict(:k => DummyCoding()))
    β = coef(m); V = vcov(m); L = LogitLink(); rhs = formula(m).rhs
    X(dd) = modelcols(rhs, Tables.columntable(dd))
    μ(Xs, b) = GLM.linkinv.(Ref(L), Xs * b)
    ∇(f) = [(f(β .+ 1e-6 .* (1:length(β) .== j)) - f(β .- 1e-6 .* (1:length(β) .== j))) / 2e-6 for j in eachindex(β)]
    se(f) = (g = ∇(f); sqrt(dot(g, V, g)))
    with(dd; kw...) = (c = copy(dd); for (k, v) in kw; c[!, k] = v isa AbstractVector ? v : fill(v, nrow(c)); end; c)
    h = 1e-5
    ame_x(dd) = (Xp = X(with(dd; x = dd.x .+ h)); Xm = X(with(dd; x = dd.x .- h));
                 b -> mean((μ(Xp, b) .- μ(Xm, b)) ./ 2h))
    contrast_k(dd, to, from) = (X1 = X(with(dd; k = to)); X0 = X(with(dd; k = from));
                                b -> mean(μ(X1, b) .- μ(X0, b)))
    chk(r, f) = (@test r.estimate ≈ f(β) rtol=1e-6; @test r.se ≈ se(f) rtol=1e-4)

    @testset "population AME, all variables ($backend)" for backend in (:ad, :fd)
        res = DataFrame(population_margins(m, d; backend))
        @test Set(res.variable) == Set(["x", "z", "k"])
        chk(only(eachrow(res[res.variable .== "x", :])), ame_x(d))
        for (to, lab) in ((2, "2 - 1"), (3, "3 - 1"))
            chk(only(eachrow(res[(res.variable .== "k") .& (res.contrast .== lab), :])), contrast_k(d, to, 1))
        end
    end

    @testset "groups = :k" begin
        res = DataFrame(population_margins(m, d; vars=[:x], groups=:k))
        @test nrow(res) == 3
        for lev in 1:3
            dd = d[d.k .== lev, :]
            chk(only(eachrow(res[string.(res.k) .== string(lev), :])), ame_x(dd))
        end
    end

    @testset "scenarios k = 2" begin
        res = DataFrame(population_margins(m, d; vars=[:x], scenarios=(k=[2],)))
        chk(only(eachrow(res)), ame_x(with(d; k = 2)))
    end

    @testset "profile: means_grid and cartesian_grid" begin
        freqs = [mean(d.k .== l) for l in 1:3]
        x̄ = mean(d.x); z̄ = mean(d.z)
        rowX(xv, kw) = sum(w .* vec(X(DataFrame(x=[xv], k=[l], z=[z̄], y=[false]))) for (l, w) in zip(1:3, kw))
        pt(xv, kw) = b -> GLM.linkinv(L, dot(rowX(xv, kw), b))
        res = DataFrame(profile_margins(m, d, means_grid(d); vars=[:x, :k]))
        chk(only(eachrow(res[res.variable .== "x", :])),
            b -> (pt(x̄ + h, freqs)(b) - pt(x̄ - h, freqs)(b)) / 2h)
        for (to, lab) in ((2, "2 - 1"), (3, "3 - 1"))
            onehot(l) = [l == j ? 1.0 : 0.0 for j in 1:3]
            chk(only(eachrow(res[(res.variable .== "k") .& (res.contrast .== lab), :])),
                b -> pt(x̄, onehot(to))(b) - pt(x̄, onehot(1))(b))
        end

        cg = DataFrame(profile_margins(m, d, cartesian_grid(k=[1, 3]); vars=[:x]))
        for r in eachrow(cg)
            oh = [r.k == j ? 1.0 : 0.0 for j in 1:3]
            chk(r, b -> (pt(x̄ + h, oh)(b) - pt(x̄ - h, oh)(b)) / 2h)
        end
        ap = DataFrame(profile_margins(m, d, cartesian_grid(k=[1, 3]); type=:predictions))
        for r in eachrow(ap)
            chk(r, pt(x̄, [r.k == j ? 1.0 : 0.0 for j in 1:3]))
        end
    end

    @testset "repeated calls reuse the cached engine" begin
        Margins.clear_engine_cache!()
        dk = select(d, :x, :k, :z, :y); dk.x = Float64.(dk.x); dk.z = Float64.(dk.z)
        population_margins(m, dk; vars=[:k])
        e1 = only(values(Margins.ENGINE_CACHE))
        population_margins(m, dk; vars=[:k])
        @test only(values(Margins.ENGINE_CACHE)) === e1
    end

    @testset "explicit grid values must be levels; typical values become mixtures" begin
        # A user-specified non-level (9) is an error naming the variable, values and levels
        err = try
            profile_margins(m, d, cartesian_grid(k=[1, 2, 9]); type=:predictions); nothing
        catch e
            e
        end
        @test err isa Margins.MarginsError
        msg = sprint(showerror, err)
        @test occursin(":k", msg) && occursin("[9]", msg) && occursin("[1, 2, 3]", msg)
        # Level values (Int or Float spelling) evaluate at those levels
        x̄ = mean(d.x); z̄ = mean(d.z)
        for g in (cartesian_grid(k=[1, 2]), cartesian_grid(k=[1.0, 2.0]))
            ap = DataFrame(profile_margins(m, d, g; type=:predictions))
            for r in eachrow(ap)
                kv = Int(r.k)
                f = b -> GLM.linkinv(L, dot(vec(X(DataFrame(x=[x̄], k=[kv], z=[z̄], y=[false]))), b))
                chk(r, f)
            end
        end
        # means_grid stores the mean code (a typical value): replaced by the mixture
        freqs = [mean(d.k .== l) for l in 1:3]
        xm = sum(w .* vec(X(DataFrame(x=[x̄], k=[l], z=[z̄], y=[false]))) for (l, w) in zip(1:3, freqs))
        r = only(eachrow(DataFrame(profile_margins(m, d, means_grid(d); type=:predictions))))
        chk(r, b -> GLM.linkinv(L, dot(xm, b)))
    end

    @testset "continuous effects are labelled dy/dx on every path" begin
        @test all(==("dy/dx"), DataFrame(population_margins(m, d; vars=[:x])).contrast)
        @test all(==("dy/dx"), DataFrame(population_margins(m, d; vars=[:x], groups=:k)).contrast)
        @test all(==("dy/dx"), DataFrame(population_margins(m, d; vars=[:x], scenarios=(z=[0.0],))).contrast)
        @test all(==("dy/dx"), DataFrame(profile_margins(m, d, means_grid(d); vars=[:x])).contrast)
    end
end

@testset "Float and Bool columns declared categorical" begin
    Random.seed!(11); n = 600
    d = DataFrame(x = randn(n), kf = Float64.(rand(1:3, n)), b = rand(Bool, n))
    d.y = d.x .+ [0.0, 1.0, -2.0][Int.(d.kf)] .+ 0.8 .* d.b .+ randn(n)
    for (v, coding) in ((:kf, nothing), (:b, nothing))
        m = lm(Term(:y) ~ Term(:x) + Term(v), d; contrasts = Dict(v => DummyCoding()))
        β = coef(m); V = vcov(m); rhs = formula(m).rhs
        X(dd) = modelcols(rhs, Tables.columntable(dd))
        lv = sort(unique(d[!, v]))
        # means_grid: v becomes its frequency mixture (no "not a level" error)
        freqs = [mean(d[!, v] .== l) for l in lv]
        row(l) = vec(X(DataFrame(:x => [mean(d.x)], v => [l], :y => [0.0])))
        xm = sum(w .* row(l) for (l, w) in zip(lv, freqs))
        r = only(eachrow(DataFrame(profile_margins(m, d, means_grid(d); type=:predictions))))
        @test r.estimate ≈ dot(xm, β) rtol=1e-10
        @test r.se ≈ sqrt(dot(xm, V, xm)) rtol=1e-8
        # population contrasts against the first level match the coefficients (lm, no interaction)
        res = DataFrame(population_margins(m, d; vars=[v]))
        @test nrow(res) == length(lv) - 1
        @test res.estimate ≈ β[3:end] rtol=1e-8
        @test res.se ≈ sqrt.(diag(V)[3:end]) rtol=1e-8
    end

    # A Bool used as a continuous regressor (no contrast) keeps its behaviour: its means_grid
    # value is the share of `true`, evaluated as a number.
    m = lm(@formula(y ~ x + b), d)
    @test coefnames(m) == ["(Intercept)", "x", "b"]
    r = only(eachrow(DataFrame(profile_margins(m, d, means_grid(d); type=:predictions))))
    @test r.estimate ≈ dot([1.0, mean(d.x), mean(d.b)], coef(m)) rtol=1e-10
    res = DataFrame(population_margins(m, d; vars=[:b]))
    @test only(res.contrast) == "true - false"
    @test only(res.estimate) ≈ coef(m)[3] rtol=1e-8
end
