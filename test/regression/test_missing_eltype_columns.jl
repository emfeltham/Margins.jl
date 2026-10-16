# test/regression/test_missing_eltype_columns.jl
#
# Regression test: columns typed `Union{Missing, T}`.
#
# CSV.jl, `allowmissing` and `dropmissing(df, cols)` (which narrows only `cols`) produce
# such columns even when they hold no missing value. They used to crash with an opaque
# `MethodError: no method matching fieldindex(::Type{Union{Missing, Float64}}, ...)`
# (or a `convert` error for predictions, or "Unsupported data type" from the grid
# builders). Columns without missing values must now give exactly the results of the
# narrowed data, and repeated calls must still reuse the cached engine. Model variables
# that DO contain missing values must raise an actionable error.

using Test
using DataFrames, CategoricalArrays, GLM, Statistics, Random
using Margins

@testset "Union{Missing,T} columns" begin
    Random.seed!(7); n = 800
    d = DataFrame(x = randn(n), z = randn(n), k = rand(1:5, n), b = rand(Bool, n),
                  g = categorical(rand(["a", "b", "c"], n)), w = rand(n) .+ 0.1)
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.5 .* d.x .+ 0.8 .* d.z .+ 0.3 .* d.b)))
    m = glm(@formula(y ~ x * z + k + b + g), d, Binomial(), LogitLink())

    du = allowmissing(copy(d))
    du.g = CategoricalArray{Union{Missing, String}}(d.g; levels=levels(d.g))
    @test all(c -> Missing <: eltype(c), eachcol(du))

    same(r1, r2) = (DataFrame(r1).estimate ≈ DataFrame(r2).estimate) &&
                   (DataFrame(r1).se ≈ DataFrame(r2).se)

    @testset "no missing values: identical results" begin
        for backend in (:ad, :fd)
            @test same(population_margins(m, du; backend), population_margins(m, d; backend))
            @test same(population_margins(m, du; backend, measure=:elasticity, vars=[:x]),
                       population_margins(m, d; backend, measure=:elasticity, vars=[:x]))
        end
        @test same(population_margins(m, du; type=:predictions), population_margins(m, d; type=:predictions))
        @test same(population_margins(m, du; vars=[:x], groups=:g, weights=:w),
                   population_margins(m, d; vars=[:x], groups=:g, weights=:w))
        @test same(population_margins(m, du; vars=[:x], scenarios=(z=[0.0, 1.0],)),
                   population_margins(m, d; vars=[:x], scenarios=(z=[0.0, 1.0],)))
        @test same(profile_margins(m, du, means_grid(du); vars=[:x, :g]),
                   profile_margins(m, d, means_grid(d); vars=[:x, :g]))
        @test same(profile_margins(m, du, cartesian_grid(x=[0.0, 1.0]); type=:predictions),
                   profile_margins(m, d, cartesian_grid(x=[0.0, 1.0]); type=:predictions))
        # the estimation-sample idiom from the review: dropmissing on the response only
        dm = allowmissing(copy(d)); dm.y[1:50] .= missing
        mm = glm(@formula(y ~ x * z + k + b + g), dm, Binomial(), LogitLink())
        ds = dropmissing(dm, :y)
        @test same(population_margins(mm, ds; vars=[:x]),
                   population_margins(mm, disallowmissing(ds); vars=[:x]))
    end

    @testset "repeated calls reuse the cached engine" begin
        # (an Int column — model variable or not — is re-converted to Float64 on every call
        # and so always rebuilds the engine; that predates this fix, so leave k out here)
        mf = glm(@formula(y ~ x * z + b + g), d, Binomial(), LogitLink())
        duf = select(du, Not(:k))
        Margins.clear_engine_cache!()
        population_margins(mf, duf; vars=[:x])
        e1 = only(values(Margins.ENGINE_CACHE))
        population_margins(mf, duf; vars=[:x])
        @test only(values(Margins.ENGINE_CACHE)) === e1
        # ...but an in-place edit of the source column is picked up
        du2 = allowmissing(select(d, Not(:k))); d2 = select(d, Not(:k))
        population_margins(mf, du2; vars=[:x])
        du2.z .+= 1; d2.z .+= 1
        @test same(population_margins(mf, du2; vars=[:x]), population_margins(mf, d2; vars=[:x]))
    end

    @testset "missing values in model variables: actionable error" begin
        dz = allowmissing(copy(d)); dz.z[1:10] .= missing
        mz = glm(@formula(y ~ x * z + k + b + g), dz, Binomial(), LogitLink())
        for f in (() -> population_margins(mz, dz; vars=[:x]),
                  () -> population_margins(mz, dz; type=:predictions),
                  () -> profile_margins(mz, dz, cartesian_grid(x=[0.0]); vars=[:x]))
            err = try f(); nothing catch e; e end
            @test err isa Margins.MarginsError
            msg = sprint(showerror, err)
            @test occursin(":z", msg) && occursin("10 missing", msg) && occursin("dropmissing", msg)
        end
        err = try means_grid(dz); nothing catch e; e end
        @test err isa Margins.MarginsError && occursin("dropmissing", sprint(showerror, err))
        # missing values only in a column the model does not use are fine
        dn = allowmissing(copy(d)); dn.extra = [isodd(i) ? missing : 1.0 for i in 1:n]
        @test same(population_margins(m, dn; vars=[:x]), population_margins(m, d; vars=[:x]))
    end
end
