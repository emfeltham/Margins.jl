# test/validation/concurrent_calls_checks.jl
#
# Checks for concurrent Margins calls, included by test_concurrent_calls.jl — directly when
# Julia runs with several threads, otherwise in a `julia -t 4` subprocess.
#
# Before 3.0.0 concurrent calls on the same model and data shared one cached engine (and
# FormulaCompiler's cached compiled formula), whose scratch buffers they overwrote: 64 of 64
# concurrent `population_margins` calls returned wrong estimates. Every concurrent result
# must now equal its serial counterpart exactly. Since the engine cache was removed (3.0),
# each call builds its own engine and the only state shared between calls is
# FormulaCompiler's compiled-formula cache (locked) and the read-only compiled operations.

using Test, Random, DataFrames, CategoricalArrays, GLM, StatsModels, Statistics
using Margins
import CovarianceMatrices

function _concurrency_fixture()
    Random.seed!(1); n = 4_000
    d = DataFrame(x = randn(n), z = randn(n), k = rand(0:5, n),
                  g = categorical(rand(["a", "b", "c"], n)))
    d.y = rand(n) .< 1 ./ (1 .+ exp.(-(0.5 .* d.x .+ 0.8 .* d.z .- 0.1 .* d.k .+ 0.3 .* (d.g .== "b"))))
    m = glm(@formula(y ~ x * z + k + g), d, Binomial(), LogitLink())
    return d, m
end

# Each job is (label, thunk) returning a DataFrame
function _concurrency_jobs(m, d)
    grid = cartesian_grid(x = [-1.0, 0.0, 1.0], g = ["a", "c"])
    hgrid = hierarchical_grid(d, :g => [(:x, [0.0, 1.0]), (:z, :mean)])
    return Tuple{String, Any}[
        ("AME ad",          () -> DataFrame(population_margins(m, d))),
        ("AME fd",          () -> DataFrame(population_margins(m, d; backend=:fd))),
        ("AME link x",      () -> DataFrame(population_margins(m, d; vars=[:x], scale=:link))),
        ("AAP",             () -> DataFrame(population_margins(m, d; type=:predictions))),
        ("AME groups",      () -> DataFrame(population_margins(m, d; vars=[:x], groups=:g))),
        ("AME scenarios",   () -> DataFrame(population_margins(m, d; vars=[:x], scenarios=(z=[-1.0, 1.0],)))),
        ("elasticity",      () -> DataFrame(population_margins(m, d; vars=[:z], measure=:elasticity))),
        ("MEM",             () -> DataFrame(profile_margins(m, d, grid; vars=[:x, :g]))),
        ("APM",             () -> DataFrame(profile_margins(m, d, grid; type=:predictions))),
        ("means grid",      () -> DataFrame(profile_margins(m, d, means_grid(d); vars=[:x]))),
        ("hierarchical",    () -> DataFrame(profile_margins(m, d, hgrid; type=:predictions))),
    ]
end

_same(a::DataFrame, b::DataFrame) = isequal(a.estimate, b.estimate) && isequal(a.se, b.se)

function _run_concurrently(jobs, refs, n_calls; spawn::Bool)
    # Vector{Bool}, not a BitVector: concurrent writes to bits sharing a word would race
    ok = fill(false, n_calls); errs = Vector{Any}(nothing, n_calls)
    body(i) = try
        j = mod1(i, length(jobs))
        r = jobs[j][2]()
        ok[i] = _same(r, refs[j])
        ok[i] || println(stderr, "MISMATCH ", jobs[j][1], " Δest=", maximum(abs.(r.estimate .- refs[j].estimate)), " Δse=", maximum(abs.(r.se .- refs[j].se)))
    catch e
        errs[i] = e
        println(stderr, "ERRORED ", jobs[mod1(i, length(jobs))][1], ": ", sprint(showerror, e, catch_backtrace())[1:min(end, 3000)])
    end
    if spawn
        tasks = [Threads.@spawn body(i) for i in 1:n_calls]
        foreach(wait, tasks)
    else
        Threads.@threads for i in 1:n_calls
            body(i)
        end
    end
    return ok, errs
end

@testset "concurrent Margins calls (nthreads = $(Threads.nthreads()))" begin
    @test Threads.nthreads() > 1
    d, m = _concurrency_fixture()
    jobs = _concurrency_jobs(m, d)
    Margins.clear_engine_cache!()
    refs = [f() for (_, f) in jobs]   # serial counterparts

    @testset "same model and data: $mode" for mode in (:spawn, :threads)
        ok, errs = _run_concurrently(jobs, refs, 132; spawn = mode === :spawn)
        @test count(!isnothing, errs) == 0
        @test all(ok)
        isempty(filter(!isnothing, errs)) || showerror(stderr, first(filter(!isnothing, errs)))
    end

    @testset "same schema, different data" begin
        subsets = [d[(i - 1) * 500 + 1:i * 500, :] for i in 1:8]
        sjobs = Tuple{String, Any}[("AME subset $i", () -> DataFrame(population_margins(m, s; backend = isodd(i) ? :ad : :fd)))
                 for (i, s) in enumerate(subsets)]
        append!(sjobs, [("APM subset $i", () -> DataFrame(profile_margins(m, s, means_grid(s); type=:predictions)))
                        for (i, s) in enumerate(subsets)])
        srefs = [f() for (_, f) in sjobs]
        for mode in (:spawn, :threads)
            ok, errs = _run_concurrently(sjobs, srefs, 128; spawn = mode === :spawn)
            @test count(!isnothing, errs) == 0
            @test all(ok)
        end
    end

    @testset "concurrent first calls (cold compiled-formula cache)" begin
        # Every call builds its own engine; the only shared state is FormulaCompiler's
        # compiled-formula cache, which compiles outside its lock. Start cold so that
        # concurrent calls race to compile the same formula.
        for mode in (:spawn, :threads)
            Margins.clear_engine_cache!()
            ok, errs = _run_concurrently(jobs, refs, 44; spawn = mode === :spawn)
            @test count(!isnothing, errs) == 0
            @test all(ok)
        end
    end

    @testset "concurrent calls with a robust vcov and a vcov matrix" begin
        V = 1.5 .* GLM.vcov(m)
        vjobs = Tuple{String, Any}[
            ("AME HC1",     () -> DataFrame(population_margins(m, d; vcov = CovarianceMatrices.HC1()))),
            ("AME matrix",  () -> DataFrame(population_margins(m, d; vcov = V))),
            ("APM HC1",     () -> DataFrame(profile_margins(m, d, means_grid(d); type = :predictions,
                                                          vcov = CovarianceMatrices.HC1()))),
        ]
        vrefs = [f() for (_, f) in vjobs]
        ok, errs = _run_concurrently(vjobs, vrefs, 48; spawn = true)
        @test count(!isnothing, errs) == 0
        @test all(ok)
    end
end
