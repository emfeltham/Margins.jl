# test/regression/test_mixedmodels_link_without_main.jl
#
# Regression test: `_auto_link` looked MixedModels up as `Main.MixedModels`, so a linear
# mixed model failed with "Cannot determine link function" unless the name `MixedModels`
# was bound in `Main` (e.g. `import MixedModels as MM` broke it). The link now comes from
# a package extension that dispatches on the model type.
#
# runtests.jl does `using MixedModels`, which binds the name in Main and would hide the
# bug, so the check runs in a fresh process.

using Test

@testset "MixedModels link without MixedModels bound in Main" begin
    code = """
    import MixedModels as MM
    using Margins, GLM, DataFrames, Random, Statistics, StatsModels, Tables
    @assert !isdefined(Main, :MixedModels)
    Random.seed!(3); n = 600
    d = DataFrame(x = randn(n), u = rand(n), s = string.(rand(1:20, n)))
    re = randn(20)[parse.(Int, d.s)] .* 0.5
    d.yc = -0.3 .+ 0.7 .* d.x .+ 0.4 .* d.x .* d.u .+ re .+ randn(n)
    m = MM.fit(MM.LinearMixedModel, @formula(yc ~ x * u + (1 | s)), d; progress=false)
    r = DataFrame(population_margins(m, d; vars=[:x]))
    β = MM.fixef(m); ix = findfirst(==("x"), MM.fixefnames(m)); ixu = findfirst(==("x & u"), MM.fixefnames(m))
    oracle = β[ix] + β[ixu] * mean(d.u)
    println(isapprox(r.estimate[1], oracle; rtol=1e-8) ? "OK" : "MISMATCH \$(r.estimate[1]) \$oracle")
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    out = withenv("JULIA_LOAD_PATH" => join(LOAD_PATH, Sys.iswindows() ? ";" : ":")) do
        read(pipeline(ignorestatus(cmd); stderr=stdout), String)
    end
    @test occursin("OK", out)
    occursin("OK", out) || println(out)
end
