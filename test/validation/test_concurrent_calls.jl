# test/validation/test_concurrent_calls.jl
#
# Concurrent Margins calls must be safe (see concurrent_calls_checks.jl). The checks need
# several threads: they run inline when this process has them, otherwise in a subprocess
# started with `-t 4`.

using Test

if Threads.nthreads() > 1
    include("concurrent_calls_checks.jl")
else
    @testset "concurrent Margins calls (subprocess, -t 4)" begin
        script = joinpath(@__DIR__, "concurrent_calls_checks.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 --project=$(Base.active_project()) $script`
        out = IOBuffer()
        proc = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
        output = String(take!(out))
        success(proc) || print(output)
        @test success(proc)
    end
end
