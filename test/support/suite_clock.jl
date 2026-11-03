# test/support/suite_clock.jl
#
# Run-environment header and per-testset clock report for runtests.jl, so that a slow run can be
# told apart from a slow test. For each top-level testset it prints the wall time, the process CPU
# time and the time the machine was asleep. A run whose wall time far exceeds its CPU time and
# whose "asleep" time is large was suspended by the OS (on a laptop: lid closed, i.e. clamshell
# sleep, which caffeinate/Amphetamine-style idle-sleep assertions do not prevent); nothing in the
# tests was slow. Diagnostics only: nothing here affects a test result.

using LinearAlgebra: BLAS

# clock_gettime ids. "awake" excludes time asleep, "all" includes it.
const _CLOCK_IDS = Sys.isapple() ? (awake = 8, all = 6, cpu = 12) :   # UPTIME_RAW, MONOTONIC, PROCESS_CPUTIME_ID
                   Sys.islinux() ? (awake = 1, all = 7, cpu = 2) :    # MONOTONIC, BOOTTIME, PROCESS_CPUTIME_ID
                   nothing

function _clock_s(id::Integer)
    ts = Ref((Int64(0), Int64(0)))   # struct timespec on 64-bit macOS and Linux
    ccall(:clock_gettime, Cint, (Cint, Ptr{Cvoid}), id, ts) == 0 || return NaN
    return ts[][1] + ts[][2] / 1e9
end

# (wall, cpu, asleep) in seconds; cpu and asleep are NaN where the clocks are not available
function _suite_clock()
    wall = time()
    (_CLOCK_IDS === nothing || Sys.WORD_SIZE != 64) && return (wall, NaN, NaN)
    return (wall, _clock_s(_CLOCK_IDS.cpu), _clock_s(_CLOCK_IDS.all) - _clock_s(_CLOCK_IDS.awake))
end

function print_suite_environment(io::IO = stdout)
    println(io, "Test environment: Julia ", VERSION, ", threads = ", Threads.nthreads(),
            " (GC threads ", Threads.ngcthreads(), "), CPU threads = ", Sys.CPU_THREADS,
            ", BLAS threads = ", BLAS.get_num_threads(),
            ", OPENBLAS_NUM_THREADS = ", get(ENV, "OPENBLAS_NUM_THREADS", "unset"),
            ", check-bounds = ", Base.JLOptions().check_bounds)
    println(io, "  BLAS: ", BLAS.get_config())
    flush(io)
end

const _SUITE_START = Ref((NaN, NaN, NaN))
const _PHASE = Ref{Union{Nothing, Tuple{String, NTuple{3, Float64}}}}(nothing)

_fmt(x) = isnan(x) ? "n/a" : string(round(x; digits = 1), "s")

function _report_phase(io::IO, name, t0, t1)
    asleep = max(t1[3] - t0[3], 0.0)
    println(io, "[clock] ", name, ": wall ", _fmt(t1[1] - t0[1]), ", CPU ", _fmt(t1[2] - t0[2]),
            ", machine asleep ", _fmt(asleep),
            asleep > 5 ? "  <-- the OS suspended this run; the wall time is not test time" : "")
    flush(io)
end

# suite_phase("name") closes the previous phase (printing its clock line) and opens a new one;
# suite_phase(nothing) closes the last phase and prints the whole-run total.
function suite_phase(name::Union{Nothing, AbstractString}, io::IO = stdout)
    now_ = _suite_clock()
    isnan(_SUITE_START[][1]) && (_SUITE_START[] = now_)
    _PHASE[] === nothing || _report_phase(io, _PHASE[][1], _PHASE[][2], now_)
    if name === nothing
        _report_phase(io, "whole run", _SUITE_START[], now_)
        _PHASE[] = nothing
    else
        _PHASE[] = (String(name), now_)
    end
    return nothing
end
