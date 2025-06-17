# AME.jl

"""
    AME

Immutable struct to store the result of an Average Marginal Effect calculation.

Fields:
- `ame::Float64`      — the computed AME (average ∂μ/∂x over all observations).
- `se::Float64`       — the delta-method standard error of the AME.
- `grad::Vector{Float64}` — the p-vector ∇₍β₎[AME], for downstream contrasts.
- `n::Int`            — the sample size (number of rows used).
- `η_base::Vector{Float64}` — the linear predictors η_i at the original x.
- `μ_base::Vector{Float64}` — the fitted values μ_i = invlink(η_i) at original x.
"""
struct AME
    ame    :: Float64
    se     :: Float64
    grad   :: Vector{Float64}
    n      :: Int
    η_base :: Vector{Float64}
    μ_base :: Vector{Float64}
end

function show(io::IO, ::MIME"text/plain", ame::AME)
    # 1) compute z, p, CI
    z    = ame.ame / ame.se
    p    = 2 * (1 - cdf(Normal(), abs(z)))
    ci_lo = ame.ame - 1.96*ame.se
    ci_hi = ame.ame + 1.96*ame.se

    # 2) header
    println(io, "Average Marginal Effect")
    println(io, repeat("-", 25))

    # 3) table row (you could loop if you had multiple predictors)
    @printf(io, "%-12s %10s %8s %8s %12s\n",
            "AME", "Std. Err.", "z", "P>|z|", "95% CI")
    @printf(io, "%-12.4f %10.4f %8.2f %8.3f [%7.4f, %7.4f]\n\n",
            ame.ame, ame.se, z, p, ci_lo, ci_hi)

    # 4) footer
    println(io, "Observations: ", ame.n)
end

"""
    AMEContrast

Immutable struct to store the result of contrasting two Average Marginal Effects.

Fields:
- `ame_low::Float64`      — AME of x at z_low
- `se_low::Float64`       — standard error of AME at z_low
- `ame_high::Float64`     — AME of x at z_high
- `se_high::Float64`      — standard error of AME at z_high
- `ame_diff::Float64`     — difference (AME_high − AME_low)
- `se_diff::Float64`      — standard error of the difference
- `grad_low::Vector{Float64}`  — gradient vector ∇₍β₎[AME @ z_low]
- `grad_high::Vector{Float64}` — gradient vector ∇₍β₎[AME @ z_high]
- `n::Int`                — sample size
- `names::Vector{String}` — coefficient names from the model
"""
struct AMEContrast
    ame_low   :: Float64
    se_low    :: Float64
    ame_high  :: Float64
    se_high   :: Float64
    ame_diff  :: Float64
    se_diff   :: Float64
    grad_low  :: Vector{Float64}
    grad_high :: Vector{Float64}
    n         :: Int
    names     :: Vector{String}
end

function show(io::IO, ::MIME"text/plain", c::AMEContrast)
    # 1) Compute z-stats, p-values, and 95% CIs
    zs    = (c.ame_low/c.se_low, c.ame_high/c.se_high, c.ame_diff/c.se_diff)
    ps    = map(z -> 2*(1 - cdf(Normal(), abs(z))), zs)
    cis   = [
      (c.ame_low  -1.96*c.se_low,  c.ame_low  +1.96*c.se_low),
      (c.ame_high -1.96*c.se_high, c.ame_high +1.96*c.se_high),
      (c.ame_diff -1.96*c.se_diff, c.ame_diff +1.96*c.se_diff),
    ]

    # 2) Header
    println(io, "Average Marginal–Effect Contrast")
    println(io, "-"^35)
    @printf(io, "%-6s %10s %10s %8s %8s %18s\n",
            "", "AME", "Std. Err.", "z", "P>|z|", "95% CI")
    println(io, "-"^35)

    # 3) Rows
    labels = ("low", "high", "diff")
    values = (c.ame_low, c.ame_high, c.ame_diff)
    ses    = (c.se_low,  c.se_high,  c.se_diff)
    for i in 1:3
        ame = values[i]; se = ses[i]; z = zs[i]; p = ps[i]
        (lo,hi) = cis[i]
        @printf(io, "%-6s %10.4f %10.4f %8.2f %8.3f [%7.4f, %7.4f]\n",
                labels[i], ame, se, z, p, lo, hi)
    end

    # 4) Footer
    println(io, "\nObservations: ", c.n)
end

##

function confint(c::AMEContrast)
    ame_diff = c.ame_diff
    se_diff = c.se_diff
    return tuple(sort([ame_diff + 1.96 * se_diff, ame_diff - 1.96 * se_diff])...)
end

function confint(a::AME)
    ame = a.ame
    se = a.se
    return tuple(sort([ame + 1.96 * se, ame - 1.96 * se])...)
end
