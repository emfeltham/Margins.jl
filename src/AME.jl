# AME.jl

"""
    AME

Immutable struct to store the result of an Average Marginal Effect calculation.

Fields:
- `var::Symbol`       — the focal predictor name.
- `ame::Float64`      — the computed AME (average ∂μ/∂x over all observations).
- `se::Float64`       — the delta-method standard error of the AME.
- `grad::Vector{Float64}` — the p-vector ∇₍β₎[AME], for downstream contrasts.
- `n::Int`            — the sample size (number of rows used).
- `η_base::Vector{Float64}` — the linear predictors η_i at the original x.
- `μ_base::Vector{Float64}` — the fitted values μ_i = invlink(η_i) at original x.
- `family::String`    — the response family (e.g. “Bernoulli”).
- `link::String`      — the link function (e.g. “LogitLink()”).
"""
struct AME
    var     :: Symbol
    ame     :: Float64
    se      :: Float64
    grad    :: Vector{Float64}
    n       :: Int
    η_base  :: Vector{Float64}
    μ_base  :: Vector{Float64}
    family  :: String
    link    :: String
end

# ─────────────────────────────────────────────────────────────────────────────
#  Display for AME
# ─────────────────────────────────────────────────────────────────────────────
function show(io::IO, ::MIME"text/plain", ame::AME)
    # Compute z, p-value, 95% CI
    z    = ame.ame / ame.se
    p    = 2 * (1 - cdf(Normal(), abs(z)))
    lo   = ame.ame - 1.96 * ame.se
    hi   = ame.ame + 1.96 * ame.se

    # Header with family & link
    println(io, "Average Marginal Effect of `" * string(ame.var) * "` (Family: ", ame.family,
            "; Link: ", ame.link, ")")
    println(io, "-"^60)

    # Column titles
    @printf(io, "%-12s %10s %10s %8s %8s %18s\n",
            "", "AME", "Std.Err", "z", "P>|z|", "95% CI")
    println(io, "-"^60)

    # Single row (since AME is for one variable)
    @printf(io, "%-12s %10.4f %10.4f %8.2f %8.3f [%7.4f, %7.4f]\n\n",
            "Δ" * string(ame.var), ame.ame, ame.se, z, p, lo, hi)

    # Footer
    println(io, "Observations: ", ame.n)
end

"""
    AMEContrast

Immutable struct to store the result of contrasting two Average Marginal Effects.

Fields:
- `var::Symbol`       — the focal predictor name.
- `ame_low::Float64`  — AME of `var` at the lower slice.
- `se_low::Float64`   — standard error of AME at the lower slice.
- `ame_high::Float64` — AME of `var` at the higher slice.
- `se_high::Float64`  — standard error of AME at the higher slice.
- `ame_diff::Float64` — difference (AME_high − AME_low).
- `se_diff::Float64`  — standard error of the difference.
- `grad_low::Vector{Float64}`  — gradient ∇₍β₎[AME @ low].
- `grad_high::Vector{Float64}` — gradient ∇₍β₎[AME @ high].
- `n::Int`            — sample size.
- `family::String`    — the response family.
- `link::String`      — the link function.
"""
struct AMEContrast
    var        :: Symbol
    ame_low    :: Float64
    se_low     :: Float64
    ame_high   :: Float64
    se_high    :: Float64
    ame_diff   :: Float64
    se_diff    :: Float64
    grad_low   :: Vector{Float64}
    grad_high  :: Vector{Float64}
    n          :: Int
    family     :: String
    link       :: String
end

# ─────────────────────────────────────────────────────────────────────────────
#  Display for AMEContrast
# ─────────────────────────────────────────────────────────────────────────────
function show(io::IO, ::MIME"text/plain", c::AMEContrast)
    # Compute z’s, p’s, CIs
    zs  = (c.ame_low/c.se_low, c.ame_high/c.se_high, c.ame_diff/c.se_diff)
    ps  = map(z -> 2*(1 - cdf(Normal(), abs(z))), zs)
    cis = [
      (c.ame_low  -1.96*c.se_low,  c.ame_low  +1.96*c.se_low),
      (c.ame_high -1.96*c.se_high, c.ame_high +1.96*c.se_high),
      (c.ame_diff -1.96*c.se_diff, c.ame_diff +1.96*c.se_diff),
    ]

    # Header with var, family & link
    println(io, "AME Contrast for `", string(c.var), "`  (Family: ", c.family,
            "; Link: ", c.link, ")")
    println(io, "-"^70)

    # Column titles
    @printf(io, "%-8s %10s %10s %8s %8s %18s\n",
            "", "AME", "Std.Err", "z", "P>|z|", "95% CI")
    println(io, "-"^70)

    # Rows: low, high, diff
    labels = ("low", "high", "diff")
    values = (c.ame_low, c.ame_high, c.ame_diff)
    ses    = (c.se_low,  c.se_high,  c.se_diff)

    for i in 1:3
        ame_v, se_v, z, p = values[i], ses[i], zs[i], ps[i]
        lo, hi = cis[i]
        @printf(io, "%-8s %10.4f %10.4f %8.2f %8.3f [%7.4f, %7.4f]\n",
                labels[i], ame_v, se_v, z, p, lo, hi)
    end

    # Footer
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
