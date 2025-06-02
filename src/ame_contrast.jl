

# ─────────────────────────────────────────────────────────────────────────────
# 1) Define an immutable struct to hold the AME‐contrast result
# ─────────────────────────────────────────────────────────────────────────────
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

"""
    ame_contrast_numeric(
      df::DataFrame,
      model;
      x::Symbol,
      z::Symbol,
      z_vals::Tuple{<:Real,<:Real},
      δ::Real = 1e-6,
      typical = mean,
      invlink = (η -> 1/(1 + exp(-η))),
      dinvlink = nothing,
      d2invlink = nothing,
      vcov = StatsBase.vcov
    ) -> NamedTuple

Compute the **difference of two AMEs**,
    ΔAME = AME(x | z = z_high) − AME(x | z = z_low)
and its standard error by calling `ame_numeric` at each `z_low`/`z_high` and using the delta‐method.

# Arguments
- `df::DataFrame`
- `model`
- `x::Symbol`
- `z::Symbol`
- `z_vals::Tuple{(z_low, z_high)}`
- `δ::Real = 1e-6`
- `typical`, `invlink`, `dinvlink`, `d2invlink`, `vcov` same as in `ame_numeric`.

# Returns
A `NamedTuple` with:
- `ame_low`   :: Float64        = AME at z_low
- `se_low`    :: Float64        = SE of AME at z_low
- `ame_high`  :: Float64        = AME at z_high
- `se_high`   :: Float64        = SE of AME at z_high
- `ame_diff`  :: Float64        = ame_high − ame_low
- `se_diff`   :: Float64        = SE of ame_diff
- `grad_low`  :: Vector{Float64} = ∇₍β₎[AME @ z_low]
- `grad_high` :: Vector{Float64} = ∇₍β₎[AME @ z_high]
- `n`         :: Int            = sample size
- `names`     :: Vector{String} = coefnames(model)
"""
function ame_contrast_numeric(
    df::DataFrame,
    model;
    x::Symbol,
    z::Symbol,
    z_vals::Tuple{<:Real,<:Real},
    δ::Real = 1e-6,
    typical = mean,
    invlink = (η -> 1/(1 + exp(-η))),
    dinvlink = nothing,
    d2invlink = nothing,
    vcov = StatsBase.vcov
)
    z_low, z_high = z_vals

    out_low = ame_numeric(
        df, model;
        x        = x,
        z        = z,
        z_val    = z_low,
        δ        = δ,
        typical  = typical,
        invlink  = invlink,
        dinvlink = dinvlink,
        d2invlink= d2invlink,
        vcov     = vcov
    )

    out_high = ame_numeric(
        df, model;
        x        = x,
        z        = z,
        z_val    = z_high,
        δ        = δ,
        typical  = typical,
        invlink  = invlink,
        dinvlink = dinvlink,
        d2invlink= d2invlink,
        vcov     = vcov
    )

    ame_low   = out_low.ame
    se_low    = out_low.se
    ame_high  = out_high.ame
    se_high   = out_high.se

    ame_diff  = ame_high - ame_low
    grad_diff = out_high.grad .- out_low.grad
    Σβ        = vcov(model)
    var_diff  = dot(grad_diff, Σβ * grad_diff)
    se_diff   = sqrt(var_diff)

    return (
        ame_low    = ame_low,
        se_low     = se_low,
        ame_high   = ame_high,
        se_high    = se_high,
        ame_diff   = ame_diff,
        se_diff    = se_diff,
        grad_low   = out_low.grad,
        grad_high  = out_high.grad,
        n          = out_low.n,
        names      = coefnames(model)
    )
end

function ame_contrast_numeric(model, out_low::AME, out_high::AME)

    ## add checks to ensure comparability

    ame_low   = out_low.ame
    se_low    = out_low.se
    ame_high  = out_high.ame
    se_high   = out_high.se

    ame_diff  = ame_high - ame_low
    grad_diff = out_high.grad .- out_low.grad
    Σβ        = vcov(model)
    var_diff  = dot(grad_diff, Σβ * grad_diff)
    se_diff   = sqrt(var_diff)

    return AMEContrast(
        ame_low,
        se_low,
        ame_high,
        se_high,
        ame_diff,
        se_diff,
        out_low.grad,
        out_high.grad,
        out_low.n,
        coefnames(model)
    )
end
