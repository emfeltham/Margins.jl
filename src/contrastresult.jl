# contrastresult.jl

# -----------------------------------------------------------------------------
# ContrastResult holds one or more contrasts
# -----------------------------------------------------------------------------
"""
    ContrastResult

Container for results of one or more contrasts between average marginal effects (AMEs).

# Fields
- `vars::Vector{Symbol}`: Variables involved in each contrast (e.g. a single predictor or two predictors).
- `comps::Vector{Tuple}`: Keys or index pairs for each comparison (e.g. `(:a,:b)`, level tuples).
- `estimate::Vector{Float64}`: Difference in AME for each comparison (θ₁−θ₂).
- `se::Vector{Float64}`: Standard error of each difference, via the Δ-method.
- `t::Vector{Float64}`: t-statistics (estimate ÷ se).
- `p::Vector{Float64}`: Two-sided p-values from Student’s t-distribution.
- `df_residual::Real`: Degrees of freedom used for all tests.
"""
struct ContrastResult
    vars        :: Vector{Symbol}
    comps       :: Vector{Tuple}
    estimate    :: Vector{Float64}
    se          :: Vector{Float64}
    t           :: Vector{Float64}
    p           :: Vector{Float64}
    df_residual :: Real
end

function Base.show(io::IO, ::MIME"text/plain", cr::ContrastResult)
    # Title
    println(io, "Contrasts of Average Marginal Effects (df = $(cr.df_residual))")
    println(io)

    # Frame & header
    println(io, "─────────────────────────────────────────────────────────────────────────")
    @printf(io, "%-20s %9s %11s %6s %9s %9s %9s\n",
        "Comparison", "Estimate", "Std.Error", "t", "Pr(>|t|)", "Lower 95%", "Upper 95%")
    println(io, "─────────────────────────────────────────────────────────────────────────")

    # critical t for 95% CI
    crit = quantile(TDist(cr.df_residual), 0.975)

    # Body rows
    for i in eachindex(cr.estimate)
        # label: e.g. (:a,:b) → "a–b"
        cmp = cr.comps[i]
        label = join(string.(cmp), "–")   # separate components

        est = cr.estimate[i]
        se  = cr.se[i]
        t   = cr.t[i]
        p   = cr.p[i]
        lo  = est - crit*se
        hi  = est + crit*se

        @printf(io, "%-20s %9.6f %11.6f %6.2f %9s %9.6f %9.6f\n",
            label, est, se, t, format_p(p), lo, hi)
    end

    # Footer
    println(io, "─────────────────────────────────────────────────────────────────────────")
    println(io)
end
