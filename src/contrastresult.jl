# contrastresult.jl — container and printer

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

format_p(p) = p < 1e-24 ? "<1e-24" : p < 1e-16 ? "<1e-16" :
              p < 1e-4  ? "<1e-04" : @sprintf("%9.3f", p)

function Base.show(io::IO, ::MIME"text/plain", cr::ContrastResult)
    println(io, "Contrasts (df = $(cr.df_residual))\n")
    println(io, "─────────────────────────────────────────────────────────────────────────")
    @printf(io, "%-20s %9s %11s %6s %9s %9s %9s\n",
            "Comparison", "Estimate", "Std.Error", "t", "Pr(>|t|)", "Lower 95%", "Upper 95%")
    println(io, "─────────────────────────────────────────────────────────────────────────")
    crit = quantile(TDist(cr.df_residual), 0.975)
    for i in eachindex(cr.estimate)
        label = join(string.(cr.comps[i]), "–")
        est, se, t, p = cr.estimate[i], cr.se[i], cr.t[i], cr.p[i]
        lo, hi = est - crit*se, est + crit*se
        @printf(io, "%-20s %9.6f %11.6f %6.2f %9s %9.6f %9.6f\n",
                label, est, se, t, format_p(p), lo, hi)
    end
    println(io, "─────────────────────────────────────────────────────────────────────────\n")
end
