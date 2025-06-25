# confint.jl

"""
    confint(res::AMEResult; level=0.95) -> Dict{Symbol,Tuple{Float64,Float64}}

Compute Wald-style confidence intervals for each variable’s AME in an AMEResult.
- `level` is the desired coverage (default 0.95).
Returns a Dict mapping each variable ⇒ (lower, upper).
"""
function confint(res::AMEResult; level::Real=0.95)
    α = 1 - level
    z = quantile(Normal(), 1 - α/2)
    cis = Dict{Symbol, Tuple{Float64,Float64}}()
    for v in res.vars
        ame = res.ame[v]
        se  = res.se[v]
        lo  = ame - z * se
        hi  = ame + z * se
        cis[v] = (lo, hi)
    end
    return cis
end

"""
    confint(res::AMEResult, var::Symbol; level=0.95) -> Tuple{Float64,Float64}

Extract the confidence interval for a single variable’s AME.
"""
function confint(res::AMEResult, var::Symbol; level::Real=0.95)
    @assert var in res.vars "Variable $(var) not in AMEResult"
    ame = res.ame[var]
    se  = res.se[var]
    α = 1 - level
    z = quantile(Normal(), 1 - α/2)
    return (ame - z * se, ame + z * se)
end
