using Printf, Distributions

# — unified confint, using Student’s t —
function confint(res::AMEResult, var::Symbol; level::Real=0.95)
    @assert var in res.vars
    ame = res.ame[var]
    se  = res.se[var]
    α   = 1 - level
    crit = quantile(TDist(res.df_residual), 1 - α/2)

    if isa(ame, Number)
        return (ame - crit*se, ame + crit*se)
    else
        sub = Dict{Tuple,Tuple{Float64,Float64}}()
        for (combo,ame_val) in ame
            se_val = se[combo]
            sub[combo] = (ame_val - crit*se_val, ame_val + crit*se_val)
        end
        return sub
    end
end

function confint(res::AMEResult; level::Real=0.95)
    Dict(v => confint(res, v; level=level) for v in res.vars)
end
