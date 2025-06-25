# AMER.jl

"""
AMEResult holds average marginal effects (AME) output for one or more predictors.

# Fields
- `vars::Vector{Symbol}`: focal predictors
- `ame::Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}`: AME estimates (scalar or per-combo)
- `se::Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}`: standard errors (scalar or per-combo)
- `grad::Dict{Symbol, Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}`: Δ-method gradients (vector or per-combo)
- `n::Int`: sample size
- `family::String`, `link::String`: model family/link info
"""
struct AMEResult
    vars        :: Vector{Symbol}
    repvals     :: Dict{Symbol,AbstractVector}
    ame         :: Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}
    se          :: Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}
    grad        :: Dict{Symbol, Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}
    n           :: Int
    df_residual :: Real
    family      :: String
    link        :: String
end

using Printf, Distributions

function Base.show(io::IO, ::MIME"text/plain", res::AMEResult)
    # Header
    println(io, "Average Marginal Effects (Family: $(res.family); Link: $(res.link))")
    println(io, "-"^80)

    # moderator columns
    repvars = collect(keys(res.repvals))
    nrep = length(repvars)

    # column headings
    for rv in repvars
        @printf(io, "%-12s ", string(rv))
    end
    @printf(io, "%-12s %10s %10s %8s %8s %18s\n",
        "Effect", "AME", "Std.Err", "z", "P>|z|", "95% CI")
    println(io, "-"^80)

    # get all CIs in one go
    cis = confint(res; level=0.95)

    # helper for p-value formatting
    # — exactly GLM.jl’s p-value formatting —
    format_p(p) =
        p < 1e-24 ? "<1e-24" :
        p < 1e-16 ? "<1e-16" :
        p < 1e-4  ? "<1e-04" :
                @sprintf("%9.3f", p)

    # body
    for v in res.vars
        ame_entry = res.ame[v]
        se_entry  = res.se[v]
        ci_entry  = cis[v]

        if isa(ame_entry, Number)
            # blank moderators
            for _ in 1:nrep
                @printf(io, "%-12s ", "")
            end
            # stats
            z    = ame_entry / se_entry
            p    = 2*(1 - cdf(Normal(), abs(z)))
            (lo,hi) = ci_entry
            pstr = format_p(p)

            @printf(io, "%-12s %10.4f %10.4f %8.2f %8s [%7.4f, %7.4f]\n",
                "Δ"*string(v), ame_entry, se_entry, z, pstr, lo, hi)

        else
            # one row per combo
            for combo in sort(collect(keys(ame_entry)))
                for val in combo
                    @printf(io, "%-12s ", string(val))
                end
                ame_val = ame_entry[combo]
                se_val  = se_entry[combo]
                z       = ame_val / se_val
                p       = 2*(1 - cdf(Normal(), abs(z)))
                (lo,hi) = ci_entry[combo]
                pstr    = format_p(p)

                @printf(io, "%-12s %10.4f %10.4f %8.2f %8s [%7.4f, %7.4f]\n",
                    "Δ"*string(v), ame_val, se_val, z, pstr, lo, hi)
            end
        end
    end

    # footer
    println(io)
    println(io, "Observations: ", res.n)
end

