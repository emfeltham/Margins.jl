# Marginsresult.jl

abstract type AbstractMarginsResult end
# an abstract supertype so both results share a common interface

"""
    MarginsResult{Kind}

Stores either average marginal effects (`Kind = :dydx`) or average predictions
(`Kind = :predict`).  When `repvals` is empty the effect for a predictor is a
scalar `Float64`; otherwise it is a `Dict{Tuple,Float64}` mapping each
representative‑value combination to its estimate.  The same convention applies
to standard errors and Δ‑method gradients.
"""
struct MarginsResult{Kind} <: AbstractMarginsResult
    vars        :: Vector{Symbol}
    repvals     :: AbstractDict
    effects     :: AbstractDict # values may be Float64 or Dict{Tuple,Float64}
    ses         :: AbstractDict
    grad        :: AbstractDict
    n           :: Int
    df_residual :: Real
    family      :: String
    link        :: String

end

# pull the Kind tag (:dydx or :predict) out of the type
Kind(::MarginsResult{K}) where {K} = K

############################
# Pretty printing          #
############################

function Base.show(io::IO, ::MIME"text/plain", res::MarginsResult)
    header = Kind(res) == :dydx ? "Average Marginal Effects" : "Average Predictions"
    println(io, header, " (Family: $(res.family); Link: $(res.link))")
    println(io, "-"^80)

    repvars = collect(keys(res.repvals))
    nrep    = length(repvars)
    for rv in repvars
        @printf(io, "%-12s ", string(rv))
    end
    @printf(io, "%-12s %10s %10s %8s %8s %18s
",
            "Effect", "Estimate", "Std.Err", "z", "P>|z|", "95% CI")
    println(io, "-"^80)

    cis = confint(res; level=0.95)
    format_p(p) = p < 1e-24 ? "<1e-24" :
                  p < 1e-16 ? "<1e-16" :
                  p < 1e-4  ? "<1e-04" : @sprintf("%9.3f", p)

    for v in res.vars
        ame_entry = res.effects[v]
        se_entry  = res.ses[v]
        ci_entry  = cis[v]

        if ame_entry isa Number
            # no repvals: single row with blank moderator cells
            for _ in 1:nrep
                @printf(io, "%-12s ", "")
            end
            z   = ame_entry / se_entry
            p   = 2*(1 - cdf(Normal(), abs(z)))
            lo,hi = ci_entry
            @printf(io, "%-12s %10.4f %10.4f %8.2f %8s [%7.4f, %7.4f]
",
                    "Δ"*string(v), ame_entry, se_entry, z, format_p(p), lo, hi)
        else
            # repvals: one row per combo
            for combo in sort!(collect(keys(ame_entry)))
                for val in combo
                    @printf(io, "%-12s ", string(val))
                end
                ame_val = ame_entry[combo]
                se_val  = se_entry[combo]
                z       = ame_val / se_val
                p       = 2*(1 - cdf(Normal(), abs(z)))
                lo,hi   = ci_entry[combo]
                @printf(io, "%-12s %10.4f %10.4f %8.2f %8s [%7.4f, %7.4f]
",
                        "Δ"*string(v), ame_val, se_val, z, format_p(p), lo, hi)
            end
        end
    end

    println(io, "
Observations: ", res.n)
end
