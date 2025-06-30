# margins_to_df.jl — updated without merge! calls
using DataFrames, Statistics, Distributions, Tables

"""
    margins_to_df(res::MarginsResult; level=0.95)

Convert a `MarginsResult` into a tidy `DataFrame`: one row per effect or
rep-value combination.
"""
function margins_to_df(res::MarginsResult; level=0.95)
    kind     = typeof(res).parameters[1]         # :dydx or :predict
    prefix   = kind === :dydx ? "Δ" : "ŷ"
    cis      = confint(res; level=level)
    repvars  = collect(keys(res.repvals))
    rows     = Vector{Dict{Symbol,Any}}()

    for v in res.vars
        est_map = res.effects[v]
        se_map  = res.ses[v]
        ci_map  = cis[v]
        effect_sym = Symbol(prefix, v)

        if est_map isa Number
            # scalar case
            d = Dict{Symbol,Any}(rv => missing for rv in repvars)
            d[:Effect]   = effect_sym
            d[:Estimate] = est_map
            d[:StdErr]   = se_map
            d[:z]        = est_map / se_map
            d[:p]        = 2*(1 - cdf(Normal(), abs(d[:z])))
            lo,hi        = ci_map
            d[:lower]    = lo
            d[:upper]    = hi
            push!(rows, d)
        else
            # dict case
            for combo in sort(collect(keys(est_map)))
                d = Dict{Symbol,Any}()
                for (i, rv) in enumerate(repvars)
                    d[rv] = combo[i]
                end
                est = est_map[combo]
                se  = se_map[combo]
                d[:Effect]   = effect_sym
                d[:Estimate] = est
                d[:StdErr]   = se
                d[:z]        = est / se
                d[:p]        = 2*(1 - cdf(Normal(), abs(d[:z])))
                lo,hi        = ci_map[combo]
                d[:lower]    = lo
                d[:upper]    = hi
                push!(rows, d)
            end
        end
    end

    df = DataFrame(rows)
    select!(df, vcat(repvars, [:Effect, :Estimate, :StdErr, :z, :p, :lower, :upper]))
    return df
end

# extend DataFrame constructor and Tables integration
DataFrame(res::MarginsResult; kwargs...) = margins_to_df(res; kwargs...)
Tables.istable(::Type{MarginsResult}) = true
Tables.schema(res::MarginsResult)     = Tables.schema(margins_to_df(res))
Tables.rows(res::MarginsResult)       = Tables.rows(margins_to_df(res))
