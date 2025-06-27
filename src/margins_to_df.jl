# margins_to_df.jl

"""
    margins_to_df(res::MarginsResult; level=0.95)

Convert a MarginsResult into a `DataFrame` with one row per effect (and per
rep‐value combination, if any). Columns are:

  • each key of `res.repvals`  
  • `:Effect`   — symbol like `:Δx1`  
  • `:AME`      — marginal‐effect estimate  
  • `:StdErr`   — its standard error  
  • `:z`        — score = AME/StdErr  
  • `:p`        — two‐sided p‐value  
  • `:lower,:upper` — CI bounds at the given level  
"""
function margins_to_df(res::MarginsResult; level=0.95)
    # pull CIs
    cis = confint(res; level=level)
    repvars = collect(keys(res.repvals))
    rows = Vector{Dict{Symbol,Any}}()

    for v in res.vars
        ame_e = res.effects[v]
        se_e  = res.ses[v]
        ci_e  = cis[v]

        if isa(ame_e, Number)
            # one row, no combos
            d = Dict{Symbol,Any}(rv => missing for rv in repvars)
            ame, se = ame_e, se_e
            z = ame/se
            p = 2 * (1 - cdf(Normal(), abs(z)))
            lo, hi = ci_e

            merge!(d,
                :Effect => Symbol("Δ", v),
                :AME    => ame,
                :StdErr => se,
                :z      => z,
                :p      => p,
                :lower  => lo,
                :upper  => hi,
            )
            push!(rows, d)

        else
            # one row per combo tuple
            for combo in sort(collect(keys(ame_e)))
                d = Dict{Symbol,Any}()
                for (i, rv) in enumerate(repvars)
                    d[rv] = combo[i]
                end

                ame = ame_e[combo]
                se  = se_e[combo]
                z   = ame/se
                p   = 2 * (1 - cdf(Normal(), abs(z)))
                lo, hi = ci_e[combo]

                merge!(d,
                    :Effect => Symbol("Δ", v),
                    :AME    => ame,
                    :StdErr => se,
                    :z      => z,
                    :p      => p,
                    :lower  => lo,
                    :upper  => hi,
                )
                push!(rows, d)
            end
        end
    end

    df = DataFrame(rows)
    select!(df, vcat(repvars, [:Effect, :AME, :StdErr, :z, :p, :lower, :upper]))
    return df
end

# ——————————————————————————————
# 1. DataFrame constructor extension
DataFrame(res::MarginsResult; kwargs...) = margins_to_df(res; kwargs...)

# Now you can do:
#    df = DataFrame(my_margins_result)
#    df2 = DataFrame(my_margins_result; level=0.90)

# ——————————————————————————————
# 2. Full Tables.jl integration
Tables.istable(::Type{MarginsResult}) = true

function Tables.schema(res::MarginsResult)
    Tables.schema(margins_to_df(res))
end

Tables.rows(res::MarginsResult) = Tables.rows(margins_to_df(res))

# Now ANY Tables.jl consumer should work:
#    CSV.write("out.csv", my_margins_result)
#    Arrow.write("out.arrow", my_margins_result)
#    Query(@from r in my_margins_result begin ... end)
