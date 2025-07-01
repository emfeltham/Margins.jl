using DataFrames, Distributions

function margins_to_df(res::MarginsResult; level=0.95)
    # 1) Setup
    kind    = Kind(res)             # :dydx or :predict
    prefix  = kind === :dydx ? "Δ" : "ŷ"
    cis     = confint(res; level=level)
    repvars = collect(keys(res.repvals))
    Nrep    = length(repvars)

    # 2) Figure out how many "level*" columns we need
    max_extra = 0
    for v in res.vars
        eff = res.effects[v]
        if eff isa Dict
            # each key is a Tuple; extra = length(key) - Nrep
            max_extra = max(max_extra,
                            maximum(length, keys(eff)) - Nrep)
        end
    end

    # 3) Build the full column order
    level_cols = Symbol.(["level$i" for i in 1:max_extra])
    stat_cols  = [:Effect, :Estimate, :StdErr, :z, :p, :lower, :upper]
    colorder   = vcat(repvars, [:Variable], level_cols, stat_cols)

    # 4) Iterate focal variables, collect rows
    rows = Vector{Dict{Symbol,Any}}()
    for v in res.vars
        eff_map = res.effects[v]
        se_map  = res.ses[v]
        ci_map  = cis[v]
        eff_sym = Symbol(prefix, v)

        if eff_map isa Number
            # Scalar AME (no repvals)
            d = Dict{Symbol,Any}()
            for rv in repvars
                d[rv] = missing
            end
            d[:Variable] = v
            for lc in level_cols
                d[lc] = missing
            end
            d[:Effect]   = eff_sym
            d[:Estimate] = eff_map
            d[:StdErr]   = se_map
            d[:z]        = eff_map / se_map
            d[:p]        = 2*(1 - cdf(Normal(), abs(d[:z])))
            lo,hi        = ci_map
            d[:lower]    = lo
            d[:upper]    = hi
            push!(rows, d)

        else
            # Dict AMEs: keys are Tuples
            for key in sort(collect(keys(eff_map)))
                d = Dict{Symbol,Any}()
                # repvals
                for (i,rv) in enumerate(repvars)
                    d[rv] = key[i]
                end
                d[:Variable] = v

                # categorical level fields
                extra = length(key) - Nrep
                for j in 1:max_extra
                    col = level_cols[j]
                    if j <= extra
                        d[col] = key[Nrep + j]
                    else
                        d[col] = missing
                    end
                end

                # stats
                est = eff_map[key]
                se  = se_map[key]
                lo,hi = ci_map[key]
                d[:Effect]   = eff_sym
                d[:Estimate] = est
                d[:StdErr]   = se
                d[:z]        = est / se
                d[:p]        = 2*(1 - cdf(Normal(), abs(d[:z])))
                d[:lower]    = lo
                d[:upper]    = hi

                push!(rows, d)
            end
        end
    end

    # 5) Build the DataFrame
    df = DataFrame(rows)
    # ensure all columns exist
    for c in colorder
        if c ∉ Symbol.(names(df))
            df[!, c] = fill(missing, nrow(df))
        end
    end
    select!(df, colorder)
end

# extend DataFrame constructor and Tables integration
DataFrame(res::MarginsResult; kwargs...) = margins_to_df(res; kwargs...)
Tables.istable(::Type{MarginsResult}) = true
Tables.schema(res::MarginsResult)     = Tables.schema(margins_to_df(res))
Tables.rows(res::MarginsResult)       = Tables.rows(margins_to_df(res))
