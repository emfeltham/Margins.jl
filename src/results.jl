struct MarginsResult
    table::DataFrame
    metadata::NamedTuple
end

function _new_result(table::DataFrame; kwargs...)
    md = (; kwargs...)
    return MarginsResult(table, md)
end

function _add_ci!(df::DataFrame; level::Real=0.95)
    if !haskey(df, :se) || !haskey(df, :dydx)
        return df
    end
    # Normal z by default; t-based can be added later with dof
    using Distributions
    z = quantile(Normal(), 0.5 + level/2)
    df[!, :ci_lo] = df.dydx .- z .* df.se
    df[!, :ci_hi] = df.dydx .+ z .* df.se
    return df
end
