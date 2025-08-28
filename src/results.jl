struct MarginsResult
    table::DataFrame
    metadata::NamedTuple
end

function _new_result(table::DataFrame; kwargs...)
    md = (; kwargs...)
    return MarginsResult(table, md)
end

function _add_ci!(df::DataFrame; level::Real=0.95, dof::Union{Nothing,Real}=nothing, mcompare::Symbol=:noadjust, groupcols::Vector{Symbol}=Symbol[])
    if !haskey(df, :se) || !haskey(df, :dydx)
        return df
    end
    # Compute test statistic and p-values
    df[!, :z] = df.dydx ./ df.se
    if dof === nothing
        # Normal
        df[!, :p] = 2 .* (1 .- cdf.(Distributions.Normal(), abs.(df.z)))
        α = 1 - level
        q = Distributions.quantile(Distributions.Normal(), 1 - α/2)
    else
        # t distribution
        tdist = Distributions.TDist(Float64(dof))
        df[!, :p] = 2 .* (1 .- cdf.(tdist, abs.(df.z)))
        α = 1 - level
        q = Distributions.quantile(tdist, 1 - α/2)
    end
    # Multiple-comparison adjustment (Bonferroni/Sidak) for groups
    if mcompare != :noadjust && !isempty(groupcols)
        _adjust_multiple!(df; method=mcompare, groupcols)
        # Group-specific critical values for CIs
        keys = eachrow(select(df, groupcols...))
        idxs_by_group = Dict{Any, Vector{Int}}()
        for (i, key) in enumerate(keys)
            push!(get!(idxs_by_group, key, Int[]), i)
        end
        df[!, :ci_lo] = similar(df.dydx)
        df[!, :ci_hi] = similar(df.dydx)
        for idxs in values(idxs_by_group)
            k = length(idxs)
            α = 1 - level
            α_adj = mcompare === :bonferroni ? (α / k) : mcompare === :sidak ? (1 - (1 - α)^(1 / k)) : α
            qg = dof === nothing ? Distributions.quantile(Distributions.Normal(), 1 - α_adj/2) : Distributions.quantile(Distributions.TDist(Float64(dof)), 1 - α_adj/2)
            @inbounds for i in idxs
                df.ci_lo[i] = df.dydx[i] - qg * df.se[i]
                df.ci_hi[i] = df.dydx[i] + qg * df.se[i]
            end
        end
    else
        df[!, :ci_lo] = df.dydx .- q .* df.se
        df[!, :ci_hi] = df.dydx .+ q .* df.se
    end
    return df
end

function _adjust_multiple!(df::DataFrame; method::Symbol=:bonferroni, groupcols::Vector{Symbol}=Symbol[])
    # Define groups over which to adjust: default all rows together
    keys = isempty(groupcols) ? [namedtuple()] : eachrow(select(df, groupcols...))
    # Build mapping from group key to indices
    idxs_by_group = Dict{Any, Vector{Int}}()
    for (i, key) in enumerate(keys)
        push!(get!(idxs_by_group, key, Int[]), i)
    end
    # Adjust p-values within each group
    for idxs in values(idxs_by_group)
        k = length(idxs)
        if k <= 1; continue; end
        @inbounds for i in idxs
            p = df.p[i]
            p_adj = if method === :bonferroni
                min(p * k, 1.0)
            elseif method === :sidak
                1 - (1 - p)^k
            else
                p  # scheffe/noadjust fallback
            end
            df.p[i] = p_adj
        end
    end
    return df
end
