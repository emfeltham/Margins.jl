# core/grouping.jl - Grouping and stratification utilities

"""
    _build_groups(data_nt, over, within=nothing, idxs=nothing)

Build group index (labels→row indices) for over/within variables, optionally limited to idxs.
"""
function _build_groups(data_nt::NamedTuple, over, within=nothing, idxs=nothing)
    if over === nothing && within === nothing
        return nothing
    end
    vars = Symbol[]
    if over !== nothing
        append!(vars, over isa Symbol ? [over] : collect(over))
    end
    if within !== nothing
        append!(vars, within isa Symbol ? [within] : collect(within))
    end
    # Build mapping from group label -> row indices
    groups = Dict{NamedTuple, Vector{Int}}()
    rows = idxs === nothing ? collect(1 : _nrows(data_nt)) : idxs
    for i in rows
        key = NamedTuple{Tuple(vars)}(Tuple(getproperty(data_nt, v)[i] for v in vars))
        push!(get!(groups, key, Int[]), i)
    end
    return collect(groups)
end

"""
    _split_by(data_nt, by)

Return a list of (bylabels, idxs) or nothing if `by` is nothing.
"""
function _split_by(data_nt::NamedTuple, by)
    if by === nothing
        return nothing
    end
    vars = by isa Symbol ? [by] : collect(by)
    groups = Dict{NamedTuple, Vector{Int}}()
    n = _nrows(data_nt)
    for i in 1:n
        key = NamedTuple{Tuple(vars)}(Tuple(getproperty(data_nt, v)[i] for v in vars))
        push!(get!(groups, key, Int[]), i)
    end
    return collect(groups)
end

"""
    _subset_data(data_nt, idxs)

Create a subset of the NamedTuple data using the provided indices.
"""
function _subset_data(data_nt::NamedTuple, idxs::Vector{Int})
    subset_data = NamedTuple{keys(data_nt)}(
        Tuple(getproperty(data_nt, k)[idxs] for k in keys(data_nt))
    )
    return subset_data
end