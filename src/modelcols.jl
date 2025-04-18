# modelcols_alt_extension.jl

# this is from Effects.jl
# function StatsModels.modelcols(t::TypicalTerm, d::NamedTuple)
#     cols = ones(length(first(d)), width(t))
#     for (idx, v) in enumerate(t.values)
#         cols[:, idx] .*= v
#     end
#     return cols
# end

# existing version of modelcols as defined in StatsBase for a CategoricalTerm
_modelcols(t::CategoricalTerm, d::NamedTuple) = t.contrasts[d[t.sym], :]

# modified version for specified values of each level of a CategoricalTerm variable
function _modelcols_vec(t::CategoricalTerm, d::NamedTuple)

    c = t.contrasts
    # find the levels in the term's dummy coding
    lix = [c.invindex[x] for x in c.coefnames]

    # this is what we want:
    # the relevant typical values
    return permutedims(reduce(hcat, [x[lix] for x in d[t.sym]]))
end

# choose the relevant version based on the input data (reference grid) structure
function modelcols_alt(t::CategoricalTerm, d::NamedTuple)
    return if typeof(d[t.sym]) <: Vector{Vector{T}} where T <: Real
        _modelcols_vec(t::CategoricalTerm, d::NamedTuple)
    elseif typeof(d[t.sym]) <: CategoricalVector
        _modelcols(t::CategoricalTerm, d::NamedTuple)
    else typeof(d[t.sym])
        error("incorrect type. check data. it is likely you need to convert a string to a categorical type.")
    end
end
