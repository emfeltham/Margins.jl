"""
    _build_profiles(at, data_nt)

Return a vector of profile dicts. Supports:
- :none → empty vector (signals per-row)
- :means → single Dict with each Real column set to mean; categoricals use first level
- Dict{Symbol,Vector} → Cartesian product profiles
- Vector{Dict} → passthrough
"""
function _build_profiles(at, data_nt::NamedTuple)
    if at === :none
        return Vector{Dict{Symbol,Any}}()
    elseif at === :means
        prof = Dict{Symbol,Any}()
        for (k, col) in pairs(data_nt)
            if eltype(col) <: Real
                prof[k] = mean(col)
            elseif Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
                prof[k] = levels(col)[1]
            end
        end
        return [prof]
    elseif at isa Dict{Symbol,<:AbstractVector}
        keys_vec = collect(keys(at))
        vals_vec = [collect(at[k]) for k in keys_vec]
        profs = Dict{Symbol,Any}[]
        for combo in Iterators.product(vals_vec...)
            d = Dict{Symbol,Any}()
            for (i, k) in enumerate(keys_vec)
                d[k] = combo[i]
            end
            push!(profs, d)
        end
        return profs
    elseif at isa Vector{<:Dict}
        return Dict{Symbol,Any}.(at)
    else
        error("Unsupported at specification: $(typeof(at))")
    end
end
