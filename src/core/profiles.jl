# profile.jl

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
            if eltype(col) <: Real && !(eltype(col) <: Bool)
                prof[k] = mean(col)
            elseif Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
                prof[k] = levels(col)[1]
            elseif eltype(col) <: Bool
                # For Bool columns, use the first level (false) as default
                prof[k] = false
            end
        end
        return [prof]
    elseif at isa Dict{Symbol,<:Any}
        # General-to-specific precedence with optional :all key 
        # Build a merged map var=>values
        merged = Dict{Symbol,Vector{Any}}()
        # Handle :all summary spec across real columns
        if haskey(at, :all)
            spec_all = at[:all]
            for (k, col) in pairs(data_nt)
                if eltype(col) <: Real && !(eltype(col) <: Bool)
                    merged[k] = Any[_expand_at_values(data_nt, k, spec_all)...]
                end
            end
        end
        # Overlay specific variable settings
        for (k, v) in pairs(at)
            k === :all && continue
            merged[k] = Any[_expand_at_values(data_nt, k, v)...]
        end
        # Produce Cartesian product over merged keys
        keys_vec = collect(keys(merged))
        vals_vec = [merged[k] for k in keys_vec]
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
        # Multiple at() blocks → concatenate profile sets from each block
        profs = Dict{Symbol,Any}[]
        for blk in at
            append!(profs, _build_profiles(blk, data_nt))
        end
        return profs
    else
        error("Unsupported at specification: $(typeof(at))")
    end
end

function _expand_at_values(data_nt::NamedTuple, var::Symbol, spec)
    if spec isa CategoricalMixture
        # Validate mixture against actual data
        col = getproperty(data_nt, var)
        _validate_mixture_against_data(spec, col, var)
        return [spec]  # Return mixture object directly for special handling
    elseif spec isa AbstractVector
        return collect(spec)
    elseif spec isa AbstractString
        # Check if this is a categorical variable - if so, treat string as categorical level
        col = getproperty(data_nt, var)
        if col isa CategoricalArray || eltype(col) <: AbstractString
            return [spec]  # Categorical level string, don't parse as number
        else
            return _parse_numlist(spec)  # Numeric string for continuous variables
        end
    elseif spec isa Symbol
        col = getproperty(data_nt, var)
        if spec === :mean
            return [mean(col)]
        elseif spec === :median
            return [Statistics.median(col)]
        elseif startswith(String(spec), "p")
            p = parse(Float64, String(spec)[2:end]) / 100
            return [Statistics.quantile(col, p)]
        else
            error("Unknown summary spec: $spec for $var")
        end
    else
        return [spec]
    end
end

function _parse_numlist(s::AbstractString)
    # Parse forms like "10(5)30" -> 10:5:30, allow comma-separated values too
    s = strip(s)
    if occursin('(', s) && occursin(')', s)
        m = match(r"^\s*([+-]?\d*\.?\d+)\((\d*\.?\d+)\)([+-]?\d*\.?\d+)\s*$", s)
        if m !== nothing
            a = parse(Float64, m.captures[1])
            step = parse(Float64, m.captures[2])
            b = parse(Float64, m.captures[3])
            n = Int(floor((b - a) / step)) + 1
            return [a + i*step for i in 0:(n-1)]
        end
    end
    # Fallback: comma/space separated
    parts = split(replace(s, ',' => ' '))
    return [parse(Float64, p) for p in parts if !isempty(strip(p))]
end

# ========================================================================================
# Phase 1: Streaming Profile Source Infrastructure
# ========================================================================================

"""
    to_profile_iterator(at::NamedTuple, data_nt::NamedTuple)

Create a streaming iterator over profile rows in nested key order without materializing a DataFrame.
Keys are iterated in the order they appear in the NamedTuple.
"""
function to_profile_iterator(at::NamedTuple, data_nt::NamedTuple)
    # Build expanded values for each key in NamedTuple order
    keys_vec = collect(keys(at))
    vals_vec = [_expand_at_values(data_nt, k, getproperty(at, k)) for k in keys_vec]
    
    # Return iterator that yields Dict for each combination
    return (Dict{Symbol,Any}(k => combo[i] for (i, k) in enumerate(keys_vec)) 
            for combo in Iterators.product(vals_vec...))
end

"""
    to_profile_iterator(at::Vector{Pair{Symbol,<:Any}}, data_nt::NamedTuple)

Create a streaming iterator respecting pair order.
"""
function to_profile_iterator(at::Vector{Pair{Symbol,<:Any}}, data_nt::NamedTuple)
    keys_vec = [pair.first for pair in at]
    vals_vec = [_expand_at_values(data_nt, pair.first, pair.second) for pair in at]
    
    return (Dict{Symbol,Any}(k => combo[i] for (i, k) in enumerate(keys_vec)) 
            for combo in Iterators.product(vals_vec...))
end

"""
    to_profile_iterator(at::Dict{Symbol,<:Any}, data_nt::NamedTuple)

Create a streaming iterator using documented stable key order.
Uses sorted key order for deterministic behavior when Dict order is not guaranteed.
"""
function to_profile_iterator(at::Dict{Symbol,<:Any}, data_nt::NamedTuple)
    # Reject bare Dict for ordered grids - suggest alternatives
    error("Use NamedTuple or Vector{Pair} for ordered profile grids instead of Dict. " *
          "For deterministic behavior, try: (x=[...], z=[...]) or [:x => [...], :z => [...]]")
end

"""
    to_profile_iterator(at::Symbol, data_nt::NamedTuple)

Handle special symbols like :means.
"""
function to_profile_iterator(at::Symbol, data_nt::NamedTuple)
    if at === :means
        prof = Dict{Symbol,Any}()
        for (k, col) in pairs(data_nt)
            if eltype(col) <: Real && !(eltype(col) <: Bool)
                prof[k] = mean(col)
            elseif Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
                prof[k] = levels(col)[1]
            elseif eltype(col) <: Bool
                prof[k] = false
            end
        end
        return (prof for _ in 1:1)  # Single-item iterator
    else
        error("Unknown symbol specification: $at")
    end
end

"""
    to_profile_iterator(reference_grid::AbstractDataFrame)

Stream rows of user-provided table, preserving input row order.
"""
function to_profile_iterator(reference_grid::AbstractDataFrame)
    return (Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(row)) 
            for row in Tables.rows(reference_grid))
end

"""
    to_profile_iterator(profiles::Vector{<:Dict})

Pass-through for pre-enumerated profiles, preserving row order.
"""
function to_profile_iterator(profiles::Vector{<:Dict})
    return (prof for prof in profiles)
end
