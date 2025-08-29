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
