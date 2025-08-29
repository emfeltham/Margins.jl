# core/utilities.jl - General utility functions

"""
    _nrows(data_nt)
"""
_nrows(data_nt::NamedTuple) = length(first(data_nt))

"""
    _is_categorical(data_nt, var::Symbol)

Detect categorical variable by column type.
"""
function _is_categorical(data_nt::NamedTuple, var::Symbol)
    col = getproperty(data_nt, var)
    return (Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)) || (eltype(col) <: Bool)
end

"""
    _vcov_model(model, p)

Fetch `vcov(model)` or return an identity fallback if unavailable.
"""
function _vcov_model(model, p::Integer)
    try
        return StatsBase.vcov(model)
    catch
        return Matrix{Float64}(I, p, p)
    end
end

"""
    _resolve_vcov(vcov_spec, model, p)

Resolve the `vcov` keyword into a covariance matrix Σ.
- :model → vcov(model)
- AbstractMatrix → returned as-is
- Function → vcov_spec(model)
- Estimator (CovarianceMatrices) → tries vcov(model, estimator)
"""
function _resolve_vcov(vcov_spec, model, p::Integer)
    if vcov_spec === :model || vcov_spec === nothing
        return _vcov_model(model, p)
    elseif vcov_spec isa AbstractMatrix
        return Matrix{Float64}(vcov_spec)
    elseif vcov_spec isa Function
        Σ = vcov_spec(model)
        return Matrix{Float64}(Σ)
    else
        # Try StatsBase.vcov(model, estimator) (e.g., CovarianceMatrices estimator)
        try
            Σ = StatsBase.vcov(model, vcov_spec)
            return Matrix{Float64}(Σ)
        catch
            error("Unsupported vcov specification: $(typeof(vcov_spec)). Provide :model, a matrix, a function, or a CovarianceMatrices estimator.")
        end
    end
end

"""
    _resolve_weights(weights, data_nt, idxs)

Resolve weights specification. Accepts:
- nothing → returns nothing
- AbstractVector → normalized to sum 1 over idxs
- Symbol → look up column in data_nt; normalize over idxs
Returns a Vector{Float64} aligned with idxs, or nothing.
"""
function _resolve_weights(weights, data_nt::NamedTuple, idxs)
    if weights === nothing
        return nothing
    elseif weights isa AbstractVector
        w = Float64.(weights[idxs])
    elseif weights isa Symbol
        col = getproperty(data_nt, weights)
        w = Float64.(col[idxs])
    else
        error("Unsupported weights spec: $(typeof(weights))")
    end
    s = sum(w)
    if s <= 0
        error("Nonpositive sum of weights")
    end
    return w ./ s
end

"""
    _balanced_weights(data_nt, idxs)

Compute row weights that balance categorical level combinations equally.
Returns a normalized Vector{Float64} aligned with `idxs`, or nothing if no categorical columns.
"""
function _balanced_weights(data_nt::NamedTuple, idxs, subset::Union{Nothing,Vector{Symbol}}=nothing)
    n = length(idxs)
    # Identify categorical columns
    cat_syms = Symbol[]
    if subset === nothing
        for (k, col) in pairs(data_nt)
            if (Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)) || (eltype(col) <: Bool)
                push!(cat_syms, k)
            end
        end
    else
        for k in subset
            col = getproperty(data_nt, k)
            if (Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)) || (eltype(col) <: Bool)
                push!(cat_syms, k)
            end
        end
    end
    isempty(cat_syms) && return nothing
    # Build key per row of concatenated levels
    counts = Dict{Tuple,Int}()
    keys_arr = Vector{Tuple}(undef, n)
    for (j, row) in enumerate(idxs)
        key = Tuple(getproperty(data_nt, s)[row] for s in cat_syms)
        keys_arr[j] = key
        counts[key] = get(counts, key, 0) + 1
    end
    w = zeros(Float64, n)
    for j in 1:n
        c = counts[keys_arr[j]]
        w[j] = 1 / c
    end
    w ./= sum(w)
    return w
end

"""
    _merge_weights(user_weights, balanced_weights, data_nt, idxs)

Combine user-provided weights and asbalanced weights (if any), normalizing to sum 1.
Returns Vector{Float64} or nothing.
"""
function _merge_weights(user_weights, balanced_weights, data_nt::NamedTuple, idxs)
    if user_weights === nothing && balanced_weights === nothing
        return nothing
    elseif user_weights === nothing
        return balanced_weights
    else
        base = _resolve_weights(user_weights, data_nt, idxs)
        if balanced_weights === nothing
            return base
        else
            w = base .* balanced_weights
            s = sum(w)
            s > 0 || error("Nonpositive combined weights")
            return w ./ s
        end
    end
end