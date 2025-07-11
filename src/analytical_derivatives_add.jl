
# analytical_derivatives_add.jl

"""
    evaluate_term(term::ZScoredTerm, data::NamedTuple) -> Vector{Float64}

Evaluate a Z-scored term at the given data points.
"""
function evaluate_term(term::ZScoredTerm, data::NamedTuple)
    # Evaluate the underlying term
    underlying_values = evaluate_term(term.term, data)
    
    # ZScoredTerm should only wrap single-column terms
    if !(underlying_values isa Vector)
        error("ZScoredTerm is expected to wrap single-column terms only, got $(typeof(underlying_values))")
    end
    
    # Apply Z-score transformation
    return zscore_transform_vector(underlying_values, term.center, term.scale)
end

"""
    analytical_derivative(term::ZScoredTerm, variable::Symbol, data::NamedTuple) -> Vector{Float64}

Compute analytical derivative of a Z-scored term.
For Z-scored term f(x) = (g(x) - center) / scale, the derivative is:
∂f/∂x = (1/scale) * ∂g/∂x
"""
function analytical_derivative(term::ZScoredTerm, variable::Symbol, data::NamedTuple)
    # Get derivative of underlying term
    underlying_derivative = analytical_derivative(term.term, variable, data)
    
    # ZScoredTerm should only produce single-column derivatives
    if !(underlying_derivative isa Vector)
        error("ZScoredTerm derivative should be single-column, got $(typeof(underlying_derivative))")
    end
    
    # Apply scale factor: d/dx[(g(x) - c)/s] = (1/s) * dg/dx
    if term.scale isa Number
        return underlying_derivative ./ term.scale
    elseif term.scale isa AbstractVector
        if length(term.scale) == 1
            return underlying_derivative ./ term.scale[1]
        else
            error("ZScoredTerm scale vector should have length 1 for single-column terms, got length $(length(term.scale))")
        end
    else
        error("Unsupported scale type for ZScoredTerm: $(typeof(term.scale))")
    end
end

"""
    zscore_transform_vector(values::AbstractVector, center, scale) -> Vector{Float64}

Apply Z-score transformation to a vector: (x - center) / scale
"""
function zscore_transform_vector(values::AbstractVector, center, scale)
    n = length(values)
    result = Vector{Float64}(undef, n)
    
    # Extract scalar values from center and scale
    c = center isa Number ? center : (length(center) == 1 ? center[1] : error("Center should be scalar or length-1 vector"))
    s = scale isa Number ? scale : (length(scale) == 1 ? scale[1] : error("Scale should be scalar or length-1 vector"))
    
    # Apply transformation efficiently
    if c == 0
        inv_s = 1.0 / s
        @inbounds @simd for i in 1:n
            result[i] = values[i] * inv_s
        end
    else
        inv_s = 1.0 / s
        @inbounds @simd for i in 1:n
            result[i] = (values[i] - c) * inv_s
        end
    end
    
    return result
end
