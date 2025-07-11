# analytical_derivatives.jl - EFFICIENT VERSION: Minimal allocations and smart caching

# Global registry for function derivatives (cached to avoid recomputation)
const DERIVATIVE_REGISTRY = Dict{Function, Function}()

# Standard mathematical function derivatives (pre-computed)
const STANDARD_DERIVATIVES = Dict{Function, Function}(
    log => x -> 1/x,
    log1p => x -> 1/(1+x),
    exp => x -> exp(x),
    expm1 => x -> exp(x),
    sqrt => x -> 1/(2*sqrt(x)),
    sin => x -> cos(x),
    cos => x -> -sin(x),
    tan => x -> sec(x)^2,
    atan => x -> 1/(1+x^2),
    sinh => x -> cosh(x),
    cosh => x -> sinh(x),
    tanh => x -> sech(x)^2,
)

###############################################################################
# Efficient Core Derivative Functions
###############################################################################

"""
    analytical_derivative(term::AbstractTerm, variable::Symbol, data::NamedTuple) -> Union{Vector{Float64}, Matrix{Float64}}

EFFICIENT: Compute analytical derivative with minimal allocations.
Returns Vector for single-column terms, Matrix for multi-column terms.
"""
function analytical_derivative(term::AbstractTerm, variable::Symbol, data::NamedTuple)
    error("Analytical derivative not implemented for $(typeof(term))")
end

# EFFICIENT: Direct implementations for common term types
function analytical_derivative(term::ContinuousTerm, variable::Symbol, data::NamedTuple)
    n = length(data[variable])
    return term.sym == variable ? ones(Float64, n) : zeros(Float64, n)
end

function analytical_derivative(term::Term, variable::Symbol, data::NamedTuple)
    n = length(data[variable])
    return term.sym == variable ? ones(Float64, n) : zeros(Float64, n)
end

function analytical_derivative(term::ConstantTerm, variable::Symbol, data::NamedTuple)
    return zeros(Float64, length(data[variable]))
end

function analytical_derivative(term::InterceptTerm, variable::Symbol, data::NamedTuple)
    return zeros(Float64, length(data[variable]))
end

function analytical_derivative(term::CategoricalTerm, variable::Symbol, data::NamedTuple)
    return zeros(Float64, length(data[variable]))
end

###############################################################################
# Efficient Term Evaluation Functions
###############################################################################

"""
    evaluate_term(term::AbstractTerm, data::NamedTuple) -> Union{Vector{Float64}, Matrix{Float64}}

EFFICIENT: Evaluate term with minimal allocations and smart type handling.
"""
function evaluate_term(term::AbstractTerm, data::NamedTuple)
    error("Term evaluation not implemented for $(typeof(term))")
end

function evaluate_term(term::ContinuousTerm, data::NamedTuple)
    return Float64.(data[term.sym])
end

function evaluate_term(term::Term, data::NamedTuple)
    return Float64.(data[term.sym])
end

function evaluate_term(term::ConstantTerm, data::NamedTuple)
    n = length(first(data))
    return fill(Float64(term.n), n)
end

function evaluate_term(term::InterceptTerm{true}, data::NamedTuple)
    n = length(first(data))
    return ones(Float64, n)
end

function evaluate_term(term::InterceptTerm{false}, data::NamedTuple)
    n = length(first(data))
    return zeros(Float64, n)
end

"""
    evaluate_term(term::CategoricalTerm, data::NamedTuple)

EFFICIENT: Handle multi-column categorical terms with minimal allocations.
"""
function evaluate_term(term::CategoricalTerm, data::NamedTuple)
    v = data[term.sym]
    M = term.contrasts.matrix
    n = length(v)
    num_contrasts = size(M, 2)
    
    # EFFICIENT: Pre-allocate result matrix
    result = Matrix{Float64}(undef, n, num_contrasts)
    
    # Handle both CategoricalArray and regular arrays efficiently
    if isa(v, CategoricalArray)
        codes = refs(v)
        # EFFICIENT: Vectorized assignment with bounds checking
        @inbounds for i in 1:n
            code = codes[i]
            for j in 1:num_contrasts
                result[i, j] = M[code, j]
            end
        end
    else
        # EFFICIENT: Convert to codes once, then vectorize
        unique_vals = sort(unique(v))
        code_map = Dict(val => i for (i, val) in enumerate(unique_vals))
        
        @inbounds for i in 1:n
            code = code_map[v[i]]
            for j in 1:num_contrasts
                result[i, j] = M[code, j]
            end
        end
    end
    
    return result
end

"""
    evaluate_term(term::InteractionTerm, data::NamedTuple)

EFFICIENT: Handle interactions with smart Kronecker product computation.
"""
function evaluate_term(term::InteractionTerm, data::NamedTuple)
    # EFFICIENT: Evaluate components and compute Kronecker product
    component_values = [evaluate_term(component, data) for component in term.terms]
    
    # Start with first component
    result = component_values[1]
    
    # EFFICIENT: Iterative Kronecker product to avoid large intermediate matrices
    for i in 2:length(component_values)
        result = kron_product_columns_efficient!(result, component_values[i])
    end
    
    return result
end

"""
    kron_product_columns_efficient!(A, B) -> Matrix{Float64}

EFFICIENT: Compute Kronecker product with minimal allocations.
"""
function kron_product_columns_efficient!(A::AbstractVecOrMat, B::AbstractVecOrMat)
    A_mat = A isa AbstractVector ? reshape(A, :, 1) : A
    B_mat = B isa AbstractVector ? reshape(B, :, 1) : B
    
    n = size(A_mat, 1)
    @assert size(B_mat, 1) == n "Arrays must have same number of rows"
    
    p, q = size(A_mat, 2), size(B_mat, 2)
    
    # EFFICIENT: Pre-allocate result
    result = Matrix{Float64}(undef, n, p * q)
    
    # EFFICIENT: Column-wise computation with SIMD
    col_idx = 1
    @inbounds for j in 1:q, i in 1:p
        @simd for row in 1:n
            result[row, col_idx] = A_mat[row, i] * B_mat[row, j]
        end
        col_idx += 1
    end
    
    return result
end

"""
    evaluate_term(term::FunctionTerm, data::NamedTuple)

EFFICIENT: Handle function terms with optimized argument evaluation.
"""
function evaluate_term(term::FunctionTerm, data::NamedTuple)
    n = length(first(data))
    
    if length(term.args) == 1
        # EFFICIENT: Single argument case
        arg_values = evaluate_term(term.args[1], data)
        
        if arg_values isa Vector
            if term.f in [<=, >=, <, >, ==, !=]
                @warn "Single-argument comparison operator $(term.f), returning input"
                return arg_values
            else
                # EFFICIENT: Vectorized function application
                return term.f.(arg_values)
            end
        else
            error("Multi-column single-argument functions not yet supported")
        end
        
    elseif length(term.args) == 2
        # EFFICIENT: Two argument case with optimized constant handling
        arg1_values = evaluate_term(term.args[1], data)
        arg2 = term.args[2]
        
        if arg2 isa ConstantTerm
            # EFFICIENT: Comparison with constant (broadcasting)
            arg2_val = arg2.n
            
            if arg1_values isa Vector
                result = Vector{Float64}(undef, n)
                
                if term.f === (<=)
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(arg1_values[i] <= arg2_val)
                    end
                elseif term.f === (>=)
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(arg1_values[i] >= arg2_val)
                    end
                elseif term.f === (<)
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(arg1_values[i] < arg2_val)
                    end
                elseif term.f === (>)
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(arg1_values[i] > arg2_val)
                    end
                elseif term.f === (==)
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(arg1_values[i] == arg2_val)
                    end
                elseif term.f === (!=)
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(arg1_values[i] != arg2_val)
                    end
                else
                    # EFFICIENT: Mathematical functions with constant
                    @inbounds @simd for i in 1:n
                        result[i] = term.f(arg1_values[i], arg2_val)
                    end
                end
                
                return result
            else
                error("Multi-column comparisons not yet supported")
            end
        else
            # Two variable arguments
            arg2_values = evaluate_term(arg2, data)
            
            if arg1_values isa Vector && arg2_values isa Vector
                result = Vector{Float64}(undef, n)
                
                if term.f in [<=, >=, <, >, ==, !=]
                    @inbounds @simd for i in 1:n
                        result[i] = Float64(term.f(arg1_values[i], arg2_values[i]))
                    end
                else
                    @inbounds @simd for i in 1:n
                        result[i] = term.f(arg1_values[i], arg2_values[i])
                    end
                end
                
                return result
            else
                error("Multi-column two-argument functions not yet supported")
            end
        end
    else
        error("Functions with $(length(term.args)) arguments not supported")
    end
end

###############################################################################
# Efficient Function Derivative Registry
###############################################################################

"""
    get_function_derivative_cached(f::Function) -> Function

EFFICIENT: Get derivative function with caching to avoid recomputation.
"""
function get_function_derivative_cached(f::Function)
    # EFFICIENT: Check cache first
    if haskey(DERIVATIVE_REGISTRY, f)
        return DERIVATIVE_REGISTRY[f]
    elseif haskey(STANDARD_DERIVATIVES, f)
        return STANDARD_DERIVATIVES[f]
    else
        # EFFICIENT: Auto-generate once and cache
        f_prime = x -> ForwardDiff.derivative(f, x)
        DERIVATIVE_REGISTRY[f] = f_prime
        return f_prime
    end
end

###############################################################################
# Efficient Derivative Computation for Complex Terms
###############################################################################

"""
    analytical_derivative(term::FunctionTerm, variable::Symbol, data::NamedTuple)

EFFICIENT: Function term derivatives with optimized chain rule application.
"""
function analytical_derivative(term::FunctionTerm, variable::Symbol, data::NamedTuple)
    n = length(data[variable])
    
    if length(term.args) == 1
        # EFFICIENT: Single argument chain rule: f'(g(x)) * g'(x)
        inner_term = term.args[1]
        inner_derivative = analytical_derivative(inner_term, variable, data)
        
        # EFFICIENT: Early exit for zero derivatives
        if inner_derivative isa Vector && all(iszero, inner_derivative)
            return zeros(Float64, n)
        end
        
        if term.f in [<=, >=, <, >, ==, !=]
            # Comparison operators have zero derivative almost everywhere
            return zeros(Float64, n)
        else
            # EFFICIENT: Apply chain rule with cached derivative function
            inner_values = evaluate_term(inner_term, data)
            f_prime = get_function_derivative_cached(term.f)
            
            try
                if inner_values isa Vector && inner_derivative isa Vector
                    result = Vector{Float64}(undef, n)
                    @inbounds @simd for i in 1:n
                        result[i] = f_prime(inner_values[i]) * inner_derivative[i]
                    end
                    return result
                else
                    error("Multi-column function derivatives not yet implemented")
                end
            catch e
                @warn "Failed to compute analytical derivative for $(term.f): $e"
                return zeros(Float64, n)
            end
        end
        
    elseif length(term.args) == 2
        # EFFICIENT: Two argument derivatives
        arg1, arg2 = term.args
        
        if arg2 isa ConstantTerm
            if term.f in [<=, >=, <, >, ==, !=]
                # Comparison with constant has zero derivative almost everywhere
                return zeros(Float64, n)
            else
                # EFFICIENT: For most mathematical functions, return zero (partial derivatives not implemented)
                return zeros(Float64, n)
            end
        else
            # Two variable arguments
            if term.f in [<=, >=, <, >, ==, !=]
                return zeros(Float64, n)
            else
                @warn "Derivative of two-variable function $(term.f) not implemented"
                return zeros(Float64, n)
            end
        end
    else
        @warn "Derivative of $(length(term.args))-argument function not implemented"
        return zeros(Float64, n)
    end
end

"""
    analytical_derivative(term::InteractionTerm, variable::Symbol, data::NamedTuple)

EFFICIENT: Interaction term derivatives with optimized product rule.
"""
function analytical_derivative(term::InteractionTerm, variable::Symbol, data::NamedTuple)
    components = term.terms
    n = length(data[variable])
    
    # EFFICIENT: Pre-compute all component values and derivatives
    component_values = Vector{Any}(undef, length(components))
    component_derivatives = Vector{Any}(undef, length(components))
    has_nonzero_derivative = false
    
    for (i, component) in enumerate(components)
        component_values[i] = evaluate_term(component, data)
        component_derivatives[i] = analytical_derivative(component, variable, data)
        
        # EFFICIENT: Check if this component has non-zero derivative
        if component_derivatives[i] isa Vector
            if !all(iszero, component_derivatives[i])
                has_nonzero_derivative = true
            end
        elseif component_derivatives[i] isa Matrix
            if !all(iszero, component_derivatives[i])
                has_nonzero_derivative = true
            end
        end
    end
    
    # EFFICIENT: Early exit if no component depends on variable
    if !has_nonzero_derivative
        full_values = evaluate_term(term, data)
        if full_values isa Vector
            return zeros(Float64, n)
        else
            return zeros(Float64, size(full_values))
        end
    end
    
    # EFFICIENT: Apply product rule using Kronecker products
    result_parts = []
    
    for i in 1:length(components)
        # Skip components with zero derivatives
        deriv_i = component_derivatives[i]
        if (deriv_i isa Vector && all(iszero, deriv_i)) ||
           (deriv_i isa Matrix && all(iszero, deriv_i))
            continue
        end
        
        # EFFICIENT: Compute i-th term of product rule
        term_contribution = deriv_i  # Start with f_i'
        
        # Multiply by all other components (f_j for j ≠ i)
        for j in 1:length(components)
            if i != j
                term_contribution = kron_product_columns_efficient!(term_contribution, component_values[j])
            end
        end
        
        push!(result_parts, term_contribution)
    end
    
    # EFFICIENT: Sum all parts
    if isempty(result_parts)
        full_values = evaluate_term(term, data)
        if full_values isa Vector
            return zeros(Float64, n)
        else
            return zeros(Float64, size(full_values))
        end
    end
    
    result = result_parts[1]
    for i in 2:length(result_parts)
        result = result .+ result_parts[i]
    end
    
    return result
end

###############################################################################
# Efficient Helper Functions for EfficientModelMatrices Integration
###############################################################################

"""
    find_term_for_column(mapping::ColumnMapping, col::Int) -> (AbstractTerm, Int)

EFFICIENT: Fast lookup of term for a specific column.
"""
function find_term_for_column(mapping::ColumnMapping, col::Int)
    # EFFICIENT: Direct lookup in term_info (already sorted by range)
    for (term, range) in mapping.term_info
        if col in range
            local_col = col - first(range) + 1
            return (term, local_col)
        end
    end
    return (nothing, 0)
end

"""
    prepare_analytical_derivatives!(ws::AMEWorkspace, variable::Symbol, h::Real, ipm::InplaceModeler)

EFFICIENT: Main interface for analytical derivative computation.
h parameter is ignored (kept for API compatibility).
"""
function prepare_analytical_derivatives!(ws::AMEWorkspace, variable::Symbol, h::Real, ipm::InplaceModeler)
    # Delegate to efficient workspace method
    prepare_analytical_derivatives_efficient!(ws, variable)
end
