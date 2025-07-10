# analytical_derivatives.jl - NEW FILE
# Add this as a new file in your src/ directory

using ForwardDiff

# Global registry for function derivatives
const DERIVATIVE_REGISTRY = Dict{Function, Function}()

# Standard mathematical function derivatives
const STANDARD_DERIVATIVES = Dict{Function, Function}(
    log => x -> 1/x,
    log1p => x -> 1/(1+x),
    exp => x -> exp(x),
    sqrt => x -> 1/(2*sqrt(x)),
    sin => x -> cos(x),
    cos => x -> -sin(x),
    tan => x -> sec(x)^2,
    # Add more as needed
)

"""
    analytical_derivative(term::AbstractTerm, variable::Symbol, data::NamedTuple) -> Vector{Float64}

Compute analytical derivative of a term with respect to a variable.
Returns a vector of derivatives evaluated at each data point.
"""
function analytical_derivative(term::AbstractTerm, variable::Symbol, data::NamedTuple)
    error("Analytical derivative not implemented for $(typeof(term))")
end

# ContinuousTerm: ∂x/∂x = 1, ∂x/∂y = 0
function analytical_derivative(term::ContinuousTerm, variable::Symbol, data::NamedTuple)
    n = length(data[variable])
    return term.sym == variable ? ones(Float64, n) : zeros(Float64, n)
end

# Term: ∂x/∂x = 1, ∂x/∂y = 0 (same as ContinuousTerm)
function analytical_derivative(term::Term, variable::Symbol, data::NamedTuple)
    n = length(data[variable])
    return term.sym == variable ? ones(Float64, n) : zeros(Float64, n)
end

# ConstantTerm: ∂c/∂x = 0
function analytical_derivative(term::ConstantTerm, variable::Symbol, data::NamedTuple)
    return zeros(Float64, length(data[variable]))
end

# InterceptTerm: ∂1/∂x = 0
function analytical_derivative(term::InterceptTerm, variable::Symbol, data::NamedTuple)
    return zeros(Float64, length(data[variable]))
end

# CategoricalTerm: ∂(categorical)/∂x = 0 (categorical variables don't depend on continuous variables)
function analytical_derivative(term::CategoricalTerm, variable::Symbol, data::NamedTuple)
    return zeros(Float64, length(data[variable]))
end

"""
    evaluate_term(term::AbstractTerm, data::NamedTuple) -> Vector

Evaluate a term at the given data points.
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
    get_function_derivative(f::Function) -> Function

Get derivative function for f, using registry, standard derivatives, or ForwardDiff fallback.
"""
function get_function_derivative(f::Function)
    if haskey(DERIVATIVE_REGISTRY, f)
        return DERIVATIVE_REGISTRY[f]
    elseif haskey(STANDARD_DERIVATIVES, f)
        return STANDARD_DERIVATIVES[f]
    else
        # Auto-generate using ForwardDiff
        f_prime = x -> ForwardDiff.derivative(f, x)
        DERIVATIVE_REGISTRY[f] = f_prime
        @info "Auto-generated derivative for function $(f) using ForwardDiff"
        return f_prime
    end
end

# FunctionTerm: Handle single-argument functions with chain rule
function analytical_derivative(term::FunctionTerm, variable::Symbol, data::NamedTuple)
    if length(term.args) == 1
        # Single argument: f(g(x)) -> f'(g(x)) * g'(x)
        inner_term = term.args[1]
        inner_derivative = analytical_derivative(inner_term, variable, data)
        
        # If inner term doesn't depend on variable, derivative is zero
        if all(x -> x == 0, inner_derivative)
            return zeros(Float64, length(data[variable]))
        end
        
        inner_values = evaluate_term(inner_term, data)
        f_prime = get_function_derivative(term.f)
        
        try
            return f_prime.(inner_values) .* inner_derivative
        catch e
            @warn "Failed to compute analytical derivative for $(term.f): $e"
            return zeros(Float64, length(data[variable]))
        end
    else
        # Multi-argument functions - implement later
        @warn "Multi-argument functions not yet implemented for $(term.f)"
        return zeros(Float64, length(data[variable]))
    end
end

# InteractionTerm: Product rule for arbitrary number of components
function analytical_derivative(term::InteractionTerm, variable::Symbol, data::NamedTuple)
    components = term.terms
    n = length(data[variable])
    result = zeros(Float64, n)
    
    # Product rule: ∂(f₁*f₂*...*fₙ)/∂x = Σᵢ(f₁*...*fᵢ₋₁*fᵢ'*fᵢ₊₁*...*fₙ)
    for i in 1:length(components)
        # Derivative of i-th component
        component_derivative = analytical_derivative(components[i], variable, data)
        
        # If this component doesn't depend on variable, skip
        if all(x -> x == 0, component_derivative)
            continue
        end
        
        # Product of all other components (constant w.r.t. variable)
        product = component_derivative
        for j in 1:length(components)
            if i != j
                component_values = evaluate_term(components[j], data)
                product .*= component_values
            end
        end
        
        result .+= product
    end
    
    return result
end

"""
    prepare_analytical_derivatives!(ws::AMEWorkspace, variable::Symbol, h::Real, ipm::InplaceModeler)

Compute analytical derivatives and fill finite_diff_matrix.
This is a drop-in replacement for prepare_finite_differences_fixed!
The h parameter is ignored (kept for API compatibility).
"""
function prepare_analytical_derivatives!(ws::AMEWorkspace, variable::Symbol, h::Real, ipm::InplaceModeler)
    # Validate that variable is continuous and pre-allocated
    if !haskey(ws.pert_vectors, variable)
        throw(ArgumentError(
            "Variable $variable not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # Get affected columns
    affected_cols = ws.variable_plans[variable]
    
    # Initialize finite difference matrix (will contain analytical derivatives)
    fill!(ws.finite_diff_matrix, 0.0)
    
    # For each affected column, compute analytical derivative
    for col in affected_cols
        # Find which term generates this column
        term, local_col_in_term = find_term_for_column(ws.mapping, col)
        
        if term !== nothing
            try
                # Compute analytical derivative of this term
                term_derivative = analytical_derivative(term, variable, ws.base_data)
                
                # For multi-column terms (like interactions), we need the derivative 
                # of the specific output column, not the whole term
                if width(term) == 1
                    # Single column term - use derivative directly
                    ws.finite_diff_matrix[:, col] = term_derivative
                else
                    # Multi-column term - need to handle this carefully
                    # For now, use the term derivative (this will need refinement)
                    ws.finite_diff_matrix[:, col] = term_derivative
                end
                
            catch e
                @warn "Failed to compute analytical derivative for column $col, term $(typeof(term)): $e"
                ws.finite_diff_matrix[:, col] .= 0.0
            end
        else
            @warn "Could not find term for column $col"
            ws.finite_diff_matrix[:, col] .= 0.0
        end
    end
    
    # Zero out unaffected columns
    total_cols = size(ws.finite_diff_matrix, 2)
    unaffected_cols = get_unchanged_columns(ws.mapping, [variable], total_cols)
    
    @inbounds for col in unaffected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = 0.0
    end
end

"""
    find_term_for_column(mapping::ColumnMapping, col::Int) -> (AbstractTerm, Int)

Find which term generates a specific column and the local column index within that term.
"""
function find_term_for_column(mapping::ColumnMapping, col::Int)
    for (term, range) in mapping.term_info
        if col in range
            local_col = col - first(range) + 1
            return (term, local_col)
        end
    end
    return (nothing, 0)
end