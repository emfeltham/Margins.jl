# analytical_derivatives.jl - NEW FILE
# Add this as a new file in your src/ directory

# Global registry for function derivatives
const DERIVATIVE_REGISTRY = Dict{Function, Function}()

# Standard mathematical function derivatives
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
    evaluate_term(term::InteractionTerm, data::NamedTuple)

REPLACED: Fixed interaction term evaluation for 3-way interactions.
"""
function evaluate_term(term::InteractionTerm, data::NamedTuple)
    n = length(first(data))
    
    # Get all component values
    component_values = [evaluate_term(component, data) for component in term.terms]
    
    # Compute element-wise product
    result = ones(Float64, n)
    for comp_vals in component_values
        result .*= comp_vals
    end
    
    return result
end

"""
    evaluate_term(term::CategoricalTerm, data::NamedTuple)

REPLACED: Fixed categorical term evaluation with proper contrast coding.
"""
function evaluate_term(term::CategoricalTerm, data::NamedTuple)
    v = data[term.sym]
    M = term.contrasts.matrix
    n = length(v)
    
    # Handle both CategoricalArray and regular arrays
    if isa(v, CategoricalArray)
        codes = refs(v)
    else
        unique_vals = sort(unique(v))
        code_map = Dict(val => i for (i, val) in enumerate(unique_vals))
        codes = [code_map[val] for val in v]
    end
    
    # For single-column case (most common with dummy coding)
    if size(M, 2) == 1
        result = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            result[i] = M[codes[i], 1]
        end
        return result
    else
        # Multi-column case - return first column for now
        # (Multi-column analytical derivatives will be handled by fallback)
        result = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            result[i] = M[codes[i], 1]
        end
        return result
    end
end

function evaluate_term(term::FunctionTerm, data::NamedTuple)
    if length(term.args) == 1
        # Single argument function
        arg_values = evaluate_term(term.args[1], data)
        if arg_values isa Vector
            return term.f.(arg_values)
        else
            # Multi-column argument - not supported yet
            error("Multi-column function arguments not yet supported")
        end
    else
        # Multi-argument functions
        arg_values = [evaluate_term(arg, data) for arg in term.args]
        
        # Check if all arguments are vectors
        if all(arg -> arg isa Vector, arg_values)
            return [term.f(args...) for args in zip(arg_values...)]
        else
            error("Multi-column function arguments not yet supported")
        end
    end
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
"""
    analytical_derivative(term::InteractionTerm, variable::Symbol, data::NamedTuple)

REPLACED: Fixed 3-way interaction analytical derivative using proper product rule.
"""
function analytical_derivative(term::InteractionTerm, variable::Symbol, data::NamedTuple)
    components = term.terms
    n = length(data[variable])
    
    # Pre-compute all component values and derivatives
    component_values = Vector{Vector{Float64}}(undef, length(components))
    component_derivatives = Vector{Vector{Float64}}(undef, length(components))
    any_depends = false
    
    for (i, component) in enumerate(components)
        component_values[i] = evaluate_term(component, data)
        component_derivatives[i] = analytical_derivative(component, variable, data)
        
        if !all(x -> x == 0, component_derivatives[i])
            any_depends = true
        end
    end
    
    if !any_depends
        return zeros(Float64, n)
    end
    
    # Apply product rule: d/dx(f₁*f₂*...*fₙ) = Σᵢ(f₁*...*fᵢ₋₁*fᵢ'*fᵢ₊₁*...*fₙ)
    result = zeros(Float64, n)
    
    for i in 1:length(components)
        # Skip if this component doesn't depend on the variable
        if all(x -> x == 0, component_derivatives[i])
            continue
        end
        
        # Compute the i-th term of the product rule
        term_contribution = copy(component_derivatives[i])  # Start with fᵢ'
        
        # Multiply by all other components (fⱼ for j ≠ i)
        for j in 1:length(components)
            if i != j
                term_contribution .*= component_values[j]
            end
        end
        
        result .+= term_contribution
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
    
    # Group affected columns by their generating term
    terms_to_process = Dict{AbstractTerm, Vector{Int}}()
    for col in affected_cols
        term, local_col_in_term = find_term_for_column(ws.mapping, col)
        if term !== nothing
            if !haskey(terms_to_process, term)
                terms_to_process[term] = Int[]
            end
            push!(terms_to_process[term], col)
        end
    end
    
    # Process each term and its columns
    for (term, cols) in terms_to_process
        try
            if width(term) == 1
                # Single column term - compute derivative directly
                term_derivative = analytical_derivative(term, variable, ws.base_data)
                @assert length(cols) == 1 "Single-width term should affect exactly one column"
                ws.finite_diff_matrix[:, cols[1]] = term_derivative
                
            else
                # Multi-column term - need to compute derivative of each output column
                term_derivatives = compute_term_column_derivatives(term, variable, ws.base_data, ipm)
                
                for (i, col) in enumerate(cols)
                    if i <= length(term_derivatives)
                        ws.finite_diff_matrix[:, col] = term_derivatives[i]
                    else
                        @warn "Not enough derivatives computed for term $(typeof(term)), column $col"
                        ws.finite_diff_matrix[:, col] .= 0.0
                    end
                end
            end
            
        catch e
            @warn "Failed to compute analytical derivative for term $(typeof(term)): $e"
            for col in cols
                ws.finite_diff_matrix[:, col] .= 0.0
            end
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
    compute_term_column_derivatives(term::InteractionTerm{T}, variable, data, ipm) where T<:Tuple

Fully‐analytic, per‐column derivatives for an InteractionTerm.
Works for any # of components (continuous×continuous, continuous×factor, etc.).
"""
function compute_term_column_derivatives(
    term::InteractionTerm{T},
    variable::Symbol,
    data::NamedTuple,
    ipm::InplaceModeler
) where T<:Tuple

    n = length(data[variable])
    w = width(term)

    # 1) Build the full design‐matrix for this interaction (n×w)
    X_int = Matrix{Float64}(undef, n, w)
    evaluate_term_to_matrix!(term, data, X_int, ipm)

    # 2) Find the continuous subterm f(x) whose dériv we want
    cont_vals = nothing
    for comp in term.terms
        if variable in termvars(comp)
            # evaluate_term on a single‐column term → Vector{Float64}
            cont_vals = evaluate_term(comp, data)
            break
        end
    end
    @assert cont_vals !== nothing "No continuous component in $term to differentiate"

    # 3) For each column j, ∂(f(x)*other_j)/∂x = other_j
    derivs = Vector{Vector{Float64}}(undef, w)
    for j in 1:w
        col = view(X_int, :, j)

        # product of *all* components except the focal one
        other_vals = col ./ cont_vals            # good when |cont_vals|>0
        zero_rows  = cont_vals .== 0.0
        if any(zero_rows)
            other_vals[zero_rows] .= view(X_int, zero_rows, j)  # == product(other components)
        end
        derivs[j] = other_vals
    end

    return derivs
end

"""
    compute_term_column_derivatives(term::AbstractTerm, variable::Symbol, data::NamedTuple, ipm::InplaceModeler) -> Vector{Vector{Float64}}

For multi-column terms, compute the derivative of each output column.
Returns a vector of derivative vectors, one for each output column.
"""
function compute_term_column_derivatives(term::AbstractTerm, variable::Symbol,
                                        data::NamedTuple, ipm::InplaceModeler)
    # Directly dispatch to the analytic‐derivative routines, then split into columns
    M = analytical_derivative(term, variable, data)  
    # assume this returns either an n×w Matrix or Vector{Vector{Float64}}

    # how many output cols should this term have?
    w = width(term)
    if M isa AbstractMatrix
        @assert size(M,2) == w
        return [view(M, :, j) for j in 1:w]
    elseif M isa AbstractVector
        # single‐column derivative – replicate for each column
        # (e.g. categoricalTerm or constantTerm → zeros(n))
        return [copy(M) for _ in 1:w]
    else
        throw(ArgumentError(
            "analytical_derivative returned $(typeof(M)), expected Vector or Matrix"
        ))
    end
end

"""
    evaluate_term_to_matrix!(term::AbstractTerm, data::NamedTuple, matrix::Matrix{Float64}, ipm::InplaceModeler)

Evaluate a term and fill the provided matrix with its output.
"""
function evaluate_term_to_matrix!(term::AbstractTerm, data::NamedTuple, matrix::Matrix{Float64}, ipm::InplaceModeler)
    # Use the existing _cols! infrastructure to evaluate the term
    fn_i = Ref(1)
    int_i = Ref(1)
    
    try
        _cols!(term, data, matrix, 1, ipm, fn_i, int_i)
    catch e
        @warn "Failed to evaluate term $(typeof(term)): $e"
        fill!(matrix, 0.0)
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
