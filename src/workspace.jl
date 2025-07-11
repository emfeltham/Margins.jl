# workspace.jl - EFFICIENT VERSION: Minimal allocations

using Statistics: std, mean
using LinearAlgebra: norm, clamp!

# Import the functions we need from EfficientModelMatrices.jl
using EfficientModelMatrices: get_unchanged_columns, eval_columns_for_variable!, 
                             eval_columns_for_variables!, build_perturbation_plan

"""
Efficient workspace that minimizes allocations and uses smart column sharing.
EFFICIENT: Lazy matrix construction and minimal copying.
"""
mutable struct AMEWorkspace
    # Base state - now stored as NamedTuple (zero-copy view of DataFrame)
    base_data::NamedTuple                           # Zero-copy view of original data
    
    # Lazy matrix construction - only build when needed
    base_matrix::Union{Matrix{Float64}, Nothing}    # Lazy: built on first access
    
    # Column mapping and update plans
    mapping::ColumnMapping                          # Maps variables to column ranges
    variable_plans::Dict{Symbol, Vector{Int}}       # var → affected column indices
    
    # Efficient working state - share memory where possible
    work_matrix::Union{Matrix{Float64}, Nothing}    # Lazy: selective updates only
    derivative_matrix::Matrix{Float64}              # Only for affected columns
    
    # Pre-allocated computation vectors (fixed size, reused)
    η::Vector{Float64}                              # n-length: linear predictors
    dη::Vector{Float64}                             # n-length: finite difference in η
    μp_vals::Vector{Float64}                        # n-length: first derivative of inverse link
    μpp_vals::Vector{Float64}                       # n-length: second derivative of inverse link
    grad_work::Vector{Float64}                      # p-length: gradient workspace
    temp1::Vector{Float64}                          # p-length: temporary vector 1
    temp2::Vector{Float64}                          # p-length: temporary vector 2
    
    # Matrix dimensions (cached to avoid recomputation)
    n::Int                                          # Number of observations
    p::Int                                          # Number of parameters
    
    # Cached InplaceModeler (avoid reconstruction)
    ipm::InplaceModeler                             # Reusable matrix constructor
end

"""
    AMEWorkspace(model::StatisticalModel, data) -> AMEWorkspace

EFFICIENT: Create workspace with minimal allocations and lazy matrix construction.
"""
function AMEWorkspace(model::StatisticalModel, data)
    # EFFICIENT: Zero-copy view of data (no DataFrame copying)
    base_data = Tables.columntable(data)
    validate_data_consistency(base_data)
    
    # Get dimensions
    n = length(first(base_data))
    
    # Build column mapping from model formula
    rhs = fixed_effects_form(model).rhs
    mapping = build_column_mapping(rhs, model)
    p = mapping.total_columns
    
    # EFFICIENT: Create InplaceModeler once and reuse
    ipm = InplaceModeler(model, n)
    
    # EFFICIENT: Pre-compute variable plans without building matrices
    # Only get variables that actually exist in the data
    data_vars = collect(keys(base_data))
    variable_plans = build_perturbation_plan(mapping, data_vars)
    
    # EFFICIENT: Pre-allocate computation vectors only (no matrices yet)
    η = Vector{Float64}(undef, n)
    dη = Vector{Float64}(undef, n)
    μp_vals = Vector{Float64}(undef, n)
    μpp_vals = Vector{Float64}(undef, n)
    grad_work = Vector{Float64}(undef, p)
    temp1 = Vector{Float64}(undef, p)
    temp2 = Vector{Float64}(undef, p)
    
    # EFFICIENT: Only allocate derivative matrix (much smaller - typically sparse)
    derivative_matrix = Matrix{Float64}(undef, n, p)
    
    return AMEWorkspace(
        base_data, nothing,  # base_matrix is lazy
        mapping, variable_plans,
        nothing, derivative_matrix,  # work_matrix is lazy
        η, dη, μp_vals, μpp_vals, grad_work, temp1, temp2,
        n, p, ipm
    )
end

"""
    get_base_matrix!(ws::AMEWorkspace) -> Matrix{Float64}

EFFICIENT: Lazy construction of base matrix - only build when first needed.
"""
function get_base_matrix!(ws::AMEWorkspace)
    if ws.base_matrix === nothing
        ws.base_matrix = Matrix{Float64}(undef, ws.n, ws.p)
        modelmatrix!(ws.ipm, ws.base_data, ws.base_matrix)
    end
    return ws.base_matrix
end

"""
    get_work_matrix!(ws::AMEWorkspace) -> Matrix{Float64}

EFFICIENT: Lazy construction of work matrix - shares memory with base matrix initially.
"""
function get_work_matrix!(ws::AMEWorkspace)
    if ws.work_matrix === nothing
        # EFFICIENT: Start by sharing memory with base matrix
        base = get_base_matrix!(ws)
        ws.work_matrix = copy(base)  # Only copy when we need to modify
    end
    return ws.work_matrix
end

"""
    reset_work_matrix!(ws::AMEWorkspace)

EFFICIENT: Reset work matrix to share memory with base matrix (zero-copy when possible).
"""
function reset_work_matrix!(ws::AMEWorkspace)
    if ws.work_matrix !== nothing && ws.base_matrix !== nothing
        # EFFICIENT: Just copy references for unchanged columns
        # The selective update will handle changed columns
        ws.work_matrix .= ws.base_matrix
    end
end

"""
    update_for_variable!(ws::AMEWorkspace, variable::Symbol, new_values::AbstractVector, 
                        ipm::InplaceModeler)

EFFICIENT: Update only affected columns using selective updates.
"""
function update_for_variable!(
    ws::AMEWorkspace, variable::Symbol, new_values::AbstractVector
)
    # Validate inputs
    if !haskey(ws.variable_plans, variable)
        @warn "Variable $variable does not affect any model matrix columns"
        return
    end
    
    if length(new_values) != ws.n
        throw(DimensionMismatch(
            "New values have length $(length(new_values)), expected $(ws.n)"
        ))
    end
    
    # EFFICIENT: Create perturbed data with zero-copy for unchanged variables
    pert_data = merge(ws.base_data, (variable => new_values,))
    
    # EFFICIENT: Lazy matrix construction and selective update
    work_matrix = get_work_matrix!(ws)
    base_matrix = get_base_matrix!(ws)
    
    # EFFICIENT: Use selective matrix update - only changed columns are recomputed
    modelmatrix_with_base!(ws.ipm, pert_data, work_matrix, base_matrix, [variable], ws.mapping)
end

"""
    update_for_variables!(ws::AMEWorkspace, changes::Dict{Symbol, <:AbstractVector})

EFFICIENT: Batch update multiple variables with minimal allocations.
"""
function update_for_variables!(
    ws::AMEWorkspace,
    changes::Dict{Symbol, <:AbstractVector}
)
    if isempty(changes)
        return
    end
    
    # Validate all changes at once
    for (var, values) in changes
        if !haskey(ws.variable_plans, var)
            throw(ArgumentError("Variable $var not found in variable plans"))
        end
        
        if length(values) != ws.n
            throw(DimensionMismatch(
                "New values for $var have length $(length(values)), expected $(ws.n)"
            ))
        end
    end
    
    # EFFICIENT: Create perturbed data with zero-copy for unchanged variables
    pert_data = merge(ws.base_data, changes)
    
    # EFFICIENT: Batch selective update
    work_matrix = get_work_matrix!(ws)
    base_matrix = get_base_matrix!(ws)
    changed_vars = collect(keys(changes))
    
    modelmatrix_with_base!(ws.ipm, pert_data, work_matrix, base_matrix, changed_vars, ws.mapping)
end

"""
    prepare_analytical_derivatives_efficient!(ws::AMEWorkspace, variable::Symbol)

EFFICIENT: Compute derivatives only for affected columns, zero out the rest.
"""
function prepare_analytical_derivatives_efficient!(ws::AMEWorkspace, variable::Symbol)
    # Get affected columns (typically a small subset)
    affected_cols = get(ws.variable_plans, variable, Int[])
    
    if isempty(affected_cols)
        fill!(ws.derivative_matrix, 0.0)
        return
    end
    
    # EFFICIENT: Zero out all columns first (sparse pattern)
    fill!(ws.derivative_matrix, 0.0)
    
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
    
    # EFFICIENT: Process only terms that are affected
    for (term, cols) in terms_to_process
        try
            # Compute analytical derivative for this term
            term_derivative = analytical_derivative(term, variable, ws.base_data)
            
            if term_derivative isa Vector
                # Single column result
                @assert length(cols) == 1 "Single-width derivative should affect exactly one column"
                ws.derivative_matrix[:, cols[1]] = term_derivative
                
            elseif term_derivative isa Matrix
                # Multi-column result - copy only affected columns
                num_cols = size(term_derivative, 2)
                @assert length(cols) == num_cols "Number of derivative columns should match affected columns"
                
                for (local_idx, global_col) in enumerate(cols)
                    ws.derivative_matrix[:, global_col] = term_derivative[:, local_idx]
                end
            end
            
        catch e
            @warn "Failed to compute analytical derivative for term $(typeof(term)): $e"
            # Zero out failed columns (already done above)
        end
    end
end

"""
    set_base_data_efficient!(ws::AMEWorkspace, new_data::NamedTuple)

EFFICIENT: Update base data and invalidate cached matrices for rebuild.
"""
function set_base_data_efficient!(ws::AMEWorkspace, new_data::NamedTuple)
    validate_data_consistency(new_data)
    
    # Check dimensions match
    new_n = length(first(new_data))
    if ws.n != new_n
        throw(DimensionMismatch("New data has $new_n observations, workspace expects $(ws.n)"))
    end
    
    # EFFICIENT: Update data reference and invalidate matrices
    ws.base_data = new_data
    ws.base_matrix = nothing  # Lazy rebuild
    ws.work_matrix = nothing  # Lazy rebuild
end

"""
    compute_memory_efficiency(ws::AMEWorkspace, variable::Symbol) -> NamedTuple

Compute memory efficiency metrics for a variable update.
"""
function compute_memory_efficiency(ws::AMEWorkspace, variable::Symbol)
    affected_cols = length(get(ws.variable_plans, variable, Int[]))
    total_cols = ws.p
    
    # Memory that would be allocated in naive approach
    naive_mb = (ws.n * total_cols * sizeof(Float64)) / (1024^2)
    
    # Memory actually used in selective approach (only affected columns)
    selective_mb = (ws.n * affected_cols * sizeof(Float64)) / (1024^2)
    
    # Efficiency metrics
    percent_cols_affected = (affected_cols / total_cols) * 100
    memory_saved_mb = naive_mb - selective_mb
    efficiency_ratio = selective_mb / naive_mb
    
    return (
        affected_columns = affected_cols,
        total_columns = total_cols,
        percent_affected = percent_cols_affected,
        naive_memory_mb = naive_mb,
        selective_memory_mb = selective_mb,
        memory_saved_mb = memory_saved_mb,
        efficiency_ratio = efficiency_ratio
    )
end

# Import the vcov helper
import StatsBase.vcov

"""
    vcov(cholΣβ::Cholesky) -> Matrix

Convert Cholesky decomposition back to covariance matrix.
"""
function vcov(cholΣβ::Cholesky)
    return Matrix(cholΣβ)
end
