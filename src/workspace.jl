# workspace.jl - FIXED VERSION addressing test failures

using Statistics: std, mean
using LinearAlgebra: norm, clamp!

# Import the functions we need from EfficientModelMatrices.jl
using EfficientModelMatrices: get_unchanged_columns, eval_columns_for_variable!, 
                             eval_columns_for_variables!, build_perturbation_plan

"""
Selective update workspace that reuses memory for unchanged columns.
FIXED: Better state management and initialization.
"""

mutable struct AMEWorkspace
    # Base state (now mutable to support representative values)
    base_data::NamedTuple                           # Current data state (may change for repvals)
    base_matrix::Matrix{Float64}                    # n × p design matrix for base data
    
    # Column mapping and update plans
    mapping::ColumnMapping                          # Maps variables to column ranges
    variable_plans::Dict{Symbol, Vector{Int}}       # var → affected column indices
    
    # Working state (selective updates)
    work_matrix::Matrix{Float64}                    # n × p working matrix (selective updates)
    derivative_matrix::Matrix{Float64}            # n × p matrix for finite differences
    
    # Perturbation data management
    pert_vectors::Dict{Symbol, Vector{Float64}}     # Pre-allocated vectors for each continuous var
    pert_cache::Dict{Symbol, NamedTuple}           # Cached perturbed NamedTuples
    
    # Standard computation vectors
    η::Vector{Float64}                              # n-length: linear predictors
    dη::Vector{Float64}                             # n-length: finite difference in η
    μp_vals::Vector{Float64}                        # n-length: first derivative of inverse link
    μpp_vals::Vector{Float64}                       # n-length: second derivative of inverse link
    grad_work::Vector{Float64}                      # p-length: gradient workspace
    temp1::Vector{Float64}                          # p-length: temporary vector 1
    temp2::Vector{Float64}                          # p-length: temporary vector 2
end

"""
    AMEWorkspace(model::StatisticalModel, data) -> AMEWorkspace

Create selective update workspace for efficient AME computation.
FIXED: Better initialization and validation.
"""
function AMEWorkspace(model::StatisticalModel, data)
    # Convert data to NamedTuple (column table format)
    base_data = Tables.columntable(data)
    validate_data_consistency(base_data)
    
    # Get dimensions
    n = length(first(base_data))
    
    # Build column mapping from model formula
    rhs = fixed_effects_form(model).rhs
    mapping = build_column_mapping(rhs, model)
    p = mapping.total_columns
    
    # Create InplaceModeler for matrix construction
    ipm = InplaceModeler(model, n)
    
    # Build base model matrix
    base_matrix = Matrix{Float64}(undef, n, p)
    modelmatrix!(ipm, base_data, base_matrix)
    
    # Pre-compute variable plans (which columns each variable affects)
    all_vars = Set{Symbol}()
    for (name, _) in pairs(base_data)
        push!(all_vars, name)
    end
    # Use the function from EfficientModelMatrices.jl
    variable_plans = build_perturbation_plan(mapping, collect(all_vars))
    
    # Pre-allocate perturbation vectors for continuous variables
    pert_vectors = Dict{Symbol, Vector{Float64}}()
    pert_cache = Dict{Symbol, NamedTuple}()
    
    for (name, col) in pairs(base_data)
        # Check if variable is continuous (Real but not Bool)
        if eltype(col) <: Real && eltype(col) != Bool
            pert_vectors[name] = Vector{Float64}(undef, n)
            # Pre-create cached NamedTuple structure for this variable
            pert_cache[name] = merge(base_data, (name => pert_vectors[name],))
        end
    end
    
    # Allocate working matrices
    work_matrix = Matrix{Float64}(undef, n, p)
    derivative_matrix = Matrix{Float64}(undef, n, p)
    
    # Initialize work_matrix to base_matrix
    work_matrix .= base_matrix
    
    # Allocate computation vectors
    η = Vector{Float64}(undef, n)
    dη = Vector{Float64}(undef, n)
    μp_vals = Vector{Float64}(undef, n)
    μpp_vals = Vector{Float64}(undef, n)
    grad_work = Vector{Float64}(undef, p)
    temp1 = Vector{Float64}(undef, p)
    temp2 = Vector{Float64}(undef, p)
    
    return AMEWorkspace(
        base_data, base_matrix,
        mapping, variable_plans,
        work_matrix, derivative_matrix,
        pert_vectors, pert_cache,
        η, dη, μp_vals, μpp_vals, grad_work, temp1, temp2
    )
end

"""
    update_for_variable!(ws::AMEWorkspace, variable::Symbol, new_values::AbstractVector, 
                        imp::InplaceModeler)

Update workspace matrices for a single perturbed variable using efficient selective updates.
"""
function update_for_variable!(ws::AMEWorkspace, variable::Symbol, new_values::AbstractVector, 
                             ipm::InplaceModeler)
    # Validate inputs
    if !haskey(ws.variable_plans, variable)
        @warn "Variable $variable does not affect any model matrix columns"
        return
    end
    
    if length(new_values) != length(first(ws.base_data))
        throw(DimensionMismatch(
            "New values have length $(length(new_values)), " *
            "expected $(length(first(ws.base_data)))"
        ))
    end
    
    # Create perturbed data structure
    pert_data = create_perturbed_data(ws.base_data, variable, new_values)
    
    # EFFICIENT: Use selective matrix update with base sharing
    modelmatrix_with_base!(ipm, pert_data, ws.work_matrix, ws.base_matrix, [variable], ws.mapping)
end

"""
    update_for_variables!(ws::AMEWorkspace, changes::Dict{Symbol, <:AbstractVector}, 
                         ipm::InplaceModeler)

Update workspace matrices for multiple variables simultaneously using efficient selective updates.
"""
function update_for_variables!(ws::AMEWorkspace, changes::Dict{Symbol, <:AbstractVector}, 
                              ipm::InplaceModeler)
    if isempty(changes)
        return
    end
    
    # Create perturbed data structure
    pert_data = batch_perturb_data(ws.base_data, changes)
    
    # Get changed variables
    changed_vars = collect(keys(changes))
    
    # EFFICIENT: Use selective matrix update with base sharing
    modelmatrix_with_base!(ipm, pert_data, ws.work_matrix, ws.base_matrix, changed_vars, ws.mapping)
end

"""
    reset_to_base!(ws::AMEWorkspace)

Reset work matrix to base matrix state (copy base_matrix to work_matrix).
"""
function reset_to_base!(ws::AMEWorkspace)
    ws.work_matrix .= ws.base_matrix
end

"""
    rebuild_base_matrix!(ws::AMEWorkspace, ipm::InplaceModeler)

Rebuild base matrix from current base_data.
FIXED: New function to support representative values computation.
"""
function rebuild_base_matrix!(ws::AMEWorkspace, ipm::InplaceModeler)
    modelmatrix!(ipm, ws.base_data, ws.base_matrix)
    ws.work_matrix .= ws.base_matrix
end

"""
    set_base_data!(ws::AMEWorkspace, new_data::NamedTuple, ipm::InplaceModeler)

Set new base data and rebuild matrices.
FIXED: New function to support representative values computation.
"""
function set_base_data!(ws::AMEWorkspace, new_data::NamedTuple, ipm::InplaceModeler)
    validate_data_consistency(new_data)
    
    # Check that dimensions match
    old_n = length(first(ws.base_data))
    new_n = length(first(new_data))
    
    if old_n != new_n
        throw(DimensionMismatch(
            "New data has $new_n observations, workspace expects $old_n"
        ))
    end
    
    # Update base data and rebuild matrices
    ws.base_data = new_data
    rebuild_base_matrix!(ws, ipm)
end

import StatsBase.vcov

"""
    vcov(cholΣβ::Cholesky) -> Matrix

Convert Cholesky decomposition back to covariance matrix.
"""
function vcov(cholΣβ::Cholesky)
    return Matrix(cholΣβ)
end

"""
    get_memory_info(ws::AMEWorkspace) -> NamedTuple

Get memory usage information for the workspace.
"""
function get_memory_info(ws::AMEWorkspace)
    matrices = Dict(
        "base_matrix" => ws.base_matrix,
        "work_matrix" => ws.work_matrix, 
        "derivative_matrix" => ws.derivative_matrix
    )
    
    vectors_size = sum(sizeof(v) for v in values(ws.pert_vectors)) + 
                   sizeof(ws.η) + sizeof(ws.dη) + sizeof(ws.μp_vals) + 
                   sizeof(ws.μpp_vals) + sizeof(ws.grad_work) + 
                   sizeof(ws.temp1) + sizeof(ws.temp2)
    
    total_matrix_size = sum(sizeof(m) for m in values(matrices))
    total_size = total_matrix_size + vectors_size
    
    return (
        total_mb = total_size / (1024^2),
        matrices_mb = total_matrix_size / (1024^2),
        vectors_mb = vectors_size / (1024^2),
        matrix_dims = size(ws.base_matrix),
        n_variables = length(ws.variable_plans),
        n_continuous = length(ws.pert_vectors)
    )
end
