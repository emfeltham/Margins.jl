# workspace.jl - CORRECTED for Batch 3 integration

"""
Selective update workspace that reuses memory for unchanged columns.
Key changes from old approach:
- Uses ColumnMapping to identify which columns each variable affects
- Pre-allocates perturbation vectors for all continuous variables
- Shares memory for unchanged columns during updates
- Single unified workspace for all AME types
"""

mutable struct AMEWorkspace
    # Base state (immutable references)
    base_data::NamedTuple                           # Original data (never modified)
    base_matrix::Matrix{Float64}                    # n × p design matrix for base data
    
    # Column mapping and update plans
    mapping::ColumnMapping                          # Maps variables to column ranges
    variable_plans::Dict{Symbol, Vector{Int}}       # var → affected column indices
    
    # Working state (selective updates)
    work_matrix::Matrix{Float64}                    # n × p working matrix (selective updates)
    finite_diff_matrix::Matrix{Float64}            # n × p matrix for finite differences
    
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

# Arguments
- `model`: Fitted statistical model 
- `data`: Data used to fit the model (DataFrame, NamedTuple, etc.)

# Returns
- `AMEWorkspace` with pre-allocated matrices and column mapping

# Details
This constructor:
1. Builds column mapping from model formula
2. Creates base model matrix
3. Pre-allocates perturbation vectors for all continuous variables
4. Sets up selective update infrastructure
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
    finite_diff_matrix = Matrix{Float64}(undef, n, p)
    
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
        work_matrix, finite_diff_matrix,
        pert_vectors, pert_cache,
        η, dη, μp_vals, μpp_vals, grad_work, temp1, temp2
    )
end

"""
    update_for_variable!(ws::AMEWorkspace, variable::Symbol, new_values::AbstractVector, 
                        ipm::InplaceModeler)

Update workspace matrices for a single perturbed variable using selective updates.
UPDATED: Now uses enhanced create_perturbed_data from selective_updates.jl

# Arguments
- `ws`: AMEWorkspace to update
- `variable`: Symbol of variable being perturbed
- `new_values`: New values for the variable
- `ipm`: InplaceModeler for matrix construction

# Details
This function:
1. Creates perturbed data using memory sharing (with proper categorical handling)
2. Updates only columns affected by the variable
3. Shares memory for all unchanged columns
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
    
    # UPDATED: Use enhanced function from selective_updates.jl 
    # (automatically handles categorical types properly)
    pert_data = create_perturbed_data(ws.base_data, variable, new_values)
    
    # Get affected and unaffected columns
    affected_cols = ws.variable_plans[variable]
    total_cols = size(ws.work_matrix, 2)
    unchanged_cols = get_unchanged_columns(ws.mapping, [variable], total_cols)
    
    # Update only affected columns in work matrix
    if !isempty(affected_cols)
        eval_columns_for_variable!(variable, pert_data, ws.work_matrix, ws.mapping, ipm)
    end
    
    # Share memory for unchanged columns
    share_unchanged_columns!(ws.work_matrix, ws.base_matrix, unchanged_cols)
end

"""
    update_for_variables!(ws::AMEWorkspace, changes::Dict{Symbol, <:AbstractVector}, 
                         imp::InplaceModeler)

Update workspace matrices for multiple variables simultaneously.
UPDATED: Now uses enhanced batch_perturb_data from selective_updates.jl

# Arguments
- `ws`: AMEWorkspace to update
- `changes`: Dictionary mapping variable names to new values
- `ipm`: InplaceModeler for matrix construction

# Details
More efficient than multiple calls to update_for_variable! when several variables change.
"""
function update_for_variables!(ws::AMEWorkspace, changes::Dict{Symbol, <:AbstractVector}, 
                              ipm::InplaceModeler)
    if isempty(changes)
        return
    end
    
    # UPDATED: Use enhanced function from selective_updates.jl
    # (automatically handles categorical types properly)
    pert_data = batch_perturb_data(ws.base_data, changes)
    
    # Get all affected columns
    changed_vars = collect(keys(changes))
    all_affected_cols = Set{Int}()
    
    for var in changed_vars
        if haskey(ws.variable_plans, var)
            union!(all_affected_cols, ws.variable_plans[var])
        end
    end
    
    affected_cols = sort(collect(all_affected_cols))
    total_cols = size(ws.work_matrix, 2)
    unchanged_cols = get_unchanged_columns(ws.mapping, changed_vars, total_cols)
    
    # Update affected columns
    if !isempty(affected_cols)
        eval_columns_for_variables!(changed_vars, pert_data, ws.work_matrix, ws.mapping, ipm)
    end
    
    # Share memory for unchanged columns  
    share_unchanged_columns!(ws.work_matrix, ws.base_matrix, unchanged_cols)
end

"""
    prepare_finite_differences!(ws::AMEWorkspace, variable::Symbol, h::Real, 
                               ipm::InplaceModeler)

Prepare finite difference matrix for AME computation of a continuous variable.

# Arguments
- `ws`: AMEWorkspace
- `variable`: Continuous variable for finite differences  
- `h`: Step size for finite differences
- `ipm`: InplaceModeler

# Details
Creates X_perturbed and computes (X_perturbed - X_current) / h for finite difference AME.
Uses selective updates - only columns affected by variable are computed.
Works correctly whether workspace is at base state or representative values.
"""
function prepare_finite_differences!(ws::AMEWorkspace, variable::Symbol, h::Real, 
                                    ipm::InplaceModeler)
    # Validate that variable is continuous and pre-allocated
    if !haskey(ws.pert_vectors, variable)
        throw(ArgumentError(
            "Variable $variable not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # Store current work matrix state (might be at repvals)
    current_matrix = copy(ws.work_matrix)
    
    # Create perturbed values: current variable values + h
    current_var_values = ws.base_data[variable]
    pert_vector = ws.pert_vectors[variable]
    
    # Fill perturbation vector
    @inbounds for i in eachindex(pert_vector)
        pert_vector[i] = current_var_values[i] + h
    end
    
    # Update work matrix with perturbed values  
    update_for_variable!(ws, variable, pert_vector, ipm)
    
    # Compute finite differences: (X_perturbed - X_current) / h
    # Only for columns affected by this variable
    affected_cols = ws.variable_plans[variable]
    invh = 1.0 / h
    
    @inbounds for col in affected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = (ws.work_matrix[row, col] - current_matrix[row, col]) * invh
    end
    
    # For unaffected columns, finite difference is zero
    total_cols = size(ws.finite_diff_matrix, 2)
    unaffected_cols = get_unchanged_columns(ws.mapping, [variable], total_cols)
    
    @inbounds for col in unaffected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = 0.0
    end
    
    # Restore current state
    ws.work_matrix .= current_matrix
end

"""
    reset_to_base!(ws::AMEWorkspace)

Reset work matrix to base matrix state (copy base_matrix to work_matrix).
"""
function reset_to_base!(ws::AMEWorkspace)
    ws.work_matrix .= ws.base_matrix
end

"""
    vcov(cholΣβ::Cholesky) -> Matrix

Convert Cholesky decomposition back to covariance matrix.
Helper function for compatibility with batch 3 functions.
"""
function vcov(cholΣβ::Cholesky)
    return Matrix(cholΣβ)
end

"""
    set_to_repvals!(ws::AMEWorkspace, repvals::Dict{Symbol, <:Any}, ipm::InplaceModeler)

Set workspace work matrix to representative values state.
Used when computing AMEs at representative values.

# Arguments
- `ws`: AMEWorkspace to update
- `repvals`: Dictionary mapping variables to representative values (scalars)
- `ipm`: InplaceModeler for matrix construction
"""
function set_to_repvals!(ws::AMEWorkspace, repvals::Dict{Symbol, <:Any}, ipm::InplaceModeler)
    if isempty(repvals)
        reset_to_base!(ws)
        return
    end
    
    n = length(first(ws.base_data))
    changes = Dict{Symbol, Vector{Float64}}()
    
    # Convert scalar repvals to vectors
    for (var, val) in repvals
        if val isa AbstractVector
            changes[var] = val
        else
            changes[var] = fill(Float64(val), n)
        end
    end
    
    update_for_variables!(ws, changes, ipm)
end

"""
    get_memory_info(ws::AMEWorkspace) -> NamedTuple

Get memory usage information for the workspace.
Useful for debugging and performance monitoring.
"""
function get_memory_info(ws::AMEWorkspace)
    matrices = Dict(
        "base_matrix" => ws.base_matrix,
        "work_matrix" => ws.work_matrix, 
        "finite_diff_matrix" => ws.finite_diff_matrix
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