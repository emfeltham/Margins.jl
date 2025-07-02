# build_continuous_design.jl - OPTIMIZED NUMERICAL-ONLY VERSION
# Focus on optimizing the proven numerical approach using modelmatrix! 

###############################################################################
# Strategy: Minimize modelmatrix! calls and optimize the ones we must make
###############################################################################

"""
Optimized numerical differentiation that:
1. Uses pre-allocated buffers extensively  
2. Minimizes data copying and DataFrame operations
3. Leverages your modelmatrix! for in-place operations
4. Processes variables in batches when possible
"""
function build_continuous_design(df, fe_rhs, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    if k == 0
        return Matrix{Float64}(undef, n, 0), [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end

    # Build base design matrix once
    tbl0 = Tables.columntable(df)
    X_base = modelmatrix(fe_rhs, tbl0)
    p = size(X_base, 2)
    
    # Pre-allocate ALL working memory up front
    workspace = create_numerical_workspace(n, p, k)
    
    # Identify affected columns for all variables in one pass
    all_affected_cols = find_all_affected_columns_batch_optimized(
        fe_rhs, cts_vars, X_base, df, workspace
    )
    
    # Pre-allocate derivative matrices
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Process variables with minimal allocations
    for (j, var) in enumerate(cts_vars)
        compute_numerical_derivatives_optimized!(
            Xdx[j], X_base, fe_rhs, df, var, 
            all_affected_cols[j], workspace
        )
    end

    return X_base, Xdx
end

"""
Pre-allocated workspace to eliminate allocations during derivatives.
Using mutable struct so we can update the DataFrame reference.
"""
mutable struct NumericalWorkspace
    # DataFrame operations - use efficient typed storage
    work_df::DataFrame                  # Working DataFrame with proper types
    
    # Matrix operations  
    X_buffer::Matrix{Float64}           # For modelmatrix! output
    
    # Variable perturbation
    original_values::Vector{Float64}    # Store original variable values
    
    # Column detection
    sample_df::DataFrame                # Small sample for column detection
    sample_X::Matrix{Float64}           # Small matrix for column detection
    sample_X_pert::Matrix{Float64}      # Perturbed small matrix
end

function create_numerical_workspace(n::Int, p::Int, k::Int)
    # Create a small sample for efficient column detection
    sample_size = min(50, n)
    
    return NumericalWorkspace(
        DataFrame(),  # Will be initialized with proper types
        Matrix{Float64}(undef, n, p),
        Vector{Float64}(undef, n),
        DataFrame(),  # Will be initialized later
        Matrix{Float64}(undef, sample_size, p),
        Matrix{Float64}(undef, sample_size, p)
    )
end

"""
Optimized batch column detection using pre-allocated workspace.
"""
function find_all_affected_columns_batch_optimized(
    fe_rhs, cts_vars::Vector{Symbol}, X_base::Matrix, df::DataFrame, workspace::NumericalWorkspace
)
    n, p = size(X_base)
    all_affected = Vector{Vector{Int}}(undef, length(cts_vars))
    
    # Use small sample for column detection (major speedup)
    sample_size = min(50, n)
    # Create proper sample indices - ensure we get exactly sample_size elements
    step_size = max(1, div(n, sample_size))
    sample_idx = collect(1:step_size:n)[1:min(sample_size, div(n, step_size) + 1)]
    actual_sample_size = length(sample_idx)
    
    # Use views for the sample data (no copying, exact dimensions)
    sample_X_view = view(X_base, sample_idx, :)
    workspace.sample_X[1:actual_sample_size, :] .= sample_X_view
    workspace.sample_df = df[sample_idx, :]
    
    for (j, var) in enumerate(cts_vars)
        all_affected[j] = find_affected_columns_optimized(
            fe_rhs, var, workspace, actual_sample_size, 1e-10
        )
    end
    
    return all_affected
end

"""
Optimized affected column detection using workspace.
"""
function find_affected_columns_optimized(
    fe_rhs, var::Symbol, workspace::NumericalWorkspace, sample_size::Int, tolerance::Float64
)
    # Create working copy of sample DataFrame  
    work_df = DataFrame(workspace.sample_df, copycols=true)
    
    # Get original values for the focal variable
    original_col = work_df[!, var]
    n_sample = length(original_col)
    
    # Ensure sample_size matches what we actually have
    @assert n_sample == sample_size "Sample size mismatch: expected $sample_size, got $n_sample"
    
    # Convert focal variable to Float64 and store original
    original_float = Float64.(original_col)
    workspace.original_values[1:n_sample] .= original_float
    
    # Create perturbation
    h = 1e-4 * max(1.0, maximum(abs, original_float))
    
    # Perturb the focal variable (this is safe because we converted to Float64)
    work_df[!, var] = original_float .+ h
    
    # Use modelmatrix! to build perturbed matrix
    work_tbl = Tables.columntable(work_df)
    modelmatrix!(view(workspace.sample_X_pert, 1:n_sample, :), fe_rhs, work_tbl)
    
    # Find affected columns efficiently
    affected_cols = Int[]
    p = size(workspace.sample_X, 2)
    @inbounds for col in 1:p
        max_change = 0.0
        @simd for i in 1:n_sample
            change = abs(workspace.sample_X_pert[i, col] - workspace.sample_X[i, col])
            max_change = max(max_change, change)
        end
        
        if max_change > tolerance
            push!(affected_cols, col)
        end
    end
    
    return affected_cols
end

"""
Optimized numerical derivative computation using workspace.
"""
function compute_numerical_derivatives_optimized!(
    Xdx::Matrix{Float64},
    X_base::Matrix{Float64}, 
    fe_rhs,
    df::DataFrame,
    var::Symbol,
    affected_cols::Vector{Int},
    workspace::NumericalWorkspace
)
    n, p = size(X_base)
    
    # Initialize to zero
    fill!(Xdx, 0.0)
    
    if isempty(affected_cols)
        return nothing
    end
    
    # Create working DataFrame with proper types
    workspace.work_df = DataFrame(df, copycols=true)
    
    # Get original values for the focal variable
    original_col = workspace.work_df[!, var]
    original_float = Float64.(original_col)
    
    # Store original values
    resize!(workspace.original_values, n)
    workspace.original_values .= original_float
    
    # Compute step size
    h = sqrt(eps(Float64)) * max(1.0, maximum(abs, original_float))
    
    # Perturb the focal variable (safe because we converted to Float64)
    workspace.work_df[!, var] = original_float .+ h
    
    # Use modelmatrix! for in-place computation
    work_tbl = Tables.columntable(workspace.work_df)
    modelmatrix!(workspace.X_buffer, fe_rhs, work_tbl)
    
    # Compute derivatives only for affected columns
    inv_h = 1.0 / h
    @inbounds for col in affected_cols
        @simd for i in 1:n
            Xdx[i, col] = (workspace.X_buffer[i, col] - X_base[i, col]) * inv_h
        end
    end
    
    return nothing
end

###############################################################################
# Alternative: Batch processing approach
###############################################################################

"""
Process multiple variables simultaneously to amortize modelmatrix! costs.
This works when variables don't interact with each other.
"""
function build_continuous_design_batched(df, fe_rhs, cts_vars::Vector{Symbol})
    n, k = length(df), length(cts_vars)
    if k == 0
        return Matrix{Float64}(undef, n, 0), [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end

    # Build base design matrix
    tbl0 = Tables.columntable(df)
    X_base = modelmatrix(fe_rhs, tbl0)
    p = size(X_base, 2)
    
    # Group variables that can be processed together
    # (This is conservative - only batch variables that definitely don't interact)
    var_batches = create_safe_variable_batches(cts_vars, fe_rhs)
    
    # Pre-allocate all derivative matrices
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Process each batch
    processed_vars = Set{Symbol}()
    for batch in var_batches
        if length(batch) == 1
            # Single variable - use regular approach
            var = batch[1]
            j = findfirst(==(var), cts_vars)
            compute_single_variable_derivatives!(Xdx[j], X_base, fe_rhs, df, var)
        else
            # Multiple variables - batch process
            compute_batch_derivatives!(Xdx, X_base, fe_rhs, df, batch, cts_vars)
        end
        
        union!(processed_vars, batch)
    end
    
    return X_base, Xdx
end

"""
Create batches of variables that can safely be processed together.
Conservative approach: only batch variables that appear in completely separate terms.
"""
function create_safe_variable_batches(cts_vars::Vector{Symbol}, fe_rhs)
    # For now, use conservative single-variable batches
    # TODO: Implement interaction analysis to find safe batches
    return [[var] for var in cts_vars]
end

"""
Compute derivatives for a single variable using optimized approach.
"""
function compute_single_variable_derivatives!(
    Xdx::Matrix{Float64}, X_base::Matrix{Float64}, fe_rhs, df::DataFrame, var::Symbol
)
    # This is the core single-variable computation
    # Uses the same approach as compute_numerical_derivatives_optimized! 
    # but without the workspace overhead for single calls
    
    n, p = size(X_base)
    fill!(Xdx, 0.0)
    
    # Find affected columns efficiently
    affected_cols = find_affected_columns_simple(fe_rhs, var, X_base, df)
    
    if isempty(affected_cols)
        return nothing
    end
    
    # Create working DataFrame with proper types
    work_df = DataFrame(df, copycols=true)
    
    # Get original values and perturb
    original_col = work_df[!, var]
    original_float = Float64.(original_col)
    h = sqrt(eps(Float64)) * max(1.0, std(original_float))
    
    # Perturb the focal variable
    work_df[!, var] = original_float .+ h
    
    # Build perturbed matrix
    work_tbl = Tables.columntable(work_df)
    X_pert = Matrix{Float64}(undef, n, p)
    modelmatrix!(X_pert, fe_rhs, work_tbl)
    
    # Compute derivatives
    inv_h = 1.0 / h
    @inbounds for col in affected_cols
        @simd for i in 1:n
            Xdx[i, col] = (X_pert[i, col] - X_base[i, col]) * inv_h
        end
    end
end

"""
Simple affected column detection for single variables.
"""
function find_affected_columns_simple(fe_rhs, var::Symbol, X_base::Matrix, df::DataFrame)
    n, p = size(X_base)
    sample_size = min(20, n)  # Even smaller sample for speed
    sample_idx = 1:div(n, sample_size):n
    
    df_sample = df[sample_idx, :]
    X_sample = X_base[sample_idx, :]
    
    # Quick perturbation test with type safety
    work_df = DataFrame(df_sample, copycols=true)
    original_col = work_df[!, var]
    original_float = Float64.(original_col)
    h = 1e-4 * max(1.0, std(original_float))
    work_df[!, var] = original_float .+ h
    
    X_pert = Matrix{Float64}(undef, size(X_sample))
    modelmatrix!(X_pert, fe_rhs, Tables.columntable(work_df))
    
    # Find affected columns
    affected_cols = Int[]
    tolerance = 1e-10
    
    @inbounds for col in 1:p
        max_change = 0.0
        @simd for i in 1:size(X_sample, 1)
            change = abs(X_pert[i, col] - X_sample[i, col])
            max_change = max(max_change, change)
        end
        
        if max_change > tolerance
            push!(affected_cols, col)
        end
    end
    
    return affected_cols
end

###############################################################################
# Single-variable helper (uses optimized approach)
###############################################################################

function build_continuous_design_single!(
    df::DataFrame,
    fe_rhs,
    focal::Symbol,
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
)
    X_full, Xdx_list = build_continuous_design(df, fe_rhs, [focal])
    copyto!(X, X_full)
    copyto!(Xdx, Xdx_list[1])
    return nothing
end
