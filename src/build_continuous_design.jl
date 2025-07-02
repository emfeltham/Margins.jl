# build_continuous_design.jl - DROP-IN REPLACEMENT
# Major performance optimizations targeting the key bottlenecks

###############################################################################
# Ultra-optimized continuous design matrix builder
###############################################################################

"""
Ultra-optimized numerical differentiation that:
1. Uses minimal sample for affected column detection
2. Pre-allocates and reuses all buffers
3. Minimizes DataFrame operations and matrix rebuilds
4. Leverages SIMD vectorization
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
    
    # Create optimized workspace
    workspace = OptimizedNumericalWorkspace(n, p)
    
    # Batch analyze affected columns with minimal sample
    all_affected_cols = batch_analyze_affected_columns(fe_rhs, cts_vars, X_base, df, workspace)
    
    # Pre-allocate derivative matrices
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Process each variable with optimized computation
    for (j, var) in enumerate(cts_vars)
        compute_derivatives_optimized!(
            Xdx[j], X_base, fe_rhs, df, var, 
            all_affected_cols[j], workspace
        )
    end

    return X_base, Xdx
end

"""
Optimized workspace with minimal allocations
"""
mutable struct OptimizedNumericalWorkspace
    # Main working DataFrame (reference, not copy)
    work_df::DataFrame
    
    # Matrix buffers
    X_buffer::Matrix{Float64}
    
    # Variable perturbation buffers
    original_values::Vector{Float64}
    
    # Small sample for column detection
    sample_df::DataFrame
    sample_X_base::Matrix{Float64}
    sample_X_pert::Matrix{Float64}
    sample_size::Int
    
    # Step size cache
    step_sizes::Dict{Symbol, Float64}
    
    function OptimizedNumericalWorkspace(n::Int, p::Int)
        # Use larger sample size for better accuracy with complex interactions
        sample_size = min(100, max(50, div(n, 100)))  # More conservative
        
        new(
            DataFrame(),  # Will reference original
            Matrix{Float64}(undef, n, p),
            Vector{Float64}(undef, n),
            DataFrame(),
            Matrix{Float64}(undef, sample_size, p),
            Matrix{Float64}(undef, sample_size, p),
            sample_size,
            Dict{Symbol, Float64}()
        )
    end
end

"""
Batch analysis of affected columns using minimal sample
"""
function batch_analyze_affected_columns(
    fe_rhs, cts_vars::Vector{Symbol}, X_base::Matrix, df::DataFrame, workspace::OptimizedNumericalWorkspace
)
    n, p = size(X_base)
    all_affected = Vector{Vector{Int}}(undef, length(cts_vars))
    
    # Use larger sample for complex interactions (more conservative)
    target_sample_size = min(100, max(50, div(n, 100)))  # At least 50, up to 100
    step_size = max(1, div(n, target_sample_size))
    sample_idx = 1:step_size:n
    actual_sample_size = min(target_sample_size, length(sample_idx))
    sample_idx = sample_idx[1:actual_sample_size]
    
    # Sample data and base matrix
    workspace.sample_df = df[sample_idx, :]
    workspace.sample_X_base[1:actual_sample_size, :] .= view(X_base, sample_idx, :)
    
    # Analyze each variable
    for (j, var) in enumerate(cts_vars)
        all_affected[j] = find_affected_columns_fast(
            fe_rhs, var, workspace, actual_sample_size
        )
    end
    
    return all_affected
end

"""
Fast affected column detection with minimal operations
"""
function find_affected_columns_fast(
    fe_rhs, var::Symbol, workspace::OptimizedNumericalWorkspace, sample_size::Int
)
    # Work with sample DataFrame
    sample_df = workspace.sample_df
    original_col = sample_df[!, var]
    
    # Convert to Float64 and compute step size (more robust for interactions)
    original_float = Float64.(original_col)
    # More conservative step size for complex interactions
    h = sqrt(eps(Float64)) * max(1.0, maximum(abs, original_float) * 0.01)
    workspace.step_sizes[var] = h
    
    # Temporary perturbation
    sample_df[!, var] = original_float .+ h
    
    # Build perturbed matrix in minimal buffer
    sample_tbl = Tables.columntable(sample_df)
    modelmatrix!(view(workspace.sample_X_pert, 1:sample_size, :), fe_rhs, sample_tbl)
    
    # Restore original immediately
    sample_df[!, var] = original_col
    
    # Find affected columns with more conservative tolerance for interactions
    affected_cols = Int[]
    tolerance = 1e-12  # Tighter tolerance for complex interactions
    p = size(workspace.sample_X_base, 2)
    
    @inbounds for col in 1:p
        max_change = 0.0
        @simd for i in 1:sample_size
            change = abs(workspace.sample_X_pert[i, col] - workspace.sample_X_base[i, col])
            max_change = max(max_change, change)
        end
        
        if max_change > tolerance
            push!(affected_cols, col)
        end
    end
    
    return affected_cols
end

"""
Optimized derivative computation with minimal matrix operations
"""
function compute_derivatives_optimized!(
    Xdx::Matrix{Float64},
    X_base::Matrix{Float64}, 
    fe_rhs,
    df::DataFrame,
    var::Symbol,
    affected_cols::Vector{Int},
    workspace::OptimizedNumericalWorkspace
)
    n, p = size(X_base)
    
    # Initialize derivative matrix
    fill!(Xdx, 0.0)
    
    if isempty(affected_cols)
        return nothing
    end
    
    # Use reference to original DataFrame (no copying)
    workspace.work_df = df
    original_col = workspace.work_df[!, var]
    
    # Get cached step size
    h = workspace.step_sizes[var]
    
    # Convert and perturb values
    copyto!(workspace.original_values, Float64.(original_col))
    
    # Temporary perturbation for matrix building
    workspace.work_df[!, var] = workspace.original_values .+ h
    work_tbl = Tables.columntable(workspace.work_df)
    
    # Build perturbed matrix
    modelmatrix!(workspace.X_buffer, fe_rhs, work_tbl)
    
    # Restore original data immediately
    workspace.work_df[!, var] = original_col
    
    # Compute derivatives only for affected columns (major speedup)
    inv_h = 1.0 / h
    @inbounds for col in affected_cols
        @simd for i in 1:n
            Xdx[i, col] = (workspace.X_buffer[i, col] - X_base[i, col]) * inv_h
        end
    end
    
    return nothing
end

###############################################################################
# Single-variable helper (optimized)
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
