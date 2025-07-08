# build_continuous_design.jl - CLEAN REPLACEMENT WITH MINIMAL ALLOCATIONS

###############################################################################
# Matrix reuse with no DataFrame modification approach
###############################################################################

# Fixed build_continuous_design.jl

function build_design_matrices_optimized(model, df, fe_rhs, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    
    # Extract existing matrix from model using standard interface
    X_base = extract_model_matrix(model, df)
    p = size(X_base, 2)
    
    if k == 0
        return X_base, [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end
    
    # OPTIMIZATION: Cache Tables.columntable (expensive operation)
    tbl0 = Tables.columntable(df)
    
    # OPTIMIZATION: Pre-analyze which terms actually need derivatives
    plan = analyze_dependencies(fe_rhs, cts_vars, tbl0)
    
    # Check if any variables actually affect any terms
    if isempty(plan.var_to_terms)
        # No continuous variables affect the model - all derivatives are zero
        return X_base, [zeros(Float64, n, p) for _ in 1:k]
    end
    
    # Pre-allocate derivative matrices
    Xdx_list = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Only compute derivatives for variables that affect terms  
    for (j, var) in enumerate(cts_vars)
        if haskey(plan.var_to_terms, var) && !isempty(plan.var_to_terms[var])
            compute_derivatives_no_df_modification!(Xdx_list[j], X_base, df, fe_rhs, var, 
                                                  plan.var_to_terms[var], tbl0)
        else
            fill!(Xdx_list[j], 0.0)
        end
    end
    
    return X_base, Xdx_list
end

# Also update this function:
function build_continuous_design_single_fast!(
    df::DataFrame,
    fe_rhs,
    focal::Symbol,
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
    active_terms::Union{Nothing, Dict{Symbol, Vector{Tuple{Int, UnitRange{Int}}}}}
)
    tbl0 = Tables.columntable(df)
    
    # Build base matrix efficiently
    modelmatrix!(X, fe_rhs, tbl0)
    
    # Smart derivative computation
    fill!(Xdx, 0.0)
    
    if isnothing(active_terms) || focal ∉ keys(tbl0) || !haskey(active_terms, focal)
        return nothing
    end
    
    term_info = active_terms[focal]
    if isempty(term_info)
        return nothing
    end
    
    # Use the optimized derivative computation with no DataFrame modification
    compute_derivatives_no_df_modification!(Xdx, X, df, fe_rhs, focal, term_info, tbl0)
    
    return nothing
end

"""
Avoid DataFrame modification entirely - use NamedTuple approach
"""
function compute_derivatives_no_df_modification!(
    Xdx::Matrix{Float64},
    X_base::Matrix{Float64},
    df::DataFrame,
    fe_rhs,
    var::Symbol,
    term_info::Vector{Tuple{Int, UnitRange{Int}}},
    tbl0::NamedTuple
)
    fill!(Xdx, 0.0)
    
    if var ∉ keys(tbl0) || isempty(term_info)
        return nothing
    end
    
    # Don't modify the DataFrame at all
    original_vals = Float64.(tbl0[var])
    h = sqrt(eps(Float64)) * max(1.0, maximum(abs, original_vals) * 0.01)
    inv_h = 1.0 / h
    
    # Create perturbed data as NamedTuple (should be much cheaper)
    perturbed_vals = original_vals .+ h
    perturbed_data = merge(tbl0, (var => perturbed_vals,))
    
    # Process term by term
    if isa(fe_rhs, StatsModels.MatrixTerm)
        terms = collect(fe_rhs.terms)
        
        for (term_idx, col_range) in term_info
            term = terms[term_idx]
            
            # Extract base columns
            base_cols = X_base[:, col_range]
            
            # Compute ONLY this specific term using NamedTuple data
            perturbed_cols = StatsModels.modelcols(term, perturbed_data)
            
            # Update derivatives
            update_derivatives_fast!(Xdx, col_range, base_cols, perturbed_cols, inv_h)
        end
    else
        # Single term case
        base_result = X_base
        perturbed_result = StatsModels.modelcols(fe_rhs, perturbed_data)
        col_range = 1:size(Xdx, 2)
        update_derivatives_fast!(Xdx, col_range, base_result, perturbed_result, inv_h)
    end
    
    return nothing
end

"""
Fast derivative update with proper type handling
"""
function update_derivatives_fast!(Xdx, col_range, base_result, perturbed_result, inv_h)
    # Handle all combinations of base_result and perturbed_result shapes
    if isa(base_result, AbstractMatrix) && isa(perturbed_result, AbstractMatrix)
        # Matrix to Matrix
        @inbounds for (local_col, global_col) in enumerate(col_range)
            @simd for row in axes(Xdx, 1)
                Xdx[row, global_col] = (perturbed_result[row, local_col] - base_result[row, local_col]) * inv_h
            end
        end
    elseif isa(base_result, AbstractVector) && isa(perturbed_result, AbstractVector)
        # Vector to Vector (single column)
        global_col = first(col_range)
        @inbounds @simd for row in axes(Xdx, 1)
            Xdx[row, global_col] = (perturbed_result[row] - base_result[row]) * inv_h
        end
    elseif isa(base_result, AbstractVector) && isa(perturbed_result, AbstractMatrix)
        # Vector base, Matrix perturbed (single column)
        global_col = first(col_range)
        @inbounds @simd for row in axes(Xdx, 1)
            Xdx[row, global_col] = (perturbed_result[row, 1] - base_result[row]) * inv_h
        end
    elseif isa(base_result, AbstractMatrix) && isa(perturbed_result, AbstractVector)
        # Matrix base, Vector perturbed (single column)
        global_col = first(col_range)
        @inbounds @simd for row in axes(Xdx, 1)
            Xdx[row, global_col] = (perturbed_result[row] - base_result[row, 1]) * inv_h
        end
    else
        # Scalar case or other combinations
        global_col = first(col_range)
        if isa(base_result, Number) && isa(perturbed_result, Number)
            diff = (perturbed_result - base_result) * inv_h
            @inbounds for row in axes(Xdx, 1)
                Xdx[row, global_col] = diff
            end
        else
            # Convert to common type and handle
            base_val = isa(base_result, Number) ? base_result : base_result[1]
            pert_val = isa(perturbed_result, Number) ? perturbed_result : perturbed_result[1]
            diff = (pert_val - base_val) * inv_h
            @inbounds for row in axes(Xdx, 1)
                Xdx[row, global_col] = diff
            end
        end
    end
end