# build_continuous_design.jl - CLEAN REPLACEMENT WITH MINIMAL ALLOCATIONS

###############################################################################
# Matrix reuse with no DataFrame modification approach
###############################################################################

"""
Ultra-fast approach that reuses the model's existing design matrix and avoids DataFrame modifications
"""
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
    active_var_to_terms = analyze_variable_dependencies_fast(fe_rhs, cts_vars, tbl0)
    
    if isempty(active_var_to_terms)
        # No continuous variables affect the model - all derivatives are zero
        return X_base, [zeros(Float64, n, p) for _ in 1:k]
    end
    
    # Pre-allocate derivative matrices
    Xdx_list = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Only compute derivatives for variables that affect terms  
    for (j, var) in enumerate(cts_vars)
        if haskey(active_var_to_terms, var) && !isempty(active_var_to_terms[var])
            compute_derivatives_no_df_modification!(Xdx_list[j], X_base, df, fe_rhs, var, 
                                                  active_var_to_terms[var], tbl0)
        else
            fill!(Xdx_list[j], 0.0)
        end
    end
    
    return X_base, Xdx_list
end

"""
Analyze which terms are affected by which continuous variables WITHOUT calling modelcols
"""
function analyze_variable_dependencies_fast(fe_rhs, cts_vars::Vector{Symbol}, tbl0::NamedTuple)
    if isa(fe_rhs, StatsModels.MatrixTerm)
        terms = collect(fe_rhs.terms)
    else
        terms = [fe_rhs]
    end
    
    var_to_term_info = Dict{Symbol, Vector{Tuple{Int, UnitRange{Int}}}}()
    
    # OPTIMIZATION: Determine column ranges from term structure, not by evaluation
    col_offset = 1
    for (i, term) in enumerate(terms)
        # Get column count without calling modelcols
        ncols = estimate_term_width_fast(term, tbl0)
        col_range = col_offset:(col_offset + ncols - 1)
        
        # Find which variables affect this term using lightweight analysis
        affected_vars = find_variables_in_term_fast(term, cts_vars)
        for var in affected_vars
            if !haskey(var_to_term_info, var)
                var_to_term_info[var] = Tuple{Int, UnitRange{Int}}[]
            end
            push!(var_to_term_info[var], (i, col_range))
        end
        
        col_offset += ncols
    end
    
    return var_to_term_info
end

"""
Estimate term width without expensive modelcols evaluation
"""
function estimate_term_width_fast(term, tbl0::NamedTuple)
    if isa(term, StatsModels.InterceptTerm)
        return 1
    elseif isa(term, StatsModels.ContinuousTerm)
        return 1
    elseif isa(term, StatsModels.CategoricalTerm)
        # Get number of levels - 1 for contrast coding
        col_data = tbl0[term.sym]
        if isa(col_data, CategoricalArray)
            return length(levels(col_data)) - 1
        else
            return length(unique(col_data)) - 1
        end
    elseif isa(term, StatsModels.InteractionTerm)
        # Product of component widths
        width = 1
        for subterm in term.terms
            width *= estimate_term_width_fast(subterm, tbl0)
        end
        return width
    elseif hasfield(typeof(term), :sym) && isa(term.sym, Symbol)
        # For terms with a symbol field (like ZScoredTerm), treat as continuous
        return 1
    else
        # For completely unknown term types, assume 1 column
        return 1
    end
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

"""
Smart variable dependency analysis
"""
function find_variables_in_term_fast(term, cts_vars::Vector{Symbol})
    # Use a more efficient approach for common term types
    if isa(term, StatsModels.ContinuousTerm)
        return term.sym in cts_vars ? [term.sym] : Symbol[]
    elseif isa(term, StatsModels.InterceptTerm)
        return Symbol[]
    elseif isa(term, StatsModels.InteractionTerm)
        result = Symbol[]
        for subterm in term.terms
            append!(result, find_variables_in_term_fast(subterm, cts_vars))
        end
        return unique!(result)
    else
        # Conservative fallback
        return intersect(cts_vars, StatsModels.termvars(term))
    end
end

###############################################################################
# KEEP ORIGINAL FUNCTIONS UNCHANGED for backward compatibility
###############################################################################

"""
Original function signature maintained for backward compatibility
"""
function build_continuous_design(df, fe_rhs, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    if k == 0
        X_base = modelmatrix(fe_rhs, Tables.columntable(df))
        return X_base, [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end

    # Fallback to original behavior but with some optimizations
    tbl0 = Tables.columntable(df)
    X_base = modelmatrix(fe_rhs, tbl0)
    p = size(X_base, 2)
    
    # Use fast analysis
    active_var_to_terms = analyze_variable_dependencies_fast(fe_rhs, cts_vars, tbl0)
    
    if isempty(active_var_to_terms)
        return X_base, [zeros(Float64, n, p) for _ in 1:k]
    end
    
    # Pre-allocate derivative matrices
    Xdx = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Process variables
    for (j, var) in enumerate(cts_vars)
        if haskey(active_var_to_terms, var) && !isempty(active_var_to_terms[var])
            compute_derivatives_no_df_modification!(Xdx[j], X_base, df, fe_rhs, var, 
                                                  active_var_to_terms[var], tbl0)
        else
            fill!(Xdx[j], 0.0)
        end
    end

    return X_base, Xdx
end
