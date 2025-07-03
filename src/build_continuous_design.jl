# build_continuous_design.jl - OPTIMIZED VERSION

###############################################################################
# Matrix reuse with smart derivative computation
###############################################################################

"""
Build design matrices with aggressive matrix reuse and minimal StatsModels calls
"""
function build_design_matrices_optimized(model, df, fe_rhs, cts_vars::Vector{Symbol})
    n, k = nrow(df), length(cts_vars)
    
    # Extract existing matrix from model using standard interface
    X_base = extract_model_matrix(model, df)
    
    p = size(X_base, 2)
    
    if k == 0
        return X_base, [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end
    
    # Smart analysis of which terms need derivatives
    tbl0 = Tables.columntable(df)
    active_var_to_terms = analyze_variable_dependencies(fe_rhs, cts_vars, tbl0)
    
    # Pre-allocate derivative matrices
    Xdx_list = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    # Only compute derivatives for variables that affect terms
    for (j, var) in enumerate(cts_vars)
        if haskey(active_var_to_terms, var) && !isempty(active_var_to_terms[var])
            compute_smart_derivatives!(Xdx_list[j], X_base, df, fe_rhs, var, 
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
function analyze_variable_dependencies(fe_rhs, cts_vars::Vector{Symbol}, tbl0::NamedTuple)
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
        ncols = estimate_term_width(term, tbl0)
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
function estimate_term_width(term, tbl0::NamedTuple)
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
            width *= estimate_term_width(subterm, tbl0)
        end
        return width
    elseif hasfield(typeof(term), :sym) && isa(term.sym, Symbol)
        # For terms with a symbol field (like ZScoredTerm), treat as continuous
        return 1
    else
        # For completely unknown term types, fall back to 1 column
        # This is safer than calling modelcols and allocating massive arrays
        return 1
    end
end

"""
Compute derivatives only for terms that actually depend on the variable
"""
function compute_smart_derivatives!(
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
        return
    end
    
    # Adaptive step size
    original_vals = Float64.(tbl0[var])
    h = sqrt(eps(Float64)) * max(1.0, maximum(abs, original_vals) * 0.01)
    inv_h = 1.0 / h
    
    # OPTIMIZATION: Minimize DataFrame operations and copying
    original_col_data = copy(df[!, var])  # Only copy the modified column
    
    # Perturb variable in-place in original dataframe (restore later)
    df[!, var] .= original_vals .+ h
    perturbed_table = Tables.columntable(df)
    
    # Process only affected terms with minimal StatsModels calls
    if isa(fe_rhs, StatsModels.MatrixTerm)
        terms = collect(fe_rhs.terms)
        
        # Extract base columns directly from X_base instead of recomputing
        for (term_idx, col_range) in term_info
            term = terms[term_idx]
            
            # Get base result from existing matrix with proper shape
            if length(col_range) == 1
                base_result = X_base[:, first(col_range)]  # Vector for single column
            else
                base_result = X_base[:, col_range]  # Matrix for multiple columns
            end
            
            # Only compute perturbed result
            perturbed_result = StatsModels.modelcols(term, perturbed_table)
            update_derivatives!(Xdx, col_range, base_result, perturbed_result, inv_h)
        end
    else
        # Single term case - reuse existing matrix
        base_result = X_base
        perturbed_result = StatsModels.modelcols(fe_rhs, perturbed_table)
        col_range = 1:size(Xdx, 2)
        update_derivatives!(Xdx, col_range, base_result, perturbed_result, inv_h)
    end
    
    # Restore original values
    df[!, var] = original_col_data
end

###############################################################################
# Keep original function for backward compatibility but redirect
###############################################################################

function build_continuous_design(df, fe_rhs, cts_vars::Vector{Symbol})
    # Create a dummy model-like object for the optimized path
    # This is a fallback when called without model
    tbl0 = Tables.columntable(df)
    X_base = modelmatrix(fe_rhs, tbl0)
    n, p = size(X_base)
    k = length(cts_vars)
    
    if k == 0
        return X_base, [Matrix{Float64}(undef, n, 0) for _ in 1:k]
    end
    
    # Use simplified version without model matrix reuse
    active_var_to_terms = analyze_variable_dependencies(fe_rhs, cts_vars, tbl0)
    
    Xdx_list = [Matrix{Float64}(undef, n, p) for _ in 1:k]
    
    for (j, var) in enumerate(cts_vars)
        if haskey(active_var_to_terms, var) && !isempty(active_var_to_terms[var])
            compute_smart_derivatives!(Xdx_list[j], X_base, DataFrame(df), fe_rhs, var, 
                                     active_var_to_terms[var], tbl0)
        else
            fill!(Xdx_list[j], 0.0)
        end
    end
    
    return X_base, Xdx_list
end
