# optimization_helpers.jl - Helper functions for matrix reuse optimizations

###############################################################################
# Helper functions needed for the optimized matrix reuse system
###############################################################################

"""
Analyze which terms actually contain continuous variables and their column ranges
"""
function analyze_active_terms(fe_rhs, cts_vars::Vector{Symbol}, tbl0::NamedTuple)
    if isa(fe_rhs, StatsModels.MatrixTerm)
        terms = collect(fe_rhs.terms)
    else
        terms = [fe_rhs]
    end
    
    active_indices = Int[]
    term_ranges = UnitRange{Int}[]
    
    col_offset = 1
    for (i, term) in enumerate(terms)
        # Determine column range for this term
        term_result = StatsModels.modelcols(term, tbl0)
        ncols = get_result_width(term_result)
        col_range = col_offset:(col_offset + ncols - 1)
        push!(term_ranges, col_range)
        
        # Check if this term contains any continuous variables
        term_vars = find_variables_in_term_optimized(term, cts_vars)
        if !isempty(term_vars)
            push!(active_indices, i)
        end
        
        col_offset += ncols
    end
    
    return active_indices, term_ranges
end

"""
Get the width (number of columns) of a modelcols result
"""
function get_result_width(result)
    if isa(result, AbstractMatrix)
        return size(result, 2)
    elseif isa(result, AbstractVector)
        return 1
    else
        return 1  # Scalar
    end
end

"""
Optimized variable dependency analysis
"""
function find_variables_in_term_optimized(term, cts_vars::Vector{Symbol})
    # Use a more efficient approach for common term types
    if isa(term, StatsModels.ContinuousTerm)
        return term.sym in cts_vars ? [term.sym] : Symbol[]
    elseif isa(term, StatsModels.InterceptTerm)
        return Symbol[]
    elseif isa(term, StatsModels.InteractionTerm)
        result = Symbol[]
        for subterm in term.terms
            append!(result, find_variables_in_term_optimized(subterm, cts_vars))
        end
        return unique!(result)
    else
        # Conservative fallback
        return intersect(cts_vars, StatsModels.termvars(term))
    end
end

"""
Update derivative matrix for a specific column range
"""
function update_derivative_range!(Xdx, col_range, base_result, perturbed_result, inv_h)
    if isa(base_result, AbstractMatrix) && isa(perturbed_result, AbstractMatrix)
        @inbounds for (local_col, global_col) in enumerate(col_range)
            @simd for row in axes(Xdx, 1)
                Xdx[row, global_col] = (perturbed_result[row, local_col] - base_result[row, local_col]) * inv_h
            end
        end
    elseif isa(base_result, AbstractVector) && isa(perturbed_result, AbstractVector)
        global_col = first(col_range)
        @inbounds @simd for row in axes(Xdx, 1)
            Xdx[row, global_col] = (perturbed_result[row] - base_result[row]) * inv_h
        end
    else
        # Scalar case
        global_col = first(col_range)
        diff = (perturbed_result - base_result) * inv_h
        @inbounds for row in axes(Xdx, 1)
            Xdx[row, global_col] = diff
        end
    end
end

###############################################################################
# Backward compatibility structures (if needed for original ultra-optimized code)
###############################################################################

"""
Ultra-optimized workspace that pre-allocates everything (backward compatibility)
"""
mutable struct UltraOptimizedWorkspace
    # Pre-allocated Tables.ColumnTable structures
    base_table::NamedTuple
    modified_table_cache::Dict{Symbol, NamedTuple}
    
    # Pre-allocated working vectors for each continuous variable
    original_vectors::Dict{Symbol, Vector{Float64}}
    perturbed_vectors::Dict{Symbol, Vector{Float64}}
    
    # Pre-allocated matrix buffers
    term_matrix_cache::Dict{Int, Matrix{Float64}}
    term_vector_cache::Dict{Int, Vector{Float64}}
    
    # Step sizes computed once
    step_sizes::Dict{Symbol, Float64}
    
    # Reference to base matrix
    base_matrix_ref::Matrix{Float64}
    
    function UltraOptimizedWorkspace(n::Int, p::Int, k::Int, base_table::NamedTuple, base_matrix::Matrix{Float64})
        new(
            base_table,
            Dict{Symbol, NamedTuple}(),
            Dict{Symbol, Vector{Float64}}(),
            Dict{Symbol, Vector{Float64}}(),
            Dict{Int, Matrix{Float64}}(),
            Dict{Int, Vector{Float64}}(),
            Dict{Symbol, Float64}(),
            base_matrix
        )
    end
end

"""
Optimized term structure information with pre-computed data (backward compatibility)
"""
struct OptimizedTermInfo
    var_to_terms::Dict{Symbol, Vector{Int}}
    terms::Vector{Any}
    term_column_ranges::Vector{UnitRange{Int}}
    base_term_results::Vector{Any}  # Pre-computed base results
    step_sizes::Dict{Symbol, Float64}
end

"""
Analyze term structure with pre-computation and caching (backward compatibility)
"""
function analyze_term_structure_optimized(fe_rhs, tbl0, cts_vars::Vector{Symbol}, workspace::UltraOptimizedWorkspace)
    # Extract terms
    if isa(fe_rhs, StatsModels.MatrixTerm)
        terms = collect(fe_rhs.terms)
    else
        terms = [fe_rhs]
    end
    
    # Pre-compute everything we need
    base_term_results = []
    term_column_ranges = UnitRange{Int}[]
    var_to_terms = Dict{Symbol, Vector{Int}}()
    
    col_start = 1
    for (i, term) in enumerate(terms)
        # Compute base columns for this term
        term_result = StatsModels.modelcols(term, tbl0)
        push!(base_term_results, term_result)
        
        # Determine column range and cache appropriate matrix/vector
        ncols = get_result_width(term_result)
        col_range = col_start:(col_start + ncols - 1)
        push!(term_column_ranges, col_range)
        col_start += ncols
        
        # Pre-allocate working space for this term
        if isa(term_result, AbstractMatrix) && ncols > 1
            workspace.term_matrix_cache[i] = Matrix{Float64}(undef, size(term_result))
        elseif ncols == 1
            workspace.term_vector_cache[i] = Vector{Float64}(undef, length(tbl0[1]))
        end
        
        # Analyze variable dependencies
        affected_vars = find_variables_in_term_optimized(term, cts_vars)
        for var in affected_vars
            if !haskey(var_to_terms, var)
                var_to_terms[var] = Int[]
            end
            push!(var_to_terms[var], i)
        end
    end
    
    # Pre-compute step sizes and working vectors
    for var in cts_vars
        if haskey(var_to_terms, var)
            values = Float64.(tbl0[var])
            workspace.step_sizes[var] = sqrt(eps(Float64)) * max(1.0, maximum(abs, values) * 0.01)
            workspace.original_vectors[var] = copy(values)
            workspace.perturbed_vectors[var] = Vector{Float64}(undef, length(values))
        end
    end
    
    return OptimizedTermInfo(
        var_to_terms, terms, term_column_ranges, base_term_results, workspace.step_sizes
    )
end

"""
Zero-allocation derivative computation (backward compatibility)
"""
function compute_derivatives_zero_alloc!(
    Xdx::Matrix{Float64},
    X_base::Matrix{Float64},
    terms_info::OptimizedTermInfo,
    var::Symbol,
    workspace::UltraOptimizedWorkspace
)
    fill!(Xdx, 0.0)
    
    if !haskey(terms_info.var_to_terms, var)
        return nothing
    end
    
    affected_term_indices = terms_info.var_to_terms[var]
    h = terms_info.step_sizes[var]
    inv_h = 1.0 / h
    
    # Prepare perturbed values (reuse pre-allocated vector)
    original_vals = workspace.original_vectors[var]
    perturbed_vals = workspace.perturbed_vectors[var]
    @inbounds @simd for i in eachindex(original_vals)
        perturbed_vals[i] = original_vals[i] + h
    end
    
    # Create modified table with minimal allocation
    if !haskey(workspace.modified_table_cache, var)
        workspace.modified_table_cache[var] = merge(workspace.base_table, (var => perturbed_vals,))
    else
        # Reuse existing NamedTuple structure, just update the reference
        workspace.modified_table_cache[var] = merge(workspace.base_table, (var => perturbed_vals,))
    end
    modified_table = workspace.modified_table_cache[var]
    
    # Process only affected terms
    for term_idx in affected_term_indices
        term = terms_info.terms[term_idx]
        col_range = terms_info.term_column_ranges[term_idx]
        base_result = terms_info.base_term_results[term_idx]
        
        # Compute perturbed result
        perturbed_result = StatsModels.modelcols(term, modified_table)
        
        # Update derivative matrix
        update_derivative_range!(Xdx, col_range, base_result, perturbed_result, inv_h)
    end
    
    return nothing
end
