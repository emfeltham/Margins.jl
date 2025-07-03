# matrix_reuse_helpers.jl - CLEAN REPLACEMENT

"""
Extract design matrix from fitted model using standard interface
"""
function extract_model_matrix(model, df)
    existing_X = modelmatrix(model)
    @assert size(existing_X, 1) == nrow(df) "Model matrix rows ($(size(existing_X, 1))) don't match data rows ($(nrow(df)))"
    return existing_X
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
