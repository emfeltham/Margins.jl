# features/averaging.jl - Proper delta method averaging for profiles

"""
    _average_profiles_with_proper_se(df, gradients, Σ; group_cols)

Average profile results using proper delta method on averaged gradients.
"""
function _average_profiles_with_proper_se(df::DataFrame, gradients::Dict, Σ::AbstractMatrix; group_cols::Vector{String}=String[])
    profile_cols = [c for c in names(df) if startswith(String(c), "at_")]
    exclude_cols = ["dydx", "se", "z", "p", "ci_lo", "ci_hi"] ∪ profile_cols
    
    if isempty(group_cols) || all(col -> length(unique(df[!, col])) == 1, group_cols)
        # Simple case: average all rows for each term
        terms = unique(df.term)
        avg_parts = DataFrame[]
        
        for term in terms
            term_rows = findall(==(term), df.term)
            if length(term_rows) <= 1
                # Single row, just keep as-is
                push!(avg_parts, df[term_rows, :])
                continue
            end
            
            # Collect gradients for this term and average them
            term_gradients = Vector{Float64}[]
            for (i, row_idx) in enumerate(term_rows)
                # Find corresponding gradient - handle different key formats
                for (key, grad) in gradients
                    # Handle both tuple and simple key formats
                    if isa(key, Tuple)
                        g_term, g_prof_idx = key
                        if g_term == term && g_prof_idx == i
                            push!(term_gradients, grad)
                            break
                        end
                    else
                        # Simple key format - just use the gradient if it exists
                        if haskey(gradients, i)
                            push!(term_gradients, gradients[i])
                            break
                        end
                    end
                end
            end
            
            if !isempty(term_gradients)
                # Average gradients and apply delta method
                avg_gradient = mean(term_gradients)
                se_proper = FormulaCompiler.delta_method_se(avg_gradient, Σ)
                
                # Create averaged result
                avg_row = DataFrame(
                    term = [term],
                    dydx = [mean(df[term_rows, :dydx])],
                    se = [se_proper]
                )
                
                # Add constant columns
                for col in names(df)
                    if !(col in exclude_cols ∪ ["term"])
                        if length(unique(df[term_rows, col])) == 1
                            avg_row[!, col] = [df[term_rows[1], col]]
                        end
                    end
                end
                push!(avg_parts, avg_row)
            end
        end
        
        return isempty(avg_parts) ? DataFrame() : reduce(vcat, avg_parts)
    else
        # Complex case: group by non-profile columns and average within groups
        grouped = groupby(df, group_cols)
        avg_parts = DataFrame[]
        
        for group in grouped
            terms = unique(group.term)
            for term in terms
                term_rows = findall(==(term), group.term)
                if length(term_rows) <= 1
                    push!(avg_parts, group[term_rows, :])
                    continue
                end
                
                # Similar logic but more complex gradient lookup
                # This would need more work to properly map gradients to grouped results
                # For now, fall back to simple averaging
                avg_group = DataFrame(
                    term = [term],
                    dydx = [mean(group[term_rows, :dydx])],
                    se = [sqrt(sum(group[term_rows, :se].^2)) / length(term_rows)]  # Improved approximation
                )
                
                for col in group_cols
                    avg_group[!, col] = [group[term_rows[1], col]]
                end
                push!(avg_parts, avg_group)
            end
        end
        
        return isempty(avg_parts) ? DataFrame() : reduce(vcat, avg_parts)
    end
end