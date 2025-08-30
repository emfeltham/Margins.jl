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
            
            if isempty(term_gradients)
                error("No gradients found for term $term in simple averaging case. Profile averaging requires proper gradient storage.")
            end
            
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
                
                # Collect gradients for this group×term combination using group-aware keys
                term_gradients = Vector{Float64}[]
                
                # Build group key from the group columns to match storage format
                # Note: the stored format is (term, group_key, prof_idx) where group_key is from profile.jl
                group_key_dict = Dict()
                for col in group_cols
                    if col in names(group) && !ismissing(group[term_rows[1], col])
                        group_key_dict[Symbol(col)] = group[term_rows[1], col]
                    end
                end
                group_key = NamedTuple(pairs(group_key_dict))
                
                # Debug: show what we're looking for vs what's available
                @debug "Looking for gradients with:" term=term group_key=group_key
                @debug "Available gradient keys:" keys=collect(keys(gradients))
                
                # Look for gradients matching this term and group combination
                for (i, row_idx) in enumerate(term_rows)
                    # Search for gradients with group-aware keys: (term, group_key, profile_idx)
                    for (grad_key, grad) in gradients
                        if isa(grad_key, Tuple) && length(grad_key) >= 3
                            g_term, g_group_key, g_prof_idx = grad_key[1], grad_key[2], grad_key[3]
                            if g_term == term && g_group_key == group_key && g_prof_idx == i
                                push!(term_gradients, grad)
                                break
                            end
                        elseif isa(grad_key, Tuple) && length(grad_key) == 2
                            # Fallback to old format if available
                            g_term, g_prof_idx = grad_key
                            if g_term == term && g_prof_idx == i
                                push!(term_gradients, grad)
                                break
                            end
                        end
                    end
                end
                
                if isempty(term_gradients)
                    error("No gradients found for group×term combination ($(term), $(group_key)). Profile averaging with grouping requires proper gradient storage.")
                end
                
                # Proper delta-method averaging with collected gradients
                avg_gradient = mean(term_gradients)
                se_proper = FormulaCompiler.delta_method_se(avg_gradient, Σ)
                
                # Create averaged result
                avg_group = DataFrame(
                    term = [term],
                    dydx = [mean(group[term_rows, :dydx])],
                    se = [se_proper]
                )
                
                # Add group columns
                for col in group_cols
                    if col in names(group)
                        avg_group[!, col] = [group[term_rows[1], col]]
                    end
                end
                
                # Add constant non-profile columns
                for col in names(group)
                    if !(col in exclude_cols ∪ ["term"] ∪ group_cols)
                        if length(unique(group[term_rows, col])) == 1
                            avg_group[!, col] = [group[term_rows[1], col]]
                        end
                    end
                end
                
                push!(avg_parts, avg_group)
            end
        end
        
        return isempty(avg_parts) ? DataFrame() : reduce(vcat, avg_parts)
    end
end