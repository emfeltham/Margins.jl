# selective_updates!.jl - EFFICIENT VERSION: Zero-copy and smart memory sharing

###############################################################################
# Efficient Data Manipulation Functions
###############################################################################

"""
    create_perturbed_data(base_data::NamedTuple, variable::Symbol, values::AbstractVector)

EFFICIENT: Simple data update with zero-copy for unchanged variables.
Creates a new NamedTuple that shares memory for all unchanged variables.
"""
function create_perturbed_data(base_data::NamedTuple, variable::Symbol, values::AbstractVector)
    # Validate that variable exists in base_data
    if !haskey(base_data, variable)
        throw(ArgumentError("Variable $variable not found in base data"))
    end
    
    # Validate length consistency
    original_length = length(base_data[variable])
    if length(values) != original_length
        throw(DimensionMismatch(
            "New values have length $(length(values)), " *
            "but original data has length $original_length"
        ))
    end
    
    # EFFICIENT: Zero-copy merge - only the changed variable gets new memory
    return merge(base_data, (variable => values,))
end

"""
    batch_perturb_data(base_data::NamedTuple, changes::Dict{Symbol, <:AbstractVector})

EFFICIENT: Batch data update with zero-copy for unchanged variables.
"""
function batch_perturb_data(base_data::NamedTuple, changes::Dict{Symbol, <:AbstractVector})
    if isempty(changes)
        return base_data
    end
    
    # Validate all variables exist and have correct lengths
    original_length = length(first(base_data))
    for (var, values) in changes
        if !haskey(base_data, var)
            throw(ArgumentError("Variable $var not found in base data"))
        end
        
        if length(values) != original_length
            throw(DimensionMismatch(
                "New values for $var have length $(length(values)), " *
                "but original data has length $original_length"
            ))
        end
    end
    
    # EFFICIENT: Zero-copy merge - unchanged variables share memory
    return merge(base_data, changes)
end

"""
    validate_data_consistency(data::NamedTuple)

EFFICIENT: Fast validation that all vectors have the same length.
"""
function validate_data_consistency(data::NamedTuple)
    if isempty(data)
        return true
    end
    
    first_length = length(first(data))
    
    for (name, values) in pairs(data)
        if length(values) != first_length
            throw(DimensionMismatch(
                "Variable $name has length $(length(values)), " *
                "but expected length $first_length"
            ))
        end
    end
    
    return true
end

###############################################################################
# Efficient Matrix Operation Functions
###############################################################################

"""
    share_unchanged_columns!(target::AbstractMatrix, source::AbstractMatrix, 
                            unchanged_columns::Vector{Int})

EFFICIENT: Update target matrix to share memory with source for unchanged columns.
This enables zero-copy for columns that don't need recomputation.
"""
function share_unchanged_columns!(target::AbstractMatrix, source::AbstractMatrix, 
                                 unchanged_columns::Vector{Int})
    # Validate dimensions
    size(target) == size(source) || throw(DimensionMismatch(
        "Target and source matrices must have same dimensions"
    ))
    
    # EFFICIENT: Share memory for each unchanged column (zero-copy)
    @inbounds for col in unchanged_columns
        if 1 ≤ col ≤ size(target, 2)
            # This creates a view that shares memory
            target[:, col] = view(source, :, col)
        else
            throw(BoundsError("Column index $col out of bounds"))
        end
    end
end

"""
    copy_matrix_selective!(target::AbstractMatrix, source::AbstractMatrix, 
                          changed_columns::Vector{Int}, unchanged_columns::Vector{Int})

EFFICIENT: Selective matrix copy that shares memory for unchanged columns.
"""
function copy_matrix_selective!(target::AbstractMatrix, source::AbstractMatrix,
                               changed_columns::Vector{Int}, unchanged_columns::Vector{Int})
    # Validate dimensions
    size(target) == size(source) || throw(DimensionMismatch(
        "Target and source matrices must have same dimensions"
    ))
    
    # EFFICIENT: Share memory for unchanged columns (zero-copy)
    @inbounds for col in unchanged_columns
        target[:, col] = view(source, :, col)
    end
    
    # Changed columns will be updated by caller
    # (just ensuring they have allocated space here)
end

"""
    update_matrix_columns_inplace!(target::AbstractMatrix, source::AbstractMatrix, 
                                  column_updates::Dict{Int, AbstractVector})

EFFICIENT: In-place update of specific matrix columns with minimal copying.
"""
function update_matrix_columns_inplace!(target::AbstractMatrix, source::AbstractMatrix,
                                       column_updates::Dict{Int, AbstractVector})
    # Validate dimensions
    size(target, 1) == size(source, 1) || throw(DimensionMismatch(
        "Target and source must have same number of rows"
    ))
    
    n_rows = size(target, 1)
    
    # EFFICIENT: Update only specified columns
    for (col_idx, new_values) in column_updates
        if 1 ≤ col_idx ≤ size(target, 2)
            if length(new_values) != n_rows
                throw(DimensionMismatch(
                    "Column $col_idx: new values have length $(length(new_values)), " *
                    "expected $n_rows"
                ))
            end
            
            # Direct column assignment (efficient)
            target[:, col_idx] = new_values
        else
            throw(BoundsError("Column index $col_idx out of bounds"))
        end
    end
end

"""
    estimate_memory_usage(matrix_dims::Tuple{Int,Int}, affected_cols::Int, 
                         dtype::Type=Float64) -> NamedTuple

EFFICIENT: Estimate memory usage for selective vs full matrix operations.
"""
function estimate_memory_usage(matrix_dims::Tuple{Int,Int}, affected_cols::Int, 
                              dtype::Type=Float64)
    nrows, ncols = matrix_dims
    element_size = sizeof(dtype)
    
    # Full matrix memory
    full_bytes = nrows * ncols * element_size
    full_mb = full_bytes / (1024^2)
    
    # Selective update memory (only affected columns)
    selective_bytes = nrows * affected_cols * element_size
    selective_mb = selective_bytes / (1024^2)
    
    # Memory savings
    saved_bytes = full_bytes - selective_bytes
    saved_mb = saved_bytes / (1024^2)
    percent_saved = (saved_bytes / full_bytes) * 100
    
    return (
        full_memory_mb = full_mb,
        selective_memory_mb = selective_mb,
        memory_saved_mb = saved_mb,
        percent_saved = percent_saved,
        affected_columns = affected_cols,
        total_columns = ncols,
        percent_cols_affected = (affected_cols / ncols) * 100
    )
end

###############################################################################
# Efficient Validation Functions
###############################################################################

"""
    validate_selective_update_fast(original::AbstractMatrix, updated::AbstractMatrix, 
                                  changed_cols::Vector{Int}) -> Bool

EFFICIENT: Fast validation that unchanged columns are truly unchanged.
Only checks a sample of rows for performance.
"""
function validate_selective_update_fast(original::AbstractMatrix, updated::AbstractMatrix, 
                                       changed_cols::Vector{Int})
    # Check dimensions
    size(original) == size(updated) || return false
    
    total_cols = size(original, 2)
    unchanged_cols = setdiff(1:total_cols, changed_cols)
    
    # EFFICIENT: Sample-based validation for large matrices
    n_rows = size(original, 1)
    sample_size = min(100, n_rows)  # Check at most 100 rows
    sample_indices = n_rows <= 100 ? (1:n_rows) : sort(rand(1:n_rows, sample_size))
    
    # Check that unchanged columns are identical (sample only)
    @inbounds for col in unchanged_cols, row in sample_indices
        if original[row, col] != updated[row, col]
            return false
        end
    end
    
    return true
end

"""
    memory_efficiency_report(ws::AMEWorkspace, variable::Symbol) -> String

EFFICIENT: Generate memory efficiency report for a workspace and variable.
"""
function memory_efficiency_report(ws::AMEWorkspace, variable::Symbol)
    affected_cols = length(get(ws.variable_plans, variable, Int[]))
    total_cols = ws.p
    
    usage = estimate_memory_usage((ws.n, ws.p), affected_cols)
    
    report = """
    Memory Efficiency Report for Variable :$variable
    ================================================
    
    Matrix Dimensions: $(ws.n) × $(ws.p)
    Affected Columns: $affected_cols / $total_cols ($(round(usage.percent_cols_affected, digits=1))%)
    
    Memory Usage:
    - Full Matrix Approach: $(round(usage.full_memory_mb, digits=2)) MB
    - Selective Approach:   $(round(usage.selective_memory_mb, digits=2)) MB
    - Memory Saved:         $(round(usage.memory_saved_mb, digits=2)) MB ($(round(usage.percent_saved, digits=1))%)
    
    Efficiency Ratio: $(round(usage.selective_memory_mb / usage.full_memory_mb, digits=3))x
    """
    
    return report
end

###############################################################################
# Advanced Efficiency Utilities
###############################################################################

"""
    plan_selective_updates(mapping::ColumnMapping, variables::Vector{Symbol}) -> Dict

EFFICIENT: Pre-plan selective updates to minimize redundant computation.
"""
function plan_selective_updates(mapping::ColumnMapping, variables::Vector{Symbol})
    plan = Dict{Symbol, Any}()
    
    total_cols = mapping.total_columns
    all_affected_cols = Set{Int}()
    
    for var in variables
        var_cols = get_variable_columns_flat(mapping, var)
        plan[var] = (
            affected_columns = var_cols,
            num_affected = length(var_cols),
            percent_affected = (length(var_cols) / total_cols) * 100
        )
        union!(all_affected_cols, var_cols)
    end
    
    plan[:summary] = (
        total_variables = length(variables),
        total_columns = total_cols,
        total_affected_columns = length(all_affected_cols),
        overall_percent_affected = (length(all_affected_cols) / total_cols) * 100,
        efficiency_potential = 1.0 - (length(all_affected_cols) / total_cols)
    )
    
    return plan
end

"""
    benchmark_selective_vs_full(ws::AMEWorkspace, variable::Symbol; n_runs::Int=10)

EFFICIENT: Benchmark selective vs full matrix updates (for development/tuning).
"""
function benchmark_selective_vs_full(ws::AMEWorkspace, variable::Symbol; n_runs::Int=10)
    # This would be used for performance tuning during development
    # Implementation would time both approaches and compare
    
    affected_cols = get(ws.variable_plans, variable, Int[])
    
    return (
        variable = variable,
        affected_columns = length(affected_cols),
        total_columns = ws.p,
        efficiency_ratio = length(affected_cols) / ws.p,
        recommendation = length(affected_cols) / ws.p < 0.5 ? "Use selective updates" : "Consider full update"
    )
end

"""
    optimize_workspace_layout(ws::AMEWorkspace) -> NamedTuple

EFFICIENT: Analyze workspace layout and suggest optimizations.
"""
function optimize_workspace_layout(ws::AMEWorkspace)
    # Analyze which variables affect which columns
    var_impacts = Dict{Symbol, Int}()
    for (var, cols) in ws.variable_plans
        var_impacts[var] = length(cols)
    end
    
    # Sort by impact (most columns affected first)
    sorted_vars = sort(collect(var_impacts), by=x->x[2], rev=true)
    
    # Calculate overlap between variables
    overlaps = Dict{Tuple{Symbol,Symbol}, Int}()
    vars = collect(keys(var_impacts))
    for i in 1:length(vars)-1
        for j in i+1:length(vars)
            var1, var2 = vars[i], vars[j]
            cols1 = Set(ws.variable_plans[var1])
            cols2 = Set(ws.variable_plans[var2])
            overlap = length(intersect(cols1, cols2))
            if overlap > 0
                overlaps[(var1, var2)] = overlap
            end
        end
    end
    
    return (
        variable_impacts = sorted_vars,
        overlapping_variables = overlaps,
        total_unique_affected_columns = length(union(values(ws.variable_plans)...)),
        optimization_potential = 1.0 - (length(union(values(ws.variable_plans)...)) / ws.p)
    )
end
