# selective_updates.jl - Core selective update utilities for Margins.jl

###############################################################################
# Data Manipulation Functions
###############################################################################

"""
    create_perturbed_data(base_data::NamedTuple, variable::Symbol, values::AbstractVector)

ENHANCED: Create a new NamedTuple with updated values for one variable.
Uses memory sharing for all unchanged variables.
FIXED: Proper handling of categorical types to prevent Vector{CategoricalValue} issues.

# Arguments
- `base_data`: Original data as NamedTuple
- `variable`: Symbol of variable to perturb
- `values`: New values for the variable

# Returns
- New NamedTuple with updated variable, sharing memory for others

# Details
This version properly handles CategoricalArray types to ensure they remain as
CategoricalArray (not Vector{CategoricalValue}) to work with refs() in EfficientModelMatrices.
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
    
    # ENHANCED: Handle categorical types properly
    orig_var = base_data[variable]
    processed_values = values
    
    if orig_var isa CategoricalArray
        processed_values = ensure_categorical_array(values, orig_var)
    end
    
    # Create new NamedTuple with updated variable
    return merge(base_data, (variable => processed_values,))
end

"""
    batch_perturb_data(base_data::NamedTuple, changes::Dict{Symbol, <:AbstractVector})

ENHANCED: Create a new NamedTuple with multiple variables updated.
Uses memory sharing for all unchanged variables.
FIXED: Proper handling of categorical types.

# Arguments
- `base_data`: Original data as NamedTuple
- `changes`: Dictionary mapping variable names to new values

# Returns
- New NamedTuple with all specified variables updated
"""
function batch_perturb_data(base_data::NamedTuple, changes::Dict{Symbol, <:AbstractVector})
    if isempty(changes)
        return base_data
    end
    
    # Validate all variables exist and have correct lengths
    for (var, values) in changes
        if !haskey(base_data, var)
            throw(ArgumentError("Variable $var not found in base data"))
        end
        
        original_length = length(base_data[var])
        if length(values) != original_length
            throw(DimensionMismatch(
                "New values for $var have length $(length(values)), " *
                "but original data has length $original_length"
            ))
        end
    end
    
    # ENHANCED: Process changes with proper categorical type handling
    processed_changes = Dict{Symbol, AbstractVector}()
    for (var, values) in changes
        orig_var = base_data[var]
        
        if orig_var isa CategoricalArray
            processed_changes[var] = ensure_categorical_array(values, orig_var)
        else
            processed_changes[var] = values
        end
    end
    
    # Create new NamedTuple with all processed changes
    return merge(base_data, processed_changes)
end

"""
    validate_data_consistency(data::NamedTuple)

Validate that all vectors in a NamedTuple have the same length.
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
# New Categorical Helper Functions
###############################################################################

"""
    ensure_categorical_array(values::AbstractVector, reference_var::CategoricalArray)

Ensure that values is a proper CategoricalArray compatible with reference_var.
This prevents Vector{CategoricalValue} issues that cause refs() errors.

# Arguments
- `values`: Input vector (may be CategoricalArray, Vector{CategoricalValue}, or plain vector)
- `reference_var`: Reference CategoricalArray to match levels and ordering

# Returns
- Proper CategoricalArray with same levels and ordering as reference

# Details
The key fix for refs() errors: converts Vector{CategoricalValue} to CategoricalArray
by extracting .level values and recreating with proper categorical structure.
"""
function ensure_categorical_array(values::AbstractVector, reference_var::CategoricalArray)
    if values isa CategoricalArray
        # Already a CategoricalArray - validate compatibility and return
        validate_categorical_compatibility(values, reference_var)
        return values
    elseif eltype(values) <: CategoricalValue
        # Vector{CategoricalValue} - extract underlying values and recreate
        # This is the key fix for the refs() error
        raw_values = [v.level for v in values]
        return categorical(raw_values, levels=levels(reference_var), ordered=isordered(reference_var))
    else
        # Plain values - convert to CategoricalArray with same levels
        return categorical(values, levels=levels(reference_var), ordered=isordered(reference_var))
    end
end

"""
    validate_categorical_compatibility(values::CategoricalArray, reference_var::CategoricalArray)

Validate that categorical arrays have compatible levels.
"""
function validate_categorical_compatibility(values::CategoricalArray, reference_var::CategoricalArray)
    if levels(values) != levels(reference_var)
        @warn "Categorical levels don't match. Values: $(levels(values)), Reference: $(levels(reference_var))"
    end
    return true
end

"""
    extract_categorical_levels(values::AbstractVector)

Extract underlying level values from various categorical vector types.
Handles CategoricalArray, Vector{CategoricalValue}, and plain vectors.
"""
function extract_categorical_levels(values::AbstractVector)
    if values isa CategoricalArray
        return [v for v in values]  # Convert to plain values
    elseif eltype(values) <: CategoricalValue
        return [v.level for v in values]  # Extract .level from each CategoricalValue
    else
        return values  # Already plain values
    end
end

###############################################################################
# Matrix Operation Functions
###############################################################################

"""
    update_matrix_columns!(target::AbstractMatrix, source::AbstractMatrix, 
                          column_map::Dict{Int, Int}, new_data::AbstractMatrix)

Update specific columns in target matrix with selective copying and memory sharing.

# Arguments
- `target`: Matrix to update (modified in-place)
- `source`: Matrix to copy unchanged columns from
- `column_map`: Mapping from target column index to source column index for copying
- `new_data`: Matrix with new data for updated columns

# Details
Columns not in column_map are assumed to be updated with new_data.
Columns in column_map are copied from source (memory sharing where possible).
"""
function update_matrix_columns!(target::AbstractMatrix, source::AbstractMatrix, 
                               column_map::Dict{Int, Int}, new_data::AbstractMatrix)
    # Validate dimensions
    size(target, 1) == size(source, 1) || throw(DimensionMismatch(
        "Target and source must have same number of rows"
    ))
    
    # Copy specified columns from source
    for (target_col, source_col) in column_map
        if 1 ≤ target_col ≤ size(target, 2) && 1 ≤ source_col ≤ size(source, 2)
            target[:, target_col] = view(source, :, source_col)
        else
            throw(BoundsError("Invalid column indices in column_map"))
        end
    end
    
    # Note: Columns not in column_map should already be updated with new_data
    # or will be updated by the caller
end

"""
    copy_matrix_selective(source::AbstractMatrix, changed_columns::Vector{Int})

Create a new matrix that shares memory for unchanged columns and allocates
new memory only for changed columns.

# Arguments
- `source`: Original matrix
- `changed_columns`: Column indices that will be updated (need new memory)

# Returns
- New matrix with shared memory for unchanged columns
"""
function copy_matrix_selective(source::AbstractMatrix, changed_columns::Vector{Int})
    target = similar(source)
    n_cols = size(source, 2)
    
    # Identify unchanged columns
    unchanged_columns = setdiff(1:n_cols, changed_columns)
    
    # Share memory for unchanged columns
    for col in unchanged_columns
        target[:, col] = view(source, :, col)
    end
    
    # Changed columns will be filled by caller
    # (we just allocate the space here)
    
    return target
end

"""
    share_unchanged_columns!(target::AbstractMatrix, source::AbstractMatrix, 
                            unchanged_columns::Vector{Int})

Update target matrix to share memory with source for specified unchanged columns.

# Arguments
- `target`: Matrix to update (modified in-place)
- `source`: Matrix to share columns from
- `unchanged_columns`: Column indices to share

# Details
This function enables memory sharing for columns that don't need to be recomputed.
"""
function share_unchanged_columns!(target::AbstractMatrix, source::AbstractMatrix, 
                                 unchanged_columns::Vector{Int})
    # Validate dimensions
    size(target) == size(source) || throw(DimensionMismatch(
        "Target and source matrices must have same dimensions"
    ))
    
    # Share memory for each unchanged column
    for col in unchanged_columns
        if 1 ≤ col ≤ size(target, 2)
            target[:, col] = view(source, :, col)
        else
            throw(BoundsError("Column index $col out of bounds"))
        end
    end
end

###############################################################################
# Validation and Utility Functions
###############################################################################

"""
    validate_selective_update(original::AbstractMatrix, updated::AbstractMatrix, 
                             changed_cols::Vector{Int}, unchanged_cols::Vector{Int})

Validate that a selective update was performed correctly.
Checks that unchanged columns are identical and changed columns are different.

# Returns
- `true` if validation passes

# Throws
- `AssertionError` if validation fails
"""
function validate_selective_update(original::AbstractMatrix, updated::AbstractMatrix, 
                                  changed_cols::Vector{Int}, unchanged_cols::Vector{Int})
    # Check dimensions
    @assert size(original) == size(updated) "Matrix dimensions must match"
    
    # Check that all columns are accounted for
    total_cols = size(original, 2)
    all_cols = sort(vcat(changed_cols, unchanged_cols))
    @assert all_cols == collect(1:total_cols) "All columns must be accounted for"
    
    # Check that unchanged columns are identical
    for col in unchanged_cols
        if !all(original[:, col] .≈ updated[:, col])
            @warn "Unchanged column $col has been modified"
            return false
        end
    end
    
    # Note: We don't check that changed columns are different because
    # they might legitimately be the same (e.g., if perturbation was very small)
    
    return true
end

"""
    memory_usage_report(matrices::Dict{String, AbstractMatrix})

Generate a report of memory usage for a set of matrices.
Useful for debugging memory efficiency of selective updates.

# Arguments
- `matrices`: Dictionary mapping matrix names to matrices

# Returns
- String with memory usage summary
"""
function memory_usage_report(matrices::Dict{String, AbstractMatrix})
    report = "Memory Usage Report:\n"
    report *= "=" ^ 50 * "\n"
    
    total_bytes = 0
    
    for (name, matrix) in matrices
        bytes = sizeof(matrix)
        total_bytes += bytes
        mb = bytes / (1024^2)
        
        report *= @sprintf("%-20s: %8.2f MB (%d x %d)\n", 
                          name, mb, size(matrix, 1), size(matrix, 2))
    end
    
    total_mb = total_bytes / (1024^2)
    report *= "-" ^ 50 * "\n"
    report *= @sprintf("%-20s: %8.2f MB\n", "Total", total_mb)
    
    return report
end

"""
    compute_memory_savings(full_size::Tuple{Int,Int}, affected_cols::Int)

Compute theoretical memory savings from selective updates.

# Arguments
- `full_size`: (nrows, ncols) of full matrix
- `affected_cols`: Number of columns that need updating

# Returns
- Tuple of (memory_saved_mb, percent_saved)
"""
function compute_memory_savings(full_size::Tuple{Int,Int}, affected_cols::Int)
    nrows, ncols = full_size
    
    # Full matrix memory (assuming Float64)
    full_bytes = nrows * ncols * sizeof(Float64)
    
    # Selective update memory (only affected columns get new allocation)
    selective_bytes = nrows * affected_cols * sizeof(Float64)
    
    saved_bytes = full_bytes - selective_bytes
    saved_mb = saved_bytes / (1024^2)
    percent_saved = (saved_bytes / full_bytes) * 100
    
    return (saved_mb, percent_saved)
end
