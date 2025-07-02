
# ------------------------------------------------ public helper -----------

"""
    modelmatrix!(X, rhs_with_schema, data)

Overwrite `X` with the design matrix that `StatsModels.modelmatrix` would
return *without allocating a new matrix*.
"""
function modelmatrix!(
    X::AbstractMatrix,
    rhs,
    data; # data should be a Tables.jl-compatible source, like a DataFrame
)
    # StatsModels.has_schema(rhs) || error("`rhs` has no schema")
    Tables.istable(data) || error("`data` is not Tables-compatible")

    # The schema is already applied to `rhs`, so we pass the new data `tbl`
    # to the workhorse function.
    modelcols!(X, rhs, data)

    return X
end

# ------------------------------------------------ internal work-horse -----

"""
    modelcols!(dest::AbstractMatrix, rhs_with_schema, tbl)

Fill `dest` with the numeric columns for `rhs_with_schema` in place.
This version robustly handles the complex return types from `StatsModels.modelcols`,
which can be a Vector, a Matrix, or a Tuple of arrays.
"""
function modelcols!(
    dest::AbstractMatrix,
    rhs,
    tbl
)
    # Let StatsModels build the columns.
    matrix_parts = StatsModels.modelcols(rhs, tbl)

    # Normalize the result to always be a single matrix.
    final_matrix = if matrix_parts isa Tuple
        # If it's a tuple of vectors/matrices, horizontally concatenate them.
        hcat(matrix_parts...)
    elseif matrix_parts isa AbstractVector
        # If it's a single vector, reshape it into a one-column matrix.
        reshape(matrix_parts, :, 1)
    else
        # Otherwise, it's already a matrix.
        matrix_parts
    end

    # Sanity check the dimensions before copying.
    if size(dest) != size(final_matrix)
        throw(
            DimensionMismatch(
                "Destination matrix is size $(size(dest)), but StatsModels created a matrix of size $(size(final_matrix))"
            )
        )
    end

    # Copy the generated matrix into the destination buffer.
    dest .= final_matrix

    return dest
end

