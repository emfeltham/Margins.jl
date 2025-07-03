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
Update derivative matrix efficiently with comprehensive type handling
"""
function update_derivatives!(Xdx, col_range, base_result, perturbed_result, inv_h)
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
