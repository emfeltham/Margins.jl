# effects_buffers.jl - Reusable buffer container for zero-allocation marginal effects computation

"""
    EffectsBuffers

Reusable buffer container for zero-allocation marginal effects computation.
Contains pre-allocated arrays for estimates, standard errors, gradients, and metadata.

Fields
------
- `estimates::Vector{Float64}`: Marginal effects estimates
- `standard_errors::Vector{Float64}`: Standard errors via delta method
- `gradients::Matrix{Float64}`: Parameter gradients (n_vars × n_params)
- `var_indices::Vector{Int}`: Variable indices for batch operations
- `row_count::Ref{Int}`: Number of rows processed
- `capacity_vars::Int`: Current capacity for variables
- `capacity_params::Int`: Current capacity for parameters

The buffers are designed to be reused across multiple calls to avoid allocations.
Use `ensure_capacity!` to resize buffers when needed, and `fill!` to zero them
before accumulation.
"""
mutable struct EffectsBuffers
    estimates::Vector{Float64}
    standard_errors::Vector{Float64}
    gradients::Matrix{Float64}
    var_indices::Vector{Int}
    variables::Vector{Symbol}
    row_count::Ref{Int}
    capacity_vars::Int
    capacity_params::Int
end

"""
    EffectsBuffers(n_vars::Int, n_params::Int)

Create EffectsBuffers with initial capacity for `n_vars` variables and `n_params` parameters.
"""
function EffectsBuffers(n_vars::Int, n_params::Int)
    return EffectsBuffers(
        Vector{Float64}(undef, n_vars),
        Vector{Float64}(undef, n_vars),
        Matrix{Float64}(undef, n_vars, n_params),
        Vector{Int}(undef, n_vars),
        Vector{Symbol}(undef, n_vars),
        Ref(0),
        n_vars,
        n_params
    )
end

"""
    EffectsBuffers(engine_vars, β)

Create EffectsBuffers sized for the given engine variables and model coefficients.
"""
function EffectsBuffers(engine_vars, β::Vector{Float64})
    n_vars = length(engine_vars)
    n_params = length(β)
    return EffectsBuffers(n_vars, n_params)
end

"""
    ensure_capacity!(buffers::EffectsBuffers, n_vars::Int, n_params::Int)

Ensure the buffers have at least the specified capacity for variables and parameters.
Resizes arrays if needed, avoiding allocations when capacity is sufficient.

Returns `true` if resizing occurred, `false` if capacity was already sufficient.
"""
function ensure_capacity!(buffers::EffectsBuffers, n_vars::Int, n_params::Int)
    resized = false

    # Check if we need to resize for variables
    if n_vars > buffers.capacity_vars
        resize!(buffers.estimates, n_vars)
        resize!(buffers.standard_errors, n_vars)
        resize!(buffers.var_indices, n_vars)
        resize!(buffers.variables, n_vars)
        buffers.capacity_vars = n_vars
        resized = true
    end

    # Check if we need to resize for parameters (affects gradients matrix)
    if n_vars > size(buffers.gradients, 1) || n_params > size(buffers.gradients, 2)
        new_rows = max(n_vars, size(buffers.gradients, 1))
        new_cols = max(n_params, size(buffers.gradients, 2))

        # Create new matrix with larger dimensions
        new_gradients = Matrix{Float64}(undef, new_rows, new_cols)

        # Copy existing data if any
        old_rows, old_cols = size(buffers.gradients)
        if old_rows > 0 && old_cols > 0
            copy_rows = min(old_rows, new_rows)
            copy_cols = min(old_cols, new_cols)
            new_gradients[1:copy_rows, 1:copy_cols] = buffers.gradients[1:copy_rows, 1:copy_cols]
        end

        buffers.gradients = new_gradients
        buffers.capacity_params = size(new_gradients, 2)
        resized = true
    end

    return resized
end

"""
    reset!(buffers::EffectsBuffers, n_vars::Int)

Reset buffers for a new computation with `n_vars` variables.
Zeros the relevant portions of the arrays and resets row count.
"""
function reset!(buffers::EffectsBuffers, n_vars::Int)
    # Zero only the portions we'll use
    fill!(view(buffers.estimates, 1:n_vars), 0.0)
    fill!(view(buffers.standard_errors, 1:n_vars), 0.0)
    fill!(view(buffers.gradients, 1:n_vars, :), 0.0)

    # Reset metadata
    buffers.row_count[] = 0

    return buffers
end

"""
    get_results_view(buffers::EffectsBuffers, n_vars::Int)

Get views into the buffer arrays for the first `n_vars` variables.
Returns `(estimates_view, standard_errors_view, gradients_view)`.

This avoids copying data when extracting results.
"""
function get_results_view(buffers::EffectsBuffers, n_vars::Int)
    estimates_view = view(buffers.estimates, 1:n_vars)
    standard_errors_view = view(buffers.standard_errors, 1:n_vars)
    gradients_view = view(buffers.gradients, 1:n_vars, :)

    return estimates_view, standard_errors_view, gradients_view
end
