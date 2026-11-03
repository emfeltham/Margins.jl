# population/continuous_effects.jl
# Continuous variable processing for population effects

"""
    _ame_calculate(engine, data_nt, scale, backend, measure, contrasts, weights)

Main entry point for population-level marginal effects computation.
Uses decomposed functions from the modular architecture.

use `weights = nothing` for standard unweighted scenario
"""
function _ame_calculate(
    engine::MarginsEngine{L,U,D,C}, data_nt::NamedTuple,
    scale::Symbol, backend::Symbol, measure::Symbol,
    contrasts::Symbol, weights
) where {L,U,D,C}

    # Step 1: Validate engine using new validation system
    validate_engine_for_computation(engine, backend)

    # Step 2: Get variable lists (use cached lists from engine - zero allocation)
    rows = 1:length(first(values(data_nt)))

    # Use pre-computed continuous/categorical lists from engine
    # These were already filtered during engine construction - no recomputation needed
    continuous_requested = engine.continuous_vars
    categorical_requested = engine.categorical_vars

    # Step 3: Calculate result size and allocate
    n_continuous = length(continuous_requested)
    n_categorical = count_categorical_rows(categorical_requested, engine, contrasts)
    total_terms = n_continuous + n_categorical
    n_obs = length(rows)
    n_params = length(engine.β)

    # Step 4: Pre-allocate results using modular function
    results, G = build_results_dataframe(total_terms, n_obs, n_params)

    # Step 5: Process continuous variables
    next_idx = 1
    if !isempty(continuous_requested)
        next_idx = _process_continuous_variables!(
            results, G, next_idx, engine, continuous_requested,
            data_nt, rows, scale, measure, weights
        )
    end

    # Step 6: Process categorical variables
    if !isempty(categorical_requested)
        _process_categorical_variables!(
            results, G, next_idx, engine, categorical_requested,
            data_nt, rows, scale, backend, contrasts, weights
        )
    end

    return (results, G)
end

"""
    _process_continuous_variables!(
        results, G, start_idx, engine, continuous_requested,
        data_nt, rows, scale, measure, weights
    ) -> next_idx

Process continuous variables using decomposed functions with proper separation of concerns.
Uses type-stable buffer management and focused computation kernels.

# Arguments
- `results`: Pre-allocated results DataFrame (modified in-place)
- `G`: Pre-allocated gradient matrix (modified in-place)
- `start_idx`: Starting row index in results DataFrame
- `engine`: MarginsEngine with model parameters and derivatives (requires HasDerivatives)
- `continuous_requested`: Vector of continuous variable names to process
- `data_nt`: Named tuple containing data
- `rows`: Row indices to process
- `scale`: Scale for marginal effects (:response or :link)
- `measure`: Econometric measure (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
- `weights`: Optional observation weights

# Returns
- `Int`: Next available row index in results DataFrame

# Processing Pipeline
1. Engine validation: Uses Fix 4 function barriers for type stability
2. Buffer access: Reuses the engine's pre-allocated batch buffers
3. Response computation: Compute average response if needed for measure transformations
4. Row processing: Unweighted or weighted processing using extracted kernels
5. Result storage: Apply measure transformations and store with standard errors

# Performance Benefits
- Type-stable buffer management eliminates Union type penalties
- Function barriers ensure compiler optimization
- Pre-allocated structures prevent allocation during computation
- Focused kernels enable @fastmath and @inbounds optimizations
- Conditional response computation optimizes for :effect measure

# Supported Measures
Each measure is the (weighted) average over `rows` of a unit-level quantity, where
gᵢ = ∂ŷᵢ/∂x and ŷᵢ is the prediction on `scale` — the definition used by Stata's
`margins, eyex()/dyex()/eydx()` and R's `avg_slopes(slope = ...)`:
- `:effect`: mean(gᵢ)
- `:elasticity` (eyex): mean(gᵢ xᵢ / ŷᵢ)
- `:semielasticity_dyex` (dy/d ln x): mean(gᵢ xᵢ)
- `:semielasticity_eydx` (d ln y/dx): mean(gᵢ / ŷᵢ)
"""
function _process_continuous_variables!(
    results::DataFrame, G::Matrix, start_idx::Int,
    engine::MarginsEngine{L,U,HasDerivatives,C},
    continuous_requested::Vector{Symbol},
    data_nt::NamedTuple, rows, scale::Symbol,
    measure::Symbol, weights
) where {L,U,C}

    if isempty(continuous_requested)
        return start_idx
    end

    # Hoist loop-invariant length calls
    n_rows = length(rows)

    # Step 1: Initialize engine's pre-allocated buffers
    fill!(engine.batch_ame_values, 0.0)
    fill!(engine.batch_gradients, 0.0)

    # Step 2: Total weight of the rows being averaged (errors if every weight is zero)
    total_weight = _total_weight(rows, weights)

    # Step 3: One pass over the rows for all variables (the continuous accumulator)
    _compute_continuous_effects!(engine, continuous_requested, data_nt, rows, scale,
                                 measure, weights, total_weight)

    # Step 4: Store results using engine's accumulated values
    cont_idx = start_idx
    for var_idx in eachindex(continuous_requested)
        var = continuous_requested[var_idx]
        final_val = engine.batch_ame_values[var_idx]
        gradient = view(engine.batch_gradients, var_idx, :)

        # Store result using extracted function
        store_continuous_result!(results, G, cont_idx, var, final_val, gradient, engine.Σ, n_rows)
        cont_idx += 1
    end

    return cont_idx
end

"""
    _total_weight(rows, weights, what="marginal effects") -> Float64

Normalizer for averaging over `rows`: `length(rows)` when unweighted, otherwise the sum of
the weights of `rows`.

Throws a `MarginsError` when weights are given and none of `rows` has a positive weight.
Such a set of rows (typically a `groups` cell whose observations all have weight zero) has
no weighted average; reporting `0.0 ± 0.0` for it would read as a precisely estimated null.
Categorical effects and predictions raise the same error for the same situation.
"""
function _total_weight(rows, weights, what::AbstractString="marginal effects")
    isnothing(weights) && return Float64(length(rows))
    s = 0.0
    @inbounds for row in rows
        w = weights[row]
        w > 0 && (s += w)
    end
    s > 0 || _throw_zero_weights(what)
    return s
end

@noinline _throw_zero_weights(what::AbstractString="marginal effects") = throw(MarginsError(
    "All weights are zero for the observations being averaged (for example, every " *
    "observation in a `groups` cell has weight 0); cannot compute weighted $what. " *
    "Drop that subgroup or give its observations positive weights."))

"""
    _compute_continuous_effects!(engine, continuous_requested, data_nt, rows, scale, measure,
                                 weights, total_weight)

Accumulate the measure for the requested continuous variables into
`engine.batch_ame_values` / `engine.batch_gradients` with [`accumulate_continuous!`](@ref),
using the engine's derivative evaluator and workspace. Allocation-free per row.
"""
function _compute_continuous_effects!(engine::MarginsEngine, continuous_requested::Vector{Symbol},
                                      data_nt::NamedTuple, rows, scale::Symbol, measure::Symbol,
                                      weights, total_weight::Float64)
    # Indices of the requested variables in de.vars. The buffer is sized to the request (a
    # plain Vector rather than a view, so the dynamic dispatch on the abstractly typed
    # `engine.de` below does not box it).
    indices_buf = resize!(engine.cont_var_indices_buf, length(continuous_requested))
    for i in eachindex(continuous_requested)
        indices_buf[i] = engine.var_index_map[continuous_requested[i]]  # KeyError if absent
    end
    # Measures that multiply by x read the variables' data columns
    xcols = measure === :effect ? () : Tuple(getproperty(data_nt, v) for v in continuous_requested)
    _accumulate_measure!(engine.batch_ame_values, engine.batch_gradients, engine.me_ws, engine.de,
                         engine.β, scale, engine.link, measure,
                         indices_buf, xcols, rows, weights, total_weight)
    return nothing
end
