# population/categorical_effects.jl
# Population-level categorical effects using FormulaCompiler primitives (Phase 4)
#
# This module implements categorical AME computation using:
# - FormulaCompiler's ContrastEvaluator (zero-allocation contrasts)
# - New kernel layer (kernels/categorical.jl)
# - Direct FC primitives (contrast_modelrow!, contrast_gradient!, delta_method_se)

"""
    _compute_categorical_contrasts!(
        results_ame, results_se, gradient_matrix,
        engine, var, rows, scale, backend, contrasts, weights
    ) -> (n_contrasts::Int, contrast_pairs)

Compute categorical contrasts using FormulaCompiler primitives with zero allocations (in-place).

**Phase 4 Optimization**: In-place operation using pre-allocated arrays for zero allocations.

# Arguments
- `results_ame::Vector{Float64}`: Pre-allocated array for AME results (from caller)
- `results_se::Vector{Float64}`: Pre-allocated array for standard errors (from caller)
- `gradient_matrix::Matrix{Float64}`: Pre-allocated matrix for gradients (from caller)
- `engine::MarginsEngine`: Engine with pre-built ContrastEvaluator
- `var::Symbol`: Categorical variable to contrast
- `rows::AbstractVector{Int}`: Rows to average over
- `scale::Symbol`: Scale for effects (:response or :linear)
- `backend::Symbol`: Backend selection (:ad or :fd) - currently unused for categorical
- `contrasts::Symbol`: Contrast type (:baseline or :pairwise)
- `weights::Union{Nothing, Vector{Float64}}`: Optional observation weights

# Returns
- `(n_contrasts::Int, contrast_pairs)`: Number of contrasts computed and the contrast pairs

# Implementation Notes
Zero-allocation architecture (Phase 4 complete):
1. Retrieve ContrastEvaluator from engine (pre-built during engine construction)
2. Generate contrast pairs using existing type-stable ContrastPair system
3. Use pre-allocated arrays passed from caller (no allocation)
4. Call categorical_contrast_ame_batch! with pre-allocated buffers (0 bytes)
5. Return count and pairs (no intermediate allocations)

# Zero-Allocation Guarantee
- **0 bytes allocated** when arrays are pre-sized correctly
- ContrastEvaluator reused from engine (no construction overhead)
- contrast_modelrow! uses pre-allocated engine buffers (0 bytes)
- contrast_gradient! uses pre-allocated engine buffers (0 bytes)
- categorical_contrast_ame_batch! stores results in-place (0 bytes)
- Uses views to avoid array copies

# Performance
- Hot path: **0 allocations** (when FC primitives are 0-alloc)
- Setup cost: O(1) contrast pair generation (minimal, unavoidable)
"""
function _compute_categorical_contrasts!(
    results_ame::Vector{Float64},
    results_se::Vector{Float64},
    gradient_matrix::Matrix{Float64},
    engine::MarginsEngine{L},
    var::Symbol,
    rows,
    scale::Symbol,
    backend::Symbol,
    contrasts::Symbol,
    weights::Union{Vector{Float64}, Nothing}=nothing
) where L

    # Validate that engine has ContrastEvaluator
    if isnothing(engine.contrast)
        error("Engine does not have ContrastEvaluator. Categorical variable $var may not be in engine.vars.")
    end

    # Generate contrast pairs using existing type-stable system
    var_col = getproperty(engine.data_nt, var)
    contrast_pairs = generate_contrast_pairs(var_col, rows, contrasts, engine.model, var, engine.data_nt)

    # Handle trivial case: single level variable
    if length(contrast_pairs) == 1 && contrast_pairs[1].level1 == contrast_pairs[1].level2
        # No contrasts to compute
        return 0, contrast_pairs
    end

    n_contrasts = length(contrast_pairs)

    # Validate array sizes
    if length(results_ame) < n_contrasts || length(results_se) < n_contrasts
        error("Pre-allocated result arrays too small: need $n_contrasts, got ame=$(length(results_ame)), se=$(length(results_se))")
    end
    if size(gradient_matrix, 1) < n_contrasts || size(gradient_matrix, 2) != length(engine.β)
        error("Pre-allocated gradient matrix wrong size: need ($n_contrasts, $(length(engine.β))), got $(size(gradient_matrix))")
    end

    # Zero-allocation computation using views (Phase 4 optimization)
    categorical_contrast_ame_batch!(
        view(results_ame, 1:n_contrasts),
        view(results_se, 1:n_contrasts),
        view(gradient_matrix, 1:n_contrasts, :),
        engine.contrast_buf, engine.contrast_grad_buf, engine.contrast_grad_accum,
        engine.contrast, var, contrast_pairs,
        engine.β, engine.Σ, engine.link,
        rows, weights
    )

    return n_contrasts, contrast_pairs
end

"""
    _process_categorical_variables!(
        results, G, start_idx, engine, categorical_requested,
        data_nt, rows, scale, backend, contrasts, weights
    ) -> next_idx

Process categorical variables using Phase 4 kernel architecture.

**Updated for Phase 4**: Simplified to use ContrastEvaluator from engine.

# Implementation Pipeline
1. Validate engine has ContrastEvaluator (function barrier for type stability)
2. Loop over categorical variables
3. Generate contrast pairs for each variable
4. Call categorical kernel for zero-allocation AME computation
5. Store results in pre-allocated DataFrame

# Performance Benefits
- **Zero allocations** via pre-allocated ContrastEvaluator
- **Type-stable** through concrete engine types
- **Direct FC primitives** eliminate intermediate layers
- **Simplified architecture** reduces complexity

# Error Handling
Provides graceful degradation:
- Missing ContrastEvaluator: Clear error message
- Empty contrast pairs: Silently skip (single-level variables)
- Individual variable errors: Log warning and continue
"""
function _process_categorical_variables!(
    results::DataFrame, G::Matrix, start_idx::Int,
    engine::MarginsEngine, categorical_requested::Vector{Symbol},
    data_nt::NamedTuple, rows, scale::Symbol, backend::Symbol,
    contrasts::Symbol, weights
)
    cont_idx = start_idx

    for var in categorical_requested
        try
            # Process single categorical variable using new kernel
            cont_idx = _process_single_categorical_variable!(
                results, G, cont_idx, engine, var,
                rows, scale, backend, contrasts, weights
            )
        catch e
            @warn "Skipping categorical variable $var due to processing error: $e"
            # Continue processing remaining variables
        end
    end

    return cont_idx
end

"""
    _process_single_categorical_variable!(
        results, G, start_idx, engine, var, rows,
        scale, backend, contrasts, weights
    ) -> next_idx

Process single categorical variable using ContrastEvaluator kernel with zero allocations.

**Phase 4 Optimization**: Pre-allocates result arrays once per variable for zero-allocation computation.

# Implementation Steps
1. Validate engine (function barrier)
2. Estimate maximum contrasts and pre-allocate result arrays
3. Compute contrasts using _compute_categorical_contrasts! (zero-allocation in-place)
4. Store results using _store_categorical_result!

# Zero-Allocation Architecture (Phase 4 Complete)
- **Pre-allocation at outer scope**: Arrays allocated once per variable (O(n_variables) cost)
- **Zero-allocation hot path**: _compute_categorical_contrasts! uses pre-allocated arrays
- **Views for storage**: No array copies when storing results
- **Type stability**: All operations concrete and predictable

# Performance
- Setup cost: O(n_variables) allocations for result arrays
- Hot path: **0 allocations** (when FC primitives are 0-alloc)
"""
function _process_single_categorical_variable!(
    results::DataFrame, G::Matrix, start_idx::Int,
    engine::MarginsEngine{L}, var::Symbol, rows,
    scale::Symbol, backend::Symbol, contrasts::Symbol,
    weights
) where L

    # Validate engine using function barriers
    validate_engine_for_computation(engine, backend)

    # Hoist loop-invariant length call
    n_rows = length(rows)

    # Estimate maximum contrasts for pre-allocation
    var_col = getproperty(engine.data_nt, var)

    # Count unique levels to estimate max contrasts (OPT.md §4.2)
    # Use cached level map from ContrastEvaluator to avoid data access
    if var_col isa AbstractVector{Bool}
        n_levels = 2
    else
        # Find level map for this variable in ContrastEvaluator's cached maps
        # (engine.contrast is guaranteed to exist when processing categorical variables)
        n_levels = nothing
        for level_map in engine.contrast.categorical_level_maps
            # Extract variable symbol from type parameter
            if typeof(level_map).parameters[1] === var
                n_levels = length(level_map.levels)
                break
            end
        end

        # Sanity check (should never fail in normal operation)
        if isnothing(n_levels)
            error("Categorical variable $var not found in ContrastEvaluator level maps. This indicates an internal consistency error.")
        end
    end

    # Estimate max contrasts based on contrast type
    max_contrasts = if contrasts == :baseline
        max(1, n_levels - 1)  # Baseline contrasts: each level minus reference
    elseif contrasts == :pairwise
        max(1, div(n_levels * (n_levels - 1), 2))  # All unique pairs
    else
        n_levels  # Conservative estimate
    end

    # Pre-allocate result arrays (Phase 4 optimization - once per variable)
    results_ame = Vector{Float64}(undef, max_contrasts)
    results_se = Vector{Float64}(undef, max_contrasts)
    gradient_matrix = Matrix{Float64}(undef, max_contrasts, length(engine.β))

    # Zero-allocation computation (Phase 4 complete)
    n_computed, contrast_pairs = _compute_categorical_contrasts!(
        results_ame, results_se, gradient_matrix,
        engine, var, rows, scale, backend, contrasts, weights
    )

    # Handle empty contrasts (single-level variable)
    if n_computed == 0
        return start_idx
    end

    # Store results in DataFrame
    cont_idx = start_idx
    for i in 1:n_computed
        if cont_idx > size(results, 1)
            @warn "Results DataFrame too small for all contrasts; some results truncated"
            break
        end

        _store_categorical_result!(
            results, G, cont_idx, var, contrast_pairs[i],
            results_ame[i],
            view(gradient_matrix, i, :),
            engine.Σ, n_rows
        )
        cont_idx += 1
    end

    return cont_idx
end

"""
    _store_categorical_result!(
        results, G, row_idx, var, pair, ame_val,
        gradient, covariance, n_obs
    )

Store categorical contrast result in pre-allocated DataFrame.

**Unchanged in Phase 4**: Storage logic is independent of computation method.

Zero additional allocations through:
- Pre-allocated DataFrame indexing
- Direct gradient storage with view operations
- Single standard error computation with dot product
"""
function _store_categorical_result!(
    results::DataFrame, G::Matrix, row_idx::Int,
    var::Symbol, pair::ContrastPair, ame_val::Float64,
    gradient::AbstractVector, covariance::Matrix, n_obs::Int
)
    # Store basic result information
    results.variable[row_idx] = string(var)
    results.contrast[row_idx] = "$(pair.level2) - $(pair.level1)"  # Treatment - Baseline format
    results.estimate[row_idx] = ame_val
    results.n[row_idx] = n_obs

    # Store gradient for SE computation
    copyto!(view(G, row_idx, :), gradient)

    # Compute standard error using delta method: SE = √(g'Σg)
    se = sqrt(max(0.0, dot(gradient, covariance, gradient)))
    results.se[row_idx] = se

    return nothing
end

# Note: count_categorical_rows function is defined in core/variable_detection.jl
# Note: generate_contrast_pairs and ContrastPair are defined in computation/scenarios.jl
