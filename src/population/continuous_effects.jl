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
- `scale`: Scale for marginal effects (:response or :linear)
- `measure`: Econometric measure (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
- `weights`: Optional observation weights

# Returns
- `Int`: Next available row index in results DataFrame

# Processing Pipeline
1. Engine validation: Uses Fix 4 function barriers for type stability
2. Buffer allocation: Type-stable ComputationBuffers from engine
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
- `:effect`: Raw marginal effects (∂μ/∂x)
- `:elasticity`: (x̄/ȳ) × (∂μ/∂x) - proportional change interpretation
- `:semielasticity_dyex`: x̄ × (∂μ/∂x) - unit change in x, percent change in y
- `:semielasticity_eydx`: (1/ȳ) × (∂μ/∂x) - percent change in x, unit change in y
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

    # Step 2: Compute average response for measure transformations (if needed)
    ȳ, total_weight = _compute_response_mean_if_needed(engine, data_nt, rows, scale, measure, weights)

    # Step 3: Main computation loop - Apply FormulaCompiler pattern for marginal effects (zero allocations)
    _compute_continuous_marginal_effects!(engine, continuous_requested, rows, scale, weights, total_weight)

    # Step 4: Store results using engine's accumulated values
    cont_idx = start_idx
    for var_idx in eachindex(continuous_requested)
        var = continuous_requested[var_idx]
        ame_val = engine.batch_ame_values[var_idx]
        gradient = view(engine.batch_gradients, var_idx, :)

        # Apply measure transformations using extracted function
        (final_val, transform_factor) = apply_measure_transformations(
            ame_val, measure, var,
            data_nt, rows, weights, ȳ, total_weight
        )

        # Transform gradient by same factor
        if transform_factor != 1.0
            gradient .*= transform_factor
        end

        # Store result using extracted function
        store_continuous_result!(results, G, cont_idx, var, final_val, gradient, engine.Σ, n_rows)
        cont_idx += 1
    end

    return cont_idx
end

"""
    _compute_response_mean_if_needed(engine, data_nt, rows, scale, measure, weights) -> (ȳ::Float64, total_weight::Float64)

Compute average response and total weight if needed for measure transformations.
Optimizes by computing only when required (measure !== :effect).

# Arguments
- `engine`: MarginsEngine with compiled formula and parameters
- `data_nt`: Named tuple containing data
- `rows`: Row indices to process
- `scale`: Scale for predictions (:response or :linear)
- `measure`: Econometric measure type
- `weights`: Optional observation weights

# Returns
- `(Float64, Float64)`: Tuple of (average_response, total_weight)

# Optimization Strategy
For `:effect` measure, returns (1.0, weight_sum) to avoid unnecessary computation.
For other measures, computes actual average response using zero-allocation prediction loop.

# Performance
- Conditional computation eliminates unnecessary work for :effect measure
- Single pass through data for response calculation
- Zero allocations through buffer reuse
"""
function _compute_response_mean_if_needed(engine, data_nt, rows, scale, measure, weights)
    # Hoist loop-invariant length call
    n_rows = length(rows)

    # Compute total weight once for all measure types (OPT.md §2.3)
    # Hoisting eliminates duplicate O(n) summation for non-effect measures
    total_weight = if isnothing(weights)
        Float64(n_rows)
    else
        # Zero-allocation weight sum
        s = 0.0
        for row in rows
            s += weights[row]
        end
        s
    end

    # Early return for :effect measure (no response computation needed)
    if measure === :effect
        return (1.0, total_weight)
    end

    # Compute average response for measure transformations
    ȳ = _average_response_over_rows(engine.compiled, engine.row_buf, engine.β, engine.link,
                                   data_nt, rows, scale, weights)
    return (ȳ, total_weight)
end

"""
    _average_response_over_rows(
        compiled, row_buf, β, link, data_nt, rows, scale, weights
    ) -> Float64

Compute average predicted response over specified rows with optional weighting.
Zero allocations through buffer reuse and single-pass computation.

# Arguments
- `compiled`: FormulaCompiler compiled formula
- `row_buf`: Pre-allocated buffer for model row evaluation
- `β`: Model parameter vector
- `link`: Link function for scale transformation
- `data_nt`: Named tuple containing data
- `rows`: Row indices to process
- `scale`: Prediction scale (:response or :linear)
- `weights`: Optional observation weights

# Returns
- `Float64`: Average predicted response across specified rows

# Mathematical Details
- Linear scale: η = Xβ (linear predictor)
- Response scale: μ = g⁻¹(η) where g is the link function
- Weighted average: Σᵢ wᵢ × yᵢ / Σᵢ wᵢ
- Unweighted average: (1/n) × Σᵢ yᵢ

# Performance
- Zero allocations through buffer reuse (row_buf)
- Single pass through specified rows
- Conditional weighting without branching inside hot loop
"""
function _average_response_over_rows(compiled, row_buf, β, link, data_nt, rows, scale, weights)
    # Hoist loop-invariant length call
    n_rows = length(rows)
    response_sum = 0.0

    if isnothing(weights)
        # Unweighted case (@inbounds for validated row indices)
        @inbounds for row in rows
            modelrow!(row_buf, compiled, data_nt, row)
            η = dot(row_buf, β)
            response = scale === :response ? GLM.linkinv(link, η) : η
            response_sum += response
        end
        return response_sum / n_rows
    else
        # Weighted case (@inbounds for validated row indices)
        total_weight = 0.0
        @inbounds for row in rows
            w = weights[row]
            if w > 0
                modelrow!(row_buf, compiled, data_nt, row)
                η = dot(row_buf, β)
                response = scale === :response ? GLM.linkinv(link, η) : η
                response_sum += w * response
                total_weight += w
            end
        end
        return total_weight > 0 ? response_sum / total_weight : 0.0
    end
end

"""
    _compute_continuous_marginal_effects!(engine, continuous_requested, rows, scale, weights, total_weight)

Process continuous variables using FormulaCompiler primitives with unified weight handling.
Zero allocations through direct use of FormulaCompiler's marginal_effects functions.

# Arguments
- `engine`: MarginsEngine with derivatives, compiled formula, and all needed buffers
- `continuous_requested`: Vector of continuous variable names
- `rows`: Row indices to process
- `scale`: Scale for marginal effects (:response or :linear)
- `weights`: Observation weight vector (or `nothing` for unweighted)
- `total_weight`: Sum of all weights (for normalization)

# Implementation
Uses FormulaCompiler primitives for correct, zero-allocation computation:
1. Allocate buffers for ALL continuous vars (FC computes all at once)
2. Dispatch to scale-specific function barrier (type-stable inner loop)
3. For each row:
   - Call marginal_effects_eta! or marginal_effects_mu! (0 bytes, handles chain rule)
   - Extract requested subset from full results
   - Accumulate with proper weighting
4. FormulaCompiler handles: Jacobian, chain rule, parameter gradients

# Performance
- Zero allocations through FormulaCompiler primitives
- Function barrier ensures type-stable inner loop
- Single function call per row replaces ~30 lines of manual computation
- FC's optimized implementations ensure correctness and speed
- Subset extraction is negligible overhead (~10ns per variable)
"""
function _compute_continuous_marginal_effects!(
    engine, continuous_requested, rows, scale, weights, total_weight
)
    # Use pre-allocated buffers from engine (zero allocations)
    g_all = engine.g_all_buf
    Gβ_all = engine.Gβ_all_buf
    indices_buf = engine.cont_var_indices_buf

    # Precompute mapping from requested vars to derivative evaluator indices
    # Note: indices_buf is pre-sized, use only first n_requested elements
    # Use O(1) hash lookup instead of O(n×m) linear search
    n_requested = length(continuous_requested)
    for i in 1:n_requested
        var = continuous_requested[i]
        idx = engine.var_index_map[var]  # Fail-fast: KeyError if var not in map
        indices_buf[i] = idx
    end

    # Extract all engine fields to pass as parameters (avoid struct field access in loop)
    de = engine.de
    β = engine.β
    link = engine.link
    batch_ame_values = engine.batch_ame_values
    batch_gradients = engine.batch_gradients

    # Function barrier: dispatch to scale-specific implementation for type stability
    if scale === :response
        _compute_me_loop_response!(g_all, Gβ_all, de, β, link, batch_ame_values, batch_gradients,
                                   indices_buf, rows, weights, total_weight)
    else
        _compute_me_loop_linear!(g_all, Gβ_all, de, β, batch_ame_values, batch_gradients,
                                indices_buf, rows, weights, total_weight)
    end

    return nothing
end

"""
    _compute_me_loop_linear!(g_all, Gβ_all, de, β, batch_ame_values, batch_gradients,
                            fc_var_indices, rows, weights, total_weight)

Type-stable function barrier for linear scale marginal effects computation.
Ensures compiler can optimize the hot loop without runtime dispatch.
All engine fields passed as explicit parameters to avoid struct field access allocations.
"""
function _compute_me_loop_linear!(
    g_all::AbstractVector{Float64},
    Gβ_all::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{Float64},
    batch_ame_values::AbstractVector{Float64},
    batch_gradients::AbstractMatrix{Float64},
    fc_var_indices::AbstractVector{Int},
    rows::Union{AbstractVector{Int}, UnitRange{Int}},
    weights::Union{Nothing, AbstractVector{<:Real}},
    total_weight::Float64
)
    n_params = length(β)

    # Pre-compute constants outside loop to avoid repeated calls
    n_rows = length(rows)
    n_vars = length(fc_var_indices)
    unweighted_weight = 1.0 / n_rows

    # @inbounds for validated row indices
    @inbounds for row in rows
        # Handle weights conditionally - Julia will specialize on weight types
        weight = if isnothing(weights)
            unweighted_weight
        else
            w = weights[row]
            if w <= 0; continue; end
            w / total_weight
        end

        # Linear scale: FormulaCompiler computes ∂η/∂x
        marginal_effects_eta!(g_all, Gβ_all, de, β, row)

        # Extract requested subset and accumulate
        @inbounds for var_idx in 1:n_vars
            fc_var_idx = fc_var_indices[var_idx]
            fc_var_idx == 0 && continue

            # Accumulate marginal effect
            batch_ame_values[var_idx] += weight * g_all[fc_var_idx]

            # Accumulate parameter gradients using BLAS AXPY (OPT.md §4.3)
            # Provides 2-4x speedup vs manual loop for n_params > 10
            # AXPY: y += α*x where α=weight, x=Gβ_all[:,fc_var_idx], y=batch_gradients[var_idx,:]
            @views BLAS.axpy!(weight, Gβ_all[:, fc_var_idx], batch_gradients[var_idx, :])
        end
    end
    return nothing
end

"""
    _compute_me_loop_response!(g_all, Gβ_all, de, β, link, batch_ame_values, batch_gradients,
                              fc_var_indices, rows, weights, total_weight)

Type-stable function barrier for response scale marginal effects computation.
Ensures compiler can optimize the hot loop without runtime dispatch.
All engine fields passed as explicit parameters to avoid struct field access allocations.
"""
function _compute_me_loop_response!(
    g_all::AbstractVector{Float64},
    Gβ_all::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{Float64},
    link::GLM.Link,
    batch_ame_values::AbstractVector{Float64},
    batch_gradients::AbstractMatrix{Float64},
    fc_var_indices::AbstractVector{Int},
    rows::Union{AbstractVector{Int}, UnitRange{Int}},
    weights::Union{Nothing, AbstractVector{<:Real}},
    total_weight::Float64
)
    n_params = length(β)

    # Pre-compute constants outside loop to avoid repeated calls
    n_rows = length(rows)
    n_vars = length(fc_var_indices)
    unweighted_weight = 1.0 / n_rows

    # @inbounds for validated row indices
    @inbounds for row in rows
        # Handle weights conditionally - Julia will specialize on weight types
        weight = if isnothing(weights)
            unweighted_weight
        else
            w = weights[row]
            if w <= 0; continue; end
            w / total_weight
        end

        # Response scale: FormulaCompiler handles full chain rule via link function
        marginal_effects_mu!(g_all, Gβ_all, de, β, link, row)

        # Extract requested subset and accumulate
        @inbounds for var_idx in 1:n_vars
            fc_var_idx = fc_var_indices[var_idx]
            fc_var_idx == 0 && continue

            # Accumulate marginal effect
            batch_ame_values[var_idx] += weight * g_all[fc_var_idx]

            # Accumulate parameter gradients using BLAS AXPY (OPT.md §4.3)
            # Provides 2-4x speedup vs manual loop for n_params > 10
            # AXPY: y += α*x where α=weight, x=Gβ_all[:,fc_var_idx], y=batch_gradients[var_idx,:]
            @views BLAS.axpy!(weight, Gβ_all[:, fc_var_idx], batch_gradients[var_idx, :])
        end
    end
    return nothing
end

"""
    _batch_compute_continuous_effects!(
        batch_ame_values, batch_gradients, de, β, link, scale,
        fc_var_indices, rows, weights, total_weight
    )

Shared zero-allocation batch computation for continuous marginal effects.
Used by both non-scenarios and scenarios/contexts code paths.

This is the core batch processing loop extracted for reuse. It computes marginal
effects for ALL requested variables in a single pass through the data rows.

# Arguments
- `batch_ame_values::AbstractVector{Float64}`: Pre-allocated vector for AME accumulation (modified in-place)
- `batch_gradients::AbstractMatrix{Float64}`: Pre-allocated matrix for gradient accumulation (modified in-place)
- `de::AbstractDerivativeEvaluator`: Derivative evaluator (built on appropriate data)
- `β::AbstractVector{Float64}`: Model coefficients
- `link::GLM.Link`: Link function for response-scale computation
- `scale::Symbol`: Either `:response` or `:link`/:linear`
- `fc_var_indices::AbstractVector{Int}`: Indices mapping requested vars to de.vars
- `rows::Union{AbstractVector{Int}, UnitRange{Int}}`: Row indices to process
- `weights::Union{Nothing, AbstractVector{<:Real}}`: Optional observation weights
- `total_weight::Float64`: Sum of weights for normalization

# Performance
- Zero allocations in hot loop (g_all, Gβ_all allocated once outside loop)
- Single pass through rows computes ALL variables
- O(N × n_vars) instead of O(N × n_vars²)

# Usage
Called by both:
- `_compute_continuous_marginal_effects!` (non-scenarios path)
- Context effects computation (scenarios path)
"""
function _batch_compute_continuous_effects!(
    batch_ame_values::AbstractVector{Float64},
    batch_gradients::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{Float64},
    link::GLM.Link,
    scale::Symbol,
    fc_var_indices::AbstractVector{Int},
    rows::Union{AbstractVector{Int}, UnitRange{Int}},
    weights::Union{Nothing, AbstractVector{<:Real}},
    total_weight::Float64
)
    # Allocate buffers for marginal_effects call (done once per context, not per variable)
    # Hoist loop-invariant length calls
    n_de_vars = length(de.vars)
    n_params = length(β)

    g_all = Vector{Float64}(undef, n_de_vars)
    Gβ_all = Matrix{Float64}(undef, n_params, n_de_vars)

    # Dispatch to scale-specific optimized loop
    if scale === :response
        _compute_me_loop_response!(g_all, Gβ_all, de, β, link,
                                   batch_ame_values, batch_gradients,
                                   fc_var_indices, rows, weights, total_weight)
    else  # :link or :linear
        _compute_me_loop_linear!(g_all, Gβ_all, de, β,
                                batch_ame_values, batch_gradients,
                                fc_var_indices, rows, weights, total_weight)
    end
    return nothing
end

"""
    apply_measure_transformations(
        ame_val::Float64, measure::Symbol, var::Symbol,
        data_nt::NamedTuple, rows, weights, ȳ::Float64,
        total_weight::Float64
    ) -> (Float64, Float64)

Apply econometric measure transformations to marginal effects.
Handles elasticity, semi-elasticity transformations with proper gradient scaling.

# Arguments
- `ame_val`: Raw marginal effect value
- `measure`: Econometric measure type
- `var`: Variable name for mean calculation
- `data_nt`: Named tuple containing data
- `rows`: Row indices to process
- `weights`: Optional observation weights
- `ȳ`: Average response value
- `total_weight`: Total weight for normalization

# Returns
- `(Float64, Float64)`: Tuple of (transformed_effect, gradient_transform_factor)

# Supported Measures
- `:effect`: No transformation (∂μ/∂x)
- `:elasticity`: (x̄/ȳ) × (∂μ/∂x) - percentage change interpretation
- `:semielasticity_dyex`: x̄ × (∂μ/∂x) - unit change in x, percent change in y
- `:semielasticity_eydx`: (1/ȳ) × (∂μ/∂x) - percent change in x, unit change in y

# Mathematical Foundation
Econometric measures provide different interpretations of marginal effects:
- Effect: Raw derivative ∂μ/∂x
- Elasticity: (∂μ/∂x) × (x/μ) ≈ percentage change in μ per 1% change in x
- Semi-elasticity (dyex): (∂μ/∂x) × x ≈ percentage change in μ per unit change in x
- Semi-elasticity (eydx): (∂μ/∂x) × (1/μ) ≈ unit change in μ per 1% change in x

# Performance
- Single pass through variable data for mean calculation
- Conditional computation based on weights
- Transform factors applied consistently to effects and gradients
"""
function apply_measure_transformations(
    ame_val::Float64, measure::Symbol, var::Symbol,
    data_nt::NamedTuple, rows, weights, ȳ::Float64,
    total_weight::Float64
)
    if measure === :effect
        return (ame_val, 1.0)
    end

    # Hoist loop-invariant length call
    n_rows = length(rows)

    # Compute variable mean for transformations
    xcol = getproperty(data_nt, var)
    x̄ = if isnothing(weights)
        # Zero-allocation manual loop instead of generator
        s = 0.0
        @inbounds for row in rows
            s += float(xcol[row])
        end
        s / n_rows
    else
        # Zero-allocation weighted sum
        s = 0.0
        @inbounds for row in rows
            s += weights[row] * float(xcol[row])
        end
        s / total_weight
    end

    # Apply transformations
    if measure === :elasticity
        transform_factor = x̄ / ȳ
        return (transform_factor * ame_val, transform_factor)
    elseif measure === :semielasticity_dyex
        transform_factor = x̄
        return (transform_factor * ame_val, transform_factor)
    elseif measure === :semielasticity_eydx
        transform_factor = 1.0 / ȳ
        return (transform_factor * ame_val, transform_factor)
    else
        throw(ArgumentError("Unknown measure: $measure"))
    end
end
