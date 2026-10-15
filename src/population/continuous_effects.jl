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

    # Step 3: Main computation loop. The plain AME path is untouched by the measure
    # machinery; elasticity measures average their unit-level quantity in a separate loop.
    if measure === :effect
        _compute_continuous_marginal_effects!(engine, continuous_requested, rows, scale, weights, total_weight)
    else
        _compute_continuous_measures!(engine, continuous_requested, data_nt, rows, scale,
                                      measure, weights, total_weight)
    end

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
    _total_weight(rows, weights) -> Float64

Normalizer for averaging over `rows`: `length(rows)` when unweighted, otherwise the sum of
the weights of `rows`.

Throws a `MarginsError` when weights are given and none of `rows` has a positive weight.
Such a set of rows (typically a `groups` cell whose observations all have weight zero) has
no weighted average; reporting `0.0 ± 0.0` for it would read as a precisely estimated null.
Categorical effects and predictions raise the same error for the same situation.
"""
function _total_weight(rows, weights)
    isnothing(weights) && return Float64(length(rows))
    s = 0.0
    @inbounds for row in rows
        w = weights[row]
        w > 0 && (s += w)
    end
    s > 0 || _throw_zero_weights()
    return s
end

@noinline _throw_zero_weights(what::AbstractString="marginal effects") = throw(MarginsError(
    "All weights are zero for the observations being averaged (for example, every " *
    "observation in a `groups` cell has weight 0); cannot compute weighted $what. " *
    "Drop that subgroup or give its observations positive weights."))

"""
    _compute_continuous_marginal_effects!(engine, continuous_requested, rows, scale, weights, total_weight)

Process continuous variables using FormulaCompiler primitives with unified weight handling.
Zero allocations through direct use of FormulaCompiler's marginal_effects functions.

# Arguments
- `engine`: MarginsEngine with derivatives, compiled formula, and all needed buffers
- `continuous_requested`: Vector of continuous variable names
- `rows`: Row indices to process
- `scale`: Scale for marginal effects (:response or :link)
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

Type-stable function barrier for link-scale marginal effects computation.
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

        # Link scale: FormulaCompiler computes ∂η/∂x
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
    _compute_me_loop_response!(g_all, Gβ_all, de::ADEvaluator, β, link, batch_ame_values,
                              batch_gradients, fc_var_indices, rows, weights, total_weight)

Response-scale loop specialized for the AD backend. Produces the same AMEs and parameter
gradients as the generic method, but accumulates the gradients without materializing the
per-row `n_params × n_vars` matrix Gβ or making strided writes into `batch_gradients`:

    ∂AME_j/∂β = Σᵢ wᵢ [g'(ηᵢ) Jᵢ[:,j] + g''(ηᵢ) (Jᵢ[:,j]'β) xᵢ] = A[:,j] + B[:,j]

with `A = Σ w g' J` and `B = Σ x (w g'' J'β)'`, both accumulated column-wise.
"""
function _compute_me_loop_response!(
    g_all::AbstractVector{Float64},
    Gβ_all::AbstractMatrix{Float64},
    de::ADEvaluator,
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
    n_vars = length(fc_var_indices)
    core = de.core
    J = Matrix{Float64}(undef, size(core.jacobian_buffer))
    xrow = Vector{Float64}(undef, length(core.xrow_buffer))
    A = zeros(n_params, n_vars)
    B = zeros(n_params, n_vars)
    c = zeros(n_vars)
    _fused_me_rows!(A, B, c, J, xrow, core.compiled_base, core.base_data, de, β, link,
                    batch_ame_values, fc_var_indices, rows, weights, total_weight,
                    1.0 / length(rows))
    @inbounds for k in 1:n_vars
        fc_var_indices[k] == 0 && continue
        for i in 1:n_params
            batch_gradients[k, i] += A[i, k] + B[i, k]
        end
    end
    return nothing
end

# Function barrier for the fused AD loop (concrete types for compiled_base/base_data)
function _fused_me_rows!(A::Matrix{Float64}, B::Matrix{Float64}, c::Vector{Float64},
                         J::Matrix{Float64}, xrow::Vector{Float64}, compiled_base, base_data,
                         de::ADEvaluator, β::AbstractVector{Float64}, link::GLM.Link,
                         batch_ame_values::AbstractVector{Float64},
                         fc_var_indices::AbstractVector{Int},
                         rows::Union{AbstractVector{Int}, UnitRange{Int}},
                         weights::Union{Nothing, AbstractVector{<:Real}},
                         total_weight::Float64, unweighted_weight::Float64)
    n_params = length(β)
    n_vars = length(fc_var_indices)
    @inbounds for row in rows
        w = if isnothing(weights)
            unweighted_weight
        else
            wr = weights[row]
            wr <= 0 && continue
            wr / total_weight
        end

        derivative_modelrow!(J, de, row)      # Jᵢ = ∂xᵢ/∂vars
        compiled_base(xrow, base_data, row)   # xᵢ
        η = dot(β, xrow)
        wg′ = w * _dmu_deta(link, η)
        wg″ = w * _d2mu_deta2(link, η)

        for k in 1:n_vars
            j = fc_var_indices[k]
            if j == 0
                c[k] = 0.0
                continue
            end
            Jβ = 0.0
            @simd for i in 1:n_params
                Jβ += J[i, j] * β[i]
            end
            batch_ame_values[k] += wg′ * Jβ
            c[k] = wg″ * Jβ
            @simd for i in 1:n_params
                A[i, k] += wg′ * J[i, j]
            end
        end
        for k in 1:n_vars
            ck = c[k]
            @simd for i in 1:n_params
                B[i, k] += ck * xrow[i]
            end
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
- `scale::Symbol`: Either `:response` or `:link`
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
    else  # :link
        _compute_me_loop_linear!(g_all, Gβ_all, de, β,
                                batch_ame_values, batch_gradients,
                                fc_var_indices, rows, weights, total_weight)
    end
    return nothing
end

"""
    _compute_continuous_measures!(engine, continuous_requested, data_nt, rows, scale,
                                  measure, weights, total_weight)

Elasticity-type measures for the requested continuous variables, accumulated into
`engine.batch_ame_values` / `engine.batch_gradients`. See
`_accumulate_continuous_measures!` for the definitions.
"""
function _compute_continuous_measures!(engine, continuous_requested, data_nt, rows, scale,
                                       measure, weights, total_weight)
    indices_buf = engine.cont_var_indices_buf
    n_requested = length(continuous_requested)
    for i in 1:n_requested
        indices_buf[i] = engine.var_index_map[continuous_requested[i]]
    end
    xcols = Tuple(getproperty(data_nt, v) for v in continuous_requested)
    _accumulate_continuous_measures!(
        engine.batch_ame_values, engine.batch_gradients, engine.g_all_buf, engine.Gβ_all_buf,
        engine.row_buf, engine.de, engine.compiled, data_nt, engine.β, engine.link,
        scale, measure, view(indices_buf, 1:n_requested), xcols, rows, weights, total_weight
    )
    return nothing
end

"""
    _batch_compute_continuous_measures!(batch_values, batch_gradients, de, compiled, data_nt,
                                        β, link, scale, measure, fc_var_indices, xcols,
                                        rows, weights, total_weight)

Allocating wrapper around `_accumulate_continuous_measures!` for the scenarios/groups path,
which builds its own derivative evaluator per context.
"""
function _batch_compute_continuous_measures!(batch_values, batch_gradients, de, compiled, data_nt,
                                             β, link, scale, measure, fc_var_indices, xcols,
                                             rows, weights, total_weight)
    n_params = length(β)
    g_all = Vector{Float64}(undef, length(de.vars))
    Gβ_all = Matrix{Float64}(undef, n_params, length(de.vars))
    row_buf = Vector{Float64}(undef, length(compiled))
    _accumulate_continuous_measures!(batch_values, batch_gradients, g_all, Gβ_all, row_buf,
                                     de, compiled, data_nt, β, link, scale, measure,
                                     fc_var_indices, xcols, rows, weights, total_weight)
    return nothing
end

"""
    _accumulate_continuous_measures!(batch_values, batch_gradients, g_all, Gβ_all, row_buf,
                                     de, compiled, data_nt, β, link, scale, measure,
                                     fc_var_indices, xcols, rows, weights, total_weight)

Average a unit-level elasticity measure over `rows`, with its delta-method gradient.

With gᵢ = ∂ŷᵢ/∂x (on `scale`), ŷᵢ the prediction on `scale` and xᵢ the variable's value
at row i, the measures are the (weighted) means of

| measure                | unit quantity qᵢ | ∇β qᵢ                                   |
|------------------------|------------------|-----------------------------------------|
| `:elasticity` (eyex)   | gᵢ xᵢ / ŷᵢ       | (xᵢ/ŷᵢ) ∇gᵢ − (gᵢ xᵢ/ŷᵢ²) ∇ŷᵢ          |
| `:semielasticity_dyex` | gᵢ xᵢ            | xᵢ ∇gᵢ                                  |
| `:semielasticity_eydx` | gᵢ / ŷᵢ          | (1/ŷᵢ) ∇gᵢ − (gᵢ/ŷᵢ²) ∇ŷᵢ               |

where ∇ŷᵢ = g⁻¹′(ηᵢ) Xᵢ on the response scale and Xᵢ on the link scale. This is what
Stata's `margins, eyex()/dyex()/eydx()` and R's `marginaleffects::avg_slopes(slope = ...)`
compute (a mean of unit-level ratios, not a ratio of means).

A row with ŷᵢ = 0 makes `:elasticity` and `:semielasticity_eydx` undefined; rather than
return Inf/NaN, this throws a `MarginsError`.
"""
function _accumulate_continuous_measures!(
    batch_values::AbstractVector{Float64}, batch_gradients::AbstractMatrix{Float64},
    g_all::AbstractVector{Float64}, Gβ_all::AbstractMatrix{Float64},
    row_buf::AbstractVector{Float64}, de, compiled, data_nt, β::AbstractVector{Float64},
    link, scale::Symbol, measure::Symbol, fc_var_indices::AbstractVector{Int},
    xcols::Tuple, rows, weights, total_weight::Float64
)
    measure in (:elasticity, :semielasticity_dyex, :semielasticity_eydx) ||
        throw(ArgumentError("Unknown measure: $measure"))
    n_params = length(β)
    n_vars = length(fc_var_indices)
    divides_by_y = measure !== :semielasticity_dyex
    uses_x = measure !== :semielasticity_eydx
    unweighted_weight = 1.0 / length(rows)

    @inbounds for row in rows
        w = if isnothing(weights)
            unweighted_weight
        else
            wr = weights[row]
            wr <= 0 && continue
            wr / total_weight
        end

        if scale === :response
            marginal_effects_mu!(g_all, Gβ_all, de, β, link, row)
        else
            marginal_effects_eta!(g_all, Gβ_all, de, β, row)
        end

        # ŷᵢ and ∇ŷᵢ = dŷ · Xᵢ (only needed for measures that divide by ŷ)
        ŷ = 1.0; dŷ = 0.0
        if divides_by_y
            modelrow!(row_buf, compiled, data_nt, row)
            η = dot(row_buf, β)
            if scale === :response
                ŷ = GLM.linkinv(link, η); dŷ = _dmu_deta(link, η)
            else
                ŷ = η; dŷ = 1.0
            end
            iszero(ŷ) && _throw_zero_prediction(measure, row, scale)
        end

        for k in 1:n_vars
            j = fc_var_indices[k]
            j == 0 && continue
            g = g_all[j]
            xi = uses_x ? Float64(xcols[k][row]) : 1.0
            # qᵢ = a·gᵢ with a = xᵢ/ŷᵢ, xᵢ or 1/ŷᵢ;  ∇qᵢ = a ∇gᵢ − (qᵢ/ŷᵢ) ∇ŷᵢ
            a = divides_by_y ? xi / ŷ : xi
            q = a * g
            batch_values[k] += w * q
            wa = w * a
            for i in 1:n_params
                batch_gradients[k, i] += wa * Gβ_all[i, j]
            end
            if divides_by_y
                c = w * (q / ŷ) * dŷ
                for i in 1:n_params
                    batch_gradients[k, i] -= c * row_buf[i]
                end
            end
        end
    end
    return nothing
end

@noinline _throw_zero_prediction(measure, row, scale) = throw(MarginsError(
    "measure=:$measure divides by the prediction ŷ, which is exactly 0 at row $row " *
    "(a data row, or a reference-grid row for profile margins) " *
    "(scale=:$scale), so the unit-level quantity is undefined there. Use " *
    "measure=:semielasticity_dyex or :effect, or exclude such observations."))
