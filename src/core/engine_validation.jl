# core/engine_validation.jl - Fix 4 engine validation with comprehensive function barriers

using FormulaCompiler: AbstractDerivativeEvaluator

"""
    validate_engine_for_computation(engine::MarginsEngine, backend::Symbol)

**Type-stable input validation** with early error detection.

Validates that engine has required components for the requested computation backend.
Uses function barrier pattern to ensure type stability and fail-fast behavior.

# Arguments
- `engine::MarginsEngine`: Engine to validate
- `backend::Symbol`: Computation backend (:ad or :fd)

# Throws
- `ArgumentError`: If engine is missing required components
- `ArgumentError`: If buffers are incorrectly sized

# Examples
```julia
validate_engine_for_computation(engine, :ad)  # Validates AD requirements
validate_engine_for_computation(engine, :fd)  # Validates FD requirements
```

This function serves as a **function barrier** - the compiler can optimize
the remaining computation knowing that all preconditions are satisfied.
"""
function validate_engine_for_computation(engine::MarginsEngine, backend::Symbol)
    # Early validation with clear error messages
    if backend === :ad
        # AD backend only requires derivative evaluator if we have continuous variables
        if !isempty(engine.continuous_vars) && isnothing(engine.de)
            throw(ArgumentError("AD backend requires derivative evaluator for continuous variables. Rebuild engine with HasDerivatives."))
        end

        # Type-stable buffer validation (function barrier) - only if we have derivatives
        if !isnothing(engine.de)
            _validate_derivative_buffers(engine.de, length(engine.β))
        end
    elseif backend === :fd
        # FD backend only requires continuous variables if we're actually doing finite differences
        # Categorical-only models can use FD backend for scenario-based contrasts
        if !isempty(engine.continuous_vars) && isnothing(engine.de)
            throw(ArgumentError("FD backend requires derivative evaluator for continuous variables."))
        end
    else
        throw(ArgumentError("Unknown backend: $backend. Must be :ad or :fd"))
    end

    # Validate basic engine consistency
    _validate_engine_consistency(engine)

    return nothing  # Explicit return for type stability
end

"""
    _validate_derivative_buffers(de::AbstractDerivativeEvaluator, n_params::Int)

**Function barrier**: Type-stable buffer validation for derivative evaluators.

The compiler knows `de` is not `nothing` when this function is called,
enabling aggressive optimization of subsequent derivative operations.

# Arguments
- `de::AbstractDerivativeEvaluator`: Derivative evaluator (known to be non-nothing)
- `n_params::Int`: Required buffer size (number of model parameters)

# Throws
- `ArgumentError`: If buffers are too small or incorrectly configured
"""
function _validate_derivative_buffers(de::AbstractDerivativeEvaluator, n_params::Int)
    # Type-stable: compiler knows de isa AbstractDerivativeEvaluator
    # Use common buffer fields that both AD and FD evaluators have
    if length(de.xrow_buffer) < n_params
        throw(ArgumentError("Derivative buffer xrow_buffer too small: $(length(de.xrow_buffer)) < $n_params"))
    end

    # Validate Jacobian buffer dimensions (both evaluators have this)
    jacobian_rows, jacobian_cols = size(de.jacobian_buffer)
    if jacobian_rows < n_params
        throw(ArgumentError("Jacobian buffer has insufficient rows: $jacobian_rows < $n_params"))
    end

    if jacobian_cols < length(de.vars)
        throw(ArgumentError("Jacobian buffer has insufficient columns: $jacobian_cols < $(length(de.vars))"))
    end

    return nothing
end

"""
    _validate_engine_consistency(engine::MarginsEngine)

**Type-stable consistency validation** for engine internal state.

# Arguments
- `engine::MarginsEngine`: Engine to validate

# Throws
- `ArgumentError`: If engine has inconsistent internal state
"""
function _validate_engine_consistency(engine::MarginsEngine)
    # Validate buffer sizes match model parameters
    if length(engine.gβ_accumulator) != length(engine.β)
        throw(ArgumentError("Gradient accumulator size mismatch: $(length(engine.gβ_accumulator)) != $(length(engine.β))"))
    end

    if length(engine.row_buf) != length(engine.compiled)
        throw(ArgumentError("Row buffer size mismatch: $(length(engine.row_buf)) != $(length(engine.compiled))"))
    end

    # Validate batch buffers for population operations
    if size(engine.batch_gradients, 2) != length(engine.β)
        throw(ArgumentError("Batch gradient buffer has wrong parameter count: $(size(engine.batch_gradients, 2)) != $(length(engine.β))"))
    end

    # Validate deta_dx_buf size matches derivative evaluator (if present)
    if engine.de !== nothing && length(engine.deta_dx_buf) != length(engine.de.vars)
        throw(ArgumentError("deta_dx_buf size mismatch: $(length(engine.deta_dx_buf)) != $(length(engine.de.vars))"))
    end

    return nothing
end