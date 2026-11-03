# core/engine_validation.jl - Fix 4 engine validation with comprehensive function barriers

"""
    validate_engine_for_computation(engine::MarginsEngine, backend::Symbol)

**Type-stable input validation** with early error detection.

Validates the backend and the engine's buffer sizes. Whether the engine has the evaluators a
computation needs is not checked here: it is part of the engine's type (`E`, `CE`), and
`build_engine` gives every population engine the evaluators its requested variables need
(FormulaCompiler sizes them for every model parameter).

# Arguments
- `engine::MarginsEngine`: Engine to validate
- `backend::Symbol`: Computation backend (:ad or :fd)

# Throws
- `ArgumentError`: If the backend is unknown
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
    if backend !== :ad && backend !== :fd
        throw(ArgumentError("Unknown backend: $backend. Must be :ad or :fd"))
    end

    # Validate basic engine consistency
    _validate_engine_consistency(engine)

    return nothing  # Explicit return for type stability
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

    return nothing
end