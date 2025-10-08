# core/buffer_management.jl - Centralized buffer allocation and management with type barriers

"""
    ComputationBuffers{T}

Type-stable buffer container for marginal effects computation.
Eliminates Union types and provides compile-time guarantees for buffer access.
"""
struct ComputationBuffers{T}
    jacobian_buffer::Matrix{T}
    xrow_buffer::Vector{T}
    ame_values::AbstractVector{T}
    gradients::AbstractMatrix{T}
    result_storage::Matrix{T}
end

# CategoricalBuffers removed - obsolete DataScenario-based buffer type
# Categorical effects now use ContrastEvaluator kernel system with direct buffer access

"""
    ProfileBuffers{T}

Type-stable buffer container for profile margins computation.
Provides function barriers for profile-specific buffer access.
"""
struct ProfileBuffers{T}
    marginal_effects_buffer::AbstractVector{T}
    gradient_buffer::Vector{T}
    row_buffer::Vector{T}
    result_buffer::Vector{T}
end

"""
    get_computation_buffers(engine::MarginsEngine{L,U,HasDerivatives}, n_vars::Int, n_params::Int) -> ComputationBuffers

Extract type-stable computation buffers from engine with size validation.
Provides function barrier for buffer access, eliminating Union type instability.

# Arguments
- `engine`: MarginsEngine with derivative support
- `n_vars`: Number of continuous variables to process
- `n_params`: Number of model parameters

# Returns
- `ComputationBuffers`: Type-stable buffer container

# Function Barrier Benefits
- Compiler knows engine.de exists (HasDerivatives type parameter)
- All buffer access is type-stable within this function
- Size validation prevents runtime bounds errors
"""
function get_computation_buffers(engine::MarginsEngine{L,U,HasDerivatives}, n_vars::Int, n_params::Int) where {L,U}
    # Validate buffer capacity (function barrier ensures type stability)
    if n_vars > length(engine.batch_ame_values) || n_params > size(engine.batch_gradients, 2)
        throw(ArgumentError("Engine batch buffers too small for $n_vars variables, $n_params parameters. Consider rebuilding engine."))
    end

    # Extract buffers with type stability
    return ComputationBuffers(
        engine.de.jacobian_buffer,                              # Type: Matrix{Float64}
        engine.de.xrow_buffer,                                  # Type: Vector{Float64}
        view(engine.batch_ame_values, 1:n_vars),               # Type: SubArray{Float64, 1, ...}
        view(engine.batch_gradients, 1:n_vars, 1:n_params),   # Type: SubArray{Float64, 2, ...}
        Matrix{Float64}(undef, n_vars, n_params)               # Type: Matrix{Float64}
    )
end

# get_categorical_buffers removed - obsolete DataScenario-based buffer accessor
# Categorical effects now access engine buffers directly (contrast_buf, contrast_grad_buf, etc.)

"""
    get_profile_buffers(engine::MarginsEngine{L,U,HasDerivatives}, max_vars::Int) -> ProfileBuffers

Extract type-stable profile computation buffers with validation.
Provides function barrier for buffer access, ensuring type stability.
"""
function get_profile_buffers(engine::MarginsEngine{L,U,HasDerivatives}, max_vars::Int) where {L,U}
    # Validate buffer capacity (function barrier ensures type stability)
    if isnothing(engine.de)
        throw(ArgumentError("Profile margins require derivative evaluator. Rebuild engine with HasDerivatives."))
    end

    if length(engine.g_buf) < max_vars
        throw(ArgumentError("Engine g_buf too small for $max_vars variables. Consider using ProfileUsage for larger buffers."))
    end

    return ProfileBuffers(
        view(engine.g_buf, 1:max_vars),           # Type: SubArray{Float64, 1, ...}
        engine.gβ_accumulator,                    # Type: Vector{Float64}
        engine.row_buf,                           # Type: Vector{Float64}
        Vector{Float64}(undef, length(engine.β))  # Type: Vector{Float64}
    )
end

"""
    prepare_computation_buffers!(buffers::ComputationBuffers)

Initialize computation buffers to zero state.
Ensures clean starting state for accumulation operations.
"""
function prepare_computation_buffers!(buffers::ComputationBuffers)
    fill!(buffers.ame_values, 0.0)
    fill!(buffers.gradients, 0.0)
    return nothing
end

# verify_and_repair_engine_buffers! functions moved to engine/core.jl
# to avoid duplication and maintain single source of truth