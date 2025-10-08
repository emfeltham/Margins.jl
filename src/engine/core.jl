# engine/core.jl - MarginsEngine struct and construction logic

"""
    MarginsUsage

Abstract type for specifying the usage pattern of a MarginsEngine.
Different usage patterns require different buffer sizing strategies for optimal performance.

See also: [`PopulationUsage`](@ref), [`ProfileUsage`](@ref)
"""
abstract type MarginsUsage end

"""
    DerivativeSupport

Abstract type for specifying whether a MarginsEngine supports derivative computation.
Enables compile-time specialization and eliminates Union type overhead.

See also: [`HasDerivatives`](@ref), [`NoDerivatives`](@ref)
"""
abstract type DerivativeSupport end

"""
    HasDerivatives <: DerivativeSupport

Indicates that the MarginsEngine has derivative computation capability.
Used for engines that need to compute marginal effects for continuous variables.

Provides:
- `de::AbstractDerivativeEvaluator` field (abstract type)
- Compile-time dispatch for derivative-based computations
- Type safety: prevents calling derivative functions on engines without support
"""
struct HasDerivatives <: DerivativeSupport end

"""
    NoDerivatives <: DerivativeSupport

Indicates that the MarginsEngine has no derivative computation capability.
Used for engines that only handle categorical/boolean variables or predictions.

Provides:
- No derivative evaluator field
- Minimal memory footprint
- Compile-time dispatch for non-derivative computations
- Type safety: compile-time error if derivative functions are called
"""
struct NoDerivatives <: DerivativeSupport end

"""
    PopulationUsage <: MarginsUsage

Usage pattern for population margins (AME, AAP).

Optimizes for:
- Minimal memory footprint (g_buf sized only for FormulaCompiler continuous variables)
- Zero allocation fallbacks for FormulaCompiler derivative operations
- Efficient row-wise processing across sample data

Used by `population_margins()` function.
"""
struct PopulationUsage <: MarginsUsage end

"""
    ProfileUsage <: MarginsUsage

Usage pattern for profile margins (MEM, APM, MER, APR).

Optimizes for:
- Larger g_buf to handle multiple profiles without allocation fallbacks
- Zero allocation fallbacks for typical reference grid sizes (up to 100 profiles)
- Efficient SE computation across profile grids

Used by `profile_margins()` function.
"""
struct ProfileUsage <: MarginsUsage end

"""
    MarginsEngine{L<:GLM.Link, U<:MarginsUsage, D<:DerivativeSupport}

Zero-allocation engine for marginal effects computation, optimized for specific usage patterns
and derivative computation requirements.

Built on jl with usage-specific pre-allocated buffers for maximum performance.

Type Parameters:
- `L<:GLM.Link`: GLM link function type for the underlying model
- `U<:MarginsUsage`: Usage pattern determining buffer sizing strategy
- `D<:DerivativeSupport`: Derivative computation capability (HasDerivatives or NoDerivatives)

Fields:
- `compiled::UnifiedCompiled`: Pre-compiled FormulaCompiler formula
- `de::Union{AbstractDerivativeEvaluator, Nothing}`: Derivative evaluator (concrete when D <: HasDerivatives, Nothing when D <: NoDerivatives)
- `contrast::Union{ContrastEvaluator, Nothing}`: Categorical contrast evaluator for discrete effects
- `g_buf::Vector{Float64}`: Usage-optimized buffer for marginal effects results
- `gβ_accumulator::Vector{Float64}`: Pre-allocated buffer for AME gradient accumulation
- `η_buf::Vector{Float64}`: Pre-allocated buffer for linear predictor computations
- `row_buf::Vector{Float64}`: Pre-allocated buffer for model row (design matrix row) computations
- `contrast_buf::Vector{Float64}`: Pre-allocated buffer for categorical contrast vectors (Phase 4)
- `contrast_grad_buf::Vector{Float64}`: Pre-allocated buffer for per-row categorical gradients (Phase 4)
- `contrast_grad_accum::Vector{Float64}`: Pre-allocated buffer for accumulated categorical gradients (Phase 4)
- `continuous_vars::Vector{Symbol}`: Detected continuous variables
- `categorical_vars::Vector{Symbol}`: Detected categorical variables
- `model::Any`: Reference to original model
- `β::Vector{Float64}`: Model coefficients
- `Σ::Matrix{Float64}`: Model covariance matrix
- `link::L`: GLM link function
- `vars::Vector{Symbol}`: Variables for analysis
- `data_nt::NamedTuple`: Reference data for scenarios/refgrids

# Buffer Sizing Strategy
- `PopulationUsage`: Minimal g_buf (sized for FormulaCompiler continuous variables only)
- `ProfileUsage`: Larger g_buf (sized for typical profile counts ~100 to avoid allocation fallbacks)

# Derivative Support Strategy
- `HasDerivatives`: de field contains AbstractDerivativeEvaluator for continuous variable computation
- `NoDerivatives`: de field is nothing, minimal memory footprint for categorical-only analysis

# Examples
```julia
# Population margins engine with derivatives (continuous variables)
engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:x1, :x2], vcov)

# Profile margins engine without derivatives (categorical-only)  
engine = build_engine(ProfileUsage, NoDerivatives, model, data_nt, [:category], vcov)
```
"""
struct MarginsEngine{L<:GLM.Link, U<:MarginsUsage, D<:DerivativeSupport, C<:UnifiedCompiled}
    # FormulaCompiler components (pre-compiled)
    compiled::C
    de::Union{AbstractDerivativeEvaluator, Nothing} # Type-parameterized access via dispatch

    # Variable index mapping
    # Pre-computed mapping from variable Symbol to index in de.vars
    # Eliminates O(n×m) linear searches in favor of O(1) hash lookups
    var_index_map::Dict{Symbol, Int}

    contrast::Union{ContrastEvaluator, Nothing} # Categorical contrast evaluator

    # Usage-optimized buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Usage-optimized marginal effects results buffer
    gβ_accumulator::Vector{Float64}     # AME gradient accumulation
    η_buf::Vector{Float64}              # Linear predictor computation buffer
    row_buf::Vector{Float64}            # Model row (design matrix row) buffer

    # Batch operation buffers (for zero-allocation batch marginal effects)
    batch_ame_values::Vector{Float64}   # Averaged marginal effects accumulator
    batch_gradients::Matrix{Float64}    # Parameter gradients accumulator (n_vars × n_params)
    batch_var_indices::Vector{Int}      # Variable indices scratch buffer
    deta_dx_buf::Vector{Float64}        # Marginal effects on linear predictor scale
    cont_var_indices_buf::Vector{Int}   # Scratch mapping for continuous variable indices

    # Continuous effects buffers (for zero-allocation population marginal effects)
    g_all_buf::Vector{Float64}          # Buffer for all continuous variable marginal effects
    Gβ_all_buf::Matrix{Float64}         # Buffer for all continuous variable parameter gradients (n_params × n_de_vars)

    # Categorical effects buffers (Phase 4 zero-allocation optimization)
    contrast_buf::Vector{Float64}           # Contrast vector buffer (length n_params)
    contrast_grad_buf::Vector{Float64}      # Per-row gradient buffer (length n_params)
    contrast_grad_accum::Vector{Float64}    # Accumulated gradient across rows (length n_params)

    # Phase 2.1: Persistent EffectsBuffers for zero-allocation population_margins_raw!
    effects_buffers::EffectsBuffers  # Reusable buffer container for raw API

    # Cache continuous and categorical variable metadata
    continuous_vars::Vector{Symbol}
    categorical_vars::Vector{Symbol}

    # Model parameters
    model::Any # Reference to original model
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple # Reference for scenarios/refgrids
end

"""
    build_engine(usage::Type{U}, deriv::Type{D}, model, data_nt, vars, vcov) where {U<:MarginsUsage, D<:DerivativeSupport} -> MarginsEngine

Construct zero-allocation margins engine with usage-specific optimization and derivative support.

# Arguments
- `usage::Type{<:MarginsUsage}`: Usage pattern (PopulationUsage or ProfileUsage)
- `deriv::Type{<:DerivativeSupport}`: Derivative support (HasDerivatives or NoDerivatives)
- `model`: Fitted statistical model (GLM.jl, MLJ.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format from Tables.jl
- `vars::Vector{Symbol}`: Variables to analyze for marginal effects
- `vcov`: Covariance estimator (function or CovarianceMatrices estimator)

# Returns
- `MarginsEngine{L, U, D}`: Pre-compiled engine with usage and derivative support optimization

# Usage-Specific Buffer Sizing Strategy
- **PopulationUsage**: Minimal g_buf sized for FormulaCompiler continuous variables only
  - Optimizes for minimal memory footprint
  - Eliminates allocation fallbacks for derivative computation
  - Perfect for AME/AAP operations across sample data

- **ProfileUsage**: Larger g_buf sized for typical profile counts (~100)  
  - Optimizes for multiple profiles without allocation fallbacks
  - Handles most reference grids without buffer reallocations
  - Perfect for MEM/APM/MER/APR operations across profile grids

# Derivative Support Strategy
- **HasDerivatives**: Includes AbstractDerivativeEvaluator for continuous variable computation
  - `de` field contains concrete AbstractDerivativeEvaluator
  - Enables compile-time dispatch for derivative-based computations
  - Type safety: can call derivative functions

- **NoDerivatives**: No derivative evaluator, minimal memory footprint
  - `de` field is nothing
  - Optimized for categorical/boolean variables only
  - Type safety: compile-time error if derivative functions are called

# Buffer Management
Pre-allocates four key buffers with usage-specific optimization:
- `g_buf`: Usage-optimized for marginal effects results
- `gβ_accumulator`: For AME gradient accumulation (sized by coefficients)
- `η_buf`: For linear predictor computations (sized by observations)
- `row_buf`: For design matrix row computations (sized by model columns)

When buffer size is insufficient, functions allocate additional memory
to prevent bounds errors while maintaining correctness. An @info message
is logged when buffer allocation fallback occurs for performance visibility.

# Performance Notes
- Compilation is expensive (~milliseconds), so cache engines when possible
- Usage-specific sizing eliminates major allocation bottlenecks
- Derivative support specialization eliminates Union type overhead
- PopulationUsage + HasDerivatives: Optimal for continuous AME
- ProfileUsage + NoDerivatives: Optimal for categorical-only analysis

# Examples
```julia
# Population margins with derivatives - continuous variables
data_nt = Tables.columntable(data)
engine = build_engine(PopulationUsage, HasDerivatives, model, data_nt, [:x1, :x2], GLM.vcov)

# Profile margins without derivatives - categorical-only
engine = build_engine(ProfileUsage, NoDerivatives, model, data_nt, [:category], GLM.vcov)

# Mixed case: profile margins with derivatives
engine = build_engine(ProfileUsage, HasDerivatives, model, data_nt, [:x1, :category], GLM.vcov)
```
"""
function build_engine(usage::Type{U}, deriv::Type{D}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov, backend::Symbol) where {U<:MarginsUsage, D<:DerivativeSupport}
    # Input validation (delegated to utilities.jl)
    _validate_variables(data_nt, vars)

    # Compile formula with FormulaCompiler (using cache for performance)
    compiled = get_or_compile_formula(model, data_nt)
    all_continuous = continuous_variables(compiled, data_nt)

    # Filter continuous and categorical vars from requested vars (zero allocation)
    # IMPORTANT: Only include vars that were REQUESTED, not all model vars
    # Count continuous and categorical separately
    n_continuous = 0
    n_categorical = 0
    for v in vars
        if v ∈ all_continuous
            n_continuous += 1
        else
            n_categorical += 1
        end
    end

    # Pre-allocate with exact sizes
    continuous_vars = Vector{Symbol}(undef, n_continuous)
    categorical_vars = Vector{Symbol}(undef, n_categorical)

    cont_idx = 1
    cat_idx = 1
    for v in vars
        if v ∈ all_continuous
            continuous_vars[cont_idx] = v
            cont_idx += 1
        else
            categorical_vars[cat_idx] = v
            cat_idx += 1
        end
    end

    # Build derivative evaluator based on DerivativeSupport parameter
    de = if D <: HasDerivatives
        # HasDerivatives: Build evaluator for REQUESTED continuous variables only
        if isempty(continuous_vars)
            nothing
        else
            # Use FormulaCompiler's derivativeevaluator (dispatches to :ad or :fd backend)
            derivativeevaluator(backend, compiled, data_nt, continuous_vars)
        end
    else  # D <: NoDerivatives
        # NoDerivatives: No derivative evaluator, minimal memory footprint
        nothing
    end

    # Build contrast evaluator for categorical variables (Phase 4)
    contrast = if !isempty(categorical_vars)
        # Build ContrastEvaluator for categorical effects using FC primitives
        contrastevaluator(compiled, data_nt, categorical_vars)
    else
        # No categorical variables requested
        nothing
    end

    # Build variable index map
    # Populates mapping from variable Symbol to index in de.vars
    # Eliminates O(n×m) linear searches during marginal effects computation
    var_index_map = if !isnothing(de)
        Dict{Symbol, Int}(var => i for (i, var) in enumerate(de.vars))
    else
        # Empty map if no derivative evaluator
        Dict{Symbol, Int}()
    end

    # Pre-allocate buffers with usage-specific optimization
    n_continuous = length(continuous_vars)
    n_coef = length(compiled)
    n_obs = length(first(data_nt))
    n_cols = length(compiled)  # Number of columns in design matrix
    
    # Usage-specific g_buf sizing strategy
    if U <: PopulationUsage
        # PopulationUsage: Minimal g_buf sized only for FormulaCompiler continuous variables
        # Optimizes for minimal memory footprint and zero FormulaCompiler allocation fallbacks
        g_buf_size = max(n_continuous, 1)  # At least size 1 to avoid bounds errors

        # Additional optimization: NoDerivatives engines can use smaller g_buf for PopulationUsage
        if D <: NoDerivatives
            # For categorical-only computations, we don't need large g_buf
            g_buf_size = 1  # Minimal size
        end
    elseif U <: ProfileUsage
        # ProfileUsage: Larger g_buf sized for typical profile counts (~100)
        # Optimizes for multiple profiles without allocation fallbacks in SE computation
        # NOTE: ProfileUsage needs this size even for NoDerivatives (predictions use g_buf)
        g_buf_size = max(n_continuous, 100)  # Handle up to 100 profiles without fallback
    else
        error("Unknown MarginsUsage type: $U")
    end
    
    g_buf = Vector{Float64}(undef, g_buf_size)
    gβ_accumulator = Vector{Float64}(undef, n_coef)
    η_buf = Vector{Float64}(undef, max(n_obs, 1))  # Buffer for linear predictor computations
    row_buf = Vector{Float64}(undef, n_cols)       # Buffer for design matrix rows

    # Initialize batch operation buffers
    # Size for continuous variables that may be requested for batch operations
    batch_ame_values = Vector{Float64}(undef, max(n_continuous, 1))
    batch_gradients = Matrix{Float64}(undef, max(n_continuous, 1), n_coef)
    batch_var_indices = Vector{Int}(undef, max(n_continuous, 1))

    # Allocate new buffer based on derivative evaluator size
    n_de_vars = de === nothing ? 0 : length(de.vars)
    deta_dx_buf = Vector{Float64}(undef, max(n_de_vars, 1))

    # Initialize continuous effects buffers for zero-allocation population marginal effects
    # These buffers hold marginal effects and gradients for ALL variables in the derivative evaluator
    g_all_buf = Vector{Float64}(undef, max(n_de_vars, 1))
    Gβ_all_buf = Matrix{Float64}(undef, n_coef, max(n_de_vars, 1))

    # Initialize categorical effects buffers (Phase 4 zero-allocation optimization)
    # These buffers enable zero-allocation categorical AME computation
    contrast_buf = Vector{Float64}(undef, n_coef)        # Contrast vector buffer
    contrast_grad_buf = Vector{Float64}(undef, n_coef)   # Per-row gradient buffer
    contrast_grad_accum = Vector{Float64}(undef, n_coef) # Accumulated gradient buffer

    # Initialize persistent EffectsBuffers for zero-allocation raw API
    # Size for maximum possible variables (all engine vars)
    n_vars_max = max(length(vars), n_continuous, 1)
    effects_buffers = EffectsBuffers(n_vars_max, n_coef)
    
    # vcov should always be a function or CovarianceMatrices estimator
    actual_vcov = vcov
    
    # Handle both functions (GLM.vcov) and CovarianceMatrices estimators (HC1())
    if isa(actual_vcov, Function)
        covariance_matrix = actual_vcov(model)
    else
        # Handle CovarianceMatrices estimators as optional dependency
        vcov_module = Base.require(Main, :CovarianceMatrices)
        covariance_matrix = Base.invokelatest(vcov_module.vcov, actual_vcov, model)
    end
    
    # Create engine instance
    engine = MarginsEngine{typeof(_auto_link(model)), U, D, typeof(compiled)}(
        compiled, de, var_index_map, contrast, g_buf, gβ_accumulator, η_buf, row_buf,
        batch_ame_values, batch_gradients, batch_var_indices, deta_dx_buf,
        Vector{Int}(undef, max(n_continuous, 1)),  # cont_var_indices_buf
        g_all_buf, Gβ_all_buf,
        contrast_buf, contrast_grad_buf, contrast_grad_accum,  # Categorical buffers (Phase 4)
        effects_buffers,  # persistent EffectsBuffers for zero-allocation raw API
        copy(continuous_vars), copy(categorical_vars),
        model, coef(model), covariance_matrix, _auto_link(model), vars, data_nt
    )

    # Fix 4: Verify and repair buffers if needed (root cause prevention)
    engine = verify_and_repair_engine_buffers!(engine)

    return engine
end

"""
    _auto_link(model) -> GLM.Link

Automatically determine link function from model.

Supports GLM.jl models and provides sensible fallbacks for other model types.

# Arguments
- `model`: Fitted statistical model

# Returns
- `GLM.Link`: Link function (IdentityLink for non-GLM models)

# Examples
```julia
link = _auto_link(glm_model)  # Returns actual link (LogitLink, etc.)
link = _auto_link(other_model)  # Returns IdentityLink() fallback
```
"""
function _auto_link(model)
    # Try to extract link from GLM.jl and MixedModels.jl using official APIs
    try
        # Use GLM.Link() function for proper link extraction
        if hasmethod(GLM.Link, (typeof(model),))
            return GLM.Link(model)
        elseif hasfield(typeof(model), :model) && hasmethod(GLM.Link, (typeof(model.model),))
            # Handle wrapped models (TableRegressionModel)
            return GLM.Link(model.model)
        end
    catch e
        error("Failed to extract link function from model: $e. " *
              "Statistical correctness cannot be guaranteed without proper link function.")
    end
    
    # Handle MixedModels.jl models
    if isdefined(Main, :MixedModels)  # Check if MixedModels is available
        # LinearMixedModel: Always uses IdentityLink (like linear regression)
        if model isa Main.MixedModels.LinearMixedModel
            return GLM.IdentityLink()
        end
        
        # GeneralizedLinearMixedModel: Extract from resp.link field
        if model isa Main.MixedModels.GeneralizedLinearMixedModel
            if hasfield(typeof(model), :resp) && hasfield(typeof(model.resp), :link)
                return model.resp.link
            end
        end
    end
    
    # Check for linear models (IdentityLink is correct for these)
    if hasfield(typeof(model), :model) && hasfield(typeof(model.model), :rr)
        # Linear models (LmResp) don't have a distribution, only GLM models do
        if typeof(model.model.rr) <: GLM.LmResp
            return GLM.IdentityLink()  # Linear models use identity link
        end
    end
    
    # Error for unknown model types - statistical correctness first
    error("Cannot determine link function for model type $(typeof(model)). " *
          "Statistical correctness cannot be guaranteed without proper link function. " *
          "Supported: GLM.jl models (lm, glm) and MixedModels.jl (LinearMixedModel, GeneralizedLinearMixedModel).")
end


# Core validation functions
"""
    _validate_variables(data_nt, vars)

Validate that requested variables exist and are analyzable.
"""
function _validate_variables(data_nt::NamedTuple, vars::Vector{Symbol})
    for var in vars
        if !haskey(data_nt, var)
            throw(MarginsError("Variable $var not found in data"))
        end

        col = getproperty(data_nt, var)
        if length(col) == 0
            throw(MarginsError("Variable $var has no observations"))
        end
    end
end

"""
    verify_and_repair_engine_buffers!(engine::MarginsEngine{L,U,HasDerivatives,C}) where {L,U,C}

Verify and repair engine derivative buffers to ensure sufficient size.
Function barrier ensures compiler optimization for HasDerivatives engines.
"""
function verify_and_repair_engine_buffers!(engine::MarginsEngine{L,U,HasDerivatives,C}) where {L,U,C}
    # Function barrier: compiler knows engine has HasDerivatives
    required_size = length(engine.β)
    needs_rebuild = false
    new_de = engine.de
    new_contrast_buf = engine.contrast_buf
    new_contrast_grad_buf = engine.contrast_grad_buf
    new_contrast_grad_accum = engine.contrast_grad_accum

    # Validate derivative evaluator buffers
    if !isnothing(engine.de)
        # Check if current buffers are sufficient using common buffer field
        current_buffer_size = length(engine.de.xrow_buffer)  # Both AD and FD have this field

        if current_buffer_size < required_size
            @warn "Engine derivative buffers too small: $current_buffer_size < $required_size. Rebuilding derivative evaluator."
            needs_rebuild = true

            # Rebuild derivative evaluator with correct size
            # Detect backend from existing evaluator type
            backend = engine.de isa FDEvaluator ? :fd : :ad
            new_de = derivativeevaluator(backend, engine.compiled, engine.data_nt, engine.continuous_vars)
        end
    end

    # Validate categorical buffers (Phase 4)
    if length(engine.contrast_buf) != required_size
        @warn "Categorical contrast_buf incorrectly sized: $(length(engine.contrast_buf)) != $required_size. Rebuilding."
        needs_rebuild = true
        new_contrast_buf = Vector{Float64}(undef, required_size)
    end

    if length(engine.contrast_grad_buf) != required_size
        @warn "Categorical contrast_grad_buf incorrectly sized: $(length(engine.contrast_grad_buf)) != $required_size. Rebuilding."
        needs_rebuild = true
        new_contrast_grad_buf = Vector{Float64}(undef, required_size)
    end

    if length(engine.contrast_grad_accum) != required_size
        @warn "Categorical contrast_grad_accum incorrectly sized: $(length(engine.contrast_grad_accum)) != $required_size. Rebuilding."
        needs_rebuild = true
        new_contrast_grad_accum = Vector{Float64}(undef, required_size)
    end

    # Rebuild engine if any buffers were invalid
    if needs_rebuild
        return MarginsEngine{L,U,HasDerivatives,C}(
            engine.compiled, new_de, engine.var_index_map, engine.contrast,
            engine.g_buf, engine.gβ_accumulator, engine.η_buf, engine.row_buf,
            engine.batch_ame_values, engine.batch_gradients, engine.batch_var_indices,
            engine.deta_dx_buf, engine.cont_var_indices_buf,
            engine.g_all_buf, engine.Gβ_all_buf,
            new_contrast_buf, new_contrast_grad_buf, new_contrast_grad_accum,  # Categorical buffers (Phase 4)
            engine.effects_buffers,
            engine.continuous_vars, engine.categorical_vars,
            engine.model, engine.β, engine.Σ, engine.link, engine.vars, engine.data_nt
        )
    end

    # Buffers are adequate
    return engine
end

"""
    verify_and_repair_engine_buffers!(engine::MarginsEngine{L,U,NoDerivatives,C}) where {L,U,C}

Verify and repair categorical buffers for engines without derivatives.
Engines without derivatives can still have categorical variables.
"""
function verify_and_repair_engine_buffers!(engine::MarginsEngine{L,U,NoDerivatives,C}) where {L,U,C}
    # No derivative buffers to verify, but validate categorical buffers (Phase 4)
    required_size = length(engine.β)
    needs_rebuild = false
    new_contrast_buf = engine.contrast_buf
    new_contrast_grad_buf = engine.contrast_grad_buf
    new_contrast_grad_accum = engine.contrast_grad_accum

    # Validate categorical buffers (Phase 4)
    if length(engine.contrast_buf) != required_size
        @warn "Categorical contrast_buf incorrectly sized: $(length(engine.contrast_buf)) != $required_size. Rebuilding."
        needs_rebuild = true
        new_contrast_buf = Vector{Float64}(undef, required_size)
    end

    if length(engine.contrast_grad_buf) != required_size
        @warn "Categorical contrast_grad_buf incorrectly sized: $(length(engine.contrast_grad_buf)) != $required_size. Rebuilding."
        needs_rebuild = true
        new_contrast_grad_buf = Vector{Float64}(undef, required_size)
    end

    if length(engine.contrast_grad_accum) != required_size
        @warn "Categorical contrast_grad_accum incorrectly sized: $(length(engine.contrast_grad_accum)) != $required_size. Rebuilding."
        needs_rebuild = true
        new_contrast_grad_accum = Vector{Float64}(undef, required_size)
    end

    # Rebuild engine if any buffers were invalid
    if needs_rebuild
        return MarginsEngine{L,U,NoDerivatives,C}(
            engine.compiled, engine.de, engine.var_index_map, engine.contrast,
            engine.g_buf, engine.gβ_accumulator, engine.η_buf, engine.row_buf,
            engine.batch_ame_values, engine.batch_gradients, engine.batch_var_indices,
            engine.deta_dx_buf, engine.cont_var_indices_buf,
            engine.g_all_buf, engine.Gβ_all_buf,
            new_contrast_buf, new_contrast_grad_buf, new_contrast_grad_accum,  # Categorical buffers (Phase 4)
            engine.effects_buffers,
            engine.continuous_vars, engine.categorical_vars,
            engine.model, engine.β, engine.Σ, engine.link, engine.vars, engine.data_nt
        )
    end

    # Buffers are adequate
    return engine
end
