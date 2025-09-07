# engine/core.jl - MarginsEngine struct and construction logic

"""
    MarginsUsage

Abstract type for specifying the usage pattern of a MarginsEngine.
Different usage patterns require different buffer sizing strategies for optimal performance.

See also: [`PopulationUsage`](@ref), [`ProfileUsage`](@ref)
"""
abstract type MarginsUsage end

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
    MarginsEngine{L<:GLM.Link, U<:MarginsUsage}

Zero-allocation engine for marginal effects computation, optimized for specific usage patterns.

Built on FormulaCompiler.jl with usage-specific pre-allocated buffers for maximum performance.

Type Parameters:
- `L<:GLM.Link`: GLM link function type for the underlying model
- `U<:MarginsUsage`: Usage pattern determining buffer sizing strategy

Fields:
- `compiled::FormulaCompiler.UnifiedCompiled`: Pre-compiled FormulaCompiler formula
- `de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}`: Derivative evaluator for continuous vars
- `g_buf::Vector{Float64}`: Usage-optimized buffer for marginal effects results
- `gβ_accumulator::Vector{Float64}`: Pre-allocated buffer for AME gradient accumulation
- `η_buf::Vector{Float64}`: Pre-allocated buffer for linear predictor computations
- `row_buf::Vector{Float64}`: Pre-allocated buffer for model row (design matrix row) computations
- `model::Any`: Reference to original model
- `β::Vector{Float64}`: Model coefficients
- `Σ::Matrix{Float64}`: Model covariance matrix
- `link::L`: GLM link function
- `vars::Vector{Symbol}`: Variables for analysis
- `data_nt::NamedTuple`: Reference data for scenarios/refgrids

# Buffer Sizing Strategy
- `PopulationUsage`: Minimal g_buf (sized for FormulaCompiler continuous variables only)
- `ProfileUsage`: Larger g_buf (sized for typical profile counts ~100 to avoid allocation fallbacks)

# Examples
```julia
# Population margins engine (minimal memory footprint)
engine = build_engine(PopulationUsage, model, data_nt, [:x1, :x2], vcov)

# Profile margins engine (optimized for multiple profiles)  
engine = build_engine(ProfileUsage, model, data_nt, [:x1, :x2], vcov)
```
"""
struct MarginsEngine{L<:GLM.Link, U<:MarginsUsage}
    # FormulaCompiler components (pre-compiled)
    compiled::FormulaCompiler.UnifiedCompiled
    de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}
    
    # Usage-optimized buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Usage-optimized marginal effects results buffer
    gβ_accumulator::Vector{Float64}     # AME gradient accumulation
    η_buf::Vector{Float64}              # Linear predictor computation buffer
    row_buf::Vector{Float64}            # Model row (design matrix row) buffer
    
    # Model parameters
    model::Any                          # Reference to original model
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple  # Reference for scenarios/refgrids
end

"""
    build_engine(usage::Type{U}, model, data_nt, vars, vcov) where {U<:MarginsUsage} -> MarginsEngine

Construct zero-allocation margins engine with usage-specific optimization.

# Arguments
- `usage::Type{<:MarginsUsage}`: Usage pattern (PopulationUsage or ProfileUsage)
- `model`: Fitted statistical model (GLM.jl, MLJ.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format from Tables.jl
- `vars::Vector{Symbol}`: Variables to analyze for marginal effects
- `vcov`: Covariance estimator (function or CovarianceMatrices estimator)

# Returns
- `MarginsEngine{L, U}`: Pre-compiled engine with usage-optimized buffers

# Usage-Specific Buffer Sizing Strategy
- **PopulationUsage**: Minimal g_buf sized for FormulaCompiler continuous variables only
  - Optimizes for minimal memory footprint
  - Eliminates allocation fallbacks for derivative computation
  - Perfect for AME/AAP operations across sample data

- **ProfileUsage**: Larger g_buf sized for typical profile counts (~100)  
  - Optimizes for multiple profiles without allocation fallbacks
  - Handles most reference grids without buffer reallocations
  - Perfect for MEM/APM/MER/APR operations across profile grids

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
- PopulationUsage: Minimal memory, zero FormulaCompiler fallbacks
- ProfileUsage: Zero fallbacks for typical profile counts (up to 100)

# Examples
```julia
# Population margins - minimal memory footprint
data_nt = Tables.columntable(data)
engine = build_engine(PopulationUsage, model, data_nt, [:x1, :x2], GLM.vcov)

# Profile margins - optimized for multiple profiles
engine = build_engine(ProfileUsage, model, data_nt, [:x1, :x2], GLM.vcov)
```
"""
function build_engine(usage::Type{U}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov) where {U<:MarginsUsage}
    # Input validation (delegated to utilities.jl)
    _validate_variables(data_nt, vars)
    
    # Compile formula with FormulaCompiler
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    
    # Build derivative evaluator for ALL continuous variables (FormulaCompiler requirement)
    # We'll use indexing to extract only the requested variables later
    de = isempty(continuous_vars) ? nothing : 
         FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=continuous_vars)
    
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
    elseif U <: ProfileUsage
        # ProfileUsage: Larger g_buf sized for typical profile counts (~100)
        # Optimizes for multiple profiles without allocation fallbacks in SE computation
        g_buf_size = max(n_continuous, 100)  # Handle up to 100 profiles without fallback
    else
        error("Unknown MarginsUsage type: $U")
    end
    
    g_buf = Vector{Float64}(undef, g_buf_size)
    gβ_accumulator = Vector{Float64}(undef, n_coef)
    η_buf = Vector{Float64}(undef, max(n_obs, 1))  # Buffer for linear predictor computations
    row_buf = Vector{Float64}(undef, n_cols)       # Buffer for design matrix rows
    
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
    
    return MarginsEngine{typeof(_auto_link(model)), U}(
        compiled, de, g_buf, gβ_accumulator, η_buf, row_buf,
        model, coef(model), covariance_matrix, _auto_link(model), vars, data_nt
    )
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


# Forward declaration - _validate_variables will be implemented in utilities.jl
function _validate_variables end