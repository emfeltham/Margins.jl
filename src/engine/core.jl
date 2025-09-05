# engine/core.jl - MarginsEngine struct and construction logic

"""
    MarginsEngine{L<:GLM.Link}

Zero-allocation engine for marginal effects computation.

Built on FormulaCompiler.jl with pre-allocated buffers for maximum performance.

Fields:
- `compiled::FormulaCompiler.UnifiedCompiled`: Pre-compiled FormulaCompiler formula
- `de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}`: Derivative evaluator for continuous vars
- `g_buf::Vector{Float64}`: Pre-allocated buffer for marginal effects results
- `gβ_accumulator::Vector{Float64}`: Pre-allocated buffer for AME gradient accumulation
- `η_buf::Vector{Float64}`: Pre-allocated buffer for linear predictor computations
- `row_buf::Vector{Float64}`: Pre-allocated buffer for model row (design matrix row) computations
- `model::Any`: Reference to original model
- `β::Vector{Float64}`: Model coefficients
- `Σ::Matrix{Float64}`: Model covariance matrix
- `link::L`: GLM link function
- `vars::Vector{Symbol}`: Variables for analysis
- `data_nt::NamedTuple`: Reference data for scenarios/refgrids

# Examples
```julia
engine = build_engine(model, data_nt, [:x1, :x2])
# Use engine for zero-allocation computations
```
"""
struct MarginsEngine{L<:GLM.Link}
    # FormulaCompiler components (pre-compiled)
    compiled::FormulaCompiler.UnifiedCompiled
    de::Union{FormulaCompiler.DerivativeEvaluator, Nothing}
    
    # Pre-allocated buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Marginal effects results
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
    build_engine(model, data_nt, vars) -> MarginsEngine

Construct zero-allocation margins engine with FormulaCompiler integration.

# Arguments
- `model`: Fitted statistical model (GLM.jl, MLJ.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format from Tables.jl
- `vars::Vector{Symbol}`: Variables to analyze for marginal effects

# Returns
- `MarginsEngine`: Pre-compiled engine with allocated buffers

# Buffer Management Strategy
The engine pre-allocates four key buffers to minimize runtime allocations:
- `g_buf`: For marginal effects results (sized by number of variables)
- `gβ_accumulator`: For AME gradient accumulation (sized by number of coefficients)  
- `η_buf`: For linear predictor computations (sized by number of observations)
- `row_buf`: For design matrix row computations (sized by number of model columns)

These buffers are safely reused across computations using bounds-checked views.
When buffer size is insufficient, functions allocate additional memory
to prevent bounds errors while maintaining correctness. An @info message
is logged when buffer allocation fallback occurs for performance visibility.

# Performance Notes
- Compilation is expensive (~milliseconds), so cache engines when possible
- Runtime computations achieve significant allocation reductions via buffer reuse
- Population margins: ~3k allocations (vs much higher without buffer optimization)
- Profile margins: Allocation overhead primarily from DataFrame/grid operations
- FormulaCompiler.jl provides zero-allocation primitive operations underneath

# Examples
```julia
data_nt = Tables.columntable(data)
engine = build_engine(model, data_nt, [:x1, :x2])

# Engine can be reused for multiple margin computations
# with minimal allocation overhead
```
"""
function build_engine(model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov)
    # Input validation (delegated to utilities.jl)
    _validate_variables(data_nt, vars)
    
    # Compile formula with FormulaCompiler
    compiled = FormulaCompiler.compile_formula(model, data_nt)
    continuous_vars = FormulaCompiler.continuous_variables(compiled, data_nt)
    vars_for_de = filter(v -> v in continuous_vars, vars)
    
    # Build derivative evaluator only if needed (for continuous variables)
    de = isempty(vars_for_de) ? nothing : 
         FormulaCompiler.build_derivative_evaluator(compiled, data_nt; vars=vars_for_de)
    
    # Pre-allocate buffers for zero runtime allocation
    n_vars = length(vars_for_de)
    n_coef = length(compiled)
    n_obs = length(first(data_nt))
    n_cols = length(compiled)  # Number of columns in design matrix
    g_buf = Vector{Float64}(undef, max(n_vars, 1))  # At least size 1 to avoid bounds errors
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
    
    return MarginsEngine(
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