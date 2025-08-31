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

# Notes
- Compilation is expensive (~milliseconds), so cache engines when possible
- Buffers are pre-allocated for zero-allocation hot paths
- Only builds derivative evaluator for continuous variables

# Examples
```julia
data_nt = Tables.columntable(data)
engine = build_engine(model, data_nt, [:x1, :x2])
```
"""
function build_engine(model, data_nt::NamedTuple, vars::Vector{Symbol})
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
    g_buf = Vector{Float64}(undef, max(n_vars, 1))  # At least size 1 to avoid bounds errors
    gβ_accumulator = Vector{Float64}(undef, n_coef)
    
    return MarginsEngine(
        compiled, de, g_buf, gβ_accumulator,
        model, coef(model), vcov(model), _auto_link(model), vars, data_nt
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
    # Try to extract link from GLM.jl models
    if hasfield(typeof(model), :model) && hasfield(typeof(model.model), :rr)
        # GLM with response distribution
        if hasfield(typeof(model.model.rr), :d)
            return model.model.rr.d.link  # GLM.jl pattern for generalized models
        else
            return GLM.IdentityLink()  # Linear models (LmResp) don't have a distribution
        end
    elseif hasfield(typeof(model), :link)
        return model.link  # Direct link field
    else
        return GLM.IdentityLink()  # Safe fallback for non-GLM models
    end
end

# Forward declaration - _validate_variables will be implemented in utilities.jl
function _validate_variables end