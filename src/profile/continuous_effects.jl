# profile/continuous_effects.jl - Profile-level continuous variable processing

"""
    ProfileContext{T}

Type-stable container for profile-specific computation context.
Eliminates dynamic type inference during profile processing.

# Type Parameter
- `T`: Type of profile values (typically Any for mixed types)

# Fields
- `profile::Dict{Symbol, T}`: Profile specification (variable values)
- `refgrid_data::NamedTuple`: Reference grid data constructed from profile
- `refgrid_compiled::Any`: Compiled FormulaCompiler evaluator for reference grid
- `refgrid_de::Union{Nothing, Any}`: Derivative evaluator for continuous variables

# Usage
Profile contexts encapsulate all information needed to compute marginal effects
at specific points (profiles) rather than averaging over the population. This
enables "marginal effects at the mean" (MEM) and other reference point analyses.

!!! note
    This type and `apply_profile_measure_transformations` below are not on any live code
    path: `profile_margins` computes marginal effects in `_mem_continuous_and_categorical`
    (`src/profile/core.jl`) and applies measures with `apply_measure_transformation`. They
    are retained pending removal — see `ISSUES.md`.
"""
struct ProfileContext{T}
    profile::Dict{Symbol, T}
    refgrid_data::NamedTuple
    refgrid_compiled::Any
    refgrid_de::Union{Nothing, Any}
end

# ProfileBuffers{T} is defined in src/core/buffer_management.jl

# get_profile_buffers function is defined in src/core/buffer_management.jl

"""
    create_profile_context(profile::Dict, engine::MarginsEngine, continuous_vars::Vector{Symbol}) -> ProfileContext

Build profile-specific context with reference grid compilation.
Creates all necessary components for efficient profile-based marginal effects computation.

# Arguments
- `profile`: Dictionary specifying profile values (variable => value)
- `engine`: MarginsEngine with model and data information
- `continuous_vars`: Vector of continuous variable names (for derivative evaluator)

# Returns
- `ProfileContext`: Complete context for profile-based computations

# Processing Pipeline
1. **Reference Grid Construction**: Build single-row data with profile values
2. **Formula Compilation**: Compile FormulaCompiler evaluator for reference grid
3. **Derivative Evaluator**: Build derivative evaluator for continuous variables
4. **Context Assembly**: Package all components into type-stable structure

# Reference Grid Logic
- **Specified variables**: Use profile values directly
- **Unspecified variables**: Use typical values (mean for continuous, mode/mixture for categorical)
- **Categorical handling**: Ensure proper CategoricalArray construction with correct levels

# Performance Benefits
- Single compilation per profile (amortized across all variables)
- Type-stable context eliminates dynamic dispatch
- Pre-built derivative evaluator for efficient gradient computation
- Optimized for repeated marginal effect calculations at the same profile

# Example
```julia
# Profile: 30-year-old with college education
profile = Dict(:age => 30.0, :education => "college")
context = create_profile_context(profile, engine, [:age, :income])
# Context now ready for efficient marginal effects computation
```
"""
function create_profile_context(profile::Dict, engine::MarginsEngine, continuous_vars::Vector{Symbol}, backend::Symbol)
    # Build reference grid data using profile values and typical values for unspecified variables
    # Create minimal DataFrame from profile Dict
    profile_df = DataFrame(profile)
    # Complete with typical values for missing model variables
    refgrid_df = complete_reference_grid(profile_df, engine.model, engine.data_nt)
    # Convert single-row DataFrame back to NamedTuple for FormulaCompiler
    refgrid_data = Tables.columntable(refgrid_df)

    # Compile reference grid (single row) for fast evaluation (using cache)
    refgrid_compiled = get_or_compile_formula(engine.model, refgrid_data)

    # Build derivative evaluator for reference grid if continuous variables are requested
    refgrid_de = if !isempty(continuous_vars)
        # Use FormulaCompiler's derivativeevaluator (dispatches to correct backend)
        derivativeevaluator(backend, refgrid_compiled, refgrid_data, continuous_vars)
    else
        nothing
    end

    return ProfileContext(profile, refgrid_data, refgrid_compiled, refgrid_de)
end


"""
    apply_profile_measure_transformations(effect_val::Float64, gradient::AbstractVector,
                                        var::Symbol, profile::Dict, context::ProfileContext,
                                        engine::MarginsEngine, scale::Symbol, measure::Symbol,
                                        continuous_vars::Vector{Symbol}) -> (Float64, Float64)

Apply econometric measure transformations for profile-based marginal effects.
Transforms raw marginal effects into interpretable econometric measures at specific profile points.

# Arguments
- `effect_val`: Raw marginal effect value at the profile
- `gradient`: Parameter gradient vector (modified in-place for transformation)
- `var`: Variable name for which effect is computed
- `profile`: Profile specification dictionary
- `context`: ProfileContext with compiled reference grid
- `engine`: MarginsEngine with model parameters and link function
- `scale`: Scale for predictions (:response or :link)
- `measure`: Econometric measure type
- `continuous_vars`: Vector of continuous variable names

# Returns
- `(Float64, Float64)`: Tuple of (transformed_effect, gradient_transform_factor)

# Supported Measures
- `:effect`: No transformation - raw marginal effect (∂μ/∂x)
- `:elasticity`: (x/μ) × (∂μ/∂x) - percentage change interpretation
- `:semielasticity_dyex`: x × (∂μ/∂x) - unit change in x, percent change in μ
- `:semielasticity_eydx`: (1/μ) × (∂μ/∂x) - percent change in x, unit change in μ

# Profile-Specific Values
- **Variable value**: Uses profile value if specified, otherwise typical value
- **Response value**: Computed at the specific profile point using reference grid
- **Categorical variables**: Use typical value (1.0) for transformation calculations

# Mathematical Foundation
All transformations computed at the profile point rather than population averages:
- **Elasticity**: (∂μ/∂x) × (x_profile/μ_profile)
- **Semi-elasticity (dyex)**: (∂μ/∂x) × x_profile
- **Semi-elasticity (eydx)**: (∂μ/∂x) × (1/μ_profile)

# Gradient Transformation
The gradient transformation factor is applied consistently to both the effect
and its parameter gradient, maintaining proper standard error calculation
through the delta method.

# Performance
- Single evaluation at profile point
- Zero allocations through buffer reuse
- Type-stable operations throughout
- Optimized for profile-specific calculations

!!! note
    Not on a live code path (see `ProfileContext` above).
"""
function apply_profile_measure_transformations(effect_val::Float64, gradient::AbstractVector,
                                             var::Symbol, profile::Dict, context::ProfileContext,
                                             engine::MarginsEngine, scale::Symbol, measure::Symbol,
                                             continuous_vars::Vector{Symbol})
    if measure === :effect
        return (effect_val, 1.0)
    end

    # Get profile-specific variable value for transformations
    if var ∈ continuous_vars && haskey(profile, var)
        x_val = float(profile[var])
    else
        # For categorical variables or unspecified continuous variables, use typical value
        col = getproperty(engine.data_nt, var)
        x_val = _is_continuous_variable(col) ? mean(col) : 1.0
    end

    # Get response value at the specific profile point
    modelrow!(engine.row_buf, context.refgrid_compiled, context.refgrid_data, 1)
    η = dot(engine.row_buf, engine.β)
    μ_val = scale === :response ? GLM.linkinv(engine.link, η) : η

    # Apply econometric transformations using profile-specific values
    if measure === :elasticity
        # Elasticity: (∂μ/∂x) × (x_profile/μ_profile)
        transform_factor = x_val / μ_val
        return (transform_factor * effect_val, transform_factor)
    elseif measure === :semielasticity_dyex
        # Semi-elasticity (unit x, percent μ): (∂μ/∂x) × x_profile
        transform_factor = x_val
        return (transform_factor * effect_val, transform_factor)
    elseif measure === :semielasticity_eydx
        # Semi-elasticity (percent x, unit μ): (∂μ/∂x) × (1/μ_profile)
        transform_factor = 1.0 / μ_val
        return (transform_factor * effect_val, transform_factor)
    else
        throw(ArgumentError("Unknown measure: $measure"))
    end
end
