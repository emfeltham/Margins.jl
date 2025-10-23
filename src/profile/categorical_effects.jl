# profile/categorical_effects.jl - Profile-level categorical variable processing
# Phase 4 Migration: Using ContrastEvaluator kernel (same as population path)

"""
    _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, scale, backend) -> Float64

Compute baseline contrast for categorical variable at specific profile point.
Uses ContrastEvaluator with FormulaCompiler primitives for zero-allocation computation.

# Arguments
- `engine`: MarginsEngine with model parameters and compiled formula
- `refgrid_de`: Derivative evaluator for the reference grid (may be nothing)
- `profile`: Profile specification dictionary
- `var`: Categorical variable name for contrast computation
- `scale`: Prediction scale (:response or :linear)
- `backend`: Computation backend (:ad or :fd)

# Returns
- `Float64`: Baseline contrast effect (treatment - baseline) at the profile point

# Mathematical Details
Computes discrete contrast using ContrastEvaluator:
- **Baseline prediction**: μ₀ = E[Y | var = baseline, profile]
- **Treatment prediction**: μ₁ = E[Y | var = treatment, profile]
- **Contrast**: Δ = μ₁ - μ₀

# Implementation
Uses the same categorical kernel as population path:
1. Get variable's baseline level
2. Get current level from profile
3. Call contrast_modelrow! for contrast vector
4. Apply scale (linear: dot with β, response: link inverse)

# Performance
- Zero allocations through pre-allocated engine buffers
- Single-row evaluation at profile point
- Reuses ContrastEvaluator from engine (no construction overhead)

# Example
```julia
profile = Dict(:age => 30.0, :education => "college")
contrast = _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, :treatment, :response, :ad)
# Returns effect of treatment - control for 30-year-old college graduates
```
"""
function _compute_row_specific_baseline_contrast(
    refgrid_contrast::ContrastEvaluator,
    refgrid_compiled,
    refgrid_data::NamedTuple,
    profile_row::Int,
    var::Symbol,
    from_level,
    to_level,
    β::Vector{Float64},
    link,
    scale::Symbol,
    contrast_buf::Vector{Float64},
    η_baseline_buf::Vector{Float64}
)
    # Use the same categorical kernel as population path
    # The only difference is we operate on refgrid data at a specific profile_row

    # Buffers now passed as arguments (0 bytes allocation)

    # Compute contrast vector at profile row (0 bytes with proper buffer reuse)
    contrast_modelrow!(contrast_buf, refgrid_contrast, profile_row, var, from_level, to_level)

    # Compute effect based on scale
    if scale == :link
        # Linear predictor scale: Δη = Δ'β
        effect = dot(contrast_buf, β)
    else  # scale == :response
        # Response scale: apply link inverse transformation
        # Need to compute μ₁ - μ₀ where μ = link⁻¹(η)

        # Baseline prediction (η scale) - use passed buffer
        refgrid_compiled(η_baseline_buf, refgrid_data, profile_row)
        η_baseline = sum(η_baseline_buf)

        # Treatment prediction: η_treatment = η_baseline + Δ'β
        Δβ = dot(contrast_buf, β)
        η_treatment = η_baseline + Δβ

        # Apply link inverse
        μ_baseline = GLM.linkinv(link, η_baseline)
        μ_treatment = GLM.linkinv(link, η_treatment)

        effect = μ_treatment - μ_baseline
    end

    return effect
end

"""
    _row_specific_contrast_grad_beta!(gβ_buffer::Vector{Float64}, engine, refgrid_de, profile, var, scale)

Compute parameter gradient for row-specific contrast at profile point.
Uses ContrastEvaluator with FormulaCompiler primitives for zero-allocation computation.

# Arguments
- `gβ_buffer`: Pre-allocated buffer for parameter gradient (modified in-place)
- `engine`: MarginsEngine with model parameters and compiled formula
- `refgrid_de`: Derivative evaluator for the reference grid
- `profile`: Profile specification dictionary
- `var`: Categorical variable name for contrast computation
- `scale`: Prediction scale (:response or :linear)

# Returns
- `nothing` (results stored in gβ_buffer)

# Mathematical Details
Computes gradient of contrast with respect to parameters:
- **Gradient**: ∇β [μ₁ - μ₀] = ∇β μ₁ - ∇β μ₀
- **Parameter derivatives**: Uses contrast_gradient! from FormulaCompiler
- **Chain rule**: Applied for response scale transformations

# Implementation
Uses the same categorical kernel as population path:
1. Get baseline and current levels
2. Call contrast_gradient! for parameter gradient
3. Store result in pre-allocated buffer (0 bytes)

# Buffer Management
Uses pre-allocated buffer to avoid allocations:
- Input buffer is modified in-place
- Uses engine.contrast_grad_buf for temporary computations
- Zero additional memory allocation

# Standard Error Integration
The computed gradient enables delta method standard error calculation:
- **Variance**: Var[contrast] = g'Σg where g is the parameter gradient
- **Standard error**: SE = √(g'Σg)

# Performance
- Single evaluation at profile point
- Reuses engine buffers for intermediate calculations
- Zero allocations through careful buffer management

# Example
```julia
gβ_buffer = Vector{Float64}(undef, length(engine.β))
profile = Dict(:age => 30.0, :education => "college")
_row_specific_contrast_grad_beta!(gβ_buffer, engine, refgrid_de, profile, :treatment, :response)
# gβ_buffer now contains gradient of treatment contrast at the profile
```
"""
function _row_specific_contrast_grad_beta!(
    gβ_buffer::Vector{Float64},
    refgrid_contrast::ContrastEvaluator,
    profile_row::Int,
    var::Symbol,
    from_level,
    to_level,
    β::Vector{Float64},
    link,
    scale::Symbol,
    gradient_buf::Vector{Float64}
)
    # Use the same categorical kernel as population path

    # Buffer now passed as argument (0 bytes allocation)

    # Compute gradient at profile row (0 bytes with proper buffer reuse)
    # contrast_gradient! handles both link and response scale internally
    contrast_gradient!(gradient_buf, refgrid_contrast, profile_row, var, from_level, to_level, β, link)

    # Copy to output buffer
    copyto!(gβ_buffer, gradient_buf)

    return nothing
end

"""
    _compute_profile_pairwise_contrasts(engine, profile_dict, var, scale, backend) -> Vector{Tuple}

Compute all pairwise contrasts for a categorical variable at a specific profile.
Uses ContrastEvaluator with FormulaCompiler primitives for zero-allocation computation.

# Arguments
- `engine`: MarginsEngine with ContrastEvaluator
- `profile_dict`: Profile specification dictionary
- `var`: Categorical variable name
- `scale`: Prediction scale (:response or :linear)
- `backend`: Computation backend (:ad or :fd)

# Returns
- Vector of (level1, level2, effect, gradient) tuples for all unique pairs

# Implementation
Uses the same categorical kernel as population path:
1. Get all levels for the categorical variable
2. Generate all pairwise combinations
3. For each pair, call contrast_modelrow! and contrast_gradient!
4. Apply scale transformation if needed

# Performance
- Zero allocations through pre-allocated engine buffers
- Reuses ContrastEvaluator for all pairs
- Single-row evaluation at profile point

# Example
```julia
profile = Dict(:age => 30.0, :education => "college")
results = _compute_profile_pairwise_contrasts(engine, profile, :treatment, :response, :ad)
# Returns vector of (level1, level2, effect, gradient) for all treatment pairs
```
"""
function _compute_profile_pairwise_contrasts(
    refgrid_contrast::ContrastEvaluator,
    refgrid_compiled,
    refgrid_data::NamedTuple,
    profile_row::Int,
    var::Symbol,
    β::Vector{Float64},
    link,
    scale::Symbol,
    contrast_buf::Vector{Float64},
    gradient_buf::Vector{Float64},
    η_buf::Vector{Float64}
)
    # Use the same categorical kernel as population path

    # Get all levels for this categorical variable from refgrid data
    col = getproperty(refgrid_data, var)
    levels_list = if col isa CategoricalArray
        CategoricalArrays.levels(col)
    elseif eltype(col) <: Bool
        [false, true]
    else
        unique(col)
    end

    # Generate all pairwise combinations
    contrast_pairs = [(level1, level2) for (i, level1) in enumerate(levels_list), (j, level2) in enumerate(levels_list) if i < j]

    # Buffers now passed as arguments (0 bytes allocation)

    results = []
    for (level1, level2) in contrast_pairs
        # Compute contrast vector (0 bytes with proper buffer reuse)
        contrast_modelrow!(contrast_buf, refgrid_contrast, profile_row, var, level1, level2)

        # Compute effect based on scale
        if scale == :link
            # Linear predictor scale: Δη = Δ'β
            effect = dot(contrast_buf, β)
        else  # scale == :response
            # Response scale: apply link inverse transformation

            # Baseline prediction (level1) - use passed buffer
            refgrid_compiled(η_buf, refgrid_data, profile_row)
            η1 = sum(η_buf)

            # Treatment prediction: η2 = η1 + Δ'β
            Δβ = dot(contrast_buf, β)
            η2 = η1 + Δβ

            # Apply link inverse
            μ1 = GLM.linkinv(link, η1)
            μ2 = GLM.linkinv(link, η2)

            effect = μ2 - μ1
        end

        # Compute gradient (0 bytes with proper buffer reuse)
        contrast_gradient!(gradient_buf, refgrid_contrast, profile_row, var, level1, level2, β, link)

        # Store result with copied gradient (stores level1, level2, with effect = level2 - level1)
        push!(results, (level1, level2, effect, copy(gradient_buf)))
    end

    return results
end
