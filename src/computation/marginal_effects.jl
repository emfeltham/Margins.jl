# computation/marginal_effects.jl
# Pure computation kernels for marginal effects
# Built on top of FormulaCompiler

"""
    compute_linear_predictor(β::Vector{Float64}, X_row::Vector{Float64}) -> Float64

Pure mathematical kernel for linear predictor computation.
Zero allocations: η = Xβ = Σ β[p] × X[p]

# Arguments
- `β`: Model parameter vector
- `X_row`: Design matrix row (feature vector for single observation)

# Returns
- `Float64`: Linear predictor η = Xβ

# Performance
- Pure mathematical computation with no allocations
- @fastmath optimization for floating-point operations
- @inbounds for bounds check elimination
- Single loop through parameters

# Usage
Used for prediction and baseline computations throughout marginal effects calculation.
"""
function compute_linear_predictor(β::Vector{Float64}, X_row::Vector{Float64})
    η = 0.0
    @inbounds @fastmath for p in 1:length(β)
        η += β[p] * X_row[p]
    end
    return η
end

# compute_categorical_contrast_effect! removed - obsolete DataScenario-based function
# Categorical effects now use ContrastEvaluator kernel system (see kernels/categorical.jl)
