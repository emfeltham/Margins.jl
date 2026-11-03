# marginal_effects.jl
# Row-level marginal effects on the linear-predictor and response scales (both backends).
# Internal test oracles for the continuous-effects accumulator
# (src/computation/continuous_accumulator.jl). Both use only FormulaCompiler's public
# evaluator API (`derivative_modelrow!`, `derivative_variables`), so one method serves the
# AD and FD backends.

"""
    marginal_effects_eta!(g, Gβ, de::AbstractDerivativeEvaluator, β, row) -> nothing

Compute both marginal effects and parameter gradients for η = Xβ - zero allocations.

The parameter gradient matrix is free: it is the Jacobian of the model row, which FC writes
straight into `Gβ`.

**Note**: Computes derivatives only for the evaluator's variables,
`derivative_variables(de)` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(derivative_variables(de))`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(derivative_variables(de)))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativeevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `row::Int`: Row index to evaluate (1-based indexing)

# Results
- `g[j] = ∂η/∂vars[j]` for all variables
- `Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i] = J[i,j]` for all model parameters and variables

# Example
```julia
vars = [:x, :z]
de = derivativeevaluator(:ad, compiled, data, vars)
g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de), length(vars))

marginal_effects_eta!(g, Gβ, de, β, 1)
# g now contains marginal effects ∂η/∂vars
# Gβ now contains parameter gradients ∂(∂η/∂vars[j])/∂β[i]
```
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    Gβ::Matrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:Real},
    row::Int
)
    # Simple bounds checks without string interpolation to avoid allocations
    n_vars = length(derivative_variables(de))
    length(g) == n_vars || throw(DimensionMismatch("gradient length mismatch"))
    size(Gβ, 1) == length(de) || throw(DimensionMismatch("Gβ first dimension must match length(de)"))
    size(Gβ, 2) == n_vars || throw(DimensionMismatch("Gβ second dimension must match the number of variables"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i] = J[i,j]
    derivative_modelrow!(Gβ, de, row)

    # Type barrier for zero-allocation matrix multiply: g = J'β
    _matrix_multiply_eta!(g, Gβ, β)

    return nothing
end

"""
    marginal_effects_mu!(g, Gβ, x, de::AbstractDerivativeEvaluator, β, link, row) -> nothing
    marginal_effects_mu!(g, Gβ, de, β, link, row) -> nothing

Compute both marginal effects and parameter gradients for μ = g⁻¹(η) - zero allocations.

Simultaneously computes marginal effects and parameter gradients for all variables using the
chain rule. `x` (length `length(de)`) is scratch; on return it holds the model row Xᵢ. The
form without `x` allocates it (one vector per call); allocation-sensitive callers pass `x`.

**Note**: Computes derivatives only for the evaluator's variables,
`derivative_variables(de)` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(derivative_variables(de))`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(derivative_variables(de)))`
- `x::Vector{Float64}`: Preallocated model-row buffer of length `length(de)`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativeevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:AbstractFloat}`: Model coefficients of length `length(de)` (floating-point types)
- `link`: GLM link function (e.g., `GLM.LogitLink()`, `GLM.LogLink()`, `GLM.IdentityLink()`)
- `row::Int`: Row index to evaluate (1-based indexing)

# Results
- `g[j] = ∂μ/∂vars[j]` for all variables
- `Gβ[i,j] = ∂(∂μ/∂vars[j])/∂β[i]` for all model parameters and variables

# Mathematical Method
Two-step computation using chain rule:
1. Jacobian J and model row X in one call: `derivative_modelrow!(Gβ, x, de, row)`; η = X'β
2. Apply full chain rule: `∂(∂μ/∂x_j)/∂β = g'(η) × J_j + (J_j'β) × g''(η) × X_row`

Supported link functions: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt

# Example
```julia
# Response-scale marginal effects + parameter gradients for logistic regression
link = GLM.LogitLink()
vars = [:x, :z]
de = derivativeevaluator(:ad, compiled, data, vars)

g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de), length(vars))
x = Vector{Float64}(undef, length(de))

marginal_effects_mu!(g, Gβ, x, de, β, link, 1)
# g contains μ-scale marginal effects
# Gβ contains parameter gradients for uncertainty quantification
```
"""
function marginal_effects_mu!(
    g::AbstractVector{Float64},
    Gβ::Matrix{Float64},
    x::AbstractVector{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:AbstractFloat},
    link,
    row::Int
)
    n_vars = length(derivative_variables(de))
    length(g) == n_vars || throw(DimensionMismatch("gradient length mismatch"))
    size(Gβ, 1) == length(de) || throw(DimensionMismatch("Gβ first dimension must match length(de)"))
    size(Gβ, 2) == n_vars || throw(DimensionMismatch("Gβ second dimension must match the number of variables"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Step 1: η-scale marginal effects + all parameter gradients (Gβ = J), and Xᵢ
    derivative_modelrow!(Gβ, x, de, row)
    _matrix_multiply_eta!(g, Gβ, β)

    # Step 2: Compute η = Xβ and link derivatives
    η = dot(β, x)
    g_prime = _dmu_deta(link, η)      # dμ/dη
    g_double_prime = _d2mu_deta2(link, η)  # d²μ/dη²

    # Step 3: Apply chain rule to marginal effects
    g .*= g_prime  # g = g'(η) × ∂η/∂x

    # Step 4: Apply FULL chain rule to all parameter gradients
    # ∂(∂μ/∂x_j)/∂β = g'(η) × J_j + (J_j'β) × g''(η) × X_row
    @inbounds for j in 1:size(Gβ, 2)  # For each variable
        # Extract j-th column (parameter gradient for variable j)
        Jj = view(Gβ, :, j)
        Jj_dot_beta = dot(Jj, β)  # J_j'β (scalar)

        # Apply chain rule to each parameter
        for i in 1:size(Gβ, 1)  # For each parameter
            Gβ[i, j] = g_prime * Gβ[i, j] + Jj_dot_beta * g_double_prime * x[i]
        end
    end

    return nothing
end

marginal_effects_mu!(g::AbstractVector{Float64}, Gβ::Matrix{Float64},
                     de::AbstractDerivativeEvaluator, β::AbstractVector{<:AbstractFloat},
                     link, row::Int) =
    marginal_effects_mu!(g, Gβ, Vector{Float64}(undef, length(de)), de, β, link, row)
