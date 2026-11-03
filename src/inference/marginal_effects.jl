# marginal_effects.jl
# Marginal effects implementations with concrete type dispatch
# Row-level marginal effects on the response scale (both backends). Internal test oracles
# for the continuous-effects accumulator (src/computation/continuous_accumulator.jl);
# the backend-specific marginal_effects_eta! methods are in the two files that follow.

"""
    marginal_effects_mu!(g, Gβ, de::AbstractDerivativeEvaluator, β, link, row) -> (g, Gβ)

Compute both marginal effects and parameter gradients for μ = g⁻¹(η) - zero allocations.

Extended version of marginal_effects_mu! that simultaneously computes marginal effects and
parameter gradients for all variables using the chain rule. More computationally intensive
than η-scale due to link function second derivatives, but still efficient.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(de.vars))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativeevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:AbstractFloat}`: Model coefficients of length `length(de)` (floating-point types)
- `link`: GLM link function (e.g., `GLM.LogitLink()`, `GLM.LogLink()`, `GLM.IdentityLink()`)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `(g, Gβ)`: Tuple containing marginal effects and parameter gradients
  - `g[j] = ∂μ/∂vars[j]` for all variables in de.vars
  - `Gβ[i,j] = ∂(∂μ/∂vars[j])/∂β[i]` for all model parameters and variables

# Performance Characteristics
- **Memory**: 0 bytes allocated (reuses η-scale computation + chain rule application)
- **Speed**: Slower than η-scale due to link function second derivatives
- **Backend**: Automatically dispatches to AD or FD implementation

# Mathematical Method
Two-step computation using chain rule:
1. Get η-scale marginal effects + parameter gradients: `marginal_effects_eta!(g, Gβ, de, β, row)`
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

marginal_effects_mu!(g, Gβ, de, β, link, 1)
# g contains μ-scale marginal effects
# Gβ contains parameter gradients for uncertainty quantification
```
"""
function marginal_effects_mu!(
    g::AbstractVector{Float64},
    Gβ::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:AbstractFloat},
    link,
    row::Int
)
    # Step 1: Get η-scale marginal effects + all parameter gradients
    marginal_effects_eta!(g, Gβ, de, β, row)  # Gβ = J

    # Step 2: Compute η = Xβ and link derivatives
    de.compiled_base(de.xrow_buffer, de.base_data, row)
    η = dot(β, de.xrow_buffer)
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
            Gβ[i, j] = g_prime * Gβ[i, j] + Jj_dot_beta * g_double_prime * de.xrow_buffer[i]
        end
    end

    return nothing
end
