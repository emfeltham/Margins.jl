# marginal_effects.jl
# Marginal effects implementations with concrete type dispatch
# Methods here apply to both finite diff and automatic diff backends

"""
    marginal_effects_eta!(g, Gβ, de::AbstractDerivativeEvaluator, β, row) -> (g, Gβ)

Compute both marginal effects and parameter gradients for η = Xβ - zero allocations.

Wrapper function that dispatches to the appropriate backend (AD or FD) implementation.
Simultaneously computes marginal effects and parameter gradients for all variables,
with parameter gradients being essentially free since they're the computed Jacobian.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(de.vars))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativeevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:AbstractFloat}`: Model coefficients of length `length(de)` (floating-point types)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `(g, Gβ)`: Tuple containing marginal effects and parameter gradients
  - `g[j] = ∂η/∂vars[j]` for all variables in de.vars
  - `Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i]` for all model parameters and variables

# Performance Characteristics
- **Memory**: 0 bytes allocated (parameter gradients are copy/transpose of computed Jacobian)
- **Speed**: Negligible overhead compared to regular marginal_effects_eta!
- **Backend**: Automatically uses AD or FD based on evaluator type

# Example
```julia
# Works with both AD and FD backends
vars = [:x, :z]
de_ad = derivativeevaluator(:ad, compiled, data, vars)
de_fd = derivativeevaluator(:fd, compiled, data, vars)

g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de_ad), length(vars))

# Both calls have identical interface
marginal_effects_eta!(g, Gβ, de_ad, β, 1)  # Uses AD
marginal_effects_eta!(g, Gβ, de_fd, β, 1)  # Uses FD
```
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    Gβ::AbstractMatrix{Float64},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:AbstractFloat},
    row::Int
)
    # Dispatch to appropriate backend implementation
    return marginal_effects_eta!(g, Gβ, de, β, row)
end

"""
    marginal_effects_eta!(G, Gβ_tensor, de::AbstractDerivativeEvaluator, β, rows) -> (G, Gβ_tensor)

Batch computation of marginal effects and parameter gradients for multiple rows - zero allocations after warmup.

Wrapper function that dispatches to the appropriate backend (AD or FD) implementation.
Computes both marginal effects ∂η/∂x and parameter gradients ∂(∂η/∂x)/∂β for multiple rows
efficiently. The parameter gradient tensor is 3D for uncertainty quantification.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `G::Matrix{Float64}`: Preallocated marginal effects matrix of size `(length(rows), length(de.vars))`
- `Gβ_tensor::Array{Float64, 3}`: Preallocated gradient tensor of size `(length(rows), length(de.vars), length(de))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativeevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:AbstractFloat}`: Model coefficients (same β used for all rows)
- `rows::AbstractVector{Int}`: Row indices to evaluate (1-based indexing)

# Returns
- `(G, Gβ_tensor)`: Tuple containing marginal effects matrix and parameter gradient tensor
  - `G[k, j] = ∂η/∂vars[j]` for row `rows[k]`
  - `Gβ_tensor[k, j, i] = ∂(∂η/∂vars[j])/∂β[i]` for row `rows[k]`

# Performance Characteristics
- **Memory**: 0 bytes allocated (parameter gradients are copy/transpose of computed Jacobians)
- **Speed**: Efficient batch processing with backend-specific optimizations
- **Backend**: Automatically uses AD or FD based on evaluator type

# Example
```julia
# Works with both AD and FD backends
rows = [1, 3, 7, 12, 15]
G_batch = Matrix{Float64}(undef, length(rows), length(vars))
Gβ_tensor = Array{Float64, 3}(undef, length(rows), length(vars), length(de))

# Both calls have identical interface
marginal_effects_eta!(G_batch, Gβ_tensor, de_ad, β, rows)  # Uses AD
marginal_effects_eta!(G_batch, Gβ_tensor, de_fd, β, rows)  # Uses FD
```
"""
function marginal_effects_eta!(
    G::AbstractMatrix{Float64},
    Gβ_tensor::AbstractArray{Float64, 3},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:AbstractFloat},
    rows::AbstractVector{Int}
)
    # Dispatch to appropriate backend implementation
    return marginal_effects_eta!(G, Gβ_tensor, de, β, rows)
end

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

"""
    marginal_effects_mu!(G, Gβ_tensor, de::AbstractDerivativeEvaluator, β, link, rows) -> (G, Gβ_tensor)

Batch computation of marginal effects and parameter gradients for μ = g⁻¹(η) for multiple rows - zero allocations after warmup.

Computes both marginal effects ∂μ/∂x and parameter gradients ∂(∂μ/∂x)/∂β for multiple rows
efficiently using the full chain rule. The parameter gradient tensor is 3D for uncertainty quantification.
More computationally intensive than η-scale due to link function second derivatives.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `G::Matrix{Float64}`: Preallocated marginal effects matrix of size `(length(rows), length(de.vars))`
- `Gβ_tensor::Array{Float64, 3}`: Preallocated gradient tensor of size `(length(rows), length(de.vars), length(de))`
- `de::AbstractDerivativeEvaluator`: Evaluator built by `derivativeevaluator(:ad/:fd, compiled, data, vars)`
- `β::AbstractVector{<:AbstractFloat}`: Model coefficients (same β used for all rows)
- `link`: GLM link function (e.g., `GLM.LogitLink()`, `GLM.LogLink()`)
- `rows::AbstractVector{Int}`: Row indices to evaluate (1-based indexing)

# Returns
- `(G, Gβ_tensor)`: Tuple containing marginal effects matrix and parameter gradient tensor
  - `G[k, j] = ∂μ/∂vars[j]` for row `rows[k]`
  - `Gβ_tensor[k, j, i] = ∂(∂μ/∂vars[j])/∂β[i]` for row `rows[k]`

# Performance Characteristics
- **Memory**: 0 bytes allocated (reuses η computation + chain rule application)
- **Speed**: Slower than η-scale due to link function second derivatives
- **Scaling**: Linear in number of rows with per-row chain rule computation

# Mathematical Method
For each row k:
1. Get η-scale marginal effects + parameter gradients: `marginal_effects_eta!(g_k, Gβ_k, de, β, row)`
2. Apply full chain rule: `∂(∂μ/∂x_j)/∂β = g'(η) × J_j + (J_j'β) × g''(η) × X_row`

Supported link functions: Identity, Log, Logit, Probit, Cloglog, Cauchit, Inverse, Sqrt

# Example
```julia
# Batch μ-scale marginal effects + parameter gradients for logistic model
link = GLM.LogitLink()
rows = [1, 3, 7, 12, 15]
G_mu_batch = Matrix{Float64}(undef, length(rows), length(vars))
Gβ_tensor = Array{Float64, 3}(undef, length(rows), length(vars), length(de))

marginal_effects_mu!(G_mu_batch, Gβ_tensor, de, β, link, rows)  # 0 bytes allocated

# Access: G_mu_batch[1, :] = μ-scale marginal effects for row 1
#         Gβ_tensor[1, :, :] = parameter gradients for row 1
```
"""
function marginal_effects_mu!(
    G::AbstractMatrix{Float64},
    Gβ_tensor::AbstractArray{Float64, 3},
    de::AbstractDerivativeEvaluator,
    β::AbstractVector{<:AbstractFloat},
    link,
    rows::AbstractVector{Int}
)
    # Validate dimensions
    size(G, 1) == length(rows) || throw(DimensionMismatch("G first dimension must match length(rows)"))
    size(G, 2) == length(de.vars) || throw(DimensionMismatch("G second dimension must match length(de.vars)"))
    size(Gβ_tensor, 1) == length(rows) || throw(DimensionMismatch("Gβ_tensor first dimension must match length(rows)"))
    size(Gβ_tensor, 2) == length(de.vars) || throw(DimensionMismatch("Gβ_tensor second dimension must match length(de.vars)"))
    size(Gβ_tensor, 3) == length(de) || throw(DimensionMismatch("Gβ_tensor third dimension must match length(de)"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Step 1: Get η-scale marginal effects + parameter gradients for all rows
    marginal_effects_eta!(G, Gβ_tensor, de, β, rows)

    # Step 2: Apply full chain rule per row
    for (k, row) in enumerate(rows)
        # Compute η = Xβ and link derivatives for this row
        de.compiled_base(de.xrow_buffer, de.base_data, row)
        η = dot(β, de.xrow_buffer)
        g_prime = _dmu_deta(link, η)      # dμ/dη
        g_double_prime = _d2mu_deta2(link, η)  # d²μ/dη²

        # Apply chain rule to marginal effects: g = g'(η) × ∂η/∂x
        G_k = view(G, k, :)  # View into k-th row
        G_k .*= g_prime

        # Apply FULL chain rule to all parameter gradients
        # ∂(∂μ/∂x_j)/∂β = g'(η) × J_j + (J_j'β) × g''(η) × X_row
        Gβ_k = view(Gβ_tensor, k, :, :)  # View into k-th slice: [vars, params]

        @inbounds for j in 1:size(Gβ_k, 1)  # For each variable
            # Extract j-th row (parameter gradient for variable j): Gβ_k[j, :] = J_j
            Jj = view(Gβ_k, j, :)
            Jj_dot_beta = dot(Jj, β)  # J_j'β (scalar)

            # Apply chain rule to each parameter
            for i in 1:size(Gβ_k, 2)  # For each parameter
                Gβ_k[j, i] = g_prime * Gβ_k[j, i] + Jj_dot_beta * g_double_prime * de.xrow_buffer[i]
            end
        end
    end

    return nothing
end
