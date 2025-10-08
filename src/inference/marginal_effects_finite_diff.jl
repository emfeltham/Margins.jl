# marginal_effects_finite_diff.jl

"""
    marginal_effects_eta!(g, Gβ, de::FDEvaluator, β, row) -> (g, Gβ)

Compute both marginal effects and parameter gradients for η = Xβ using finite differences - zero allocations.

Extended version of marginal_effects_eta! that simultaneously computes marginal effects and
parameter gradients for all variables. The parameter gradient matrix is essentially free since
it's just a copy of the already-computed Jacobian matrix.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `Gβ::Matrix{Float64}`: Preallocated parameter gradient matrix of size `(length(de), length(de.vars))`
- `de::FDEvaluator`: FD evaluator built by `derivativeevaluator_fd(compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)` (may be any numeric type)
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `(g, Gβ)`: Tuple containing marginal effects and parameter gradients
  - `g[j] = ∂η/∂vars[j]` for all variables in de.vars
  - `Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i]` for all model parameters and variables

# Performance Characteristics
- **Memory**: 0 bytes allocated (parameter gradients are copy of computed Jacobian)
- **Speed**: Negligible overhead compared to regular marginal_effects_eta!
- **Efficiency**: Parameter gradient computation is essentially free

# Mathematical Method
- Marginal effects: Same as regular marginal_effects_eta!
- Parameter gradients: `Gβ = J` where J is the Jacobian matrix

# Example
```julia
# Simultaneous marginal effects + parameter gradients for all variables
vars = [:x, :z]
de = derivativeevaluator_fd(compiled, data, vars)
g = Vector{Float64}(undef, length(vars))
Gβ = Matrix{Float64}(undef, length(de), length(vars))

# Get both marginal effects and parameter gradients
marginal_effects_eta!(g, Gβ, de, β, 1)

# g now contains marginal effects ∂η/∂vars
# Gβ now contains parameter gradients ∂(∂η/∂vars[j])/∂β[i]
```
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    Gβ::AbstractMatrix{Float64},
    de::FDEvaluator,
    β::AbstractVector{<:AbstractFloat},
    row::Int
)
    # Simple bounds checks without string interpolation to avoid allocations
    length(g) == length(de.vars) || throw(DimensionMismatch("gradient length mismatch"))
    size(Gβ, 1) == length(de) || throw(DimensionMismatch("Gβ first dimension must match length(de)"))
    size(Gβ, 2) == length(de.vars) || throw(DimensionMismatch("Gβ second dimension must match length(de.vars)"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Use derivative_modelrow! for Jacobian computation
    derivative_modelrow!(de.jacobian_buffer, de, row)

    # Extract parameter gradients (essentially free - copy of Jacobian)
    # Gβ[i,j] = ∂(∂η/∂vars[j])/∂β[i] = J[i,j]
    Gβ .= de.jacobian_buffer

    # Type barrier for zero-allocation matrix multiply: g = J'β
    _matrix_multiply_eta!(g, de.jacobian_buffer, β)

    return nothing
end

"""
    marginal_effects_eta!(g, de::FDEvaluator, β, row) -> g

Convenience method matching ADEvaluator signature - computes only marginal effects without parameter gradients.

This is a simpler interface that matches the AD backend signature, making it easier to write
backend-agnostic code. It internally allocates a temporary Gβ matrix which is discarded.

For performance-critical code where you need both marginal effects and parameter gradients,
use the full signature: `marginal_effects_eta!(g, Gβ, de, β, row)`.

# Arguments
- `g::Vector{Float64}`: Preallocated gradient buffer of length `length(de.vars)`
- `de::FDEvaluator`: FD evaluator built by `derivativeevaluator_fd(compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients of length `length(de)`
- `row::Int`: Row index to evaluate (1-based indexing)

# Returns
- `g`: Vector containing marginal effects `g[j] = ∂η/∂vars[j]`

# Example
```julia
# Simple usage matching AD backend
de_fd = derivativeevaluator_fd(compiled, data, [:x, :z])
g = Vector{Float64}(undef, length(de_fd.vars))
marginal_effects_eta!(g, de_fd, β, 1)  # Works like AD backend
```
"""
function marginal_effects_eta!(
    g::AbstractVector{Float64},
    de::FDEvaluator,
    β::AbstractVector{<:Real},
    row::Int
)
    # Allocate temporary Gβ matrix
    Gβ = Matrix{Float64}(undef, length(de), length(de.vars))

    # Call the full version
    marginal_effects_eta!(g, Gβ, de, β, row)

    return g
end

"""
    marginal_effects_eta!(G, Gβ_tensor, de::FDEvaluator, β, rows) -> (G, Gβ_tensor)

Batch computation of marginal effects and parameter gradients for multiple rows using finite differences - zero allocations after warmup.

Computes both marginal effects ∂η/∂x and parameter gradients ∂(∂η/∂x)/∂β for multiple rows
efficiently. The parameter gradient tensor is 3D: `Gβ_tensor[k, j, i] = ∂(∂η/∂vars[j])/∂β[i]` for row k.

**Note**: Computes derivatives only for variables in `de.vars` (specified during evaluator construction).

# Arguments
- `G::Matrix{Float64}`: Preallocated marginal effects matrix of size `(length(rows), length(de.vars))`
- `Gβ_tensor::Array{Float64, 3}`: Preallocated gradient tensor of size `(length(rows), length(de.vars), length(de))`
- `de::FDEvaluator`: FD evaluator built by `derivativeevaluator_fd(compiled, data, vars)`
- `β::AbstractVector{<:Real}`: Model coefficients (same β used for all rows)
- `rows::AbstractVector{Int}`: Row indices to evaluate (1-based indexing)

# Returns
- `(G, Gβ_tensor)`: Tuple containing marginal effects matrix and parameter gradient tensor
  - `G[k, j] = ∂η/∂vars[j]` for row `rows[k]`
  - `Gβ_tensor[k, j, i] = ∂(∂η/∂vars[j])/∂β[i]` for row `rows[k]`

# Performance Characteristics
- **Memory**: 0 bytes allocated (reuses evaluator buffers for each row)
- **Speed**: Efficient batch processing with optimized finite differences
- **Scaling**: Linear in number of rows with minimal per-row overhead

# Mathematical Method
For each row k:
1. Compute Jacobian J_k using `derivative_modelrow!`
2. Marginal effects: G[k, :] = J_k' * β
3. Parameter gradients: Gβ_tensor[k, :, :] = J_k

# Example
```julia
# Batch marginal effects + parameter gradients for uncertainty quantification
rows = [1, 3, 7, 12, 15]
G_batch = Matrix{Float64}(undef, length(rows), length(vars))
Gβ_tensor = Array{Float64, 3}(undef, length(rows), length(vars), length(de))

marginal_effects_eta!(G_batch, Gβ_tensor, de, β, rows)  # 0 bytes allocated

# Access: G_batch[1, :] = marginal effects for row 1
#         Gβ_tensor[1, :, :] = parameter gradients for row 1
```
"""
function marginal_effects_eta!(
    G::AbstractMatrix{Float64},
    Gβ_tensor::AbstractArray{Float64, 3},
    de::FDEvaluator,
    β::AbstractVector{<:AbstractFloat},
    rows::AbstractVector{Int}
)
    # Validate dimensions
    size(G, 1) == length(rows) || throw(DimensionMismatch("G first dimension must match length(rows)"))
    size(G, 2) == length(de.vars) || throw(DimensionMismatch("G second dimension must match length(de.vars)"))
    size(Gβ_tensor, 1) == length(rows) || throw(DimensionMismatch("Gβ_tensor first dimension must match length(rows)"))
    size(Gβ_tensor, 2) == length(de.vars) || throw(DimensionMismatch("Gβ_tensor second dimension must match length(de.vars)"))
    size(Gβ_tensor, 3) == length(de) || throw(DimensionMismatch("Gβ_tensor third dimension must match length(de)"))
    length(β) == length(de) || throw(DimensionMismatch("beta length mismatch"))

    # Batch processing: iterate rows efficiently
    for (k, row) in enumerate(rows)
        # Compute Jacobian for this row
        derivative_modelrow!(de.jacobian_buffer, de, row)

        # Compute marginal effects using type barrier: G[k, :] = J' * β
        G_k = view(G, k, :)  # View into k-th row
        _matrix_multiply_eta!(G_k, de.jacobian_buffer, β)

        # Copy parameter gradients: Gβ_tensor[k, :, :] = J
        Gβ_k = view(Gβ_tensor, k, :, :)  # View into k-th slice
        Gβ_k .= transpose(de.jacobian_buffer)
    end

    return nothing
end
