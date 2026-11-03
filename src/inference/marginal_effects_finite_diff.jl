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
