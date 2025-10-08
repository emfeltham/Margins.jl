# utilities.jl
# Utility functions for derivative operations

# Type barrier function for zero-allocation matrix multiply
# Handles arbitrary Real coefficient types with zero overhead for Float64
# (compiler eliminates Float64() conversion when β is already Float64)
@noinline function _matrix_multiply_eta!(
    g::AbstractVector{Float64},
    jacobian_buffer::Matrix{Float64},
    β::AbstractVector{<:Real}
)
    @inbounds @fastmath for j in eachindex(g)
        acc = 0.0
        for i in 1:size(jacobian_buffer, 1)
            β_i = Float64(β[i])
            acc += jacobian_buffer[i, j] * β_i
        end
        g[j] = acc
    end
    return g
end

# Type barrier function for zero-allocation matrix multiply
@noinline function _matrix_multiply_eta!(
    g::AbstractVector{Float64},
    jacobian_buffer::Matrix{Float64},
    βref::Vector{Float64}
)
    @inbounds @fastmath for j in eachindex(g)
        acc = 0.0
        for i in 1:size(jacobian_buffer, 1)
            acc += jacobian_buffer[i, j] * βref[i]

        end
        g[j] = acc
    end
    return g
end
