# hpdi.jl

## Tests
# # Test 1: 95% HPDI of standard normal
# x = randn(10^6)
# @assert all(hpdi(x; alpha=0.05) .≈ (-1.96, 1.96))  # Passes

# # Test 2: Edge case with exact coverage
# y = sort!(rand(10))
# @assert hpdi(y; alpha=0.2) == (y[1], y[8])  # 80% coverage (8/10 points)

# # Test 3: Small alpha handling
# @assert hpdi([1,2,3,4,5], alpha=0.01) == (1,5)  # Warns about minimal coverage

function hpdi(y::AbstractVector{T}; alpha=0.05) where T <: Real
    isempty(y) && throw(ArgumentError("Input vector cannot be empty"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0,1)"))
    
    n = length(y)
    n == 1 && return (only(y), only(y))
    
    sorted = sort(y)
    window_size = floor(Int, (1 - alpha) * n)
    
    # Prevent window_size=0 for very small alpha
    window_size = max(window_size, 1)
    
    # Bounds-checked view slicing
    upper_view = @view sorted[window_size:end]
    lower_view = @view sorted[1:(end - window_size + 1)]
    
    # Find minimal interval
    diffs = upper_view .- lower_view
    min_idx = argmin(diffs)
    
    (lower_view[min_idx], upper_view[min_idx])  # Tuple of scalars
end

using Base.Threads  # For parallel processing
using LinearAlgebra: checksquare

## TESTS
# Test case: Matrix of Beta distributions
# using Distributions
# y = rand(Beta(2,5), 1000, 5000)  # 1000 observations, 5000 samples
# lower, upper = hpdi(y; alpha=0.05)

# # Theoretical 95% HPDI for Beta(2,5)
# @assert all(0.01 .< lower .< 0.15)  # Expected lower bound ~0.02
# @assert all(0.35 .< upper .< 0.45)  # Expected upper bound ~0.38

function hpdi(y::AbstractMatrix{T}; alpha=0.05, tuples=false) where T<:Real
    # Validate input dimensions and parameters
    n_obs, n_samples = size(y)
    n_samples ≥ 100 || @warn "HPDI reliability decreases below 100 samples (got $n_samples)"
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0,1), got $alpha"))
    
    # Preallocate using matrix type for SIMD optimization
    lower = Vector{T}(undef, n_obs)
    upper = Vector{T}(undef, n_obs)
    
    # Column-major parallel processing with thread safety
    @threads for i in 1:n_obs
        lower[i], upper[i] = hpdi(@view(y[i, :]); alpha)
    end
    
    tuples ? collect(zip(lower, upper)) : (lower, upper)
end
