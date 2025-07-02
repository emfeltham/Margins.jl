# workspace.jl - DROP-IN REPLACEMENT
# Optimized workspace with better memory layout

"""
Enhanced workspace that maintains compatibility with existing code
while providing better memory efficiency.
"""
struct AMEWorkspace
    η::Vector{Float64}
    dη::Vector{Float64}
    arr1::Vector{Float64}
    arr2::Vector{Float64}
    buf1::Vector{Float64}
    buf2::Vector{Float64}
    
    # Internal constructor for better memory layout
    function AMEWorkspace(n::Int, p::Int)
        new(
            Vector{Float64}(undef, n),   # η
            Vector{Float64}(undef, n),   # dη
            Vector{Float64}(undef, n),   # arr1
            Vector{Float64}(undef, n),   # arr2
            Vector{Float64}(undef, p),   # buf1
            Vector{Float64}(undef, p),   # buf2
        )
    end
end

"""
Constructor function (maintains existing API).
"""
function AMEWorkspace(n, p)
    AMEWorkspace(Int(n), Int(p))
end