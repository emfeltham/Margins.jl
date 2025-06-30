struct AMEWorkspace
    η::Vector{Float64}
    dη::Vector{Float64}
    arr1::Vector{Float64}
    arr2::Vector{Float64}
    buf1::Vector{Float64}
    buf2::Vector{Float64}
end

function AMEWorkspace(n, p)
    AMEWorkspace(
        Vector{Float64}(undef,n),
        Vector{Float64}(undef,n),
        Vector{Float64}(undef,n),
        Vector{Float64}(undef,n),
        Vector{Float64}(undef,p),
        Vector{Float64}(undef,p),
    )
end
