# typicals get.jl

function get_typicals(df, variables; typical = mean)
    typs = Dict{Symbol, Union{Float64, Tuple{Vector{Float64}, Vector{String}}}}()
    for v in variables
        x = df[!, v]

        # enforce that categorical vars are CategoricalVector, for levels-order
        # these are the simple mean types
        c1 = typeof(x) <: Vector{T} where T <: Real
        c2 = typeof(x) <: Vector{T} where T <: Bool
        c3 = typeof(x) <: BitVector

        typs[v] = if c1 | c2 | c3
            typical(x)
        elseif typeof(x) <: CategoricalVector
            lv = levels(x)
            ([sum(x .== l) * inv(length(x)) for l in lv], convert(Vector{String}, lv))
        else
            error("what type?")
        end
    end
    return typs
end
