"""
    modelcols_alt(t::AbstractTerm, data)

Create a numerical "model columns" representation of data based on an
`AbstractTerm`.  `data` can either be a whole table (a property-accessible
collection of iterable columns or iterable collection of property-accessible
rows, as defined by [Tables.jl](https://github.com/JuliaData/Tables.jl) or a
single row (in the form of a `NamedTuple` of scalar values).  Tables will be
converted to a `NamedTuple` of `Vectors` (e.g., a `Tables.ColumnTable`).
"""
function modelcols_alt(t, d::D) where D
    Tables.istable(d) || throw(ArgumentError("Data of type $D is not a table!"))
    ## throw an error for t which don't have a more specific modelcols_alt method defined
    ## TODO: this seems like it ought to be handled by dispatching on something
    ## like modelcols_alt(::Any, ::NamedTuple) or modelcols_alt(::AbstractTerm, ::NamedTuple)
    ## but that causes ambiguity errors or under-constrained modelcols_alt methods for
    ## custom term types...
    d isa NamedTuple && throw(ArgumentError("don't know how to generate modelcols_alt for " *
                                            "term $t. Did you forget to call apply_schema?"))
    modelcols_alt(t, columntable(d))
end

"""
    modelcols_alt(ts::NTuple{N, AbstractTerm}, data) where N

When a tuple of terms is provided, `modelcols_alt` broadcasts over the individual
terms.  To create a single matrix, wrap the tuple in a [`MatrixTerm`](@ref).

# Example

```jldoctest
julia> using StableRNGs; rng = StableRNG(1); using Margins;

julia> d = (a = [1:9;], b = rand(rng, 9), c = repeat(["d","e","f"], 3));

julia> ts = Margins.apply_schema(term.((:a, :b, :c)), schema(d))
a(continuous)
b(continuous)
c(DummyCoding:3→2)

julia> cols = Margins.modelcols_alt(ts, d)
([1, 2, 3, 4, 5, 6, 7, 8, 9], [0.5851946422124186, 0.07733793456911231, 0.7166282400543453, 0.3203570514066232, 0.6530930076222579, 0.2366391513734556, 0.7096838914472361, 0.5577872440804086, 0.05079002172175784], [0.0 0.0; 1.0 0.0; … ; 1.0 0.0; 0.0 1.0])

julia> reduce(hcat, cols)
9×4 Matrix{Float64}:
 1.0  0.585195   0.0  0.0
 2.0  0.0773379  1.0  0.0
 3.0  0.716628   0.0  1.0
 4.0  0.320357   0.0  0.0
 5.0  0.653093   1.0  0.0
 6.0  0.236639   0.0  1.0
 7.0  0.709684   0.0  0.0
 8.0  0.557787   1.0  0.0
 9.0  0.05079    0.0  1.0

julia> Margins.modelcols_alt(MatrixTerm(ts), d)
9×4 Matrix{Float64}:
 1.0  0.585195   0.0  0.0
 2.0  0.0773379  1.0  0.0
 3.0  0.716628   0.0  1.0
 4.0  0.320357   0.0  0.0
 5.0  0.653093   1.0  0.0
 6.0  0.236639   0.0  1.0
 7.0  0.709684   0.0  0.0
 8.0  0.557787   1.0  0.0
 9.0  0.05079    0.0  1.0
```
"""
modelcols_alt(ts::TupleTerm, d::NamedTuple) = modelcols_alt.(ts, Ref(d))

modelcols_alt(t::Term, d::NamedTuple) = getproperty(d, t.sym)
modelcols_alt(t::ConstantTerm, d::NamedTuple) = t.n

modelcols_alt(ft::FunctionTerm, d::NamedTuple) =
    Base.Broadcast.materialize(lazy_modelcols_alt(ft, d))

lazy_modelcols_alt(ft::FunctionTerm, d::NamedTuple) =
    Base.Broadcast.broadcasted(ft.f, lazy_modelcols_alt.(ft.args, Ref(d))...)
lazy_modelcols_alt(x, d) = modelcols_alt(x, d)


modelcols_alt(t::ContinuousTerm, d::NamedTuple) = copy.(d[t.sym])

# replaced with function in "modelcols.jl"
# modelcols_alt(t::CategoricalTerm, d::NamedTuple) = t.contrasts[d[t.sym], :]


"""
    reshape_last_to_i(i::Int, a)

Reshape `a` so that its last dimension moves to dimension `i` (+1 if `a` is an
`AbstractMatrix`).
"""
reshape_last_to_i(i, a) = a
reshape_last_to_i(i, a::AbstractVector) = reshape(a, ones(Int, i-1)..., :)
reshape_last_to_i(i, a::AbstractMatrix) = reshape(a, size(a,1), ones(Int, i-1)..., :)

# an "inside out" kronecker-like product based on broadcasting reshaped arrays
# for a single row, some will be scalars, others possibly vectors.  for a whole
# table, some will be vectors, possibly some matrices
function kron_insideout(op::Function, args...)
    args = (reshape_last_to_i(i,a) for (i,a) in enumerate(args))
    out = broadcast(op, args...)
    # flatten array output to vector
    out isa AbstractArray ? vec(out) : out
end

function row_kron_insideout(op::Function, args...)
    rows = size(args[1], 1)
    args = (reshape_last_to_i(i,reshape(a, size(a,1), :)) for (i,a) in enumerate(args))
    # args = (reshape(a, size(a,1), ones(Int, i-1)..., :) for (i,a) in enumerate(args))
    reshape(broadcast(op, args...), rows, :)
end

# two options here: either special-case ColumnTable (named tuple of vectors)
# vs. vanilla NamedTuple, or reshape and use normal broadcasting
modelcols_alt(t::InteractionTerm, d::NamedTuple) =
    kron_insideout(*, (modelcols_alt(term, d) for term in t.terms)...)

function modelcols_alt(t::InteractionTerm, d::ColumnTable)
    row_kron_insideout(*, (modelcols_alt(term, d) for term in t.terms)...)
end

modelcols_alt(t::InterceptTerm{true}, d::NamedTuple) = ones(size(first(d)))
modelcols_alt(t::InterceptTerm{false}, d) = Matrix{Float64}(undef, size(first(d),1), 0)

modelcols_alt(t::FormulaTerm, d::NamedTuple) = (modelcols_alt(t.lhs,d), modelcols_alt(t.rhs, d))

function modelcols_alt(t::MatrixTerm, d::ColumnTable)
    mat = reduce(hcat, [modelcols_alt(tt, d) for tt in t.terms])
    reshape(mat, size(mat, 1), :)
end

modelcols_alt(t::MatrixTerm, d::NamedTuple) =
    reduce(vcat, [modelcols_alt(tt, d) for tt in t.terms])

