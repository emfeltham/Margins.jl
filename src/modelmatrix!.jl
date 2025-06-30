
# ------------------------------------------------ public helper -----------

"""
    modelmatrix!(X, rhs_with_schema, data)

Overwrite `X` with the design matrix that `StatsModels.modelmatrix` would
return *without allocating a new matrix*.  
`rhs_with_schema` **must** already have a schema (e.g. take it from
`mf.schema.rhs` or `ModelMatrix(mf).rhs`).  `data` can be any
Tables-compatible object.
"""
function modelmatrix!(
    X::AbstractMatrix,
    rhs,
    data;
)
    StatsModels.has_schema(rhs)      || error("`rhs` has no schema")
    Tables.istable(data)             || error("`data` is not Tables-compatible")

    modelcols!(X, rhs, Tables.columntable(data))
    return X
end

# convenience: pull the baked RHS from a ModelFrame ------------------------
getrhs(mf::StatsModels.ModelFrame) = mf.schema.rhs
export modelmatrix!, getrhs

# ------------------------------------------------ internal work-horse -----

"""
    modelcols!(dest, rhs_with_schema, tbl; hints=Dict(), mod=StatisticalModel)

Fill `dest` (size `n × p`) with the numeric columns for `rhs_with_schema`
*in place*.  Returns `dest`.
"""
function modelcols!(
    dest::AbstractMatrix,
    rhs,
    tbl
)
    # 1. Get the list of MatrixTerms to expand -----------------------------
    mterms =
        rhs isa StatsModels.MatrixTerm ? (rhs,) :
        StatsModels.collect_matrix_terms(rhs)

    # 2. Copy every column into the destination buffer ---------------------
    colofs = 0
    for mt in mterms
        cols = StatsModels.modelcols(mt, tbl)  # <- public API
        # `cols` can be a Vector or a Matrix
        if cols isa AbstractVector         # 1 column
            dest[:, colofs + 1] .= cols
            colofs += 1
        else                               # k columns
            k = size(cols, 2)
            dest[:, colofs+1 : colofs+k] .= cols
            colofs += k
        end
    end

    # 3. Sanity check ------------------------------------------------------
    colofs == size(dest, 2) ||
        throw(DimensionMismatch("dest has $(size(dest,2)) columns, but \
                                 RHS produced $colofs"))

    return dest
end


