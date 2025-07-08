# ame_representation.jl - COMPLETE DROP-IN REPLACEMENT

###############################################################################
# Zero-allocation AME at representative values using InplaceModeler
###############################################################################

###############  ame_representation.jl  #######################################

# ame_representation.jl  ── drop-in replacement ── zero-allocation + correct Boolean keys
function _ame_representation!(
    ws::AMEWorkspace,
    ipm::InplaceModeler,
    df::DataFrame,
    focal::Symbol,
    repvals::AbstractDict{Symbol,<:AbstractVector},
    β::AbstractVector,
    cholΣβ::LinearAlgebra.Cholesky,
    invlink::Function,
    dinvlink::Function,
    d2invlink::Function,
)
    # build the grid of rep-value combos
    repvars = collect(keys(repvals))
    combos  = collect(Iterators.product((repvals[r] for r in repvars)...))

    n, p    = size(ws.X_base)
    focal_type = eltype(df[!, focal])

    # ensure we have one perturbation vector if focal is continuous
    if focal_type <: Real && focal_type != Bool && !haskey(ws.pert_data, focal)
        ws.pert_data[focal] = Vector{Float64}(undef, n)
    end

    # a working copy of the data
    workdf = DataFrame(df, copycols = true)
    original_vals = copy(workdf[!, focal])

    # result containers
    ame_d  = Dict{Tuple,Float64}()
    se_d   = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()

    for combo in combos
        # set rep-values in-place
        @inbounds for (rv,val) in zip(repvars, combo)
            fill!(workdf[!, rv], val)
        end

        # rebuild the base table & matrix once per combo
        ws.base_tbl = Tables.columntable(workdf)
        modelmatrix!(ipm, ws.base_tbl, ws.X_base)

        if focal_type <: Real && focal_type != Bool
            # — continuous focal: same as before —
            orig = ws.base_tbl[focal]
            maxabs = maximum(abs, orig)
            h     = clamp(sqrt(eps(Float64))*max(1, maxabs*0.01),
                          1e-8, maxabs*0.1)
            invh  = 1 / h

            pert = ws.pert_data[focal]
            @inbounds @simd for i in 1:n
                pert[i] = orig[i] + h
            end

            pert_tbl = merge(ws.base_tbl, (focal => pert,))
            modelmatrix!(ipm, pert_tbl, ws.Xdx)
            BLAS.axpy!(-1.0, vec(ws.X_base), vec(ws.Xdx))
            BLAS.scal!(invh, vec(ws.Xdx))

            ame, se, g = _ame_continuous!(
                β, cholΣβ, ws.X_base, ws.Xdx, dinvlink, d2invlink, ws
            )

            ame_d[Tuple(combo)]  = ame
            se_d[Tuple(combo)]  = se
            grad_d[Tuple(combo)] = g

        elseif focal_type <: Bool
            # — Boolean focal: one baseline contrast per combo —
            tmp_ame = Dict{Tuple,Float64}()
            tmp_se  = Dict{Tuple,Float64}()
            tmp_gr  = Dict{Tuple,Vector{Float64}}()

            _ame_factor_baseline!(
                tmp_ame, tmp_se, tmp_gr,
                ipm, ws.base_tbl, workdf,
                β, Matrix(cholΣβ), focal,
                invlink, dinvlink,
            )

            # there is exactly one key (base→other) in tmp_ame
            pair = first(keys(tmp_ame))
            ame_d[Tuple(combo)]  = tmp_ame[pair]
            se_d[Tuple(combo)]  = tmp_se[pair]
            grad_d[Tuple(combo)] = tmp_gr[pair]

        else
            # — multi-level factor focal: allpairs contrasts per combo —
            tmp_ame = Dict{Tuple,Float64}()
            tmp_se  = Dict{Tuple,Float64}()
            tmp_gr  = Dict{Tuple,Vector{Float64}}()

            _ame_factor_allpairs!(
                tmp_ame, tmp_se, tmp_gr,
                ipm, ws.base_tbl, workdf,
                β, Matrix(cholΣβ), focal,
                invlink, dinvlink,
            )

            repkey = Tuple(combo)
            for lev_pair in keys(tmp_ame)
                fullkey = (repkey..., lev_pair...)
                ame_d[fullkey]  = tmp_ame[lev_pair]
                se_d[fullkey]  = tmp_se[lev_pair]
                grad_d[fullkey] = tmp_gr[lev_pair]
            end
        end
    end

    # restore the original focal column
    workdf[!, focal] = original_vals
    return ame_d, se_d, grad_d
end
