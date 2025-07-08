# ame_representation.jl - WORKSPACE REUSE VERSION

###############################################################################
# Zero-allocation AME at representative values - maximize workspace reuse
###############################################################################

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

    # OPTIMIZATION: Create single working DataFrame and reuse
    workdf = DataFrame(df, copycols = true)
    original_focal = copy(workdf[!, focal])
    
    # Pre-allocate temporary working NamedTuple storage
    work_tbl_cache = Dict{Tuple,NamedTuple}()

    # result containers
    ame_d  = Dict{Tuple,Float64}()
    se_d   = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()

    for combo in combos
        # set rep-values in-place (reuse same DataFrame)
        @inbounds for (rv,val) in zip(repvars, combo)
            fill!(workdf[!, rv], val)
        end

        # OPTIMIZATION: Cache the columntable for this combo
        combo_key = Tuple(combo)
        if !haskey(work_tbl_cache, combo_key)
            work_tbl_cache[combo_key] = Tables.columntable(workdf)
        end
        work_tbl = work_tbl_cache[combo_key]

        # Update workspace base table and matrix
        ws.base_tbl = work_tbl
        modelmatrix!(ipm, ws.base_tbl, ws.X_base)

        if focal_type <: Real && focal_type != Bool
            # — continuous focal: REUSE workspace perturbation infrastructure —
            
            # Ensure perturbation vector exists for focal variable
            if !haskey(ws.pert_data, focal)
                ws.pert_data[focal] = Vector{Float64}(undef, n)
                ws.pert_cache[focal] = merge(work_tbl, (focal => ws.pert_data[focal],))
            else
                # Update cached NamedTuple with current rep values
                ws.pert_cache[focal] = merge(work_tbl, (focal => ws.pert_data[focal],))
            end
            
            orig = ws.base_tbl[focal]
            maxabs = maximum(abs, orig)
            h     = clamp(sqrt(eps(Float64))*max(1, maxabs*0.01),
                          1e-8, maxabs*0.1)
            invh  = 1 / h

            pert = ws.pert_data[focal]
            @inbounds @simd for i in 1:n
                pert[i] = orig[i] + h
            end

            pert_tbl = ws.pert_cache[focal]
            modelmatrix!(ipm, pert_tbl, ws.Xdx)
            BLAS.axpy!(-1.0, vec(ws.X_base), vec(ws.Xdx))
            BLAS.scal!(invh, vec(ws.Xdx))

            ame, se, g_ref = _ame_continuous!(
                β, cholΣβ, ws.X_base, ws.Xdx, dinvlink, d2invlink, ws
            )

            ame_d[combo_key]  = ame
            se_d[combo_key]  = se
            grad_d[combo_key] = copy(g_ref)  # Copy since workspace will be reused

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
            ame_d[combo_key]  = tmp_ame[pair]
            se_d[combo_key]  = tmp_se[pair]
            grad_d[combo_key] = tmp_gr[pair]

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

            repkey = combo_key
            for lev_pair in keys(tmp_ame)
                fullkey = (repkey..., lev_pair...)
                ame_d[fullkey]  = tmp_ame[lev_pair]
                se_d[fullkey]  = tmp_se[lev_pair]
                grad_d[fullkey] = tmp_gr[lev_pair]
            end
        end
    end

    # restore the original focal column
    workdf[!, focal] = original_focal
    return ame_d, se_d, grad_d
end
