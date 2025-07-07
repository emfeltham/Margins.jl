# ame_representation.jl - COMPLETE DROP-IN REPLACEMENT

###############################################################################
# Zero-allocation AME at representative values using InplaceModeler
###############################################################################

function _ame_representation(
    ipm::InplaceModeler,
    df::DataFrame,
    focal::Symbol,
    repvals::AbstractDict{Symbol,<:AbstractVector},
    β::Vector{Float64},
    cholΣβ::LinearAlgebra.Cholesky,
    invlink::Function,
    dinvlink::Function,
    d2invlink::Function,
)
    # Representative value grid
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))

    nr, p = nrow(df), length(β)

    # Pre-allocate working objects - OPTIMIZED VERSION
    workdf = DataFrame(df, copycols=true)
    
    # Create single AMEWorkspace for all computations
    ws = AMEWorkspace(nr, p, workdf)
    
    # Store original focal column
    original_focal_col = copy(workdf[!, focal])

    # Result containers
    ame_dict = Dict{Tuple,Float64}()
    se_dict = Dict{Tuple,Float64}()
    grad_dict = Dict{Tuple,Vector{Float64}}()

    # Pre-analyze focal variable type
    focal_type = eltype(workdf[!, focal])
    
    # Pre-allocate perturbation array for continuous variables
    if focal_type <: Real && focal_type != Bool
        ws.pert_data[focal] = Vector{Float64}(undef, nr)
    end
    
    # Main computation loop
    for combo in combos
        # Modify representative values in-place
        @inbounds for (rv, val) in zip(repvars, combo)
            fill!(workdf[!, rv], val)
        end

        if focal_type <: Real && focal_type != Bool
            # OPTIMIZED: Continuous focal variable using zero-allocation approach
            # Update cached table in workspace
            ws.base_tbl = Tables.columntable(workdf)
            
            # Build base matrix for this representative combination
            modelmatrix!(ipm, ws.base_tbl, ws.X_base)
            
            # Get original values and compute perturbation - OPTIMIZED
            original_vals = ws.base_tbl[focal]
            max_abs = 0.0
            @inbounds @simd for i in 1:nr
                val = abs(original_vals[i])
                max_abs = val > max_abs ? val : max_abs
            end
            
            h = sqrt(eps(Float64)) * max(1.0, max_abs * 0.01)
            h = max(h, 1e-8)
            h = min(h, max_abs * 0.1)
            inv_h = 1.0 / h
            
            # Create perturbed values
            pert_vals = ws.pert_data[focal]
            @inbounds @simd for i in 1:nr
                pert_vals[i] = original_vals[i] + h
            end
            
            # Create perturbed table and compute derivatives
            pert_tbl = merge(ws.base_tbl, (focal => pert_vals,))
            modelmatrix!(ipm, pert_tbl, ws.Xdx)
            
            # Compute derivatives in-place
            @inbounds @simd for i in 1:(nr*p)
                ws.Xdx[i] = (ws.Xdx[i] - ws.X_base[i]) * inv_h
            end
            
            ame, se, grad = _ame_continuous!(
                β, cholΣβ, ws.X_base, ws.Xdx, dinvlink, d2invlink, ws
            )

            key = Tuple(combo)
            ame_dict[key] = ame
            se_dict[key] = se
            grad_dict[key] = grad

        elseif focal_type <: Bool
            # Boolean focal variable
            tbl = Tables.columntable(workdf)

            ame_b = Dict{Tuple,Float64}()
            se_b = Dict{Tuple,Float64}()
            grad_b = Dict{Tuple,Vector{Float64}}()

            _ame_factor_baseline!(
                ame_b, se_b, grad_b, ipm, tbl, workdf, β, Matrix(cholΣβ), focal, invlink, dinvlink
            )

            pair_key = first(keys(ame_b))
            key = Tuple(combo)

            ame_dict[key] = ame_b[pair_key]
            se_dict[key] = se_b[pair_key]
            grad_dict[key] = grad_b[pair_key]

        else
            # Multi-level categorical focal variable
            tbl = Tables.columntable(workdf)

            ame_sub = Dict{Tuple,Float64}()
            se_sub = Dict{Tuple,Float64}()
            grad_sub = Dict{Tuple,Vector{Float64}}()

            _ame_factor_allpairs!(
                ame_sub, se_sub, grad_sub, ipm, tbl, workdf, β, Matrix(cholΣβ), focal, invlink, dinvlink
            )

            repkey = Tuple(combo)
            for pair in keys(ame_sub)
                fullkey = (repkey..., pair...)
                ame_dict[fullkey] = ame_sub[pair]
                se_dict[fullkey] = se_sub[pair]
                grad_dict[fullkey] = grad_sub[pair]
            end
        end
    end

    # Restore original focal column
    workdf[!, focal] = original_focal_col

    return ame_dict, se_dict, grad_dict
end
