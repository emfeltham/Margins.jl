# ame_representation.jl - CLEAN REPLACEMENT

###############################################################################
# Optimized AME at representative values with minimal allocations
###############################################################################

function _ame_representation(
    df::DataFrame,
    focal::Symbol,
    repvals::AbstractDict{Symbol,<:AbstractVector},
    fe_rhs,
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

    # Pre-allocate working objects
    workdf = DataFrame(df, copycols=true)
    ws = AMEWorkspace(nr, p)
    X = Matrix{Float64}(undef, nr, p)
    Xdx = Matrix{Float64}(undef, nr, p)

    # Store original focal column
    original_focal_col = copy(workdf[!, focal])

    # Result containers
    ame_dict = Dict{Tuple,Float64}()
    se_dict = Dict{Tuple,Float64}()
    grad_dict = Dict{Tuple,Vector{Float64}}()

    # Pre-analyze for continuous variables
    base_tbl = Tables.columntable(workdf)
    focal_type = eltype(workdf[!, focal])
    
    active_terms = nothing
    if focal_type <: Real && focal_type != Bool
        active_terms = analyze_variable_dependencies_fast(fe_rhs, [focal], base_tbl)
    end
    
    # Main computation loop
    for combo in combos
        # Modify representative values in-place
        @inbounds for (rv, val) in zip(repvars, combo)
            fill!(workdf[!, rv], val)
        end

        if focal_type <: Real && focal_type != Bool
            # Continuous focal variable
            build_continuous_design_single_fast!(workdf, fe_rhs, focal, X, Xdx, active_terms)

            ame, se, grad = _ame_continuous!(
                β, cholΣβ, X, Xdx, dinvlink, d2invlink, ws
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
                ame_b, se_b, grad_b,
                tbl, fe_rhs, β, cholΣβ, focal, invlink, dinvlink
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
                ame_sub, se_sub, grad_sub,
                tbl, fe_rhs, β, cholΣβ, focal, invlink, dinvlink
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

