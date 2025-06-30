# ame_representation.jl

###############################################################################
# 3. MER helper: marginal effects at representative values
###############################################################################

"""
    _ame_representation(
      df::DataFrame,
      model,
      focal::Symbol,
      repvals::Dict{Symbol, Vector},
      fe_form,
      β::Vector{Float64},
      cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
      invlink::Function,
      dinvlink::Function,
      d2invlink::Function
    ) -> (Dict{Tuple,Float64}, Dict{Tuple,Float64}, Dict{Tuple,Vector{Float64}})

Compute the marginal effect of `focal` at all rep-value combos, in place,
without ever stacking a giant DataFrame—and safely under Threads.
"""
function _ame_representation(
    df::DataFrame,
    model,
    focal::Symbol,
    repvals::AbstractDict{Symbol,<:AbstractVector},
    fe_form,
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    invlink::Function,
    dinvlink::Function,
    d2invlink::Function,
)

    # 1️⃣  enumerate representative-value combinations
    repvars = collect(keys(repvals))
    combos  = collect(Iterators.product((repvals[r] for r in repvars)...))
    m       = length(combos)
    n       = nrow(df)

    # 2️⃣  per-thread scratch objects
    p          = size(modelmatrix(fe_form, df), 2)          # # columns of X
    nthreads   = Threads.nthreads()
    workdfs    = [DataFrame(df, copycols = true) for _ in 1:nthreads]
    workspaces = [AMEWorkspace(n, p)           for _ in 1:nthreads]
    Xs         = [Matrix{Float64}(undef, n, p) for _ in 1:nthreads]
    Xdxs       = [Matrix{Float64}(undef, n, p) for _ in 1:nthreads]

    # 3️⃣  outputs
    ames  = Vector{Float64}(undef, m)
    ses   = Vector{Float64}(undef, m)
    grads = Vector{Vector{Float64}}(undef, m)

    # 4️⃣  main loop (parallel over combos)
    Threads.@threads for idx in 1:m
        tid   = Threads.threadid()
        combo = combos[idx]
        dfl   = workdfs[tid]

        # overwrite rep-value columns *in this thread’s private copy*
        @inbounds for (rv, val) in zip(repvars, combo)
            fill!(dfl[!, rv], val)
        end

        # rebuild design for the *single* focal variable
        X    = Xs[tid]
        Xdx  = Xdxs[tid]
        build_continuous_design_single!(dfl, fe_form, focal, X, Xdx)

        ame_i, se_i, grad_i = _ame_continuous!(
            β, cholΣβ, X, Xdx,
            dinvlink, d2invlink,
            workspaces[tid]
        )

        ames[idx]  = ame_i
        ses[idx]   = se_i
        grads[idx] = grad_i
    end

    # 5️⃣  pack dictionaries
    ame_dict  = Dict{Tuple,Float64}()
    se_dict   = Dict{Tuple,Float64}()
    grad_dict = Dict{Tuple,Vector{Float64}}()
    for (i, combo) in enumerate(combos)
        ame_dict[combo]  = ames[i]
        se_dict[combo]   = ses[i]
        grad_dict[combo] = grads[i]
    end
    return ame_dict, se_dict, grad_dict
end
