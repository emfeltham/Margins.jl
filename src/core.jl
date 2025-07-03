# core.jl - OPTIMIZED VERSION

###############################################################################
# Ultra-optimized margins wrapper with matrix reuse
###############################################################################

"""
    margins(model, vars, df; kwargs...)  ->  MarginsResult

Optimized version with matrix reuse and smart derivative computation.
"""
function margins(
    model,
    vars,
    df::AbstractDataFrame;
    vcov::Function = StatsBase.vcov,
    repvals::AbstractDict{Symbol,<:AbstractVector} = Dict{Symbol,Vector{Float64}}(),
    pairs::Symbol = :allpairs,
    type::Symbol = :dydx,
)

    type ∈ (:dydx, :predict) ||
        throw(ArgumentError("`type` must be :dydx or :predict, got `$type`"))

    # Setup
    varlist = isa(vars, Symbol) ? [vars] : collect(vars)
    invlink, dinvlink, d2invlink = link_functions(model)

    fe_form = fixed_effects_form(model)
    fe_rhs = fe_form.rhs

    # Variable classification
    iscts(v) = eltype(df[!, v]) <: Real && eltype(df[!, v]) != Bool
    cts_vars = filter(iscts, union(varlist, keys(repvals)))
    
    # MAJOR OPTIMIZATION: Use matrix reuse
    X_base, Xdx_list = build_design_matrices_optimized(model, df, fe_rhs, cts_vars)

    n, p = size(X_base)
    β, Σβ = coef(model), vcov(model)
    cholΣβ = cholesky(Σβ)

    tbl0 = Tables.columntable(df)

    # Pre-allocate result containers
    result_map = Dict{Symbol,Union{Float64,Dict{Tuple,Float64}}}()
    se_map = Dict{Symbol,Union{Float64,Dict{Tuple,Float64}}}()
    grad_map = Dict{Symbol,Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}()

    ensure_dict!(v) = begin
        result_map[v] isa Dict || (result_map[v] = Dict{Tuple,Float64}())
        se_map[v] isa Dict || (se_map[v] = Dict{Tuple,Float64}())
        grad_map[v] isa Dict || (grad_map[v] = Dict{Tuple,Vector{Float64}}())
    end
    
    if type == :predict
        if isempty(repvals)
            # Optimized prediction computation
            η = X_base * β
            @inbounds @simd ivdep for i in 1:n
                η[i] = invlink(η[i])
            end
            pred = sum(η) / n

            # Efficient gradient computation
            mul!(η, X_base, β)  # Recompute η
            @inbounds @simd ivdep for i in 1:n
                η[i] = dinvlink(η[i])
            end
            grad = (X_base' * η) ./ n
            se = sqrt(dot(grad, Σβ * grad))

            for v in varlist
                result_map[v] = pred
                se_map[v] = se
                grad_map[v] = grad
            end

        else
            # Representative values prediction
            repvars = collect(keys(repvals))
            combos = collect(Iterators.product((repvals[r] for r in repvars)...))

            workdf = DataFrame(df, copycols=true)
            η_work = Vector{Float64}(undef, n)
            μ_work = Vector{Float64}(undef, n)
            μp_work = Vector{Float64}(undef, n)
            grad_work = Vector{Float64}(undef, p)
            X_buf = similar(X_base)

            for combo in combos
                for (rv, val) in zip(repvars, combo)
                    fill!(workdf[!, rv], val)
                end

                modelmatrix!(X_buf, fe_rhs, Tables.columntable(workdf))

                mul!(η_work, X_buf, β)
                @inbounds @simd ivdep for i in 1:n
                    μ_work[i] = invlink(η_work[i])
                    μp_work[i] = dinvlink(η_work[i])
                end

                pred = sum(μ_work) / n
                mul!(grad_work, X_buf', μp_work)
                grad_work ./= n
                se = sqrt(dot(grad_work, Σβ * grad_work))

                for v in varlist
                    ensure_dict!(v)
                    result_map[v][combo] = pred
                    se_map[v][combo] = se
                    grad_map[v][combo] = copy(grad_work)
                end
            end
        end
    
    else # :dydx branch
        cat_vars = setdiff(varlist, cts_vars)

        if isempty(repvals)
            # Batch continuous AMEs
            if !isempty(cts_vars)
                requested_cts_indices = Int[]
                requested_cts_vars = Symbol[]
                
                for (i, var) in enumerate(cts_vars)
                    if var in varlist
                        push!(requested_cts_indices, i)
                        push!(requested_cts_vars, var)
                    end
                end
                
                if !isempty(requested_cts_indices)
                    ames, ses, grads = compute_ames_batch(
                        β, cholΣβ, X_base, 
                        Xdx_list[requested_cts_indices],
                        dinvlink, d2invlink, n, p
                    )
                    
                    for (i, var) in enumerate(requested_cts_vars)
                        result_map[var] = ames[i]
                        se_map[var] = ses[i]
                        grad_map[var] = grads[i]
                    end
                end
            end

            # Categorical AMEs
            for v in cat_vars
                ame_d = Dict{Tuple,Float64}()
                se_d = Dict{Tuple,Float64}()
                grad_d = Dict{Tuple,Vector{Float64}}()
                
                if pairs == :baseline
                    _ame_factor_baseline!(
                        ame_d, se_d, grad_d,
                        tbl0, fe_rhs, β, Σβ, v,
                        invlink, dinvlink
                    )
                else
                    _ame_factor_allpairs!(
                        ame_d, se_d, grad_d,
                        tbl0, fe_rhs, β, Σβ, v,
                        invlink, dinvlink
                    )
                end
                
                result_map[v] = ame_d
                se_map[v] = se_d
                grad_map[v] = grad_d
            end

        else
            # AMEs at representative values
            for v in varlist
                ame_d, se_d, g_d = _ame_representation(
                    df, v, repvals,
                    fe_rhs, β, cholΣβ,
                    invlink, dinvlink, d2invlink
                )

                result_map[v] = ame_d
                se_map[v] = se_d
                grad_map[v] = g_d
            end
        end
    end

    return MarginsResult{type}(
        varlist, repvals, result_map, se_map, grad_map,
        n, dof_residual(model),
        string(family(model).dist),
        string(family(model).link),
    )
end

"""
Batch AME computation for multiple continuous variables
"""
function compute_ames_batch(
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    X_base::Matrix{Float64},
    Xdx_list::Vector{Matrix{Float64}},
    dinvlink::Function,
    d2invlink::Function,
    n::Int,
    p::Int
)
    k = length(Xdx_list)
    
    # Pre-allocate results
    ames = Vector{Float64}(undef, k)
    ses = Vector{Float64}(undef, k)
    grads = Vector{Vector{Float64}}(undef, k)
    
    # Shared workspace
    ws = AMEWorkspace(n, p)
    
    # Pre-compute base linear predictor
    mul!(ws.η, X_base, β)
    
    # Pre-compute link function values
    @inbounds @simd ivdep for i in 1:n
        ηi = ws.η[i]
        ws.arr2[i] = dinvlink(ηi)
        ws.arr1[i] = d2invlink(ηi)
    end
    
    # Process each variable
    for j in 1:k
        Xdx = Xdx_list[j]
        
        mul!(ws.dη, Xdx, β)
        
        # AME computation
        sum_ame = 0.0
        @inbounds @simd ivdep for i in 1:n
            dηi = ws.dη[i]
            mp = ws.arr2[i]
            mpp = ws.arr1[i]
            
            sum_ame += mp * dηi
            ws.arr1[i] = mpp * dηi
        end
        ames[j] = sum_ame / n
        
        # Gradient computation
        mul!(ws.buf1, X_base', ws.arr1)
        mul!(ws.buf2, Xdx', ws.arr2)
        
        inv_n = 1.0 / n
        @inbounds @simd ivdep for i in 1:p
            ws.buf1[i] = (ws.buf1[i] + ws.buf2[i]) * inv_n
        end
        
        # Standard error
        mul!(ws.buf2, cholΣβ.U, ws.buf1)
        ses[j] = norm(ws.buf2)
        
        grads[j] = copy(ws.buf1)
        
        # Restore for next iteration
        if j < k
            @inbounds @simd ivdep for i in 1:n
                ws.arr1[i] = d2invlink(ws.η[i])
            end
        end
    end
    
    return ames, ses, grads
end
