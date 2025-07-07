# core.jl - OPTIMIZED VERSION WITH EfficientModelMatrices.jl

###############################################################################
# Ultra-optimized margins wrapper with EfficientModelMatrices integration
###############################################################################

"""
    margins(model, vars, df; kwargs...)  ->  MarginsResult

Optimized version using EfficientModelMatrices.jl for zero-allocation model matrix construction.
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

    # MAJOR OPTIMIZATION: Create InplaceModeler for zero-allocation matrix construction
    n, p = nrow(df), width(fixed_effects_form(model).rhs)
    ipm = InplaceModeler(model, n)
    
    # Pre-allocate matrices that will be reused
    X_base = Matrix{Float64}(undef, n, p)
    X_work = Matrix{Float64}(undef, n, p)
    
    # Build base design matrix once
    tbl0 = Tables.columntable(df)
    modelmatrix!(ipm, tbl0, X_base)

    β, Σβ = coef(model), vcov(model)
    cholΣβ = cholesky(Σβ)

    # Variable classification
    iscts(v) = eltype(df[!, v]) <: Real && eltype(df[!, v]) != Bool
    cts_vars = filter(iscts, union(varlist, keys(repvals)))
    cat_vars = setdiff(varlist, cts_vars)

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

            for combo in combos
                for (rv, val) in zip(repvars, combo)
                    fill!(workdf[!, rv], val)
                end

                # Zero-allocation matrix construction
                work_tbl = Tables.columntable(workdf)
                modelmatrix!(ipm, work_tbl, X_work)

                mul!(η_work, X_work, β)
                @inbounds @simd ivdep for i in 1:n
                    μ_work[i] = invlink(η_work[i])
                    μp_work[i] = dinvlink(η_work[i])
                end

                pred = sum(μ_work) / n
                mul!(grad_work, X_work', μp_work)
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
        if isempty(repvals)
            # OPTIMIZED: Batch continuous AMEs using new workspace approach
            if !isempty(cts_vars)
                requested_cts_vars = filter(var -> var in varlist, cts_vars)
                
                if !isempty(requested_cts_vars)
                    ames, ses, grads = compute_continuous_ames_batch!(
                        ipm, df, requested_cts_vars, β, cholΣβ, dinvlink, d2invlink
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
                        ame_d, se_d, grad_d, ipm, tbl0, df, β, Σβ, v, invlink, dinvlink
                    )
                else
                    _ame_factor_allpairs!(
                        ame_d, se_d, grad_d, ipm, tbl0, df, β, Σβ, v, invlink, dinvlink
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
                    ipm, df, v, repvals, β, cholΣβ, invlink, dinvlink, d2invlink
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
