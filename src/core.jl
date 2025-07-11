# core.jl - EFFICIENT VERSION: Minimal allocations and smart memory sharing

###############################################################################
# Main margins() function with efficient selective updates
# EFFICIENT: Zero-copy data handling and lazy matrix construction
###############################################################################

"""
    margins(model, vars, df; kwargs...)  ->  MarginsResult

Compute average marginal effects (AMEs) or average predictions using efficient 
selective matrix updates and minimal memory allocations.
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
    # Validate arguments
    type ∈ (:dydx, :predict) ||
        throw(ArgumentError("`type` must be :dydx or :predict, got `$type`"))
    
    pairs ∈ (:allpairs, :baseline) ||
        throw(ArgumentError("`pairs` must be :allpairs or :baseline, got `$pairs`"))

    # Normalize variables to vector
    varlist = isa(vars, Symbol) ? [vars] : collect(vars)
    
    # Get model information
    n = nrow(df)
    
    # EFFICIENT: Build workspace with minimal allocations
    ws = AMEWorkspace(model, df)
    
    # Get link functions
    invlink, dinvlink, d2invlink = link_functions(model)
    
    # Get coefficients and covariance
    β = coef(model)
    Σβ = vcov(model)
    cholΣβ = cholesky(Σβ)
    
    # Classify variables by type
    iscts(v) = eltype(df[!, v]) <: Real && eltype(df[!, v]) != Bool
    cts_vars = filter(iscts, union(varlist, keys(repvals)))
    cat_vars = setdiff(varlist, cts_vars)
    
    # Result containers
    result_map = Dict{Symbol,Any}()
    se_map     = Dict{Symbol,Any}()
    grad_map   = Dict{Symbol,Any}()
    
    # Branch by computation type
    if type == :predict
        compute_predictions_efficient!(result_map, se_map, grad_map, varlist, repvals, 
                                      ws, β, Σβ, invlink, dinvlink)
    else # :dydx
        if isempty(repvals)
            # Standard AMEs without representative values
            compute_standard_ames_efficient!(
                result_map, se_map, grad_map, varlist, 
                cts_vars, cat_vars, pairs, ws, β, cholΣβ, 
                invlink, dinvlink, d2invlink, df
            )
        else
            # AMEs at representative values
            compute_repval_ames_efficient!(
                result_map, se_map, grad_map, varlist, repvals,
                ws, β, cholΣβ, invlink, dinvlink, d2invlink, df
            )
        end
    end
    
    return MarginsResult{type}(
        varlist, repvals, result_map, se_map, grad_map,
        n, dof_residual(model),
        string(family(model).dist),
        string(family(model).link),
    )
end

###############################################################################
# Efficient Standard AME computation
###############################################################################

"""
    compute_standard_ames_efficient!(...)

EFFICIENT: Compute standard AMEs with minimal allocations and smart column sharing.
"""
function compute_standard_ames_efficient!(result_map, se_map, grad_map, varlist, cts_vars, cat_vars, 
                                         pairs, ws, β, cholΣβ, invlink, dinvlink, d2invlink, df)
    # EFFICIENT: Continuous variables with analytical derivatives
    if !isempty(cts_vars)
        requested_cts = filter(v -> v in varlist, cts_vars)
        if !isempty(requested_cts)
            ames, ses, grads = compute_continuous_ames_efficient!(
                requested_cts, ws, β, cholΣβ, dinvlink, d2invlink
            )
            
            for (i, v) in enumerate(requested_cts)
                result_map[v] = ames[i]
                se_map[v] = ses[i]
                grad_map[v] = grads[i]
            end
        end
    end
    
    # EFFICIENT: Categorical variables with selective updates
    for v in cat_vars
        ame_d = Dict{Tuple,Float64}()
        se_d = Dict{Tuple,Float64}()
        grad_d = Dict{Tuple,Vector{Float64}}()
        
        if pairs == :baseline
            compute_factor_baseline_efficient!(
                ame_d, se_d, grad_d, v, ws, β, 
                vcov(cholΣβ), invlink, dinvlink, df
            )
        else # :allpairs
            compute_factor_allpairs_efficient!(
                ame_d, se_d, grad_d, v, ws, β, 
                vcov(cholΣβ), invlink, dinvlink, df
            )
        end
        
        result_map[v] = ame_d
        se_map[v] = se_d
        grad_map[v] = grad_d
    end
end

###############################################################################
# Efficient Continuous AME Computation
###############################################################################

"""
    compute_continuous_ames_efficient!(variables::Vector{Symbol}, ws::AMEWorkspace, ...)

EFFICIENT: Compute AMEs for continuous variables with analytical derivatives and minimal allocations.
"""
function compute_continuous_ames_efficient!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                          β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                          dinvlink::Function, d2invlink::Function)
    k = length(variables)
    
    # Pre-allocate results
    ames = Vector{Float64}(undef, k)
    ses = Vector{Float64}(undef, k)
    grads = Vector{Vector{Float64}}(undef, k)
    
    # EFFICIENT: Process each variable with minimal matrix operations
    for (j, variable) in enumerate(variables)
        # Validate that variable affects some columns
        if !haskey(ws.variable_plans, variable)
            throw(ArgumentError(
                "Variable $variable not found in variable plans. " *
                "Only variables that appear in the model are supported."
            ))
        end
        
        # EFFICIENT: Prepare analytical derivatives (only affected columns)
        prepare_analytical_derivatives_efficient!(ws, variable)
        
        # EFFICIENT: Compute AME using lazy matrix construction
        ame, se, grad_ref = compute_ame_analytical_efficient!(
            β, cholΣβ, ws, dinvlink, d2invlink
        )
        
        # Store results
        ames[j] = ame
        ses[j] = se
        grads[j] = copy(grad_ref)  # Copy since workspace will be reused
    end
    
    return ames, ses, grads
end

"""
    compute_ame_analytical_efficient!(β, cholΣβ, ws, dinvlink, d2invlink)

EFFICIENT: Core AME computation with lazy matrix access and optimized numerics.
"""
function compute_ame_analytical_efficient!(
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    ws::AMEWorkspace,
    dinvlink::Function,
    d2invlink::Function
)
    # EFFICIENT: Lazy matrix access - only build base matrix when needed
    X = get_base_matrix!(ws)
    Xdx = ws.derivative_matrix  # Already computed for affected columns only
    
    n, p = size(X)
    
    # Unpack workspace vectors (pre-allocated, reused)
    η, dη = ws.η, ws.dη
    μp_vals, μpp_vals = ws.μp_vals, ws.μpp_vals
    grad_work = ws.grad_work
    temp1, temp2 = ws.temp1, ws.temp2
    
    # EFFICIENT: Compute linear predictors with BLAS
    mul!(η, X, β)
    mul!(dη, Xdx, β)
    
    # EFFICIENT: Vectorized link function computation with bounds checking
    sum_ame = 0.0
    n_valid = 0
    
    @inbounds for i in 1:n
        ηi = η[i]
        dηi = dη[i]
        
        # Skip problematic observations
        if !isfinite(ηi) || !isfinite(dηi) || abs(ηi) > 50.0 || abs(dηi) > 50.0
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        # Compute link function derivatives
        local μp, μpp
        try
            μp = dinvlink(ηi)
            μpp = d2invlink(ηi)
        catch
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        # Check for reasonable outputs
        if !isfinite(μp) || !isfinite(μpp)
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        marginal_effect = μp * dηi
        
        if isfinite(marginal_effect) && abs(marginal_effect) < 1e10
            sum_ame += marginal_effect
            n_valid += 1
            μp_vals[i] = μp
            μpp_vals[i] = μpp * dηi
        else
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
        end
    end
    
    # Compute AME
    ame = sum_ame / n
    
    # EFFICIENT: Gradient computation with BLAS and pre-allocated vectors
    se = NaN
    fill!(grad_work, 0.0)
    
    try
        # Use pre-allocated temporary vectors
        mul!(temp1, X', μpp_vals)
        mul!(temp2, Xdx', μp_vals)
        
        # Check for numerical issues
        if all(isfinite, temp1) && all(isfinite, temp2)
            inv_n = 1.0 / n
            @inbounds @simd for i in 1:p
                grad_work[i] = (temp1[i] + temp2[i]) * inv_n
                if !isfinite(grad_work[i])
                    grad_work[i] = 0.0
                end
            end
            
            # EFFICIENT: SE computation using Cholesky factorization
            grad_norm = norm(grad_work)
            if grad_norm > 0.0 && isfinite(grad_norm) && grad_norm < 1e6
                mul!(temp1, cholΣβ.U, grad_work)
                se_squared = dot(temp1, temp1)
                if se_squared >= 0 && isfinite(se_squared)
                    se = sqrt(se_squared)
                end
            end
        end
        
    catch e
        @warn "Gradient computation failed: $e"
    end
    
    return ame, se, grad_work
end

###############################################################################
# Efficient Categorical AME Functions (delegated to ame_factor.jl)
###############################################################################

function compute_factor_baseline_efficient!(ame_d, se_d, grad_d, variable::Symbol,
                                          ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          df::AbstractDataFrame)
    # Delegate to efficient factor computation
    compute_factor_baseline_selective!(ame_d, se_d, grad_d, variable, ws, β, Σβ, 
                                     invlink, dinvlink, df, ws.ipm)
end

function compute_factor_allpairs_efficient!(ame_d, se_d, grad_d, variable::Symbol,
                                          ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          df::AbstractDataFrame)
    # Delegate to efficient factor computation
    compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable, ws, β, Σβ, 
                                     invlink, dinvlink, df, ws.ipm)
end

###############################################################################
# Efficient Representative Values Computation
###############################################################################

"""
    compute_repval_ames_efficient!(...)

EFFICIENT: Compute AMEs at representative values with minimal allocations.
"""
function compute_repval_ames_efficient!(result_map, se_map, grad_map, varlist, repvals,
                                       ws, β, cholΣβ, invlink, dinvlink, d2invlink, df)
    for v in varlist
        ame_d, se_d, g_d = _ame_representation_efficient!(
            ws, df, v, repvals, β, cholΣβ, invlink, dinvlink, d2invlink
        )
        
        result_map[v] = ame_d
        se_map[v] = se_d
        grad_map[v] = g_d
    end
end

"""
    _ame_representation_efficient!(...)

EFFICIENT: Representative values AME with smart memory management.
"""
function _ame_representation_efficient!(ws::AMEWorkspace, df::AbstractDataFrame,
                                       focal::Symbol, repvals::AbstractDict{Symbol,<:AbstractVector},
                                       β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                       invlink::Function, dinvlink::Function, d2invlink::Function)
    
    # Build grid of representative value combinations
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))
    
    focal_type = eltype(df[!, focal])
    
    # Store original state (reference only - no copying)
    original_base_data = ws.base_data
    
    # Result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    for combo in combos
        combo_key = Tuple(combo)
        
        # EFFICIENT: Create representative value data with zero-copy for unchanged vars
        repval_data = create_representative_data_efficient(original_base_data, repvars, combo)
        
        # EFFICIENT: Update workspace state without full reconstruction
        set_base_data_efficient!(ws, repval_data)
        
        # Compute AME at these representative values
        if focal_type <: Real && focal_type != Bool
            # Continuous focal variable
            ame, se, grad = compute_continuous_focal_at_repvals_efficient!(
                focal, ws, β, cholΣβ, dinvlink, d2invlink
            )
            
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        elseif focal_type <: Bool
            # Boolean focal variable
            ame, se, grad = compute_bool_focal_at_repvals_efficient!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink
            )
            
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        else
            # Categorical focal variable
            factor_results = compute_categorical_focal_at_repvals_efficient!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, df
            )
            
            for (level_pair, ame_val) in factor_results[:ame]
                full_key = (combo_key..., level_pair...)
                ame_d[full_key] = ame_val
                se_d[full_key] = factor_results[:se][level_pair]
                grad_d[full_key] = copy(factor_results[:grad][level_pair])
            end
        end
    end
    
    # EFFICIENT: Restore original state (reference only)
    set_base_data_efficient!(ws, original_base_data)
    
    return ame_d, se_d, grad_d
end

###############################################################################
# Efficient Helper Functions
###############################################################################

"""
    create_representative_data_efficient(base_data::NamedTuple, repvars::Vector{Symbol}, combo::Tuple)

EFFICIENT: Create representative data with zero-copy for unchanged variables.
"""
function create_representative_data_efficient(base_data::NamedTuple, repvars::Vector{Symbol}, combo::Tuple)
    n = length(first(base_data))
    
    # Start with original data (zero-copy reference)
    repval_data = base_data
    
    # Only update changed variables
    for (rv, val) in zip(repvars, combo)
        new_values = create_representative_column_efficient(base_data[rv], val, n)
        repval_data = merge(repval_data, (rv => new_values,))
    end
    
    return repval_data
end

"""
    create_representative_column_efficient(orig_col, val, n::Int)

EFFICIENT: Create representative column with appropriate type preservation.
"""
function create_representative_column_efficient(orig_col, val, n::Int)
    if orig_col isa CategoricalArray
        # EFFICIENT: Preserve categorical structure with original levels
        return categorical(
            fill(val, n);
            levels = levels(orig_col),
            ordered = isordered(orig_col)
        )
    elseif val isa CategoricalValue
        # Handle CategoricalValue input efficiently
        return categorical(
            fill(val, n);
            levels = levels(orig_col),
            ordered = isordered(orig_col)
        )
    elseif val isa Real
        # EFFICIENT: Direct type-stable allocation for continuous values
        return fill(convert(eltype(orig_col), val), n)
    else
        # Generic fallback
        return fill(val, n)
    end
end

###############################################################################
# Efficient Focal Variable Computation at Representative Values
###############################################################################

"""
    compute_continuous_focal_at_repvals_efficient!(...)

EFFICIENT: Compute continuous focal AME at rep values with analytical derivatives.
"""
function compute_continuous_focal_at_repvals_efficient!(
    focal::Symbol, ws::AMEWorkspace,
    β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
    dinvlink::Function, d2invlink::Function)

    # Validate that variable affects some columns
    if !haskey(ws.variable_plans, focal)
        throw(ArgumentError(
            "Variable $focal not found in variable plans. " *
            "Only variables that appear in the model are supported."
        ))
    end
    
    # EFFICIENT: Compute analytical derivatives at current state
    prepare_analytical_derivatives_efficient!(ws, focal)
    
    # EFFICIENT: Use existing efficient AME computation
    ame, se, grad_ref = compute_ame_analytical_efficient!(
        β, cholΣβ, ws, dinvlink, d2invlink
    )
    
    # Validation and bounds checking
    if !isfinite(ame) || abs(ame) > 1e6
        @warn "AME computation failed for focal=$focal, returning 0"
        ame = 0.0
    end
    
    if !isfinite(se) || se < 0 || se > 1e6
        se = NaN
    end
    
    return ame, se, grad_ref
end

"""
    compute_bool_focal_at_repvals_efficient!(...)

EFFICIENT: Boolean focal variable AME with selective updates.
"""
function compute_bool_focal_at_repvals_efficient!(focal::Symbol, ws::AMEWorkspace,
                                                 β::AbstractVector, Σβ::AbstractMatrix,
                                                 invlink::Function, dinvlink::Function)
    n = ws.n
    
    # EFFICIENT: Get work matrix (lazy construction)
    work_matrix = get_work_matrix!(ws)
    
    # Store current state (reference only)
    current_state = copy(work_matrix)
    
    # Compute prediction at focal = false
    false_data = fill(false, n)
    update_for_variable!(ws, focal, false_data)
    
    mul!(ws.η, work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_false = sum(ws.μp_vals)
    mul!(ws.temp1, work_matrix', ws.μpp_vals)
    
    # Compute prediction at focal = true
    true_data = fill(true, n)
    update_for_variable!(ws, focal, true_data)
    
    mul!(ws.η, work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_true = sum(ws.μp_vals)
    mul!(ws.temp2, work_matrix', ws.μpp_vals)
    
    # Compute AME and SE
    ame = (sumμ_true - sumμ_false) / n
    
    @inbounds @simd for k in 1:length(β)
        ws.grad_work[k] = (ws.temp2[k] - ws.temp1[k]) / n
    end
    
    se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
    
    # EFFICIENT: Restore state (copy back)
    work_matrix .= current_state
    
    return ame, se, ws.grad_work
end

"""
    compute_categorical_focal_at_repvals_efficient!(...)

EFFICIENT: Categorical focal variable AME with selective updates.
"""
function compute_categorical_focal_at_repvals_efficient!(focal::Symbol, ws::AMEWorkspace,
                                                        β::AbstractVector, Σβ::AbstractMatrix,
                                                        invlink::Function, dinvlink::Function,
                                                        df::AbstractDataFrame)
    # EFFICIENT: Get work matrix (lazy construction)
    work_matrix = get_work_matrix!(ws)
    
    # Store current state (reference only)
    current_state = copy(work_matrix)
    
    # Get factor levels from original DataFrame
    factor_col = df[!, focal]
    levels_list = get_factor_levels_safe(factor_col)
    
    if length(levels_list) < 2
        throw(ArgumentError("Focal variable $focal has fewer than 2 levels"))
    end
    
    # Result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    n = ws.n
    
    # EFFICIENT: Compute all pairwise comparisons with selective updates
    for i in 1:length(levels_list)-1
        for j in i+1:length(levels_list)
            level_i, level_j = levels_list[i], levels_list[j]
            
            # Compute prediction at level_i
            level_i_data = create_categorical_level_data_efficient(focal, level_i, ws, n)
            update_for_variable!(ws, focal, level_i_data)
            
            mul!(ws.η, work_matrix, β)
            
            @inbounds @simd for k in 1:n
                ws.μp_vals[k] = invlink(ws.η[k])
                ws.μpp_vals[k] = dinvlink(ws.η[k])
            end
            
            sumμ_i = sum(ws.μp_vals)
            mul!(ws.temp1, work_matrix', ws.μpp_vals)
            
            # Compute prediction at level_j
            level_j_data = create_categorical_level_data_efficient(focal, level_j, ws, n)
            update_for_variable!(ws, focal, level_j_data)
            
            mul!(ws.η, work_matrix, β)
            
            @inbounds @simd for k in 1:n
                ws.μp_vals[k] = invlink(ws.η[k])
                ws.μpp_vals[k] = dinvlink(ws.η[k])
            end
            
            sumμ_j = sum(ws.μp_vals)
            mul!(ws.temp2, work_matrix', ws.μpp_vals)
            
            # Compute AME and SE for this pair
            ame = (sumμ_j - sumμ_i) / n
            
            @inbounds @simd for k in 1:length(β)
                ws.grad_work[k] = (ws.temp2[k] - ws.temp1[k]) / n
            end
            
            se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
            
            # Store results
            key = (level_i, level_j)
            ame_d[key] = ame
            se_d[key] = se
            grad_d[key] = copy(ws.grad_work)
        end
    end
    
    # EFFICIENT: Restore state
    work_matrix .= current_state
    
    return Dict(:ame => ame_d, :se => se_d, :grad => grad_d)
end

"""
    create_categorical_level_data_efficient(variable::Symbol, level, ws::AMEWorkspace, n::Int)

EFFICIENT: Create categorical level data with proper type preservation.
"""
function create_categorical_level_data_efficient(variable::Symbol, level, ws::AMEWorkspace, n::Int)
    orig_var = ws.base_data[variable]
    
    if orig_var isa CategoricalArray
        # EFFICIENT: Preserve the original levels and ordering
        return categorical(
            fill(level, n);
            levels = levels(orig_var),
            ordered = isordered(orig_var)
        )
    else
        # For non-categorical data, create plain vector
        return fill(level, n)
    end
end

###############################################################################
# Efficient Prediction Computation
###############################################################################

"""
    compute_predictions_efficient!(...)

EFFICIENT: Compute predictions with lazy matrix construction.
"""
function compute_predictions_efficient!(result_map, se_map, grad_map, varlist, repvals, 
                                       ws, β, Σβ, invlink, dinvlink)
    if isempty(repvals)
        # EFFICIENT: Simple prediction with lazy base matrix
        base_matrix = get_base_matrix!(ws)
        
        mul!(ws.η, base_matrix, β)
        
        # Apply inverse link
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = invlink(ws.η[i])
        end
        
        pred = sum(ws.μp_vals) / length(ws.μp_vals)
        
        # Compute gradient
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = dinvlink(ws.η[i])
        end
        
        mul!(ws.grad_work, base_matrix', ws.μp_vals)
        ws.grad_work ./= length(ws.η)
        
        se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
        
        # Store results for all requested variables
        for v in varlist
            result_map[v] = pred
            se_map[v] = se
            grad_map[v] = copy(ws.grad_work)
        end
    else
        # EFFICIENT: Predictions at representative values
        compute_repval_predictions_efficient!(result_map, se_map, grad_map, varlist, repvals,
                                            ws, β, Σβ, invlink, dinvlink)
    end
end

"""
    compute_repval_predictions_efficient!(...)

EFFICIENT: Representative value predictions with selective updates.
"""
function compute_repval_predictions_efficient!(result_map, se_map, grad_map, varlist, repvals,
                                             ws, β, Σβ, invlink, dinvlink)
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))
    
    # Store original state
    original_base_data = ws.base_data
    
    for combo in combos
        combo_key = Tuple(combo)
        
        # EFFICIENT: Create representative data with zero-copy
        repval_data = create_representative_data_efficient(original_base_data, repvars, combo)
        
        # EFFICIENT: Update workspace efficiently
        set_base_data_efficient!(ws, repval_data)
        work_matrix = get_work_matrix!(ws)
        
        # Compute prediction
        mul!(ws.η, work_matrix, β)
        
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = invlink(ws.η[i])
        end
        
        pred = sum(ws.μp_vals) / length(ws.μp_vals)
        
        # Compute gradient  
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = dinvlink(ws.η[i])
        end
        
        mul!(ws.grad_work, work_matrix', ws.μp_vals)
        ws.grad_work ./= length(ws.η)
        
        se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
        
        # Store results
        for v in varlist
            if !haskey(result_map, v)
                result_map[v] = Dict{Tuple,Float64}()
                se_map[v] = Dict{Tuple,Float64}()
                grad_map[v] = Dict{Tuple,Vector{Float64}}()
            end
            result_map[v][combo_key] = pred
            se_map[v][combo_key] = se
            grad_map[v][combo_key] = copy(ws.grad_work)
        end
    end
    
    # EFFICIENT: Restore original state
    set_base_data_efficient!(ws, original_base_data)
end
