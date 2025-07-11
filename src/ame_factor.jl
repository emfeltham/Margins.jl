# ame_factor.jl - EFFICIENT VERSION: Minimal allocations and smart categorical handling

###############################################################################
# Efficient categorical AME computation with selective updates
###############################################################################

"""
    compute_factor_pair_selective!(variable::Symbol, level_i, level_j, 
                                  ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                  invlink::Function, dinvlink::Function, ipm::InplaceModeler)

EFFICIENT: Compute contrast between two levels with minimal allocations and smart state management.
"""
function compute_factor_pair_selective!(variable::Symbol, level_i, level_j, 
                                       ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                       invlink::Function, dinvlink::Function, imp::InplaceModeler)
    n = ws.n
    
    # EFFICIENT: Get original categorical variable for type preservation
    orig_var = ws.base_data[variable]
    
    # EFFICIENT: Get work matrix (lazy construction)
    work_matrix = get_work_matrix!(ws)
    
    # ---------- Compute prediction at level_i (EFFICIENT) ------------------
    level_i_data = create_level_data_efficient(orig_var, level_i, n)
    update_for_variable!(ws, variable, level_i_data)
    
    # EFFICIENT: Use pre-allocated vectors and BLAS
    mul!(ws.η, work_matrix, β)
    
    # EFFICIENT: Vectorized link function computation
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_i = sum(ws.μp_vals)
    
    # EFFICIENT: Use pre-allocated temp vector and BLAS
    mul!(ws.temp1, work_matrix', ws.μpp_vals)
    
    # ---------- Compute prediction at level_j (EFFICIENT) ------------------
    level_j_data = create_level_data_efficient(orig_var, level_j, n)
    update_for_variable!(ws, variable, level_j_data)
    
    mul!(ws.η, work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_j = sum(ws.μp_vals)
    
    # EFFICIENT: Use pre-allocated temp vector and BLAS
    mul!(ws.temp2, work_matrix', ws.μpp_vals)
    
    # ---------- Compute AME and SE (EFFICIENT) -----------------------------
    ame = (sumμ_j - sumμ_i) / n
    
    # EFFICIENT: Vectorized gradient computation
    inv_n = 1.0 / n
    @inbounds @simd for k in 1:length(β)
        ws.grad_work[k] = (ws.temp2[k] - ws.temp1[k]) * inv_n
    end
    
    # EFFICIENT: SE computation using BLAS
    se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
    
    return ame, se, ws.grad_work
end

"""
    create_level_data_efficient(orig_var, level, n::Int)

EFFICIENT: Create level data with minimal allocations and proper type preservation.
"""
function create_level_data_efficient(orig_var, level, n::Int)
    if orig_var isa CategoricalArray
        # EFFICIENT: Preserve categorical structure with reference to original levels
        return categorical(
            fill(level, n);
            levels = levels(orig_var),
            ordered = isordered(orig_var)
        )
    else
        # EFFICIENT: Direct type-stable allocation for non-categorical
        return fill(level, n)
    end
end

"""
    compute_factor_baseline_selective!(ame_d, se_d, grad_d, variable::Symbol, ...)

EFFICIENT: Baseline contrasts with minimal allocations and smart level handling.
"""
function compute_factor_baseline_selective!(ame_d, se_d, grad_d, variable::Symbol,
                                          ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          df::AbstractDataFrame, ipm::InplaceModeler)
    # EFFICIENT: Get factor levels once
    factor_col = df[!, variable]
    levels_list = get_factor_levels_efficient(factor_col)
    
    if length(levels_list) < 2
        @warn "Variable $variable has fewer than 2 levels, skipping"
        return
    end
    
    base_level = levels_list[1]
    
    # EFFICIENT: Compute baseline contrasts in batch
    for level in levels_list[2:end]
        ame, se, grad = compute_factor_pair_selective!(
            variable, base_level, level, ws, β, Σβ, invlink, dinvlink, ipm
        )
        
        key = (base_level, level)
        ame_d[key] = ame
        se_d[key] = se
        grad_d[key] = copy(grad)  # Copy since workspace vector will be reused
    end
end

"""
    compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable::Symbol, ...)

EFFICIENT: All-pairs contrasts with optimal ordering and minimal allocations.
"""
function compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable::Symbol,
                                          ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          df::AbstractDataFrame, ipm::InplaceModeler)
    # EFFICIENT: Get factor levels once
    factor_col = df[!, variable]
    levels_list = get_factor_levels_efficient(factor_col)
    
    if length(levels_list) < 2
        @warn "Variable $variable has fewer than 2 levels, skipping"
        return
    end
    
    # EFFICIENT: Compute all pairs with minimal matrix operations
    for i in 1:length(levels_list)-1
        for j in i+1:length(levels_list)
            level_i, level_j = levels_list[i], levels_list[j]
            
            ame, se, grad = compute_factor_pair_selective!(
                variable, level_i, level_j, ws, β, Σβ, invlink, dinvlink, ipm
            )
            
            key = (level_i, level_j)
            ame_d[key] = ame
            se_d[key] = se
            grad_d[key] = copy(grad)  # Copy since workspace vector will be reused
        end
    end
end

"""
    compute_single_bool_ame_selective!(variable::Symbol, ws::AMEWorkspace, ...)

EFFICIENT: Boolean variable AME with optimized true/false handling.
"""
function compute_single_bool_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                          β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          ipm::InplaceModeler)
    # EFFICIENT: Direct boolean comparison (false -> true)
    return compute_factor_pair_selective!(variable, false, true, ws, β, Σβ, invlink, dinvlink, ipm)
end

"""
    compute_single_factor_ames_selective!(variable::Symbol, ws::AMEWorkspace, ...)

EFFICIENT: Multi-level categorical AME with batch processing.
"""
function compute_single_factor_ames_selective!(variable::Symbol, ws::AMEWorkspace,
                                             β::AbstractVector, Σβ::AbstractMatrix,
                                             invlink::Function, dinvlink::Function, 
                                             df::AbstractDataFrame, ipm::InplaceModeler)
    # EFFICIENT: Pre-allocate result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    # EFFICIENT: Use existing optimized all-pairs function
    compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable, ws, β, Σβ, 
                                     invlink, dinvlink, df, ipm)
    
    return Dict(:ame => ame_d, :se => se_d, :grad => grad_d)
end

###############################################################################
# Efficient Helper Functions for Factor Level Handling
###############################################################################

"""
    get_factor_levels_efficient(col) -> Vector

EFFICIENT: Fast factor level extraction with type-specific optimizations.
"""
function get_factor_levels_efficient(col)
    if col isa CategoricalArray
        # EFFICIENT: Direct access to levels (no copying)
        return levels(col)
    elseif eltype(col) <: Bool
        # EFFICIENT: Pre-defined boolean levels
        return [false, true]
    elseif eltype(col) <: CategoricalValue
        # EFFICIENT: Extract levels once from any CategoricalValue
        return levels(first(col))
    else
        # EFFICIENT: Compute unique values once and sort
        unique_vals = unique(col)
        return sort!(unique_vals)
    end
end

"""
    create_categorical_level_data(variable::Symbol, level, ws::AMEWorkspace, n::Int)

EFFICIENT: Create categorical level data with type preservation and minimal allocations.
"""
function create_categorical_level_data(variable::Symbol, level, ws::AMEWorkspace, n::Int)
    orig_var = ws.base_data[variable]
    return create_level_data_efficient(orig_var, level, n)
end

###############################################################################
# Efficient Representative Values Support
###############################################################################

"""
    compute_factor_at_repvals_efficient!(variable::Symbol, repval_data::NamedTuple,
                                        ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                        invlink::Function, dinvlink::Function,
                                        df::AbstractDataFrame, ipm::InplaceModeler)

EFFICIENT: Factor AME computation at representative values with state management.
"""
function compute_factor_at_repvals_efficient!(variable::Symbol, repval_data::NamedTuple,
                                             ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                             invlink::Function, dinvlink::Function,
                                             df::AbstractDataFrame, ipm::InplaceModeler)
    
    # Store original state
    original_data = ws.base_data
    
    # EFFICIENT: Update workspace to representative value state
    set_base_data_efficient!(ws, repval_data)
    
    # Determine variable type
    factor_col = df[!, variable]
    
    try
        if eltype(factor_col) <: Bool
            # EFFICIENT: Boolean case
            ame, se, grad = compute_single_bool_ame_selective!(
                variable, ws, β, Σβ, invlink, dinvlink, ipm
            )
            return Dict(:ame => Dict((false, true) => ame),
                       :se => Dict((false, true) => se),
                       :grad => Dict((false, true) => copy(grad)))
        else
            # EFFICIENT: Multi-level categorical case
            return compute_single_factor_ames_selective!(
                variable, ws, β, Σβ, invlink, dinvlink, df, ipm
            )
        end
    finally
        # EFFICIENT: Always restore original state
        set_base_data_efficient!(ws, original_data)
    end
end

###############################################################################
# Optimized Batch Processing Functions
###############################################################################

"""
    batch_factor_computations!(variables::Vector{Symbol}, ws::AMEWorkspace, ...)

EFFICIENT: Process multiple categorical variables in batch to minimize state changes.
"""
function batch_factor_computations!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                   β::AbstractVector, Σβ::AbstractMatrix,
                                   invlink::Function, dinvlink::Function,
                                   df::AbstractDataFrame, ipm::InplaceModeler,
                                   pairs::Symbol = :allpairs)
    
    results = Dict{Symbol, Dict{Symbol, Any}}()
    
    for variable in variables
        # EFFICIENT: Pre-allocate result containers for this variable
        ame_d = Dict{Tuple,Float64}()
        se_d = Dict{Tuple,Float64}()
        grad_d = Dict{Tuple,Vector{Float64}}()
        
        # EFFICIENT: Choose computation method based on pairs parameter
        if pairs == :baseline
            compute_factor_baseline_selective!(
                ame_d, se_d, grad_d, variable, ws, β, Σβ, 
                invlink, dinvlink, df, ipm
            )
        else # :allpairs
            compute_factor_allpairs_selective!(
                ame_d, se_d, grad_d, variable, ws, β, Σβ, 
                invlink, dinvlink, df, ipm
            )
        end
        
        # Store results for this variable
        results[variable] = Dict(
            :ame => ame_d,
            :se => se_d,
            :grad => grad_d
        )
    end
    
    return results
end

"""
    estimate_factor_computation_cost(variable::Symbol, df::AbstractDataFrame) -> NamedTuple

EFFICIENT: Estimate computational cost for factor variable processing.
"""
function estimate_factor_computation_cost(variable::Symbol, df::AbstractDataFrame)
    factor_col = df[!, variable]
    levels_list = get_factor_levels_efficient(factor_col)
    n_levels = length(levels_list)
    n_obs = length(factor_col)
    
    # Estimate number of comparisons
    n_baseline_comparisons = max(0, n_levels - 1)
    n_allpairs_comparisons = max(0, (n_levels * (n_levels - 1)) ÷ 2)
    
    # Estimate operations per comparison (matrix multiplications, etc.)
    ops_per_comparison = 4 * n_obs  # Rough estimate
    
    return (
        variable = variable,
        n_levels = n_levels,
        n_observations = n_obs,
        baseline_comparisons = n_baseline_comparisons,
        allpairs_comparisons = n_allpairs_comparisons,
        baseline_cost = n_baseline_comparisons * ops_per_comparison,
        allpairs_cost = n_allpairs_comparisons * ops_per_comparison,
        cost_ratio = n_allpairs_comparisons / max(1, n_baseline_comparisons)
    )
end
