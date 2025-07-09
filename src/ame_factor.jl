# ame_factor.jl - FIXED for categorical data handling

###############################################################################
# Selective update categorical AME computation - FIXED
###############################################################################

"""
    compute_factor_pair_selective!(variable::Symbol, level_i, level_j, 
                                  ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                  invlink::Function, dinvlink::Function, ipm::InplaceModeler)

Compute contrast between two levels of a categorical variable using selective updates.
FIXED: Now properly handles categorical data types.

# Arguments
- `variable`: Categorical variable symbol
- `level_i`, `level_j`: Factor levels to compare
- `ws`: AMEWorkspace with selective update infrastructure
- `β`: Model coefficients
- `Σβ`: Coefficient covariance matrix
- `invlink`: Inverse link function
- `dinvlink`: First derivative of inverse link function
- `ipm`: InplaceModeler for matrix construction

# Returns
- `(ame, se, grad)`: AME difference (level_j - level_i), standard error, gradient

# Details
Uses selective updates to compute predictions at each level, then takes difference.
Only columns affected by the categorical variable are updated.
FIXED: Properly creates CategoricalVector for level data.
"""
function compute_factor_pair_selective!(variable::Symbol, level_i, level_j, 
                                       ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                       invlink::Function, dinvlink::Function, ipm::InplaceModeler)
    n = length(first(ws.base_data))
    
    # Get the original categorical variable to preserve levels and type
    orig_var = ws.base_data[variable]
    
    # ---------- Compute prediction at level_i ------------------------------
    # Create categorical data with all observations set to level_i
    if orig_var isa CategoricalArray
        # Preserve the original levels and ordering
        level_i_data = categorical(fill(level_i, n), levels=levels(orig_var), ordered=isordered(orig_var))
    else
        # For non-categorical data, create appropriate vector
        level_i_data = fill(level_i, n)
    end
    
    update_for_variable!(ws, variable, level_i_data, ipm)
    
    # Compute predictions
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])  # Store for gradient
    end
    
    sumμ_i = sum(ws.μp_vals)
    
    # Store gradient components for level_i
    mul!(ws.temp1, ws.work_matrix', ws.μpp_vals)  # temp1 = X_i' * μp_i
    
    # ---------- Compute prediction at level_j ------------------------------
    # Create categorical data with all observations set to level_j
    if orig_var isa CategoricalArray
        # Preserve the original levels and ordering
        level_j_data = categorical(fill(level_j, n), levels=levels(orig_var), ordered=isordered(orig_var))
    else
        # For non-categorical data, create appropriate vector
        level_j_data = fill(level_j, n)
    end
    
    update_for_variable!(ws, variable, level_j_data, ipm)
    
    # Compute predictions
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])  # Store for gradient
    end
    
    sumμ_j = sum(ws.μp_vals)
    
    # Store gradient components for level_j
    mul!(ws.temp2, ws.work_matrix', ws.μpp_vals)  # temp2 = X_j' * μp_j
    
    # ---------- Compute AME, gradient, and SE -------------------------------
    ame = (sumμ_j - sumμ_i) / n
    
    # Gradient: (∇μ_j - ∇μ_i) / n
    @inbounds @simd for k in 1:length(β)
        ws.grad_work[k] = (ws.temp2[k] - ws.temp1[k]) / n
    end
    
    # Standard error via delta method
    se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
    
    return ame, se, ws.grad_work
end

"""
    compute_factor_baseline_selective!(ame_d, se_d, grad_d, variable::Symbol,
                                      ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                      invlink::Function, dinvlink::Function, 
                                      df::AbstractDataFrame, ipm::InplaceModeler)

Compute baseline contrasts for categorical variable using selective updates.
Updates `ame_d`, `se_d`, `grad_d` dictionaries in-place.
"""
function compute_factor_baseline_selective!(ame_d, se_d, grad_d, variable::Symbol,
                                          ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          df::AbstractDataFrame, ipm::InplaceModeler)
    # Get factor levels from original data
    factor_col = df[!, variable]
    levels_list = get_factor_levels_safe(factor_col)
    
    if length(levels_list) < 2
        @warn "Variable $variable has fewer than 2 levels, skipping"
        return
    end
    
    base_level = levels_list[1]
    
    # Compute baseline contrasts
    for level in levels_list[2:end]
        ame, se, grad = compute_factor_pair_selective!(
            variable, base_level, level, ws, β, Σβ, invlink, dinvlink, ipm
        )
        
        key = (base_level, level)
        ame_d[key] = ame
        se_d[key] = se
        grad_d[key] = copy(grad)
    end
end

"""
    compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable::Symbol,
                                     ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                     invlink::Function, dinvlink::Function, 
                                     df::AbstractDataFrame, ipm::InplaceModeler)

Compute all-pairs contrasts for categorical variable using selective updates.
Updates `ame_d`, `se_d`, `grad_d` dictionaries in-place.
"""
function compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable::Symbol,
                                          ws::AMEWorkspace, β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          df::AbstractDataFrame, ipm::InplaceModeler)
    # Get factor levels from original data
    factor_col = df[!, variable]
    levels_list = get_factor_levels_safe(factor_col)
    
    if length(levels_list) < 2
        @warn "Variable $variable has fewer than 2 levels, skipping"
        return
    end
    
    # Compute all pairs
    for i in 1:length(levels_list)-1
        for j in i+1:length(levels_list)
            level_i, level_j = levels_list[i], levels_list[j]
            
            ame, se, grad = compute_factor_pair_selective!(
                variable, level_i, level_j, ws, β, Σβ, invlink, dinvlink, ipm
            )
            
            key = (level_i, level_j)
            ame_d[key] = ame
            se_d[key] = se
            grad_d[key] = copy(grad)
        end
    end
end

"""
    compute_single_bool_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                     β::AbstractVector, Σβ::AbstractMatrix,
                                     invlink::Function, dinvlink::Function, 
                                     ipm::InplaceModeler)

Compute AME for a Boolean variable using selective updates.
Used for representative values computation.

# Returns
- `(ame, se, grad)`: AME for true vs false contrast
"""
function compute_single_bool_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                          β::AbstractVector, Σβ::AbstractMatrix,
                                          invlink::Function, dinvlink::Function, 
                                          ipm::InplaceModeler)
    return compute_factor_pair_selective!(variable, false, true, ws, β, Σβ, invlink, dinvlink, ipm)
end

"""
    compute_single_factor_ames_selective!(variable::Symbol, ws::AMEWorkspace,
                                        β::AbstractVector, Σβ::AbstractMatrix,
                                        invlink::Function, dinvlink::Function, 
                                        df::AbstractDataFrame, ipm::InplaceModeler)

Compute all-pairs AMEs for a multi-level categorical variable using selective updates.
Used for representative values computation.

# Returns
- Dictionary with keys `:ame`, `:se`, `:grad` containing pairwise results
"""
function compute_single_factor_ames_selective!(variable::Symbol, ws::AMEWorkspace,
                                             β::AbstractVector, Σβ::AbstractMatrix,
                                             invlink::Function, dinvlink::Function, 
                                             df::AbstractDataFrame, ipm::InplaceModeler)
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    compute_factor_allpairs_selective!(ame_d, se_d, grad_d, variable, ws, β, Σβ, 
                                     invlink, dinvlink, df, ipm)
    
    return Dict(:ame => ame_d, :se => se_d, :grad => grad_d)
end

###############################################################################
# Helper functions for factor level handling - ENHANCED
###############################################################################

"""
    get_factor_levels_safe(col) → Vector

Return the full, ordered list of levels for anything that should be treated as a
factor – including `Bool`s and plain `Vector{T}` that happen to contain
`CategoricalValue`s.  Fallback: unique(col) sorted.
"""
function get_factor_levels_safe(col)
    if col isa CategoricalArray
        return levels(col)
    elseif eltype(col) <: Bool
        return [false, true]          # canonical order
    elseif eltype(col) <: CategoricalValue
        return levels!(categorical(col))  # wrap in a temporary CategoricalArray
    else
        return sort(unique(col))
    end
end

"""
    create_categorical_level_data(variable::Symbol, level, ws::AMEWorkspace, n::Int)

Create appropriately typed data vector for a categorical variable level.
Preserves CategoricalVector structure when needed.
"""
function create_categorical_level_data(variable::Symbol, level, ws::AMEWorkspace, n::Int)
    orig_var = ws.base_data[variable]
    
    if orig_var isa CategoricalArray
        # Preserve the original levels and ordering
        return categorical(
            fill(level, n);
                levels  = levels(orig_var),
                ordered = isordered(orig_var)
            )

    else
        # For non-categorical data, create plain vector
        return fill(level, n)
    end
end
