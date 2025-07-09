# ame_representation.jl - FINAL FIX for finite differences at representative values

###############################################################################
# FINAL FIX: Continuous focal computation at representative values
###############################################################################

"""
    compute_continuous_focal_at_repvals_final_fix!(focal::Symbol, ws::AMEWorkspace,
                                                  β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                                  dinvlink::Function, d2invlink::Function,
                                                  ipm::InplaceModeler)

FINAL FIX: The issue was that we weren't properly creating the base state for finite differences.

The correct approach:
1. Current workspace is at repvals (e.g., PetalWidth = quantile)
2. Create base state: SepalWidth = original, PetalWidth = quantile  
3. Create perturbed state: SepalWidth = original + h, PetalWidth = quantile
4. Finite difference = (perturbed - base) / h

The key insight: Both base and perturbed states must have the SAME representative values
for non-focal variables. We were using the wrong base state.
"""
function compute_continuous_focal_at_repvals_final_fix!(focal::Symbol, ws::AMEWorkspace,
                                                       β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                                       dinvlink::Function, d2invlink::Function,
                                                       ipm::InplaceModeler)
    # Validate focal variable
    if !haskey(ws.pert_vectors, focal)
        throw(ArgumentError(
            "Focal variable $focal not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # Get step size based on original focal values
    orig_focal_values = ws.base_data[focal]
    maxabs = maximum(abs, orig_focal_values)
    h = clamp(sqrt(eps(Float64)) * max(1, maxabs * 0.01), 1e-8, maxabs * 0.1)
    invh = 1.0 / h
    
    # CRITICAL INSIGHT: The workspace is currently at representative values
    # We need to compute finite differences while keeping other variables at repvals
    
    # Step 1: Create base state (focal = original, other vars = repvals)
    # The current work_matrix has other vars at repvals, but focal might not be original
    # We need to explicitly set focal to original values while keeping repvals
    
    # Reset focal to original values while preserving repvals for other variables
    update_for_variable!(ws, focal, orig_focal_values, ipm)
    base_matrix = copy(ws.work_matrix)  # This is our true base state at repvals
    
    # Step 2: Create perturbed state (focal = original + h, other vars = repvals)
    pert_vector = ws.pert_vectors[focal]
    @inbounds for i in eachindex(pert_vector)
        pert_vector[i] = orig_focal_values[i] + h
    end
    
    update_for_variable!(ws, focal, pert_vector, ipm)
    perturbed_matrix = copy(ws.work_matrix)  # This is perturbed state at repvals
    
    # Step 3: Compute finite differences correctly
    # Both base_matrix and perturbed_matrix have the same repvals for non-focal vars
    affected_cols = ws.variable_plans[focal]
    
    @inbounds for col in affected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = (perturbed_matrix[row, col] - base_matrix[row, col]) * invh
    end
    
    # Zero out unaffected columns
    total_cols = size(ws.finite_diff_matrix, 2)
    unaffected_cols = get_unchanged_columns(ws.mapping, [focal], total_cols)
    
    @inbounds for col in unaffected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = 0.0
    end
    
    # Step 4: Compute AME using base state at repvals
    ame, se, grad_ref = _ame_continuous_selective!(
        β, cholΣβ, base_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
    )
    
    # Restore the base state (focal = original, other vars = repvals)
    ws.work_matrix .= base_matrix
    
    return ame, se, grad_ref
end

###############################################################################
# Update the main function to use the final fix
###############################################################################

"""
    _ame_representation!(ws::AMEWorkspace, ipm::InplaceModeler, df::DataFrame,
                        focal::Symbol, repvals::AbstractDict{Symbol,<:AbstractVector},
                        β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                        invlink::Function, dinvlink::Function, d2invlink::Function)

Updated to use the final fix for continuous variables.
"""
function _ame_representation!(ws::AMEWorkspace, ipm::InplaceModeler, df::DataFrame,
                             focal::Symbol, repvals::AbstractDict{Symbol,<:AbstractVector},
                             β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                             invlink::Function, dinvlink::Function, d2invlink::Function)
    
    # Build grid of representative value combinations
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))
    
    n = length(first(ws.base_data))
    focal_type = eltype(df[!, focal])
    
    # Result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    for combo in combos
        combo_key = Tuple(combo)
        
        # Set up representative values (excluding focal variable)
        repval_changes = Dict{Symbol, Vector}()
        for (rv, val) in zip(repvars, combo)
            if rv != focal  # Don't set focal variable to repval yet
                # Handle different types appropriately
                if val isa Real
                    repval_changes[rv] = fill(Float64(val), n)
                else
                    # For categorical or other types, use as-is
                    repval_changes[rv] = fill(val, n)
                end
            end
        end
        
        # Update workspace to representative values
        if !isempty(repval_changes)
            update_for_variables!(ws, repval_changes, ipm)
        else
            # No repval changes needed, use base state
            reset_to_base!(ws)
        end
        
        # Now compute AME for focal variable at this repval combination
        if focal_type <: Real && focal_type != Bool
            # Continuous focal variable - USE FINAL FIX
            ame, se, grad = compute_continuous_focal_at_repvals_final_fix!(
                focal, ws, β, cholΣβ, dinvlink, d2invlink, ipm
            )
            
            # Key is just the repval combination
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        elseif focal_type <: Bool
            # Boolean focal variable - baseline contrast (false vs true)
            ame, se, grad = compute_bool_focal_at_repvals!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, ipm
            )
            
            # Key is just the repval combination (the contrast is implicit: false vs true)
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        else
            # Multi-level categorical focal variable - all pairs
            factor_results = compute_categorical_focal_at_repvals!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, df, ipm
            )
            
            # Combine repval keys with factor level pair keys
            for (level_pair, ame_val) in factor_results[:ame]
                full_key = (combo_key..., level_pair...)
                ame_d[full_key] = ame_val
                se_d[full_key] = factor_results[:se][level_pair]
                grad_d[full_key] = copy(factor_results[:grad][level_pair])
            end
        end
    end
    
    return ame_d, se_d, grad_d
end

# Keep all the other functions the same - they were working correctly:

function compute_bool_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                       β::AbstractVector, Σβ::AbstractMatrix,
                                       invlink::Function, dinvlink::Function,
                                       ipm::InplaceModeler)
    n = length(first(ws.base_data))
    
    # Store current repval state
    repval_matrix = copy(ws.work_matrix)
    
    # ---------- Compute prediction at focal = false ------------------------
    false_data = fill(false, n)
    update_for_variable!(ws, focal, false_data, ipm)
    
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_false = sum(ws.μp_vals)
    mul!(ws.temp1, ws.work_matrix', ws.μpp_vals)  # Gradient for false
    
    # ---------- Compute prediction at focal = true -------------------------
    true_data = fill(true, n)
    update_for_variable!(ws, focal, true_data, ipm)
    
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_true = sum(ws.μp_vals)
    mul!(ws.temp2, ws.work_matrix', ws.μpp_vals)  # Gradient for true
    
    # ---------- Compute AME and SE ------------------------------------------
    ame = (sumμ_true - sumμ_false) / n
    
    # Gradient: (∇μ_true - ∇μ_false) / n
    @inbounds @simd for k in 1:length(β)
        ws.grad_work[k] = (ws.temp2[k] - ws.temp1[k]) / n
    end
    
    se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
    
    # Restore repval state
    ws.work_matrix .= repval_matrix
    
    return ame, se, ws.grad_work
end

function compute_categorical_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                             β::AbstractVector, Σβ::AbstractMatrix,
                                             invlink::Function, dinvlink::Function,
                                             df::AbstractDataFrame, ipm::InplaceModeler)
    # Store current repval state
    repval_matrix = copy(ws.work_matrix)
    
    # Get factor levels
    factor_col = df[!, focal]
    levels_list = get_factor_levels_safe(factor_col)
    
    if length(levels_list) < 2
        throw(ArgumentError("Focal variable $focal has fewer than 2 levels"))
    end
    
    # Result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    n = length(first(ws.base_data))
    
    # Compute all pairs
    for i in 1:length(levels_list)-1
        for j in i+1:length(levels_list)
            level_i, level_j = levels_list[i], levels_list[j]
            
            # Compute prediction at level_i
            level_i_data = create_categorical_level_data(focal, level_i, ws, n)
            update_for_variable!(ws, focal, level_i_data, ipm)
            
            mul!(ws.η, ws.work_matrix, β)
            
            @inbounds @simd for k in 1:n
                ws.μp_vals[k] = invlink(ws.η[k])
                ws.μpp_vals[k] = dinvlink(ws.η[k])
            end
            
            sumμ_i = sum(ws.μp_vals)
            mul!(ws.temp1, ws.work_matrix', ws.μpp_vals)
            
            # Compute prediction at level_j
            level_j_data = create_categorical_level_data(focal, level_j, ws, n)
            update_for_variable!(ws, focal, level_j_data, ipm)
            
            mul!(ws.η, ws.work_matrix, β)
            
            @inbounds @simd for k in 1:n
                ws.μp_vals[k] = invlink(ws.η[k])
                ws.μpp_vals[k] = dinvlink(ws.η[k])
            end
            
            sumμ_j = sum(ws.μp_vals)
            mul!(ws.temp2, ws.work_matrix', ws.μpp_vals)
            
            # Compute AME and SE
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
    
    # Restore repval state
    ws.work_matrix .= repval_matrix
    
    return Dict(:ame => ame_d, :se => se_d, :grad => grad_d)
end
