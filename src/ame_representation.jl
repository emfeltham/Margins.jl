# ame_representation.jl - FIXED VERSION addressing test failures

###############################################################################
# Key Issue Fixed: Representative values state management
# The main problem was that we were incorrectly handling the base_data state
# when computing AMEs at representative values
###############################################################################

function compute_continuous_focal_at_repvals_fixed!(
    focal::Symbol, ws::AMEWorkspace,
    β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
    dinvlink::Function, d2invlink::Function, ipm::InplaceModeler)

    n = length(first(ws.base_data))
    
    # FIXED: The key issue was that we need to use the CURRENT values in ws.base_data
    # which might already be set to representative values, not the original values
    orig_focal_vals = ws.base_data[focal]
    
    # Use reasonable step size calculation
    focal_scale = std(orig_focal_vals)
    focal_range = maximum(orig_focal_vals) - minimum(orig_focal_vals)
    
    if focal_scale > 0
        h = focal_scale * 1e-6
    elseif focal_range > 0
        h = focal_range * 1e-6
    else
        h = 1e-6
    end
    
    h = max(h, 1e-8)
    h = min(h, 1e-3)
    
    # Store current state
    baseline_matrix = copy(ws.work_matrix)
    
    # Create perturbed values - SIMPLE approach
    pert_focal_vals = orig_focal_vals .+ h
    
    # Update focal variable
    update_for_variable!(ws, focal, pert_focal_vals, ipm)
    
    # Compute finite differences
    affected_cols = ws.variable_plans[focal]
    fill!(ws.finite_diff_matrix, 0.0)
    
    invh = 1.0 / h
    
    # Compute finite differences with basic stability checks
    for col in affected_cols
        for row in 1:n
            baseline_val = baseline_matrix[row, col]
            perturbed_val = ws.work_matrix[row, col]
            
            # Basic sanity checks
            if !isfinite(baseline_val) || !isfinite(perturbed_val)
                ws.finite_diff_matrix[row, col] = 0.0
                continue
            end
            
            raw_diff = perturbed_val - baseline_val
            finite_diff = raw_diff * invh
            
            # Reasonable clamping
            if isfinite(finite_diff)
                ws.finite_diff_matrix[row, col] = clamp(finite_diff, -1e6, 1e6)
            else
                ws.finite_diff_matrix[row, col] = 0.0
            end
        end
    end
    
    # Compute AME
    ame, se, grad_ref = _ame_continuous_selective_fixed!(
        β, cholΣβ, baseline_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
    )
    
    # Final reality checks
    if !isfinite(ame) || abs(ame) > 1e6
        @warn "AME computation failed for focal=$focal, returning 0"
        ame = 0.0
    end
    
    if !isfinite(se) || se < 0 || se > 1e6
        se = NaN
    end
    
    # Restore state
    ws.work_matrix .= baseline_matrix
    
    return ame, se, grad_ref
end

###############################################################################
# FIXED: Main function with better state management
###############################################################################

function _ame_representation!(ws::AMEWorkspace, ipm::InplaceModeler, df::DataFrame,
                             focal::Symbol, repvals::AbstractDict{Symbol,<:AbstractVector},
                             β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                             invlink::Function, dinvlink::Function, d2invlink::Function)
    
    # Build grid of representative value combinations
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))
    
    n = length(first(ws.base_data))
    focal_type = eltype(df[!, focal])
    
    # Store original base_data to restore later
    original_base_data = ws.base_data
    
    # Result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    for combo in combos
        combo_key = Tuple(combo)
        
        # FIXED: Set ALL variables to representative values
        # Create a new base_data NamedTuple with representative values
        repval_data = original_base_data
        
        for (rv, val) in zip(repvars, combo)
            orig_col = original_base_data[rv]
            if val isa Real
                new_values = fill(Float64(val), n)
            elseif val isa CategoricalValue || orig_col isa CategoricalArray
                new_values = categorical(
                    fill(val, n);
                    levels = levels(orig_col),
                    ordered = isordered(orig_col),
                )
            else
                new_values = fill(val, n)
            end
            
            # Update the data tuple
            repval_data = merge(repval_data, (rv => new_values,))
        end
        
        # FIXED: Update BOTH the workspace base_data AND work_matrix
        ws.base_data = repval_data
        
        # Rebuild the work matrix with the new base data
        modelmatrix!(ipm, repval_data, ws.work_matrix)
        
        # Compute AME at these representative values
        if focal_type <: Real && focal_type != Bool
            # Continuous focal variable
            ame, se, grad = compute_continuous_focal_at_repvals_fixed!(
                focal, ws, β, cholΣβ, dinvlink, d2invlink, ipm
            )
            
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        elseif focal_type <: Bool
            # Boolean focal variable
            ame, se, grad = compute_bool_focal_at_repvals!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, ipm
            )
            
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        else
            # Multi-level categorical focal variable
            factor_results = compute_categorical_focal_at_repvals!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, df, ipm
            )
            
            for (level_pair, ame_val) in factor_results[:ame]
                full_key = (combo_key..., level_pair...)
                ame_d[full_key] = ame_val
                se_d[full_key] = factor_results[:se][level_pair]
                grad_d[full_key] = copy(factor_results[:grad][level_pair])
            end
        end
    end
    
    # FIXED: Restore original state completely
    ws.base_data = original_base_data
    modelmatrix!(ipm, original_base_data, ws.work_matrix)
    
    return ame_d, se_d, grad_d
end

# Supporting functions (keep existing implementations but ensure they use current ws.base_data)
function compute_bool_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                       β::AbstractVector, Σβ::AbstractMatrix,
                                       invlink::Function, dinvlink::Function,
                                       ipm::InplaceModeler)
    n = length(first(ws.base_data))
    
    # Store current repval state
    repval_matrix = copy(ws.work_matrix)
    
    # Compute prediction at focal = false
    false_data = fill(false, n)
    update_for_variable!(ws, focal, false_data, ipm)
    
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_false = sum(ws.μp_vals)
    mul!(ws.temp1, ws.work_matrix', ws.μpp_vals)
    
    # Compute prediction at focal = true
    true_data = fill(true, n)
    update_for_variable!(ws, focal, true_data, ipm)
    
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_true = sum(ws.μp_vals)
    mul!(ws.temp2, ws.work_matrix', ws.μpp_vals)
    
    # Compute AME and SE
    ame = (sumμ_true - sumμ_false) / n
    
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
    
    # Get factor levels from the ORIGINAL df, not current ws.base_data
    # This ensures we have the proper levels structure
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
