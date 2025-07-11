# ame_representation.jl - FIXED VERSION

###############################################################################
# Representative Values AME Computation with Pure Analytical Derivatives
# FIXED: Corrected variable references and simplified logic
###############################################################################

"""
    compute_continuous_focal_at_repvals!(
        focal::Symbol, ws::AMEWorkspace,
        β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
        dinvlink::Function, d2invlink::Function, ipm::InplaceModeler)

Compute AME for a continuous focal variable at representative values using 
pure analytical derivatives. ws.work_matrix already contains the representative value state.
"""
function compute_continuous_focal_at_repvals!(
    focal::Symbol, ws::AMEWorkspace,
    β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
    dinvlink::Function, d2invlink::Function, ipm::InplaceModeler)

    # Validate that variable is continuous and supported
    if !haskey(ws.pert_vectors, focal)
        throw(ArgumentError(
            "Variable $focal not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # Compute analytical derivatives at the current representative value state
    # ws.work_matrix already contains the representative value state
    prepare_analytical_derivatives!(ws, focal, 0.0, ipm)
    
    # Compute AME using analytical derivatives
    ame, se, grad_ref = _ame_continuous_selective_fixed!(
        β, cholΣβ, ws.work_matrix, ws.derivative_matrix, dinvlink, d2invlink, ws
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
    _ame_representation!(ws::AMEWorkspace, imp::InplaceModeler, df::DataFrame,
                        focal::Symbol, repvals::AbstractDict{Symbol,<:AbstractVector},
                        β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                        invlink::Function, dinvlink::Function, d2invlink::Function)

Compute AMEs at representative values. For continuous focal variables, uses analytical 
derivatives. For Boolean/categorical focal variables, uses discrete comparisons.
"""
function _ame_representation!(ws::AMEWorkspace, imp::InplaceModeler, df::DataFrame,
                             focal::Symbol, repvals::AbstractDict{Symbol,<:AbstractVector},
                             β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                             invlink::Function, dinvlink::Function, d2invlink::Function)
    
    # Build grid of representative value combinations
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))
    
    n = length(first(ws.base_data))
    focal_type = eltype(df[!, focal])
    
    # Store original state for restoration
    original_base_data = ws.base_data
    original_base_matrix = copy(ws.base_matrix)
    
    # Result containers
    ame_d = Dict{Tuple,Float64}()
    se_d = Dict{Tuple,Float64}()
    grad_d = Dict{Tuple,Vector{Float64}}()
    
    for combo in combos
        combo_key = Tuple(combo)
        
        # Create representative value data
        repval_data = create_representative_data(original_base_data, repvars, combo, n)
        
        # Update workspace state for representative values
        ws.base_data = repval_data
        
        # Build representative value matrix using selective updates
        modelmatrix_with_base!(imp, repval_data, ws.work_matrix, original_base_matrix, repvars, ws.mapping)
        
        # Compute AME at these representative values based on focal variable type
        if focal_type <: Real && focal_type != Bool
            # Continuous focal variable - use analytical derivatives
            ame, se, grad = compute_continuous_focal_at_repvals!(
                focal, ws, β, cholΣβ, dinvlink, d2invlink, imp
            )
            
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        elseif focal_type <: Bool
            # Boolean focal variable - use discrete comparison
            ame, se, grad = compute_bool_focal_at_repvals!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, imp
            )
            
            ame_d[combo_key] = ame
            se_d[combo_key] = se
            grad_d[combo_key] = copy(grad)
            
        else
            # Multi-level categorical focal variable
            factor_results = compute_categorical_focal_at_repvals!(
                focal, ws, β, vcov(cholΣβ), invlink, dinvlink, df, imp
            )
            
            for (level_pair, ame_val) in factor_results[:ame]
                full_key = (combo_key..., level_pair...)
                ame_d[full_key] = ame_val
                se_d[full_key] = factor_results[:se][level_pair]
                grad_d[full_key] = copy(factor_results[:grad][level_pair])
            end
        end
    end
    
    # Restore original state
    ws.base_data = original_base_data
    ws.base_matrix .= original_base_matrix
    ws.work_matrix .= original_base_matrix
    
    return ame_d, se_d, grad_d
end

###############################################################################
# Helper Functions for Representative Value Data Creation
###############################################################################

"""
    create_representative_data(base_data::NamedTuple, repvars::Vector{Symbol}, 
                              combo::Tuple, n::Int) -> NamedTuple

Create data structure with representative values substituted for specified variables.
Handles both continuous and categorical variables appropriately.
"""
function create_representative_data(base_data::NamedTuple, repvars::Vector{Symbol}, 
                                   combo::Tuple, n::Int)
    repval_data = base_data
    
    for (rv, val) in zip(repvars, combo)
        orig_col = base_data[rv]
        
        new_values = create_representative_column(orig_col, val, n)
        repval_data = merge(repval_data, (rv => new_values,))
    end
    
    return repval_data
end

"""
    create_representative_column(orig_col, val, n::Int)

Create a column vector with representative value `val` repeated `n` times,
preserving the appropriate data type and structure.
"""
function create_representative_column(orig_col, val, n::Int)
    if orig_col isa CategoricalArray
        # Preserve categorical structure with original levels and ordering
        return categorical(
            fill(val, n);
            levels = levels(orig_col),
            ordered = isordered(orig_col)
        )
    elseif val isa CategoricalValue
        # Handle CategoricalValue input
        return categorical(
            fill(val, n);
            levels = levels(orig_col),
            ordered = isordered(orig_col)
        )
    elseif val isa Real
        # Continuous representative value
        return fill(Float64(val), n)
    else
        # Generic fallback
        return fill(val, n)
    end
end

###############################################################################
# Boolean and Categorical AME Functions
###############################################################################

"""
    compute_bool_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                  β::AbstractVector, Σβ::AbstractMatrix,
                                  invlink::Function, dinvlink::Function,
                                  imp::InplaceModeler)

Compute AME for Boolean focal variable at representative values.
Uses discrete comparison between true and false states.
"""
function compute_bool_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                       β::AbstractVector, Σβ::AbstractMatrix,
                                       invlink::Function, dinvlink::Function,
                                       imp::InplaceModeler)
    n = length(first(ws.base_data))
    
    # Store current representative values state
    repval_matrix = copy(ws.work_matrix)
    
    # Compute prediction at focal = false
    false_data = fill(false, n)
    update_for_variable!(ws, focal, false_data, imp)
    
    mul!(ws.η, ws.work_matrix, β)
    
    @inbounds @simd for k in 1:n
        ws.μp_vals[k] = invlink(ws.η[k])
        ws.μpp_vals[k] = dinvlink(ws.η[k])
    end
    
    sumμ_false = sum(ws.μp_vals)
    mul!(ws.temp1, ws.work_matrix', ws.μpp_vals)
    
    # Compute prediction at focal = true
    true_data = fill(true, n)
    update_for_variable!(ws, focal, true_data, imp)
    
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
    
    # Restore representative values state
    ws.work_matrix .= repval_matrix
    
    return ame, se, ws.grad_work
end

"""
    compute_categorical_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                         β::AbstractVector, Σβ::AbstractMatrix,
                                         invlink::Function, dinvlink::Function,
                                         df::AbstractDataFrame, imp::InplaceModeler)

Compute pairwise AMEs for categorical focal variable at representative values.
"""
function compute_categorical_focal_at_repvals!(focal::Symbol, ws::AMEWorkspace,
                                             β::AbstractVector, Σβ::AbstractMatrix,
                                             invlink::Function, dinvlink::Function,
                                             df::AbstractDataFrame, imp::InplaceModeler)
    # Store current representative values state
    repval_matrix = copy(ws.work_matrix)
    
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
    
    n = length(first(ws.base_data))
    
    # Compute all pairwise comparisons
    for i in 1:length(levels_list)-1
        for j in i+1:length(levels_list)
            level_i, level_j = levels_list[i], levels_list[j]
            
            # Compute prediction at level_i
            level_i_data = create_categorical_level_data(focal, level_i, ws, n)
            update_for_variable!(ws, focal, level_i_data, imp)
            
            mul!(ws.η, ws.work_matrix, β)
            
            @inbounds @simd for k in 1:n
                ws.μp_vals[k] = invlink(ws.η[k])
                ws.μpp_vals[k] = dinvlink(ws.η[k])
            end
            
            sumμ_i = sum(ws.μp_vals)
            mul!(ws.temp1, ws.work_matrix', ws.μpp_vals)
            
            # Compute prediction at level_j
            level_j_data = create_categorical_level_data(focal, level_j, ws, n)
            update_for_variable!(ws, focal, level_j_data, imp)
            
            mul!(ws.η, ws.work_matrix, β)
            
            @inbounds @simd for k in 1:n
                ws.μp_vals[k] = invlink(ws.η[k])
                ws.μpp_vals[k] = dinvlink(ws.η[k])
            end
            
            sumμ_j = sum(ws.μp_vals)
            mul!(ws.temp2, ws.work_matrix', ws.μpp_vals)
            
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
    
    # Restore representative values state
    ws.work_matrix .= repval_matrix
    
    return Dict(:ame => ame_d, :se => se_d, :grad => grad_d)
end
