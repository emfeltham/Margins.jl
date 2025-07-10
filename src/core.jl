# core.jl - FIXED VERSION addressing test failures

###############################################################################
# Main margins() function with selective update infrastructure - FIXED VERSION
# Key fixes: 
# 1. Call fixed versions of continuous AME functions
# 2. Better representative values handling
# 3. Improved error handling and validation
###############################################################################

"""
    margins(model, vars, df; kwargs...)  ->  MarginsResult

Compute average marginal effects (AMEs) or average predictions using selective 
matrix updates for memory efficiency. FIXED: Now uses corrected numerical methods.
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
    
    # Build selective update infrastructure
    ws = AMEWorkspace(model, df)
    ipm = InplaceModeler(model, n)
    
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
    
    # Helper to ensure Dict structure for categorical results
    ensure_dict!(v) = begin
        result_map[v] isa Dict || (result_map[v] = Dict{Tuple,Float64}())
        se_map[v] isa Dict || (se_map[v] = Dict{Tuple,Float64}())
        grad_map[v] isa Dict || (grad_map[v] = Dict{Tuple,Vector{Float64}}())
    end
    
    # Branch by computation type
    if type == :predict
        compute_predictions!(result_map, se_map, grad_map, varlist, repvals, 
                           ws, β, Σβ, invlink, dinvlink, ipm)
    else # :dydx
        if isempty(repvals)
            # Standard AMEs without representative values
            compute_standard_ames_fixed!(
                result_map, se_map, grad_map, varlist, 
                cts_vars, cat_vars, pairs, ws, β, cholΣβ, 
                invlink, dinvlink, d2invlink, df, ipm
            )
        else
            # AMEs at representative values
            compute_repval_ames_fixed!(
                result_map, se_map, grad_map, varlist, repvals,
                ws, β, cholΣβ, invlink, dinvlink, d2invlink, 
                df, ipm
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
# Standard AME computation (no repvals) - FIXED VERSION
###############################################################################

"""
    compute_standard_ames_fixed!(
        result_map, se_map, grad_map, varlist, cts_vars, cat_vars, 
        pairs, ws, β, cholΣβ, invlink, dinvlink, d2invlink, df, ipm
    )

Compute standard AMEs without representative values using efficient selective updates.
"""
function compute_standard_ames_fixed!(result_map, se_map, grad_map, varlist, cts_vars, cat_vars, 
                                     pairs, ws, β, cholΣβ, invlink, dinvlink, d2invlink, df, ipm)
    # Continuous variables - now uses efficient selective updates
    if !isempty(cts_vars)
        requested_cts = filter(v -> v in varlist, cts_vars)
        if !isempty(requested_cts)
            ames, ses, grads = compute_continuous_ames_selective!(
                requested_cts, ws, β, cholΣβ, dinvlink, d2invlink, ipm
            )
            
            for (i, v) in enumerate(requested_cts)
                result_map[v] = ames[i]
                se_map[v] = ses[i]
                grad_map[v] = grads[i]
            end
        end
    end
    
    # Categorical variables - unchanged
    for v in cat_vars
        ame_d = Dict{Tuple,Float64}()
        se_d = Dict{Tuple,Float64}()
        grad_d = Dict{Tuple,Vector{Float64}}()
        
        if pairs == :baseline
            compute_factor_baseline_selective!(
                ame_d, se_d, grad_d, v, ws, β, 
                vcov(cholΣβ), invlink, dinvlink, df, ipm
            )
        else # :allpairs
            compute_factor_allpairs_selective!(
                ame_d, se_d, grad_d, v, ws, β, 
                vcov(cholΣβ), invlink, dinvlink, df, ipm
            )
        end
        
        result_map[v] = ame_d
        se_map[v] = se_d
        grad_map[v] = grad_d
    end
end

###############################################################################
# Representative values AME computation - FIXED VERSION
###############################################################################

"""
    compute_repval_ames_fixed!(result_map, se_map, grad_map, varlist, repvals,
                              ws, β, cholΣβ, invlink, dinvlink, d2invlink, df, ipm)

Compute AMEs at representative values using efficient selective updates.
"""
function compute_repval_ames_fixed!(result_map, se_map, grad_map, varlist, repvals,
                                   ws, β, cholΣβ, invlink, dinvlink, d2invlink, df, ipm)
    for v in varlist
        ame_d, se_d, g_d = _ame_representation!(
            ws, ipm, df, v, repvals, β, cholΣβ, invlink, dinvlink, d2invlink
        )
        
        result_map[v] = ame_d
        se_map[v] = se_d
        grad_map[v] = g_d
    end
end

###############################################################################
# Prediction computation with selective updates (unchanged from working version)
###############################################################################

"""
    compute_predictions!(result_map, se_map, grad_map, varlist, repvals, 
                        ws, β, Σβ, invlink, dinvlink, ipm)

Compute average predictions using selective matrix updates.
"""
function compute_predictions!(result_map, se_map, grad_map, varlist, repvals, 
                             ws, β, Σβ, invlink, dinvlink, ipm)
    if isempty(repvals)
        # Simple prediction at base data
        mul!(ws.η, ws.base_matrix, β)
        
        # Apply inverse link
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = invlink(ws.η[i])
        end
        
        pred = sum(ws.μp_vals) / length(ws.μp_vals)
        
        # Compute gradient
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = dinvlink(ws.η[i])
        end
        
        mul!(ws.grad_work, ws.base_matrix', ws.μp_vals)
        ws.grad_work ./= length(ws.η)
        
        se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
        
        # Store results for all requested variables
        for v in varlist
            result_map[v] = pred
            se_map[v] = se
            grad_map[v] = copy(ws.grad_work)
        end
    else
        # Predictions at representative values
        compute_repval_predictions!(result_map, se_map, grad_map, varlist, repvals,
                                   ws, β, Σβ, invlink, dinvlink, ipm)
    end
end

"""
    compute_repval_predictions!(result_map, se_map, grad_map, varlist, repvals,
                               ws, β, Σβ, invlink, dinvlink, ipm)

Compute predictions at representative values using selective updates.
"""
function compute_repval_predictions!(result_map, se_map, grad_map, varlist, repvals,
                                    ws, β, Σβ, invlink, dinvlink, ipm)
    repvars = collect(keys(repvals))
    combos = collect(Iterators.product((repvals[r] for r in repvars)...))
    
    for combo in combos
        n = length(first(ws.base_data))
        changes = Dict{Symbol,AbstractVector}()
        for (rv, val) in zip(repvars, combo)
            orig_col = ws.base_data[rv]
            if orig_col isa CategoricalArray || val isa CategoricalValue
                # proper CategoricalArray for factor‐like repvals
                changes[rv] = categorical(
                    fill(val, n);
                    levels  = levels(orig_col),
                    ordered = isordered(orig_col),
                )
            else
                changes[rv] = fill(val, n)
            end
        end
        
        # Update matrices with representative values
        update_for_variables!(ws, changes, ipm)
        
        # Compute prediction
        mul!(ws.η, ws.work_matrix, β)
        
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = invlink(ws.η[i])
        end
        
        pred = sum(ws.μp_vals) / length(ws.μp_vals)
        
        # Compute gradient  
        @inbounds @simd for i in eachindex(ws.η)
            ws.μp_vals[i] = dinvlink(ws.η[i])
        end
        
        mul!(ws.grad_work, ws.work_matrix', ws.μp_vals)
        ws.grad_work ./= length(ws.η)
        
        se = sqrt(dot(ws.grad_work, Σβ * ws.grad_work))
        
        # Store results
        combo_key = Tuple(combo)
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
end

###############################################################################
# Utility functions (unchanged)
###############################################################################

"""
    classify_variable_type(var::Symbol, df::AbstractDataFrame) -> Symbol

Classify a variable as :continuous, :boolean, or :categorical.
"""
function classify_variable_type(var::Symbol, df::AbstractDataFrame)
    col_type = eltype(df[!, var])
    
    if col_type <: Real && col_type != Bool
        return :continuous
    elseif col_type <: Bool
        return :boolean
    else
        return :categorical
    end
end

"""
    validate_variables(varlist::Vector{Symbol}, df::AbstractDataFrame)

Validate that all requested variables exist in the data.
"""
function validate_variables(varlist::Vector{Symbol}, df::AbstractDataFrame)
    data_vars = Set(Symbol.(names(df)))
    
    for var in varlist
        if var ∉ data_vars
            throw(ArgumentError("Variable $var not found in data"))
        end
    end
    
    return true
end
