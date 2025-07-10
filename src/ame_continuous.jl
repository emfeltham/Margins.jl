# ame_continuous.jl

"""
    compute_continuous_ames_selective!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                      β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                      dinvlink::Function, d2invlink::Function, 
                                      ipm::InplaceModeler)

Compute AMEs for multiple continuous variables using selective matrix updates.
FIXED: Now uses appropriate step sizes and stable numerical methods.
"""
function compute_continuous_ames_selective!(variables::Vector{Symbol}, ws::AMEWorkspace,
                                          β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                          dinvlink::Function, d2invlink::Function, 
                                          ipm::InplaceModeler)
    n, p = size(ws.base_matrix)
    k = length(variables)
    
    # Pre-allocate results
    ames = Vector{Float64}(undef, k)
    ses = Vector{Float64}(undef, k)
    grads = Vector{Vector{Float64}}(undef, k)
    
    # Process each variable with selective updates
    for (j, variable) in enumerate(variables)
        # Validate that variable is continuous and pre-allocated
        if !haskey(ws.pert_vectors, variable)
            throw(ArgumentError(
                "Variable $variable not found in perturbation vectors. " *
                "Only continuous (non-Bool) variables are supported."
            ))
        end
        
        # FIXED: Use robust step size calculation instead of ultra-conservative 1e-6
        orig_values = ws.base_data[variable]
        
        # Compute robust scale measures
        finite_vals = filter(isfinite, orig_values)
        
        if !isempty(finite_vals)
            var_std = std(finite_vals)
            var_range = maximum(finite_vals) - minimum(finite_vals)
            var_mean = mean(finite_vals)
            
            # Use 0.1% of standard deviation (much larger than previous 0.0001%)
            if var_std > 0 && isfinite(var_std)
                h = var_std * 1e-3  # CHANGED: from 1e-6 to 1e-3
            elseif var_range > 0 && isfinite(var_range)
                h = var_range * 1e-3  # CHANGED: from 1e-6 to 1e-3
            elseif abs(var_mean) > 0 && isfinite(var_mean)
                h = abs(var_mean) * 1e-3  # CHANGED: from 1e-6 to 1e-3
            else
                h = 1e-3  # CHANGED: from 1e-6 to 1e-3
            end
        else
            h = 1e-3  # CHANGED: from 1e-6 to 1e-3
        end
        
        # Ensure reasonable bounds
        h = max(h, 1e-6)   # Not too small
        h = min(h, 1e-1)   # Not too large
        
        # Prepare finite difference matrix using selective updates
        prepare_analytical_derivatives!(ws, variable, h, ipm)
        
        # Compute AME and SE using selective finite difference matrix
        ame, se, grad_ref = _ame_continuous_selective_fixed!(
            β, cholΣβ, ws.base_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
        )
        
        # Store results (copy gradient since workspace will be reused)
        ames[j] = ame
        ses[j] = se
        grads[j] = copy(grad_ref)
    end
    
    return ames, ses, grads
end

#####

"""
    update_affected_columns_forwarddiff!(work_matrix, data, affected_cols, variable, ws, ipm)

ForwardDiff-compatible selective column update.
"""
function update_affected_columns_forwarddiff!(work_matrix::AbstractMatrix{T}, 
                                            data::NamedTuple, 
                                            affected_cols::Vector{Int}, 
                                            variable::Symbol,
                                            ws::AMEWorkspace,
                                            ipm::InplaceModeler) where T
    
    println("  === DEBUG: update_affected_columns_forwarddiff! ===")
    println("  Work matrix type: $(typeof(work_matrix))")
    println("  Target variable: $variable")
    println("  Affected cols: $affected_cols")
    
    # Get the terms that affect these columns
    terms_to_update = Set{AbstractTerm}()
    for (term, range) in ws.mapping.term_info
        if !isempty(intersect(collect(range), affected_cols))
            push!(terms_to_update, term)
            println("  Term to update: $(typeof(term)) affecting $range")
        end
    end
    
    # Re-evaluate only the affected terms using ForwardDiff-compatible operations
    fn_i = Ref(1)
    int_i = Ref(1)
    
    for term in terms_to_update
        range = ws.mapping.term_to_range[term]
        if !isempty(range)
            println("  Processing term $(typeof(term)) for range $range")
            
            try
                # Use ForwardDiff-compatible _cols! function
                _cols_forwarddiff!(term, data, work_matrix, first(range), ipm, fn_i, int_i)
                println("  Successfully processed $(typeof(term))")
            catch e
                println("  ERROR processing $(typeof(term)): $e")
                println("  Error type: $(typeof(e))")
                if isa(e, MethodError)
                    println("  Method error: $(e.f) with args types $(typeof.(e.args))")
                end
                rethrow(e)
            end
        end
    end
    
    println("  === END DEBUG: update_affected_columns_forwarddiff! ===")
end

"""
    _cols_forwarddiff!(term, data, X, j, ipm, fn_i, int_i)

ForwardDiff-compatible version of _cols! that handles dual numbers.
"""
function _cols_forwarddiff!(term::ContinuousTerm, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    copy!(view(X, :, j), data[term.sym])
    return j + 1
end

function _cols_forwarddiff!(term::Term, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    copy!(view(X, :, j), data[term.sym])
    return j + 1
end

function _cols_forwarddiff!(term::InterceptTerm{true}, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    fill!(view(X, :, j), one(T))
    return j + 1
end

function _cols_forwarddiff!(term::InterceptTerm{false}, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    return j  # No columns for false intercept
end

function _cols_forwarddiff!(term::ConstantTerm, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    fill!(view(X, :, j), T(term.n))
    return j + 1
end

function _cols_forwarddiff!(term::CategoricalTerm, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    println("    === DEBUG: _cols_forwarddiff! CategoricalTerm ===")
    println("    Term symbol: $(term.sym)")
    println("    Matrix type T: $T")
    
    v = data[term.sym]
    println("    Variable v type: $(typeof(v))")
    println("    Variable v eltype: $(eltype(v))")
    println("    Variable v sample: $(v[1:min(3, length(v))])")
    
    M = term.contrasts.matrix
    n_contrast_cols = size(M, 2)
    
    println("    Contrast matrix size: $(size(M))")
    println("    n_contrast_cols: $n_contrast_cols")
    
    # Handle both CategoricalArray and regular arrays
    if isa(v, CategoricalArray)
        println("    Using CategoricalArray path")
        codes = refs(v)
        println("    Codes type: $(typeof(codes))")
        println("    Codes sample: $(codes[1:min(3, length(codes))])")
    else
        println("    Using regular array path")
        
        # Check if we have dual numbers
        if eltype(v) <: ForwardDiff.Dual
            println("    FOUND DUAL NUMBERS in categorical variable!")
            v_values = [ForwardDiff.value(x) for x in v]
            println("    Extracted values: $(v_values[1:min(3, length(v_values))])")
        else
            v_values = v
            println("    No dual numbers, using values directly")
        end
        
        unique_vals = sort(unique(v_values))
        println("    Unique values: $unique_vals")
        code_map = Dict(val => i for (i, val) in enumerate(unique_vals))
        println("    Code map: $code_map")
        codes = [code_map[val] for val in v_values]
        println("    Generated codes: $(codes[1:min(3, length(codes))])")
    end
    
    println("    About to fill matrix...")
    
    try
        @inbounds for r in 1:length(codes)
            code = codes[r]
            @simd for k in 1:n_contrast_cols
                X[r, j + k - 1] = T(M[code, k])
            end
        end
        println("    Matrix filling completed successfully")
    catch e
        println("    ERROR in matrix filling: $e")
        println("    Error type: $(typeof(e))")
        rethrow(e)
    end
    
    println("    === END DEBUG: _cols_forwarddiff! CategoricalTerm ===")
    return j + n_contrast_cols
end

function _cols_forwarddiff!(term::FunctionTerm, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    idx = fn_i[]; fn_i[] += 1
    nargs = length(term.args)
    
    # Create temporary scratch that can handle dual numbers
    temp_scratch = Matrix{T}(undef, size(X, 1), nargs)
    
    # Fill each argument into its column
    for (arg_i, arg) in enumerate(term.args)
        _cols_forwarddiff!(arg, data, temp_scratch, arg_i, ipm, fn_i, int_i)
    end
    
    # Apply function and store result
    col = view(X, :, j)
    rows = size(col, 1)
    @inbounds @simd for r in 1:rows
        if nargs == 1
            col[r] = term.f(temp_scratch[r, 1])
        elseif nargs == 2
            col[r] = term.f(temp_scratch[r, 1], temp_scratch[r, 2])
        elseif nargs == 3
            col[r] = term.f(temp_scratch[r, 1], temp_scratch[r, 2], temp_scratch[r, 3])
        else
            col[r] = term.f(ntuple(k -> temp_scratch[r, k], nargs)...)
        end
    end
    
    return j + 1
end

function _cols_forwarddiff!(term::InteractionTerm, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    idx = int_i[]; int_i[] += 1
    sw = ipm.int_subw[idx]
    stride = ipm.int_stride[idx]
    rows = size(X, 1)
    
    # Create temporary scratch that can handle dual numbers
    total_component_width = sum(sw)
    temp_scratch = Matrix{T}(undef, rows, total_component_width)
    
    # Fill each component into temp_scratch
    ofs = 0
    for (comp, w) in zip(term.terms, sw)
        comp_view = view(temp_scratch, :, ofs+1:ofs+w)
        _cols_forwarddiff!(comp, data, comp_view, 1, ipm, fn_i, int_i)
        ofs += w
    end
    
    # Compute Kronecker product into destination
    total = prod(sw)
    dest = view(X, :, j:j+total-1)
    
    @inbounds for r in 1:rows, col in 1:total
        off = col - 1
        acc = one(T)
        ofs = 0
        for p in 1:length(sw)
            k = (off ÷ stride[p]) % sw[p]
            acc *= temp_scratch[r, ofs + k + 1]
            ofs += sw[p]
        end
        dest[r, col] = acc
    end
    
    return j + total
end

# Fallback for any other term types
function _cols_forwarddiff!(term::AbstractTerm, data::NamedTuple, X::AbstractMatrix{T}, j::Int, ipm, fn_i, int_i) where T
    # Use the regular _cols! function and convert result
    w = width(term)
    temp_matrix = Matrix{Float64}(undef, size(X, 1), w)
    next_j = _cols!(term, data, temp_matrix, 1, ipm, fn_i, int_i)
    
    # Copy and convert to target type
    n_cols = next_j - j
    for k in 1:n_cols
        for r in 1:size(X, 1)
            X[r, j + k - 1] = T(temp_matrix[r, k])
        end
    end
    
    return next_j
end

##########

function _ame_continuous_selective_fixed!(
    β::Vector{Float64},
    cholΣβ::Cholesky{Float64,<:AbstractMatrix{Float64}},
    X::AbstractMatrix{Float64},
    Xdx::AbstractMatrix{Float64},
    dinvlink::Function,
    d2invlink::Function,
    ws::AMEWorkspace
)
    n, p = size(X)
    
    # Unpack workspace vectors
    η, dη = ws.η, ws.dη
    μp_vals, μpp_vals = ws.μp_vals, ws.μpp_vals
    grad_work = ws.grad_work
    temp1, temp2 = ws.temp1, ws.temp2
    
    # Compute linear predictors
    mul!(η, X, β)
    mul!(dη, Xdx, β)
    
    # Only clamp extreme values that would cause link function failures
    @inbounds for i in 1:n
        if abs(η[i]) > 50.0  # Very generous bounds
            η[i] = sign(η[i]) * 50.0
        end
        if abs(dη[i]) > 50.0
            dη[i] = sign(dη[i]) * 50.0
        end
    end
    
    # Compute AME with improved numerical stability
    sum_ame = 0.0
    n_valid = 0
    
    @inbounds for i in 1:n
        ηi = η[i]
        dηi = dη[i]
        
        # Skip only clearly problematic observations
        if !isfinite(ηi) || !isfinite(dηi)
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        # Compute link function derivatives with error handling
        local μp, μpp
        try
            μp = dinvlink(ηi)
            μpp = d2invlink(ηi)
        catch e
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        # Check for reasonable link function outputs
        if !isfinite(μp) || !isfinite(μpp)
            μp_vals[i] = 0.0
            μpp_vals[i] = 0.0
            continue
        end
        
        marginal_effect = μp * dηi
        
        # FIXED: More reasonable bounds checking
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
    
    # FIXED: Improved gradient computation
    fill!(temp1, 0.0)
    fill!(temp2, 0.0)
    
    # Initialize se to a reasonable default
    se = NaN
    
    try
        mul!(temp1, X', μpp_vals)
        mul!(temp2, Xdx', μp_vals)
        
        # Check for problematic gradients before proceeding
        temp1_finite = all(isfinite, temp1)
        temp2_finite = all(isfinite, temp2)
        
        if temp1_finite && temp2_finite
            # Combine gradients
            inv_n = 1.0 / n
            @inbounds for i in 1:p
                grad_work[i] = (temp1[i] + temp2[i]) * inv_n
                
                # Basic sanity check
                if !isfinite(grad_work[i])
                    grad_work[i] = 0.0
                end
            end
            
            # Compute SE with improved stability
            grad_norm = norm(grad_work)
            if grad_norm > 0.0 && isfinite(grad_norm) && grad_norm < 1e6
                mul!(temp1, cholΣβ.U, grad_work)
                se_squared = dot(temp1, temp1)
                if se_squared >= 0 && isfinite(se_squared)
                    se = sqrt(se_squared)
                end
            end
        else
            fill!(grad_work, 0.0)
        end
        
    catch e
        @warn "Gradient computation failed: $e"
        fill!(grad_work, 0.0)
        se = NaN
    end
    
    return ame, se, grad_work
end

"""
    compute_single_continuous_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                           β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                           dinvlink::Function, d2invlink::Function, 
                                           ipm::InplaceModeler)

Compute AME for a single continuous variable using selective updates.
Used for representative values computation - FIXED VERSION.
"""
function compute_single_continuous_ame_selective!(variable::Symbol, ws::AMEWorkspace,
                                                β::AbstractVector, cholΣβ::LinearAlgebra.Cholesky,
                                                dinvlink::Function, d2invlink::Function, 
                                                ipm::InplaceModeler)
    # Validate variable
    if !haskey(ws.pert_vectors, variable)
        throw(ArgumentError(
            "Variable $variable not found in perturbation vectors. " *
            "Only continuous (non-Bool) variables are supported."
        ))
    end
    
    # FIXED: Use robust step size calculation for representative values too
    orig_values = ws.base_data[variable]
    
    # Compute robust scale measures
    finite_vals = filter(isfinite, orig_values)
    
    if !isempty(finite_vals)
        var_std = std(finite_vals)
        var_range = maximum(finite_vals) - minimum(finite_vals)
        var_mean = mean(finite_vals)
        
        # Use 0.1% of standard deviation (same as main computation)
        if var_std > 0 && isfinite(var_std)
            h = var_std * 1e-3  # CHANGED: from 1e-6 to 1e-3
        elseif var_range > 0 && isfinite(var_range)
            h = var_range * 1e-3  # CHANGED: from 1e-6 to 1e-3
        elseif abs(var_mean) > 0 && isfinite(var_mean)
            h = abs(var_mean) * 1e-3  # CHANGED: from 1e-6 to 1e-3
        else
            h = 1e-3  # CHANGED: from 1e-6 to 1e-3
        end
    else
        h = 1e-3  # CHANGED: from 1e-6 to 1e-3
    end
    
    h = max(h, 1e-6)
    h = min(h, 1e-1)
    
    # The workspace should already have work_matrix set to representative values
    # We need to compute finite differences from this state
    
    # Store current state
    current_matrix = copy(ws.work_matrix)
    
    # Create perturbed values: current + h
    pert_vector = ws.pert_vectors[variable]
    current_values = ws.base_data[variable]  # This might be at repvals already
    
    @inbounds for i in eachindex(pert_vector)
        pert_vector[i] = current_values[i] + h
    end
    
    # Update work matrix with perturbed values
    update_for_variable!(ws, variable, pert_vector, ipm)
    
    # Get affected columns
    affected_cols = ws.variable_plans[variable]
    
    # Compute finite differences: (X_perturbed - X_current) / h
    invh = 1.0 / h
    
    @inbounds for col in affected_cols, row in axes(ws.finite_diff_matrix, 1)
        baseline_val = current_matrix[row, col]
        perturbed_val = ws.work_matrix[row, col]
        
        # Basic sanity checks only
        if !isfinite(baseline_val) || !isfinite(perturbed_val)
            ws.finite_diff_matrix[row, col] = 0.0
            continue
        end
        
        raw_diff = perturbed_val - baseline_val
        finite_diff = raw_diff * invh
        
        # FIXED: Less aggressive clamping
        if isfinite(finite_diff)
            ws.finite_diff_matrix[row, col] = clamp(finite_diff, -1e8, 1e8)  # CHANGED: from 1e6 to 1e8
        else
            ws.finite_diff_matrix[row, col] = 0.0
        end
    end
    
    # Zero out unaffected columns
    total_cols = size(ws.finite_diff_matrix, 2)
    unaffected_cols = get_unchanged_columns(ws.mapping, [variable], total_cols)
    
    @inbounds for col in unaffected_cols, row in axes(ws.finite_diff_matrix, 1)
        ws.finite_diff_matrix[row, col] = 0.0
    end
    
    # Compute AME using current work matrix as base and finite_diff_matrix for derivative
    ame, se, grad_ref = _ame_continuous_selective_fixed!(
        β, cholΣβ, current_matrix, ws.finite_diff_matrix, dinvlink, d2invlink, ws
    )
    
    # Restore state
    ws.work_matrix .= current_matrix
    
    return ame, se, grad_ref
end
