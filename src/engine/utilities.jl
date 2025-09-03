# engine/utilities.jl - FormulaCompiler-based marginal effects computation

using Tables  # Required for the architectural rework

"""
    _validate_variables(data_nt, vars)

Validate that requested variables exist and are analyzable.
Follows REORG.md input validation pattern (MARGINS_GUIDE.md recommendation).

# Arguments
- `data_nt::NamedTuple`: Data in columntable format
- `vars::Vector{Symbol}`: Variables to validate

# Throws
- `MarginsError`: If any variable is not found in data

# Examples
```julia
_validate_variables(data_nt, [:x1, :x2])  # Validates x1, x2 exist
```
"""
function _validate_variables(data_nt::NamedTuple, vars::Vector{Symbol})
    for var in vars
        if !haskey(data_nt, var)
            throw(MarginsError("Variable $var not found in data"))
        end
        
        col = getproperty(data_nt, var)
        if length(col) == 0
            throw(MarginsError("Variable $var has no observations"))
        end
        # Both continuous and categorical variables supported
        # Auto-detection handles the dispatch - no warnings needed
    end
end

# Helper: accumulate marginal effect value across rows using concrete arguments
function _accumulate_me_value(g_buf::Vector{Float64}, de, β::Vector{Float64}, link, rows, scale::Symbol, backend::Symbol, idx::Int)
    acc = 0.0
    if scale === :response
        for row in rows
            FormulaCompiler.marginal_effects_mu!(g_buf, de, β, row; link=link, backend=backend)
            acc += g_buf[idx]
        end
    else
        for row in rows
            FormulaCompiler.marginal_effects_eta!(g_buf, de, β, row; backend=backend)
            acc += g_buf[idx]
        end
    end
    return acc
end

# Helper: average response (μ or η) over rows using concrete arguments
function _average_response_over_rows(compiled, row_buf::Vector{Float64}, β::Vector{Float64}, link, data_nt::NamedTuple, rows, scale::Symbol)
    if scale === :response
        s = 0.0
        for row in rows
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, row)
            η = dot(β, row_buf)
            s += GLM.linkinv(link, η)
        end
        return s / length(rows)
    else
        s = 0.0
        for row in rows
            FormulaCompiler.modelrow!(row_buf, compiled, data_nt, row)
            s += dot(β, row_buf)
        end
        return s / length(rows)
    end
end

"""
    _get_baseline_level(model, var) -> baseline_level

Extract baseline level from model's contrast coding (statistically principled).

This function implements the design decision to use the model's actual contrast
coding rather than making assumptions from the data.

# Arguments
- `model`: Fitted statistical model
- `var::Symbol`: Categorical variable name

# Returns
- Baseline level used in the model's contrast coding

# Throws
- `MarginsError`: If baseline level cannot be determined

# Examples
```julia
baseline = _get_baseline_level(model, :region)  # Returns "North" if that's the baseline
```
"""
function _get_baseline_level(model, var::Symbol)
    # Extract baseline from model's contrast system
    if hasfield(typeof(model), :mf) && hasfield(typeof(model.mf), :contrasts)
        if haskey(model.mf.contrasts, var)
            contrast = model.mf.contrasts[var]
            if contrast isa StatsModels.DummyCoding
                return contrast.base
            elseif contrast isa StatsModels.EffectsCoding  
                return contrast.base
            elseif contrast isa StatsModels.HelmertCoding
                return contrast.base
            elseif contrast isa StatsModels.SeqDiffCoding
                return contrast.base
            # Add other contrast types as needed
            else
                throw(MarginsError("Unsupported contrast type $(typeof(contrast)) for variable $var"))
            end
        end
    end
    
    # Fallback: For CategoricalArrays in GLM, the first level is typically the baseline
    # Extract the variable from the original data to get its levels
    if hasfield(typeof(model), :mf) && hasfield(typeof(model.mf), :data)
        data = model.mf.data
        if haskey(data, var)
            col = data[var]
            if isa(col, CategoricalVector) || isa(col, CategoricalArray)
                # First level is the baseline by GLM convention
                levels_list = levels(col)
                if !isempty(levels_list)
                    return levels_list[1]
                end
            end
        end
    end
    
    throw(MarginsError("Could not determine baseline level for variable $var from model contrasts. " *
                      "Please ensure the model has proper contrast coding information."))
end

"""
    _is_continuous_variable(col) -> Bool

Determine if a data column represents a continuous variable.

Follows the design principle: Real numbers (except Bool) are continuous,
everything else is categorical.

# Arguments
- `col`: Data column (Vector)

# Returns
- `Bool`: true if continuous, false if categorical

# Examples
```julia
_is_continuous_variable([1.0, 2.0, 3.0])  # true
_is_continuous_variable([1, 2, 3])        # true (Int64 is Real)
_is_continuous_variable([true, false])    # false (Bool is categorical)
_is_continuous_variable(["A", "B", "C"])  # false
```
"""
function _is_continuous_variable(col)
    return eltype(col) <: Real && !(eltype(col) <: Bool)
end

"""
    _ame_continuous_and_categorical(engine, data_nt; target=:mu, backend=:ad) -> (DataFrame, Matrix)

Zero-allocation population effects (AME) using FormulaCompiler's built-in APIs.
Implements REORG.md lines 290-348 with explicit backend selection and batch operations.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine
- `data_nt::NamedTuple`: Data in columntable format
- `target::Symbol`: `:eta` for link scale, `:mu` for response scale
- `backend::Symbol`: `:ad` or `:fd` backend selection

# Returns
- `(DataFrame, Matrix{Float64})`: Results table and gradient matrix G

# Examples
```julia
df, G = _ame_continuous_and_categorical(engine, data_nt; target=:mu, backend=:fd)
```
"""
function _ame_continuous_and_categorical(engine::MarginsEngine{L}, data_nt::NamedTuple; scale=:response, backend=:ad, measure=:effect, contrasts=:baseline) where L
    rows = 1:length(first(data_nt))
    n_obs = length(first(data_nt))
    
    # Auto-detect variable types and process accordingly
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, data_nt)
    
    # Determine which variables we're processing (continuous vs categorical)
    continuous_requested = engine.de === nothing ? Symbol[] : engine.de.vars
    categorical_requested = [v for v in engine.vars if v ∉ continuous_vars]
    
    # Total number of variables to process
    total_vars = length(continuous_requested) + length(categorical_requested)
    
    if total_vars == 0
        # No variables to process - return empty DataFrame
        empty_df = DataFrame(
            term = String[],
            estimate = Float64[],
            se = Float64[],
            n = Int[]
        )
        return (empty_df, Matrix{Float64}(undef, 0, length(engine.β)))
    end
    
    # PRE-ALLOCATE results DataFrame to avoid dynamic growth (PERFORMANCE FIX)
    results = DataFrame(
        term = Vector{String}(undef, total_vars),
        estimate = Vector{Float64}(undef, total_vars), 
        se = Vector{Float64}(undef, total_vars),
        n = fill(n_obs, total_vars)  # Add sample size for all variables
    )
    G = Matrix{Float64}(undef, total_vars, length(engine.β))
    
    # Process continuous variables with FC's built-in AME gradient accumulation (ZERO ALLOCATION!)
    cont_idx = 1
    # Hoist frequently used engine fields to locals to avoid per-iteration field access costs
    local_de = engine.de
    local_β = engine.β
    local_link = engine.link
    local_row_buf = engine.row_buf
    local_compiled = engine.compiled
    
    # Process continuous variables (if any)
    if engine.de !== nothing
        for var in continuous_requested
            # Direct computation with explicit backend - no fallbacks
            # Users must choose backend explicitly based on their accuracy/performance needs
            FormulaCompiler.accumulate_ame_gradient!(
                engine.gβ_accumulator, local_de, local_β, rows, var;
                link=(scale === :response ? local_link : GLM.IdentityLink()), 
                backend=backend
            )
            
            # Note: accumulate_ame_gradient! already averages the gradient
            gβ_avg = engine.gβ_accumulator  # Use directly without copying
            se = compute_se_only(gβ_avg, engine.Σ)
            
            # Compute AME value via helper with concrete arguments
            ame_val = _accumulate_me_value(engine.g_buf, local_de, local_β, local_link, rows, scale, backend, cont_idx)
            ame_val /= length(rows)
            
            # Apply elasticity transformations if requested (Phase 3)
            final_val = ame_val
            if measure !== :effect && engine.de !== nothing
                # Compute average x and y for elasticity measures (vectorized)
                xcol = getproperty(data_nt, var)
                
                # Step 1: Compute average x directly (no loop needed)
                x̄ = sum(float(xcol[row]) for row in rows) / length(rows)
                
                # Step 2: Compute η/μ averages using helper with concrete arguments
                ȳ = _average_response_over_rows(local_compiled, local_row_buf, local_β, local_link, data_nt, rows, scale)
                
                # Apply transformation based on measure type
                if measure === :elasticity
                    final_val = (x̄ / ȳ) * ame_val
                elseif measure === :semielasticity_dyex
                    final_val = x̄ * ame_val
                elseif measure === :semielasticity_eydx
                    final_val = (1 / ȳ) * ame_val
                end
            end
            
            # Direct assignment instead of push! to avoid reallocation
            results.term[cont_idx] = string(var)
            results.estimate[cont_idx] = final_val
            results.se[cont_idx] = se
            # Copy the averaged gradient to the output matrix
            for j in 1:length(gβ_avg)
                G[cont_idx, j] = gβ_avg[j]
            end
            cont_idx += 1
        end
    end
    
    # Process categorical variables (if any)
    for var in categorical_requested
        # Compute traditional baseline contrasts for population margins
        ame_val, gβ_avg = _compute_categorical_baseline_ame(engine, var, rows, scale, backend)
        se = compute_se_only(gβ_avg, engine.Σ)
        
        # Direct assignment instead of push! to avoid reallocation  
        results.term[cont_idx] = string(var)
        results.estimate[cont_idx] = ame_val
        results.se[cont_idx] = se
        G[cont_idx, :] = gβ_avg  # Still increment for output indexing
        cont_idx += 1
    end
    
    return (results, G)
end

"""
    _mem_continuous_and_categorical(engine, profiles; target=:mu, backend=:ad) -> (DataFrame, Matrix)

Profile Effects (MEM) Using Reference Grids with FormulaCompiler's built-in APIs.
Implements REORG.md lines 353-486 following FormulaCompiler guide.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine
- `profiles::Vector{Dict}`: Vector of profile dictionaries
- `target::Symbol`: `:eta` for link scale, `:mu` for response scale
- `backend::Symbol`: `:ad` or `:fd` backend selection

# Returns
- `(DataFrame, Matrix{Float64})`: Results table and gradient matrix G

# Examples
```julia
profiles = [Dict(:x1 => 0.0, :region => "North")]
df, G = _mem_continuous_and_categorical(engine, profiles; target=:mu, backend=:ad)
```
"""
function _mem_continuous_and_categorical(engine::MarginsEngine{L}, profiles::Vector; target=:mu, backend=:ad, measure=:effect) where L
    # Handle the case where we have only categorical variables (engine.de === nothing)
    # or mixed continuous/categorical variables
    
    n_profiles = length(profiles)
    
    # Auto-detect variable types ONCE (not per profile) - PERFORMANCE FIX
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, engine.data_nt)
    
    # Determine which variables we're actually processing
    requested_vars = engine.vars  # Variables requested by user
    continuous_requested = [v for v in requested_vars if v ∈ continuous_vars]
    categorical_requested = [v for v in requested_vars if v ∉ continuous_vars]
    
    # Calculate total number of terms (for gradient matrix sizing)
    total_terms = n_profiles * length(requested_vars)
    
    # PRE-ALLOCATE results DataFrame to avoid dynamic growth (PERFORMANCE FIX)
    results = DataFrame(
        term = String[],
        estimate = Float64[],
        se = Float64[]
    )
    G = Matrix{Float64}(undef, total_terms, length(engine.β))
    
    row_idx = 1
    # Hoist engine fields commonly used in inner loops
    local_row_buf = engine.row_buf
    local_compiled = engine.compiled
    local_β = engine.β
    local_link = engine.link
    for profile in profiles
        # Build minimal synthetic reference grid data (not scenarios!)
        # This creates efficient synthetic data with just the needed variables
        refgrid_data = _build_refgrid_data(profile, engine.data_nt)
        refgrid_compiled = FormulaCompiler.compile_formula(engine.model, refgrid_data)
        
        # Build derivative evaluator only if we have continuous variables
        refgrid_de = nothing
        if !isempty(continuous_requested) && engine.de !== nothing
            refgrid_de = FormulaCompiler.build_derivative_evaluator(refgrid_compiled, refgrid_data; 
                                                                   vars=continuous_requested)
        end
        
        # Process all requested variables (both continuous and categorical)
        for var in requested_vars
            if var ∈ continuous_vars
                # Continuous variable: compute derivative using FormulaCompiler
                if target === :mu
                    FormulaCompiler.marginal_effects_mu!(engine.g_buf, refgrid_de, engine.β, 1;
                                                        link=engine.link, backend=backend)
                else
                    FormulaCompiler.marginal_effects_eta!(engine.g_buf, refgrid_de, engine.β, 1;
                                                         backend=backend)
                end
                # Find the index of this variable in the continuous variables
                continuous_var_idx = findfirst(==(var), continuous_requested)
                effect_val = engine.g_buf[continuous_var_idx]
                
                # Compute parameter gradient for SE using refgrid derivative evaluator
                if target === :mu
                    FormulaCompiler.me_mu_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var;
                                                   link=engine.link)
                else
                    FormulaCompiler.me_eta_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var)
                end
            else
                # Categorical variable: compute row-specific baseline contrast
                effect_val = _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, target, backend)
                _row_specific_contrast_grad_beta!(engine.gβ_accumulator, engine, refgrid_de, profile, var, target)
            end
            
            se = compute_se_only(engine.gβ_accumulator, engine.Σ)
            
            # Apply elasticity transformations for continuous variables if requested (Phase 3)
            final_val = effect_val
            if var ∈ continuous_vars && measure !== :effect && engine.de !== nothing
                # Get x and y values at this specific profile
                x_val = float(profile[var])
                
                # Use FormulaCompiler's proper pattern for prediction with scenario
                # Note: This assumes refgrid_data is available in this context
                FormulaCompiler.modelrow!(local_row_buf, local_compiled, refgrid_data, 1)
                η = dot(local_row_buf, local_β)
                
                if target === :mu
                    y_val = GLM.linkinv(local_link, η)                # Transform to μ scale
                else
                    y_val = η                                          # Use η scale directly
                end
                
                # Apply transformation based on measure type
                if measure === :elasticity
                    final_val = (x_val / y_val) * effect_val
                elseif measure === :semielasticity_dyex
                    final_val = x_val * effect_val
                elseif measure === :semielasticity_eydx
                    final_val = (1 / y_val) * effect_val
                end
            end
            
            # Build profile description
            profile_desc = join(["$(k)=$(v)" for (k,v) in pairs(profile)], ", ")
            if var ∈ continuous_vars
                term_name = "$(var) at $(profile_desc)"
            else
                # Show the specific contrast being computed
                current_level = profile[var]
                baseline_level = _get_baseline_level(engine.model, var)
                term_name = "$(var)=$(current_level) vs $(baseline_level) at $(profile_desc)"
            end
            
            # Store results using push!
            push!(results, (term=term_name, estimate=final_val, se=se))
            G[row_idx, :] = engine.gβ_accumulator
            row_idx += 1
        end
    end
    
    return (results, G)
end

# Helper: Build minimal reference grid data for row-specific contrasts
function _build_refgrid_data(profile::Dict, original_data::NamedTuple)
    # Create minimal synthetic data with only needed variables
    refgrid = NamedTuple()
    for (var, val) in pairs(original_data)
        if haskey(profile, var)
            # Use profile value for this variable (including categorical levels)
            profile_val = profile[var]
            if val isa CategoricalArray && profile_val isa String
                # Convert string to categorical with same levels
                profile_val = CategoricalArrays.categorical([profile_val], levels=levels(val))[1]
            end
            refgrid = merge(refgrid, NamedTuple{(var,)}(([profile_val],)))
        else
            # Use representative value (mean for continuous, first level for categorical)
            typical_val = _get_typical_value(val)
            refgrid = merge(refgrid, NamedTuple{(var,)}(([typical_val],)))
        end
    end
    return refgrid
end

"""
    _get_typical_value(col) -> typical_value

Get representative value for a data column.

# Arguments
- `col`: Data column (Vector)

# Returns
- Representative value: mean for continuous, mode for categorical

# Examples
```julia
_get_typical_value([1.0, 2.0, 3.0])     # 2.0 (mean)
_get_typical_value(["A", "A", "B"])     # "A" (mode)
_get_typical_value([true, false, true]) # true (mode)
```
"""
function _get_typical_value(col)
    if _is_continuous_variable(col)
        return mean(col)
    elseif col isa CategoricalArray
        return _create_frequency_mixture(col)  # Use frequency-weighted mixture
    elseif eltype(col) <: Bool  
        return _create_frequency_mixture(col)  # Use frequency-weighted mixture for consistency
    elseif eltype(col) <: AbstractString
        return mode(col)  # Simple mode for string categoricals (not the main use case)
    else
        return first(col)  # Fallback to first value
    end
end

"""
    _create_frequency_mixture(col) -> CategoricalMixture or Float64

Create a frequency-weighted categorical mixture from data column.
This represents the actual population composition in the data, providing
a statistically principled "typical value" for categorical variables.

Special handling for Bool: returns probability of `true` as Float64.

# Arguments
- `col`: Data column (Vector of any categorical type)

# Returns
- `CategoricalMixture`: Mixture with levels and frequencies as weights
- `Float64`: For Bool columns, returns P(true)

# Examples
```julia
# For data: ["A", "A", "B", "A"] 
# Returns: mix("A" => 0.75, "B" => 0.25)

# For data: [true, false, true, true]  
# Returns: 0.75  (probability of true)
```
"""
function _create_frequency_mixture(col)
    # Special handling for Bool: return probability of true
    if eltype(col) <: Bool
        p_true = mean(col)  # Proportion of true values
        return p_true
    end
    
    # General categorical handling
    level_counts = Dict()
    total_count = length(col)
    
    for value in col
        level_counts[value] = get(level_counts, value, 0) + 1
    end
    
    # Convert to levels and weights
    levels = collect(keys(level_counts))
    weights = [level_counts[level] / total_count for level in levels]
    
    return CategoricalMixture(levels, weights)
end

"""
    _build_metadata(; type, vars, scale, backend, measure, n_obs, model_type, timestamp, at_spec, has_contexts) -> Dict

Build metadata dictionary for MarginsResult.

# Keyword Arguments
- `type::Symbol=:unknown`: Analysis type (:effects, :predictions)
- `vars::Vector{Symbol}=Symbol[]`: Variables analyzed
- `scale::Symbol=:response`: Target scale (:link, :response)  
- `backend::Symbol=:ad`: Computation backend (:ad, :fd)
- `n_obs::Int=0`: Number of observations
- `model_type=nothing`: Type of fitted model
- `timestamp=now()`: Analysis timestamp
- at_spec: Profile specification (for profile margins)
- has_contexts: Whether contexts (scenarios/groups) are used

# Returns
- `Dict{Symbol, Any}`: Metadata dictionary

# Examples
```julia
metadata = _build_metadata(
    type=:effects, vars=[:x1, :x2], scale=:response, 
    backend=:ad, n_obs=1000, model_type=LinearModel
)
```
"""
function _build_metadata(; 
    type=:unknown, 
    vars=Symbol[], 
    scale=:response, 
    backend=:ad,
    measure=:effect,
    n_obs=0,
    model_type=nothing,
    timestamp=nothing,
    at_spec=nothing,
    has_contexts=false)
    
    # Set default timestamp
    ts = timestamp === nothing ? string(now()) : timestamp
    
    return Dict{Symbol, Any}(
        :type => type,
        :vars => vars,
        :n_vars => vars === nothing ? 0 : length(vars),
        :scale => scale,
        :backend => backend,
        :measure => measure,
        :n_obs => n_obs,
        :model_type => model_type === nothing ? "unknown" : string(typeof(model_type)),
        :timestamp => ts,
        :at_spec => at_spec,
        :has_contexts => has_contexts
    )
end

# Import dependencies for utility functions
using Dates: now
using StatsBase: mode

"""
    _compute_categorical_baseline_ame(engine, var, rows, target, backend) -> (Float64, Vector{Float64})

Compute traditional baseline contrasts for categorical variables in population margins.
This computes the average marginal effect (AME) of changing from baseline to the modal level.
"""
function _compute_categorical_baseline_ame(engine::MarginsEngine{L}, var::Symbol, rows, scale::Symbol, backend::Symbol) where L
    # Get the variable data
    var_col = getproperty(engine.data_nt, var)
    
    # Get baseline level from model  
    baseline_level = _get_baseline_level(engine.model, var)
    
    # Find the modal (most frequent) non-baseline level
    level_counts = Dict()
    for row in rows
        level = var_col[row]
        if level != baseline_level
            level_counts[level] = get(level_counts, level, 0) + 1
        end
    end
    
    if isempty(level_counts)
        # All observations are at baseline level
        return (0.0, zeros(length(engine.β)))
    end
    
    # Get the most frequent non-baseline level
    modal_level = argmax(level_counts)
    
    # Compute average discrete change: E[Y|var=modal] - E[Y|var=baseline]
    ame_sum = 0.0
    grad_sum = zeros(length(engine.β))
    
    # For each observation, compute the discrete change if we switched from baseline to modal
    for row in rows
        # Create profiles for this observation
        baseline_profile = Dict(Symbol(k) => v[row] for (k, v) in pairs(engine.data_nt))
        modal_profile = copy(baseline_profile)
        
        # Set the categorical variable to baseline and modal levels
        baseline_profile[var] = baseline_level
        modal_profile[var] = modal_level
        
        # Compute predictions
        baseline_pred, baseline_grad = _profile_prediction_with_gradient(engine, baseline_profile, scale, backend)
        modal_pred, modal_grad = _profile_prediction_with_gradient(engine, modal_profile, scale, backend)
        
        # Accumulate the discrete change
        ame_sum += (modal_pred - baseline_pred)
        grad_sum .+= (modal_grad .- baseline_grad)
    end
    
    # Average over all observations
    n_obs = length(rows)
    ame_val = ame_sum / n_obs
    gβ_avg = grad_sum ./ n_obs
    
    return (ame_val, gβ_avg)
end

"""
    _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, scale, backend) -> Float64

Compute row-specific baseline contrast using the new profile/contrasts.jl implementation.
Helper function for _mem_continuous_and_categorical.
"""
function _compute_row_specific_baseline_contrast(engine::MarginsEngine{L}, refgrid_de, profile::Dict, var::Symbol, scale::Symbol, backend::Symbol) where L
    # Use the new profile contrasts implementation
    effect, _ = compute_profile_categorical_contrast(engine, profile, var, scale; backend)
    return effect
end

"""
    _row_specific_contrast_grad_beta!(gβ_buffer, engine, refgrid_de, profile, var, scale)

Compute gradient for row-specific categorical contrast using the new profile/contrasts.jl implementation.
Helper function for _mem_continuous_and_categorical.
"""
function _row_specific_contrast_grad_beta!(gβ_buffer::Vector{Float64}, engine::MarginsEngine{L}, refgrid_de, profile::Dict, var::Symbol, scale::Symbol) where L
    # Use the new profile contrasts implementation  
    _, gradient = compute_profile_categorical_contrast(engine, profile, var, scale; backend=:ad)
    copyto!(gβ_buffer, gradient)
end

"""
    _predict_with_formulacompiler(engine, profile, target) -> Float64

Make predictions using FormulaCompiler for categorical contrast computation.
Helper function that properly uses FormulaCompiler instead of manual computation.
"""
function _predict_with_formulacompiler(engine::MarginsEngine{L}, profile::Dict, target::Symbol) where L
    # Create minimal reference data for this profile  
    profile_data = _build_refgrid_data(profile, engine.data_nt)
    profile_compiled = FormulaCompiler.compile_formula(engine.model, profile_data)
    
    # Use zero-allocation FormulaCompiler's modelrow! to get design matrix row, then apply coefficients
    FormulaCompiler.modelrow!(engine.row_buf, profile_compiled, profile_data, 1)
    η = dot(engine.row_buf, engine.β)
    
    if target === :eta
        return η
    else # :mu  
        return GLM.linkinv(engine.link, η)
    end
end

"""
    _mem_continuous_and_categorical_refgrid(engine::MarginsEngine{L}, reference_grid; scale=:response, backend=:ad, measure=:effect) where L -> (DataFrame, Matrix{Float64})

**Architectural Rework**: Efficient single-compilation approach for profile marginal effects.

Replaces the problematic per-profile compilation with a single compilation approach:
1. Compile once with the complete reference grid 
2. Evaluate all profiles by iterating over rows
3. Fixes CategoricalMixture routing issues and improves performance

# Arguments
- `engine`: Pre-built MarginsEngine with original data
- `reference_grid`: DataFrame containing all profiles (with potential CategoricalMixture objects)
- `target`: Target scale (:mu or :eta)
- `backend`: Computational backend (:ad or :fd) 
- `measure`: Effect measure (:effect, :elasticity, etc.)

# Returns
- `DataFrame`: Results with estimates, standard errors, etc.
- `Matrix{Float64}`: Gradient matrix for delta-method standard errors

# Performance
- O(1) compilation instead of O(n) per-profile compilations
- Consistent mixture routing across all profiles
- Memory efficient with single compiled object
"""
function _mem_continuous_and_categorical_refgrid(engine::MarginsEngine{L}, reference_grid; scale=:response, backend=:ad, measure=:effect) where L
    n_profiles = nrow(reference_grid)
    
    # Auto-detect variable types ONCE (not per profile)
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, engine.data_nt)
    
    # Determine which variables we're actually processing
    requested_vars = engine.vars  # Variables requested by user
    continuous_requested = [v for v in requested_vars if v ∈ continuous_vars]
    categorical_requested = [v for v in requested_vars if v ∉ continuous_vars]
    
    # Calculate total number of terms (for gradient matrix sizing)
    total_terms = n_profiles * length(requested_vars)
    
    # PRE-ALLOCATE results DataFrame to avoid dynamic growth
    results = DataFrame(
        term = String[],
        estimate = Float64[],
        se = Float64[]
    )
    G = Matrix{Float64}(undef, total_terms, length(engine.β))
    
    # ARCHITECTURAL FIX: Single compilation with reference grid
    # Convert reference grid to Tables format for FormulaCompiler
    refgrid_data = Tables.columntable(reference_grid)
    
    # Single compilation with complete reference grid (fixes CategoricalMixture routing)
    refgrid_compiled = FormulaCompiler.compile_formula(engine.model, refgrid_data)
    
    # Build derivative evaluator once if we have continuous variables
    refgrid_de = nothing
    if !isempty(continuous_requested) && engine.de !== nothing
        refgrid_de = FormulaCompiler.build_derivative_evaluator(refgrid_compiled, refgrid_data; 
                                                               vars=continuous_requested)
    end
    
    row_idx = 1
    # Hoist commonly used fields
    local_β = engine.β
    local_link = engine.link
    
    # MAIN LOOP: Iterate over profile rows instead of recompiling
    for profile_idx in 1:n_profiles
        # Process all requested variables for this profile row
        for var in requested_vars
            if var ∈ continuous_vars
                # Continuous variable: compute derivative using FormulaCompiler
                if scale === :response
                    FormulaCompiler.marginal_effects_mu!(engine.g_buf, refgrid_de, local_β, profile_idx;
                                                        link=local_link, backend=backend)
                else # scale === :link
                    FormulaCompiler.marginal_effects_eta!(engine.g_buf, refgrid_de, local_β, profile_idx; 
                                                         backend=backend)
                end
                
                # Find the gradient component for this variable
                var_idx = findfirst(==(var), continuous_requested)
                if var_idx !== nothing
                    marginal_effect = engine.g_buf[var_idx]
                    
                    # Apply measure transformation
                    if measure === :effect
                        estimate = marginal_effect
                    elseif measure === :elasticity
                        # Get the variable value at this profile
                        var_value = refgrid_data[var][profile_idx]
                        # Get the predicted value (μ or η)
                        output = Vector{Float64}(undef, length(refgrid_compiled))
                        refgrid_compiled(output, refgrid_data, profile_idx)
                        pred_value = sum(output)  # Sum of all terms gives prediction
                        
                        if scale === :response
                            estimate = marginal_effect * (var_value / pred_value)
                        else # :link
                            estimate = marginal_effect * (var_value / pred_value)
                        end
                    else
                        error("Unsupported measure: $measure")
                    end
                    
                    # Compute parameter gradient for SE using refgrid derivative evaluator
                    if scale === :response
                        FormulaCompiler.me_mu_grad_beta!(engine.gβ_accumulator, refgrid_de, local_β, profile_idx, var;
                                                       link=local_link)
                    else
                        FormulaCompiler.me_eta_grad_beta!(engine.gβ_accumulator, refgrid_de, local_β, profile_idx, var)
                    end
                    
                    se = compute_se_only(engine.gβ_accumulator, engine.Σ)
                    
                    # Store results
                    push!(results, (term=string(var), estimate=estimate, se=se))
                    G[row_idx, :] = engine.gβ_accumulator
                    
                    row_idx += 1
                end
                
            else
                # Categorical variable: use existing contrast functions
                # Extract profile as Dict for compatibility 
                profile_dict = Dict(Symbol(k) => reference_grid[profile_idx, k] for k in names(reference_grid))
                
                # Use existing functions - same as the old per-profile system
                marginal_effect = _compute_row_specific_baseline_contrast(engine, refgrid_de, profile_dict, var, scale, backend)
                _row_specific_contrast_grad_beta!(engine.gβ_accumulator, engine, refgrid_de, profile_dict, var, scale)
                
                # Apply measure transformations if needed 
                final_effect = marginal_effect
                
                # Compute standard error
                se = compute_se_only(engine.gβ_accumulator, engine.Σ)
                
                # Build descriptive term name showing the specific contrast
                current_level = profile_dict[var]
                baseline_level = _get_baseline_level(engine.model, var)
                profile_parts = [string(k, "=", v) for (k, v) in pairs(profile_dict) if k != var]
                profile_desc = join(profile_parts, ", ")
                term_name = "$(var)=$(current_level) vs $(baseline_level) at $(profile_desc)"
                
                # Store results
                push!(results, (term=term_name, estimate=final_effect, se=se))
                G[row_idx, :] = engine.gβ_accumulator
                
                row_idx += 1
            end
        end
    end
    
    # Trim gradient matrix to actual size
    actual_rows = nrow(results)
    G = G[1:actual_rows, :]
    
    return (results, G)
end
