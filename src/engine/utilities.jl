# engine/utilities.jl - FormulaCompiler-based marginal effects computation

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
Implements REORG.md lines 290-348 with graceful fallbacks and batch operations.

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
function _ame_continuous_and_categorical(engine::MarginsEngine, data_nt::NamedTuple; target=:mu, backend=:ad, measure=:effect)
    engine.de === nothing && return (DataFrame(), Matrix{Float64}(undef, 0, length(engine.β)))
    
    rows = 1:length(first(data_nt))
    results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
    G = Matrix{Float64}(undef, length(engine.de.vars), length(engine.β))
    
    # Auto-detect variable types and process accordingly
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, data_nt)
    
    # Process continuous variables with FC's built-in AME gradient accumulation (ZERO ALLOCATION!)
    cont_idx = 1
    for var in engine.de.vars
        if var ∈ continuous_vars
            # Graceful backend fallback (MARGINS_GUIDE.md pattern)
            try
                # This is the key: FC already provides zero-allocation AME gradients
                FormulaCompiler.accumulate_ame_gradient!(
                    engine.gβ_accumulator, engine.de, engine.β, rows, var;
                    link=(target === :mu ? engine.link : GLM.IdentityLink()), 
                    backend=backend
                )
            catch e
                if backend === :ad
                    @warn "AD backend failed for $var, falling back to FD: $e"
                    FormulaCompiler.accumulate_ame_gradient!(
                        engine.gβ_accumulator, engine.de, engine.β, rows, var;
                        link=(target === :mu ? engine.link : GLM.IdentityLink()), 
                        backend=:fd
                    )
                else
                    rethrow(e)
                end
            end
            
            # Average the gradient and compute SE
            gβ_avg = engine.gβ_accumulator ./ length(rows)
            se = FormulaCompiler.delta_method_se(gβ_avg, engine.Σ)
            
            # Compute AME value (also zero allocation with FC)
            ame_val = 0.0
            for row in rows
                if target === :mu
                    FormulaCompiler.marginal_effects_mu!(engine.g_buf, engine.de, engine.β, row; 
                                                        link=engine.link, backend=backend)
                else
                    FormulaCompiler.marginal_effects_eta!(engine.g_buf, engine.de, engine.β, row; 
                                                         backend=backend)
                end
                ame_val += engine.g_buf[cont_idx]
            end
            ame_val /= length(rows)
            
            # Apply elasticity transformations if requested (Phase 3)
            final_val = ame_val
            if measure !== :effect && engine.de !== nothing
                # Compute average x and y for elasticity measures
                x_acc = 0.0
                y_acc = 0.0
                xcol = getproperty(data_nt, var)
                
                # Use FormulaCompiler's proper pattern: compiled_base + dot(β, X_row)
                for row in rows
                    x_acc += float(xcol[row])
                    
                    # Get design matrix row and compute linear predictor (following FC pattern)
                    engine.compiled(engine.de.xrow_buffer, data_nt, row)  # Fill design matrix row  
                    η = dot(engine.β, engine.de.xrow_buffer)              # Compute η = Xβ
                    
                    if target === :mu
                        y_acc += GLM.linkinv(engine.link, η)     # Transform to μ scale
                    else
                        y_acc += η                               # Use η scale directly
                    end
                end
                x̄ = x_acc / length(rows)
                ȳ = y_acc / length(rows)
                
                # Apply transformation based on measure type
                if measure === :elasticity
                    final_val = (x̄ / ȳ) * ame_val
                elseif measure === :semielasticity_x
                    final_val = x̄ * ame_val
                elseif measure === :semielasticity_y
                    final_val = (1 / ȳ) * ame_val
                end
            end
            
            push!(results, (term=string(var), estimate=final_val, se=se))
            G[cont_idx, :] = gβ_avg
            cont_idx += 1
            
        else  # Categorical variable
            # Compute traditional baseline contrasts for population margins
            ame_val, gβ_avg = _compute_categorical_baseline_ame(engine, var, rows, target, backend)
            se = FormulaCompiler.delta_method_se(gβ_avg, engine.Σ)
            
            push!(results, (term="$(var) (baseline contrast)", estimate=ame_val, se=se))
            G[cont_idx, :] = gβ_avg  # Still increment for output indexing
            cont_idx += 1
        end
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
function _mem_continuous_and_categorical(engine::MarginsEngine, profiles::Vector; target=:mu, backend=:ad, measure=:effect)
    # Handle the case where we have only categorical variables (engine.de === nothing)
    # or mixed continuous/categorical variables
    
    results = DataFrame(term=String[], estimate=Float64[], se=Float64[])
    n_profiles = length(profiles)
    
    # Auto-detect variable types from the compiled formula and requested variables
    continuous_vars = FormulaCompiler.continuous_variables(engine.compiled, engine.data_nt)
    
    # Determine which variables we're actually processing
    requested_vars = engine.vars  # Variables requested by user
    continuous_requested = [v for v in requested_vars if v ∈ continuous_vars]
    categorical_requested = [v for v in requested_vars if v ∉ continuous_vars]
    
    # Calculate total number of terms (for gradient matrix sizing)
    total_terms = n_profiles * length(requested_vars)
    G = Matrix{Float64}(undef, total_terms, length(engine.β))
    
    row_idx = 1
    for profile in profiles
        # Build minimal synthetic reference grid data (FormulaCompiler guide recommendation)
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
                # Find the index of this variable in the continuous variables for the gradient buffer
                continuous_var_idx = findfirst(==(var), continuous_requested)
                effect_val = engine.g_buf[continuous_var_idx]
            else
                # Categorical variable: compute row-specific baseline contrast
                # Novel approach: contrast this row's category vs baseline at this profile
                effect_val = _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, target, backend)
            end
            
            # Compute gradient for SE at reference point using FormulaCompiler
            if var ∈ continuous_vars
                if target === :mu
                    FormulaCompiler.me_mu_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var;
                                                    link=engine.link)
                else
                    FormulaCompiler.me_eta_grad_beta!(engine.gβ_accumulator, refgrid_de, engine.β, 1, var)
                end
            else
                # Compute gradient for row-specific categorical contrast
                _row_specific_contrast_grad_beta!(engine.gβ_accumulator, engine, refgrid_de, profile, var, target)
            end
            se = FormulaCompiler.delta_method_se(engine.gβ_accumulator, engine.Σ)
            
            # Apply elasticity transformations for continuous variables if requested (Phase 3)
            final_val = effect_val
            if var ∈ continuous_vars && measure !== :effect && engine.de !== nothing
                # Get x and y values at this specific profile
                x_val = float(profile[var])
                
                # Use FormulaCompiler's proper pattern for prediction
                refgrid_de.compiled_base(refgrid_de.xrow_buffer, refgrid_data, 1)  # Fill design matrix row
                η = dot(engine.β, refgrid_de.xrow_buffer)                          # Compute η = Xβ
                
                if target === :mu
                    y_val = GLM.linkinv(engine.link, η)                # Transform to μ scale
                else
                    y_val = η                                          # Use η scale directly
                end
                
                # Apply transformation based on measure type
                if measure === :elasticity
                    final_val = (x_val / y_val) * effect_val
                elseif measure === :semielasticity_x
                    final_val = x_val * effect_val
                elseif measure === :semielasticity_y
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
    _build_metadata(; kwargs...) -> Dict

Build metadata dictionary for MarginsResult.

# Keyword Arguments
- `type::Symbol=:unknown`: Analysis type (:effects, :predictions)
- `vars::Vector{Symbol}=Symbol[]`: Variables analyzed
- `target::Symbol=:mu`: Target scale (:eta, :mu)  
- `backend::Symbol=:ad`: Computation backend (:ad, :fd)
- `n_obs::Int=0`: Number of observations
- `model_type=nothing`: Type of fitted model
- `timestamp=now()`: Analysis timestamp
- Additional kwargs stored in :additional sub-dict

# Returns
- `Dict{Symbol, Any}`: Metadata dictionary

# Examples
```julia
metadata = _build_metadata(
    type=:effects, vars=[:x1, :x2], target=:mu, 
    backend=:fd, n_obs=1000, model_type=LinearModel
)
```
"""
function _build_metadata(; 
    type=:unknown, 
    vars=Symbol[], 
    target=:mu, 
    backend=:ad,
    n_obs=0,
    model_type=nothing,
    timestamp=nothing,
    kwargs...)
    
    # Set default timestamp
    ts = timestamp === nothing ? string(now()) : timestamp
    
    return Dict{Symbol, Any}(
        :type => type,
        :vars => vars,
        :n_vars => length(vars),
        :target => target,
        :backend => backend, 
        :n_obs => n_obs,
        :model_type => model_type === nothing ? "unknown" : string(typeof(model_type)),
        :timestamp => ts,
        :additional => Dict{Symbol, Any}(kwargs...)
    )
end

# Import dependencies for utility functions
using Dates: now
using StatsBase: mode

"""
    _compute_categorical_baseline_ame(engine, var, rows, target, backend) -> (Float64, Vector{Float64})

Compute traditional baseline contrasts for categorical variables in population margins.
Helper function for _ame_continuous_and_categorical.
"""
function _compute_categorical_baseline_ame(engine::MarginsEngine, var::Symbol, rows, target::Symbol, backend::Symbol)
    # Placeholder implementation - categorical AME computation
    # Would compute average discrete change from baseline level
    ame_val = 0.0  # Placeholder
    gβ_avg = zeros(length(engine.β))  # Placeholder gradient
    return (ame_val, gβ_avg)
end

"""
    _compute_row_specific_baseline_contrast(engine, refgrid_de, profile, var, target, backend) -> Float64

Compute row-specific baseline contrast using the new profile/contrasts.jl implementation.
Helper function for _mem_continuous_and_categorical.
"""
function _compute_row_specific_baseline_contrast(engine::MarginsEngine, refgrid_de, profile::Dict, var::Symbol, target::Symbol, backend::Symbol)
    # Use the new profile contrasts implementation
    effect, _ = compute_profile_categorical_contrast(engine, profile, var, target; backend)
    return effect
end

"""
    _row_specific_contrast_grad_beta!(gβ_buffer, engine, refgrid_de, profile, var, target)

Compute gradient for row-specific categorical contrast using the new profile/contrasts.jl implementation.
Helper function for _mem_continuous_and_categorical.
"""
function _row_specific_contrast_grad_beta!(gβ_buffer::Vector{Float64}, engine::MarginsEngine, refgrid_de, profile::Dict, var::Symbol, target::Symbol)
    # Use the new profile contrasts implementation  
    _, gradient = compute_profile_categorical_contrast(engine, profile, var, target; backend=:ad)
    copyto!(gβ_buffer, gradient)
end

"""
    _predict_with_formulacompiler(engine, profile, target) -> Float64

Make predictions using FormulaCompiler for categorical contrast computation.
Helper function that properly uses FormulaCompiler instead of manual computation.
"""
function _predict_with_formulacompiler(engine::MarginsEngine, profile::Dict, target::Symbol)
    # Create minimal reference data for this profile  
    profile_data = _build_refgrid_data(profile, engine.data_nt)
    profile_compiled = FormulaCompiler.compile_formula(engine.model, profile_data)
    
    # Use FormulaCompiler's modelrow to get design matrix row, then apply coefficients
    X_row = FormulaCompiler.modelrow(profile_compiled, profile_data, 1)
    η = dot(X_row, engine.β)
    
    if target === :eta
        return η
    else # :mu  
        return GLM.linkinv(engine.link, η)
    end
end