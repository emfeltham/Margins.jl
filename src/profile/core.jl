# profile/core.jl - Main profile_margins() function with reference grid approach

using Distributions: Normal, cdf

"""
    profile_margins(model, data; at=:means, type=:effects, vars=nothing, target=:mu, backend=:ad, kwargs...)

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at representative points or user-specified scenarios,
providing marginal effects at the mean (MEM) or marginal effects at representative values (MER).

# Arguments
- `model`: Fitted statistical model (GLM, etc.)
- `data`: Data table (DataFrame, NamedTuple, etc.)
- `at`: Profile specification (`:means`, Dict, Vector{Dict}, or DataFrame)
- `type::Symbol=:effects`: Type of analysis (`:effects` for MEM/MER, `:predictions` for APM/APR)
- `vars=nothing`: Variables for analysis (only needed for `:effects`; defaults to all continuous)
- `target::Symbol=:mu`: Target scale (`:mu` for response scale, `:eta` for linear predictor)
- `backend::Symbol=:ad`: Computational backend (`:ad` or `:fd`)

# Returns
- `MarginsResult`: Container with results DataFrame, gradients matrix, and metadata

# Examples
```julia
# Effects at sample means (MEM)
result = profile_margins(model, data; at=:means, type=:effects, vars=[:x1, :x2])

# Effects at specific profiles (MER)
result = profile_margins(model, data; at=Dict(:x1 => [0, 1], :x2 => [mean]), type=:effects)

# Predictions at the mean (APM)
result = profile_margins(model, data; at=:means, type=:predictions)

# Multiple explicit profiles
profiles = [Dict(:x1 => 0.0, :x2 => 1.0), Dict(:x1 => 1.0, :x2 => 0.0)]
result = profile_margins(model, data; at=profiles, type=:effects)
```
"""
function profile_margins(model, data; at=:means, type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:ad, kwargs...)
    # Single data conversion (consistent format throughout)
    data_nt = Tables.columntable(data)
    
    # Handle vars parameter (only needed for type=:effects)
    if type === :effects
        vars = vars === nothing ? :all_continuous : vars  # Default to all continuous variables
        if vars === :all_continuous
            vars = _get_continuous_variables(data_nt)
        elseif isa(vars, Symbol)
            vars = [vars]  # Convert single var to vector
        end
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Build zero-allocation engine with caching
    engine = _get_or_build_engine_for_profiles(model, data_nt, vars)
    
    # Build reference grid from at specification
    reference_grid = _build_reference_grid(at, data_nt)
    
    if type === :effects
        # Convert reference grid to profiles for processing
        profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
        df, G = _mem_continuous_and_categorical(engine, profiles; target, backend, kwargs...)  # → MEM/MER
        metadata = _build_metadata(; type, vars, target, backend, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at, kwargs...)
        return MarginsResult(df, G, metadata)
    else # :predictions  
        df, G = _profile_predictions(engine, reference_grid; target, kwargs...)  # → APM/APR
        metadata = _build_metadata(; type, vars=Symbol[], target, backend, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at, kwargs...)
        return MarginsResult(df, G, metadata)
    end
end

"""
    profile_margins(model, reference_grid::DataFrame; type=:effects, vars=nothing, target=:mu, backend=:ad, kwargs...)

Profile margins using an explicit reference grid DataFrame.

This is the most efficient method when you have already constructed your desired reference grid.
"""
function profile_margins(model, reference_grid::DataFrame; type::Symbol=:effects, vars=nothing, target::Symbol=:mu, backend::Symbol=:ad, kwargs...)
    # Convert reference grid to data format
    data_nt = Tables.columntable(reference_grid)
    
    # Handle vars parameter (only needed for type=:effects)
    if type === :effects
        vars = vars === nothing ? :all_continuous : vars  # Default to all continuous variables
        if vars === :all_continuous
            vars = _get_continuous_variables(data_nt)
        elseif isa(vars, Symbol)
            vars = [vars]  # Convert single var to vector
        end
    else # type === :predictions
        vars = nothing  # Not needed for predictions
    end
    
    # Build zero-allocation engine
    engine = build_engine(model, data_nt, vars === nothing ? Symbol[] : vars)
    
    if type === :effects
        # Convert reference grid to profiles for processing
        profiles = [Dict(pairs(row)) for row in eachrow(reference_grid)]
        df, G = _mem_continuous_and_categorical(engine, profiles; target, backend, kwargs...)  # → MEM/MER
        metadata = _build_metadata(; type, vars, target, backend, n_obs=nrow(reference_grid), 
                                  model_type=typeof(model), at_spec="explicit_grid", kwargs...)
        return MarginsResult(df, G, metadata)
    else # :predictions  
        df, G = _profile_predictions(engine, reference_grid; target, kwargs...)  # → APM/APR
        metadata = _build_metadata(; type, vars=Symbol[], target, backend, n_obs=nrow(reference_grid), 
                                  model_type=typeof(model), at_spec="explicit_grid", kwargs...)
        return MarginsResult(df, G, metadata)
    end
end

"""
    _get_or_build_engine_for_profiles(model, data_nt, vars)

Get or build engine with caching for profile computations.
Uses same cache as population margins but ensures compatibility.
"""
function _get_or_build_engine_for_profiles(model, data_nt::NamedTuple, vars)
    # Use same caching mechanism as population margins
    cache_key = hash((model, keys(data_nt), vars, :profiles))  # Add :profiles to differentiate
    if haskey(COMPILED_CACHE, cache_key)
        return COMPILED_CACHE[cache_key]
    else
        engine = build_engine(model, data_nt, vars === nothing ? Symbol[] : vars)
        COMPILED_CACHE[cache_key] = engine
        return engine
    end
end

"""
    _profile_predictions(engine, reference_grid; target, kwargs...) -> (DataFrame, Matrix{Float64})

Compute adjusted predictions at profiles (APM/APR) with delta-method standard errors.

This function evaluates predictions at each row of the reference grid, providing
adjusted predictions at the mean (APM) or at representative values (APR).
"""
function _profile_predictions(engine::MarginsEngine, reference_grid::DataFrame; target=:mu, kwargs...)
    n_profiles = nrow(reference_grid)
    n_params = length(engine.β)
    
    # Preallocate arrays
    predictions = Vector{Float64}(undef, n_profiles)
    G = zeros(n_profiles, n_params)  # One row per profile
    
    # Convert reference grid to NamedTuple format for FormulaCompiler
    data_nt = Tables.columntable(reference_grid)
    
    # Compute predictions and gradients for each profile
    for i in 1:n_profiles
        # Use FormulaCompiler modelrow to get design matrix row for profile i
        X_row = FormulaCompiler.modelrow(engine.compiled, data_nt, i)
        eta_i = dot(X_row, engine.β)
        
        if target === :mu
            # Transform to response scale
            predictions[i] = GLM.linkinv(engine.link, eta_i)
            
            # Compute gradient with chain rule: d/dβ[linkinv(Xβ)] = linkinv'(Xβ) * X
            link_deriv = GLM.mueta(engine.link, eta_i)
            G[i, :] = link_deriv .* X_row
        else # target === :eta
            # Keep on link scale
            predictions[i] = eta_i
            
            # Gradient is just the design matrix row
            G[i, :] = X_row
        end
    end
    
    # Compute delta-method SEs for each profile
    se_vals = Vector{Float64}(undef, n_profiles)
    for i in 1:n_profiles
        se_vals[i] = sqrt((G[i:i, :] * engine.Σ * G[i:i, :]')[1, 1])
    end
    
    # Create results DataFrame with profile information
    results = DataFrame()
    results.term = ["APM/APR" for _ in 1:n_profiles]
    results.estimate = predictions
    results.se = se_vals
    results.t_stat = predictions ./ se_vals
    results.p_value = 2 .* (1 .- cdf.(Normal(), abs.(predictions ./ se_vals)))
    
    # Add profile columns to show which profile each prediction corresponds to
    for (col_name, col_data) in pairs(Tables.columns(reference_grid))
        results[!, Symbol("at_$(col_name)")] = col_data
    end
    
    return results, G
end