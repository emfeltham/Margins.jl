# compute_categorical.jl

"""
    _categorical_effects(model, data_nt, engine; target=:mu, contrasts=:pairwise, rows=:all, at=:none)

Compute discrete-change effects for categorical variables in `engine.vars`.
When `at=:none`, computes average discrete change over `rows` (AME-style for categoricals).
When `at` specifies profiles, computes per-profile contrasts (no averaging).
Supports `:pairwise` (all pairs) and `:baseline` (vs first level).
Returns (df, G) where G is a Matrix{Float64} with one row per result row.
"""
function _categorical_effects(model, data_nt, engine; target::Symbol=:mu, contrasts::Symbol=:pairwise, rows=:all, at=:none)
    (; compiled, vars, β, Σ, link) = engine
    idxs = rows === :all ? (1:_nrows(data_nt)) : rows
    
    # Count the number of contrasts to pre-allocate
    n_contrasts = 0
    n_profiles = at === :none ? 1 : length(_build_profiles(at, data_nt))
    
    for var in vars
        col = getproperty(data_nt, var)
        levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
            string.(CategoricalArrays.levels(col))
        elseif eltype(col) <: Bool
            string.([false, true])
        else
            unique!(string.(collect(col)))
        end
        pairs = if contrasts === :baseline
            [(levels[1], lev) for lev in levels if lev != levels[1]]
        else
            [(levels[i], levels[j]) for i in 1:length(levels) for j in (i+1):length(levels)]
        end
        n_contrasts += length(pairs)
    end
    
    n_result_rows = n_contrasts * n_profiles
    
    # Skip if no categorical variables
    if n_result_rows == 0
        empty_df = DataFrame(term=String[], level_from=String[], level_to=String[], estimate=Float64[], se=Float64[])
        empty_G = Matrix{Float64}(undef, 0, length(β))
        return (empty_df, empty_G)
    end
    
    out = DataFrame(term=String[], level_from=String[], level_to=String[], estimate=Float64[], se=Float64[])
    G = Matrix{Float64}(undef, n_result_rows, length(β))
    
    # Buffers
    X_to = Vector{Float64}(undef, length(compiled))
    X_from = Vector{Float64}(undef, length(compiled))
    Δ = Vector{Float64}(undef, length(compiled))
    
    row_idx = 1
    for var in vars
        col = getproperty(data_nt, var)
        levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
            string.(CategoricalArrays.levels(col))
        elseif eltype(col) <: Bool
            string.([false, true])
        else
            unique!(string.(collect(col)))
        end
        pairs = if contrasts === :baseline
            [(levels[1], lev) for lev in levels if lev != levels[1]]
        else
            [(levels[i], levels[j]) for i in 1:length(levels) for j in (i+1):length(levels)]
        end
        for (from, to) in pairs
            if at === :none
                # Average across rows
                acc_val = 0.0
                acc_gβ = zeros(Float64, length(compiled))
                for row in idxs
                    # η: use ΔX; μ: need X_to/X_from and link derivatives
                    FormulaCompiler.contrast_modelrow!(Δ, compiled, data_nt, row; var=var, from=from, to=to)
                    if target === :eta
                        val_row = dot(β, Δ)
                        gβ = Δ
                    else
                        # Build X_to and X_from by evaluating compiled with overrides
                        # to
                        data_to, ov_to = FormulaCompiler.build_row_override_data(data_nt, [var], row)
                        
                        # Convert string levels to CategoricalValues if needed
                        base_col = getproperty(data_nt, var)
                        if (Base.find_package("CategoricalArrays") !== nothing) && (base_col isa CategoricalArrays.CategoricalArray)
                            levels_list = CategoricalArrays.levels(base_col)
                            temp_from = CategoricalArrays.categorical([from], levels=levels_list, ordered=CategoricalArrays.isordered(base_col))
                            temp_to = CategoricalArrays.categorical([to], levels=levels_list, ordered=CategoricalArrays.isordered(base_col))
                            ov_to[1].replacement = temp_from[1]
                            compiled(X_from, data_to, row)
                            ov_to[1].replacement = temp_to[1]
                            compiled(X_to, data_to, row)
                        else
                            ov_to[1].replacement = from
                            compiled(X_from, data_to, row)
                            ov_to[1].replacement = to
                            compiled(X_to, data_to, row)
                        end
                        η_from = dot(β, X_from)
                        η_to = dot(β, X_to)
                        μ_from = GLM.linkinv(link, η_from)
                        μ_to = GLM.linkinv(link, η_to)
                        val_row = μ_to - μ_from
                        gβ = _dmu_deta_local(link, η_to) .* X_to .- _dmu_deta_local(link, η_from) .* X_from
                    end
                    acc_val += val_row
                    acc_gβ .+= gβ
                end
                val = acc_val / length(idxs)
                gβ_mean = acc_gβ ./ length(idxs)
                se = FormulaCompiler.delta_method_se(gβ_mean, Σ)
                
                # Store averaged gradient
                G[row_idx, :] = gβ_mean
                push!(out, (term=string(var), level_from=from, level_to=to, estimate=val, se=se))
                row_idx += 1
            else
                # Profiles: compute per profile using scenario-based approach
                profiles = _build_profiles(at, data_nt)
                
                # Pre-allocate profile columns
                all_profile_keys = Set{Symbol}()
                for prof in profiles
                    for k in keys(prof)
                        push!(all_profile_keys, Symbol("at_", k))
                    end
                end
                for col_name in all_profile_keys
                    out[!, col_name] = Union{Missing,Any}[]
                end
                
                for prof in profiles
                    # Create two scenarios for the contrast: one with 'from' level, one with 'to' level
                    prof_from = copy(prof)
                    prof_to = copy(prof)
                    prof_from[var] = from
                    prof_to[var] = to
                    
                    processed_prof_from = _process_profile_for_scenario(prof_from, data_nt)
                    processed_prof_to = _process_profile_for_scenario(prof_to, data_nt)
                    
                    scen_from = FormulaCompiler.create_scenario("from", data_nt, processed_prof_from)
                    scen_to = FormulaCompiler.create_scenario("to", data_nt, processed_prof_to)
                    
                    # Evaluate at both scenarios and compute contrast
                    compiled(X_from, scen_from.data, 1)
                    compiled(X_to, scen_to.data, 1)
                    Δ .= X_to .- X_from
                    if target === :eta
                        val = dot(β, Δ)
                        gβ = copy(Δ)
                    else
                        # For μ target: compute response-scale contrast using link inverse
                        η_from = dot(β, X_from)
                        η_to = dot(β, X_to)
                        μ_from = GLM.linkinv(link, η_from)
                        μ_to = GLM.linkinv(link, η_to)
                        val = μ_to - μ_from
                        gβ = _dmu_deta_local(link, η_to) .* X_to .- _dmu_deta_local(link, η_from) .* X_from
                    end
                    se = FormulaCompiler.delta_method_se(gβ, Σ)
                    
                    # Store gradient in matrix
                    G[row_idx, :] = gβ
                    
                    # Create row data
                    row_data = Dict{Symbol,Any}(
                        :term => string(var), 
                        :level_from => from, 
                        :level_to => to, 
                        :estimate => val, 
                        :se => se
                    )
                    
                    # Add profile information
                    for (k, v) in prof
                        row_data[Symbol("at_", k)] = v
                    end
                    
                    # Fill missing profile columns with missing
                    for col_name in all_profile_keys
                        if !haskey(row_data, col_name)
                            row_data[col_name] = missing
                        end
                    end
                    
                    push!(out, row_data)
                    row_idx += 1
                end
            end
        end
    end
    return (out, G)
end

"""
    _categorical_effects_from_profiles(model, data_nt, engine, profiles; target=:mu, contrasts=:pairwise)

Compute discrete-change effects for categorical variables using pre-built profiles.
This is used by table-based profile dispatch where profiles are built from DataFrame rows.
Returns (df, G) where G is a Matrix{Float64} with one row per result row.
"""
function _categorical_effects_from_profiles(model, data_nt, engine, profiles; target::Symbol=:mu, contrasts::Symbol=:pairwise)
    (; compiled, vars, β, Σ, link) = engine
    
    # Count the number of contrasts to pre-allocate
    n_contrasts = 0
    for var in vars
        col = getproperty(data_nt, var)
        levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
            string.(CategoricalArrays.levels(col))
        elseif eltype(col) <: Bool
            string.([false, true])
        else
            unique!(string.(collect(col)))
        end
        
        pairs = if contrasts === :baseline
            [(levels[1], lev) for lev in levels if lev != levels[1]]
        else
            [(levels[i], levels[j]) for i in 1:length(levels) for j in (i+1):length(levels)]
        end
        n_contrasts += length(pairs)
    end
    
    n_result_rows = n_contrasts * length(profiles)
    
    # Skip if no categorical variables
    if n_result_rows == 0
        empty_df = DataFrame(term=String[], level_from=String[], level_to=String[], estimate=Float64[], se=Float64[])
        empty_G = Matrix{Float64}(undef, 0, length(β))
        return (empty_df, empty_G)
    end
    
    out = DataFrame(term=String[], level_from=String[], level_to=String[], estimate=Float64[], se=Float64[])
    G = Matrix{Float64}(undef, n_result_rows, length(β))
    
    # Pre-allocate all profile columns to avoid column count mismatch
    all_profile_keys = Set{Symbol}()
    for prof in profiles
        for k in keys(prof)
            push!(all_profile_keys, Symbol("at_", k))
        end
    end
    for col_name in all_profile_keys
        out[!, col_name] = Union{Missing,Any}[]
    end
    
    # Buffers
    X_to = Vector{Float64}(undef, length(compiled))
    X_from = Vector{Float64}(undef, length(compiled))
    Δ = Vector{Float64}(undef, length(compiled))
    
    row_idx = 1
    for var in vars
        col = getproperty(data_nt, var)
        levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
            string.(CategoricalArrays.levels(col))
        elseif eltype(col) <: Bool
            string.([false, true])
        else
            unique!(string.(collect(col)))
        end
        
        pairs = if contrasts === :baseline
            [(levels[1], lev) for lev in levels if lev != levels[1]]
        else
            [(levels[i], levels[j]) for i in 1:length(levels) for j in (i+1):length(levels)]
        end
        
        for (from, to) in pairs
            for (prof_idx, prof) in enumerate(profiles)
                # Create two scenarios for the contrast: one with 'from' level, one with 'to' level
                prof_from = copy(prof)
                prof_to = copy(prof)
                prof_from[var] = from
                prof_to[var] = to
                
                processed_prof_from = _process_profile_for_scenario(prof_from, data_nt)
                processed_prof_to = _process_profile_for_scenario(prof_to, data_nt)
                
                scen_from = FormulaCompiler.create_scenario("from", data_nt, processed_prof_from)
                scen_to = FormulaCompiler.create_scenario("to", data_nt, processed_prof_to)
                
                # Evaluate at both scenarios and compute contrast
                compiled(X_from, scen_from.data, 1)
                compiled(X_to, scen_to.data, 1)
                Δ .= X_to .- X_from
                
                if target === :eta
                    val = dot(β, Δ)
                    gβ = copy(Δ)
                else
                    # For μ target: compute response-scale contrast using link inverse
                    η_from = dot(β, X_from)
                    η_to = dot(β, X_to)
                    μ_from = GLM.linkinv(link, η_from)
                    μ_to = GLM.linkinv(link, η_to)
                    val = μ_to - μ_from
                    gβ = _dmu_deta_local(link, η_to) .* X_to .- _dmu_deta_local(link, η_from) .* X_from
                end
                
                se = FormulaCompiler.delta_method_se(gβ, Σ)
                
                # Store gradient in matrix
                G[row_idx, :] = gβ
                
                # Create row data
                row_data = Dict{Symbol,Any}()
                row_data[:term] = string(var)
                row_data[:level_from] = from  
                row_data[:level_to] = to
                row_data[:estimate] = val
                row_data[:se] = se
                
                # Add profile columns
                for (k, v) in prof
                    row_data[Symbol("at_", k)] = v
                end
                
                # Fill missing profile columns with missing
                for col_name in all_profile_keys
                    if !haskey(row_data, col_name)
                        row_data[col_name] = missing
                    end
                end
                
                push!(out, row_data)
                row_idx += 1
            end
        end
    end
    
    return (out, G)
end

"""
    _categorical_effects_from_profiles_streaming(model, data_nt, engine, profiles; target=:mu, contrasts=:pairwise)

Phase 2: Streaming version that accepts Vector{<:Dict} profiles (already collected from streaming source).
This is identical to _categorical_effects_from_profiles but with explicit "streaming" naming for Phase 2.
"""
function _categorical_effects_from_profiles_streaming(model, data_nt, engine, profiles; target::Symbol=:mu, contrasts::Symbol=:pairwise)
    # Delegate to existing implementation - already handles Vector{<:Dict} correctly
    return _categorical_effects_from_profiles(model, data_nt, engine, profiles; target=target, contrasts=contrasts)
end
