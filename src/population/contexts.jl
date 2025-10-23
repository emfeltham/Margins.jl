# population/contexts.jl - Clean FormulaCompiler-only context effects implementation

using FormulaCompiler:
    _dmu_deta, _d2mu_deta2, modelrow!,
    derivativeevaluator,
    contrastevaluator
using CategoricalArrays: CategoricalValue, levels, pool
using Statistics: quantile
using LinearAlgebra: dot
import GLM

"""
    _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups, weights, type, scale, backend, ci_alpha, measure)

Handle population margins with scenarios/groups contexts - clean FormulaCompiler implementation.
"""
function _population_margins_with_contexts(engine, data_nt, vars, scenarios, groups, weights, type, scale, backend, ci_alpha, measure, contrasts=:baseline)
    results = DataFrame()
    gradients_list = Matrix{Float64}[]

    # Parse specifications
    scenario_specs = isnothing(scenarios) ? [Dict()] : _parse_at_specification(scenarios)
    group_specs = isnothing(groups) ? [Dict()] : _parse_groups_specification(groups, data_nt)

    # Create all combinations of contexts
    total_combinations = length(scenario_specs) * length(group_specs)

    # Enforce hard limits
    if total_combinations > 1000
        throw(MarginsError("Combination explosion detected ($total_combinations combinations). Maximum recommended: 1000."))
    elseif total_combinations > 250
        throw(MarginsError("Large combination count detected ($total_combinations combinations). Maximum recommended: 250."))
    end

    for scenario_spec in scenario_specs, group_spec in group_specs
        context_data, context_indices, context_weights = _create_context_data_with_weights(data_nt, scenario_spec, group_spec, weights)

        if type === :effects
            # PERFORMANCE OPTIMIZATION Phase 1: Build evaluators ONCE per context
            # This prevents rebuilding evaluators n_vars times (was causing 300× overhead)
            continuous_vars = continuous_variables(engine.compiled, context_data)
            context_de = derivativeevaluator(backend, engine.compiled, context_data, continuous_vars)

            # Separate continuous and categorical variables for batch vs per-variable processing
            continuous_vars_to_compute = Symbol[]
            categorical_vars_to_compute = Symbol[]
            for var in vars
                # Skip if this var appears in scenarios/groups
                if haskey(scenario_spec, var) || haskey(group_spec, var)
                    continue
                end

                if _is_continuous_variable(context_data[var])
                    push!(continuous_vars_to_compute, var)
                else
                    push!(categorical_vars_to_compute, var)
                end
            end

            # PERFORMANCE OPTIMIZATION Phase 2: BATCH process ALL continuous variables in one pass
            # This prevents redundant derivative computation (was n_vars × N calls, now just N calls)
            if !isempty(continuous_vars_to_compute)
                cont_results, cont_gradients = _compute_all_continuous_context_effects_batch(
                    engine, context_data, context_indices, continuous_vars_to_compute,
                    scale, context_weights, context_de
                )

                # Add context identifiers to continuous results
                for (ctx_var, ctx_val) in merge(scenario_spec, group_spec)
                    display_val = if ctx_val isa NamedTuple && haskey(ctx_val, :label)
                        ctx_val.label
                    else
                        ctx_val
                    end
                    cont_results[!, Symbol("at_$(ctx_var)")] = fill(display_val, nrow(cont_results))
                end

                results = _append_results_with_missing_columns(results, cont_results)
                push!(gradients_list, cont_gradients)
            end

            # Categorical evaluator: build once for all categorical variables
            if !isempty(categorical_vars_to_compute)
                context_ce = contrastevaluator(engine.compiled, context_data, categorical_vars_to_compute)

                # Process categorical variables one-at-a-time (different computation pattern, correctly done per-variable)
                for var in categorical_vars_to_compute
                    cat_result, cat_gradients = _compute_categorical_context_effect(
                        engine, context_data, context_indices, var, scale, weights, contrasts, context_ce
                    )

                    # Add context identifiers
                    for (ctx_var, ctx_val) in merge(scenario_spec, group_spec)
                        display_val = if ctx_val isa NamedTuple && haskey(ctx_val, :label)
                            ctx_val.label
                        else
                            ctx_val
                        end
                        cat_result[!, Symbol("at_$(ctx_var)")] = fill(display_val, nrow(cat_result))
                    end

                    results = _append_results_with_missing_columns(results, cat_result)
                    push!(gradients_list, cat_gradients)
                end
            end
        else # :predictions
            # Compute prediction in this context using FormulaCompiler primitives
            pred_result, pred_gradients = _compute_population_prediction_in_context(
                engine, context_data, context_indices, scale, context_weights
            )

            # Add context identifiers
            for (ctx_var, ctx_val) in merge(scenario_spec, group_spec)
                display_val = if ctx_val isa NamedTuple && haskey(ctx_val, :label)
                    ctx_val.label
                else
                    ctx_val
                end
                pred_result[!, Symbol("at_$(ctx_var)")] = fill(display_val, nrow(pred_result))
            end

            results = _append_results_with_missing_columns(results, pred_result)
            push!(gradients_list, pred_gradients)
        end
    end

    # Combine gradients
    G = isempty(gradients_list) ? Matrix{Float64}(undef, 0, length(engine.β)) : vcat(gradients_list...)

    # Build metadata
    metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=length(first(data_nt)),
                              model_type=typeof(engine.model), has_contexts=true)

    # Store confidence interval parameters in metadata
    metadata[:alpha] = ci_alpha

    # Store scenarios and groups variable information for Context column
    scenarios_vars = isnothing(scenarios) ? Symbol[] : (scenarios isa NamedTuple ? collect(keys(scenarios)) : Symbol[])
    groups_vars = isnothing(groups) ? Symbol[] : _extract_group_variables(groups)
    metadata[:scenarios_vars] = scenarios_vars
    metadata[:groups_vars] = groups_vars

    # Add analysis_type for format auto-detection
    metadata[:analysis_type] = :population

    # Extract raw components from DataFrame
    estimates = results.estimate
    standard_errors = results.se

    # Only extract variables/terms for effects, not predictions
    if type === :effects
        variables = string.(results.variable)
        terms = string.(results.contrast)
    else
        variables = String[]
        terms = String[]
    end

    # CRITICAL: Preserve actual subgroup sizes from computation
    if "n" in names(results)
        metadata[:subgroup_n_values] = results.n
        metadata[:has_subgroup_n] = true
    end

    # Extract profile values from at_ columns (if any)
    profile_values = _extract_context_profile_values(results)

    # Return type-specific result
    if type === :effects
        return EffectsResult(estimates, standard_errors, variables, terms, profile_values, nothing, G, metadata)
    else # :predictions
        return PredictionsResult(estimates, standard_errors, profile_values, nothing, G, metadata)
    end
end

"""
    _extract_group_variables(groups) -> Vector{Symbol}

Extract a flat list of grouping variable names from the unified `groups` specification.
"""
function _extract_group_variables(groups)
    vars = Symbol[]
    if groups isa Symbol
        push!(vars, groups)
    elseif groups isa AbstractVector
        for g in groups
            append!(vars, _extract_group_variables(g))
        end
    elseif groups isa Tuple && length(groups) == 2
        var, spec = groups
        if var isa Symbol
            push!(vars, var)
        end
    elseif groups isa Pair
        append!(vars, _extract_group_variables(groups.first))
        append!(vars, _extract_group_variables(groups.second))
    end
    # Deduplicate while preserving order
    seen = Set{Symbol}()
    out = Symbol[]
    for v in vars
        if !(v in seen)
            push!(out, v)
            push!(seen, v)
        end
    end
    return out
end

"""
    _extract_context_profile_values(results) -> Union{Nothing, NamedTuple}

Extract profile values from at_ columns in results DataFrame.
"""
function _extract_context_profile_values(results)
    at_cols = [col for col in names(results) if startswith(String(col), "at_")]

    if isempty(at_cols)
        return nothing
    end

    # Extract the profile data
    profile_dict = Dict{Symbol, Vector}()
    for col in at_cols
        var_name = Symbol(String(col)[4:end])
        profile_dict[var_name] = results[!, col]
    end

    return NamedTuple(profile_dict)
end

"""
    _compute_all_continuous_context_effects_batch(
        engine, context_data, context_indices, continuous_vars_to_compute,
        scale, context_weights, context_de
    )

PHASE 2 OPTIMIZATION: Batch compute ALL continuous variables in ONE pass through rows.

This replaces the per-variable loop that was calling marginal_effects_mu! once per variable,
wasting computation by discarding n-1 derivatives on each call. Now we compute all derivatives
once per row and extract all requested variables.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine
- `context_data::NamedTuple`: Data with scenario overrides applied
- `context_indices::Vector{Int}`: Row indices for this context
- `continuous_vars_to_compute::Vector{Symbol}`: All continuous variables to process
- `scale::Symbol`: Either `:link` or `:response`
- `context_weights::Union{Vector{Float64}, Nothing}`: Observation weights for this context
- `context_de`: Pre-built derivative evaluator for all continuous variables

# Returns
- `(results_df, gradients_matrix)`: Combined results and gradients for ALL continuous variables

# Performance
- OLD: n_vars loops × N rows × compute_all_derivs = O(N × n_vars²)
- NEW: 1 loop × N rows × compute_all_derivs = O(N × n_vars)
- **Speedup: n_vars× (e.g., 55× for 55 variables)**
"""
function _compute_all_continuous_context_effects_batch(
    engine::MarginsEngine{L},
    context_data::NamedTuple,
    context_indices::AbstractVector{Int},
    continuous_vars_to_compute::Vector{Symbol},
    scale::Symbol,
    context_weights::Union{Vector{Float64}, Nothing},
    context_de
) where L

    n_vars = length(continuous_vars_to_compute)
    n_obs = length(context_indices)
    n_params = length(engine.β)

    # Pre-allocate accumulation buffers for ALL variables
    batch_ame_values = zeros(Float64, n_vars)
    batch_gradients = zeros(Float64, n_vars, n_params)

    # Build variable index mapping (which column in context_de.vars for each requested var)
    var_indices = Vector{Int}(undef, n_vars)
    for i in 1:n_vars
        var = continuous_vars_to_compute[i]
        idx = findfirst(==(var), context_de.vars)
        if isnothing(idx)
            error("Variable $var not found in derivative evaluator variables")
        end
        var_indices[i] = idx
    end

    # Compute total weight for normalization
    total_weight = if isnothing(context_weights)
        Float64(n_obs)
    else
        sum(context_weights[context_indices])
    end

    # Call SHARED batch computation function (same code as non-scenarios path)
    # This is the core optimization - single pass through rows extracts ALL variables
    _batch_compute_continuous_effects!(
        batch_ame_values, batch_gradients,
        context_de, engine.β, engine.link, scale,
        var_indices, context_indices, context_weights, total_weight
    )

    # Build results DataFrame and gradients matrix
    results = DataFrame(
        variable = String[],
        contrast = String[],
        estimate = Float64[],
        se = Float64[],
        n = Int[]
    )

    gradients_list = Matrix{Float64}[]

    for (i, var) in enumerate(continuous_vars_to_compute)
        ame_val = batch_ame_values[i]
        gβ_avg = view(batch_gradients, i, :)
        se = compute_se_only(gβ_avg, engine.Σ)

        push!(results, (
            variable = string(var),
            contrast = "derivative",
            estimate = ame_val,
            se = se,
            n = n_obs
        ))

        push!(gradients_list, reshape(collect(gβ_avg), 1, :))
    end

    gradients_matrix = vcat(gradients_list...)

    return (results, gradients_matrix)
end

"""
    _compute_population_effect_in_context_clean(engine, context_data, context_indices, var, scale, backend, full_weights, context_weights, scenario_spec, contrasts, context_de, context_ce)

Compute population marginal effect for a single variable within a context.

Routes to either continuous or categorical computation based on variable type.
Uses FormulaCompiler primitives for efficient zero-allocation computation.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine
- `context_data::NamedTuple`: Data with scenario overrides applied
- `context_indices::Vector{Int}`: Row indices for this context
- `var::Symbol`: Variable to analyze
- `scale::Symbol`: Either `:eta` or `:response`
- `backend::Symbol`: Either `:fd` or `:ad`
- `full_weights::Union{Vector{Float64}, Nothing}`: Full weights vector (for categorical kernel)
- `context_weights::Union{Vector{Float64}, Nothing}`: Subset weights (for continuous effects)
- `scenario_spec`: Scenario specification (for metadata)
- `contrasts::Symbol`: Contrast type for categorical variables (`:baseline` or `:pairwise`)
- `context_de`: Pre-built derivative evaluator (for continuous variables)
- `context_ce`: Pre-built contrast evaluator (for categorical variables, or nothing)

# Returns
- `(result_df, gradients)`: Results DataFrame and parameter gradient matrix
"""
function _compute_population_effect_in_context_clean(
    engine::MarginsEngine{L},
    context_data::NamedTuple,
    context_indices::AbstractVector{Int},
    var::Symbol,
    scale::Symbol,
    backend::Symbol,
    full_weights::Union{Vector{Float64}, Nothing},
    context_weights::Union{Vector{Float64}, Nothing},
    scenario_spec,
    contrasts::Symbol,
    context_de,  # Pre-built derivative evaluator
    context_ce   # Pre-built contrast evaluator (or nothing)
) where L

    # Check if variable is continuous or categorical
    if _is_continuous_variable(context_data[var])
        # Continuous variable: compute derivative-based marginal effect (uses context_weights)
        return _compute_continuous_context_effect_batch(
            engine, context_data, context_indices, var, scale, context_weights, context_de
        )
    else
        # Categorical variable: compute contrast-based effect (uses full_weights)
        return _compute_categorical_context_effect(
            engine, context_data, context_indices, var, scale, full_weights, contrasts, context_ce
        )
    end
end

"""
    _compute_continuous_context_effect_batch(engine, context_data, context_indices, var, scale, weights, context_de)

Clean batch implementation for continuous context effects using FormulaCompiler primitives.

Computes average marginal effects for a continuous variable within a specific context
(defined by scenario overrides and/or group filters). Uses FormulaCompiler's zero-allocation
marginal_effects_eta!/mu! functions.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine (from original data)
- `context_data::NamedTuple`: Modified data with scenario overrides applied
- `context_indices::Vector{Int}`: Row indices within this context
- `var::Symbol`: Variable to compute marginal effect for
- `scale::Symbol`: Either `:eta` (linear predictor) or `:response` (mean scale)
- `weights::Union{Vector{Float64}, Nothing}`: Optional observation weights
- `context_de`: Pre-built derivative evaluator for this context

# Returns
- `(result_df, gradients)`: DataFrame with results and parameter gradients matrix
"""
function _compute_continuous_context_effect_batch(
    engine::MarginsEngine{L},
    context_data::NamedTuple,
    context_indices::AbstractVector{Int},
    var::Symbol,
    scale::Symbol,
    weights::Union{Vector{Float64}, Nothing},
    context_de  # Pre-built derivative evaluator
) where L

    # Use pre-built derivative evaluator (no longer rebuilding per variable - PERFORMANCE FIX)
    # context_de was built once per context in _population_margins_with_contexts

    # Find variable index
    var_idx = findfirst(==(var), context_de.vars)
    isnothing(var_idx) && throw(ArgumentError("Variable $var not found in continuous variables"))

    # Pre-allocate buffers for marginal effects and parameter gradients
    g_buf = Vector{Float64}(undef, length(context_de.vars))
    Gβ_buf = Matrix{Float64}(undef, length(engine.β), length(context_de.vars))

    if isnothing(weights)
        # Unweighted case: simple averaging
        ame_sum = 0.0
        grad_sum = zeros(Float64, length(engine.β))

        for row in context_indices
            # Compute marginal effects and parameter gradients using FC primitives
            if scale === :response
                marginal_effects_mu!(g_buf, Gβ_buf, context_de, engine.β, engine.link, row)
            else
                marginal_effects_eta!(g_buf, Gβ_buf, context_de, engine.β, row)
            end

            # Accumulate marginal effect for this variable
            ame_sum += g_buf[var_idx]

            # Accumulate parameter gradient for this variable (column var_idx of Gβ_buf)
            @inbounds for p in 1:length(engine.β)
                grad_sum[p] += Gβ_buf[p, var_idx]
            end
        end

        # Average over rows
        n = length(context_indices)
        ame_val = ame_sum / n
        gβ_avg = grad_sum ./ n

    else
        # Weighted case
        context_weights = weights[context_indices]
        ame_sum = 0.0
        grad_sum = zeros(Float64, length(engine.β))
        total_weight = 0.0

        for (i, row) in enumerate(context_indices)
            w = context_weights[i]
            if w > 0
                # Compute marginal effects and parameter gradients using FC primitives
                if scale === :response
                    marginal_effects_mu!(g_buf, Gβ_buf, context_de, engine.β, engine.link, row)
                else
                    marginal_effects_eta!(g_buf, Gβ_buf, context_de, engine.β, row)
                end

                # Accumulate weighted marginal effect
                ame_sum += w * g_buf[var_idx]

                # Accumulate weighted parameter gradient
                @inbounds for p in 1:length(engine.β)
                    grad_sum[p] += w * Gβ_buf[p, var_idx]
                end

                total_weight += w
            end
        end

        if total_weight <= 0
            error("All weights are zero in this context")
        end

        # Weighted average
        ame_val = ame_sum / total_weight
        gβ_avg = grad_sum ./ total_weight
    end

    # Compute standard error
    se = compute_se_only(gβ_avg, engine.Σ)

    # Build result DataFrame
    result_df = DataFrame(
        variable = [string(var)],
        contrast = ["derivative"],
        estimate = [ame_val],
        se = [se],
        n = [length(context_indices)]
    )

    return (result_df, reshape(gβ_avg, 1, :))
end

"""
    _compute_categorical_context_effect(engine, context_data, context_indices, var, scale, weights, contrasts, context_ce)

Compute categorical marginal effects within a context using FormulaCompiler ContrastEvaluator.

Uses the zero-allocation categorical kernel from Phase 4 for efficient contrast computation.
Uses pre-built ContrastEvaluator to avoid rebuilding per variable.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine
- `context_data::NamedTuple`: Data with scenario overrides applied
- `context_indices::Vector{Int}`: Row indices for this context
- `var::Symbol`: Categorical variable to analyze
- `scale::Symbol`: Either `:eta` or `:response`
- `weights::Union{Vector{Float64}, Nothing}`: Optional weights
- `contrasts::Symbol`: Contrast type (`:baseline` or `:pairwise`)
- `context_ce`: Pre-built contrast evaluator for all categorical variables in this context

# Returns
- `(result_df, gradients)`: Results DataFrame and parameter gradient matrix
"""
function _compute_categorical_context_effect(
    engine::MarginsEngine{L},
    context_data::NamedTuple,
    context_indices::AbstractVector{Int},
    var::Symbol,
    scale::Symbol,
    weights::Union{Vector{Float64}, Nothing},
    contrasts::Symbol,
    context_ce  # Pre-built contrast evaluator
) where L

    # Use pre-built ContrastEvaluator (no longer rebuilding per variable - PERFORMANCE FIX)
    # context_ce was built once per context in _population_margins_with_contexts
    context_evaluator = context_ce

    # Generate contrast pairs for this variable
    var_col = context_data[var]
    contrast_pairs = generate_contrast_pairs(
        var_col, context_indices, contrasts, engine.model, var, context_data
    )

    # Pre-allocate buffers for kernel
    contrast_buf = Vector{Float64}(undef, length(engine.β))
    gradient_buf = Vector{Float64}(undef, length(engine.β))
    gradient_accum = Vector{Float64}(undef, length(engine.β))

    # Build results
    results = DataFrame(
        variable = String[],
        contrast = String[],
        estimate = Float64[],
        se = Float64[],
        n = Int[]
    )

    gradients_matrix = Matrix{Float64}(undef, length(contrast_pairs), length(engine.β))

    # Compute each contrast
    for (i, pair) in enumerate(contrast_pairs)
        # Use the zero-allocation categorical kernel
        ame, se = categorical_contrast_ame!(
            contrast_buf,
            gradient_buf,
            gradient_accum,
            context_evaluator,
            var,
            pair.level1,
            pair.level2,
            engine.β,
            engine.Σ,
            engine.link,
            context_indices,
            weights
        )

        # Store results
        push!(results.variable, string(var))
        push!(results.contrast, "$(pair.level2) - $(pair.level1)")
        push!(results.estimate, ame)
        push!(results.se, se)
        push!(results.n, length(context_indices))

        # Store gradient (already computed in gradient_accum by kernel)
        gradients_matrix[i, :] = gradient_accum
    end

    return (results, gradients_matrix)
end

"""
    _compute_population_prediction_in_context(engine, context_data, context_indices, scale, weights)

Compute population average prediction within a context using FormulaCompiler primitives.

Evaluates the model's predicted values for observations in a context and computes
parameter gradients for uncertainty quantification. Uses FormulaCompiler's compiled
formula evaluation and link functions.

# Arguments
- `engine::MarginsEngine`: Pre-built margins engine
- `context_data::NamedTuple`: Data with scenario overrides applied
- `context_indices::Vector{Int}`: Row indices for this context
- `scale::Symbol`: Either `:eta` (linear predictor) or `:response` (mean scale)
- `weights::Union{Vector{Float64}, Nothing}`: Optional observation weights

# Returns
- `(result_df, gradients)`: Results DataFrame and parameter gradient matrix

# Mathematical Details
- η scale: Prediction is η = Xβ, gradient is ∂η/∂β = X
- μ scale: Prediction is μ = g⁻¹(Xβ), gradient is ∂μ/∂β = g'(η) × X
  where g⁻¹ is the inverse link function and g' is its derivative
"""
function _compute_population_prediction_in_context(
    engine::MarginsEngine{L},
    context_data::NamedTuple,
    context_indices::AbstractVector{Int},
    scale::Symbol,
    weights::Union{Vector{Float64}, Nothing}
) where L

    # Pre-allocate buffer for model matrix row
    xrow_buf = Vector{Float64}(undef, length(engine.β))

    if isnothing(weights)
        # Unweighted case
        pred_sum = 0.0
        grad_sum = zeros(Float64, length(engine.β))

        for row in context_indices
            # Evaluate model matrix row: X[row, :]
            engine.compiled(xrow_buf, context_data, row)

            # Compute linear predictor: η = Xβ
            η = dot(xrow_buf, engine.β)

            if scale === :response
                # Apply link inverse: μ = g⁻¹(η)
                μ = GLM.linkinv(engine.link, η)
                pred_sum += μ

                # Gradient: ∂μ/∂β = g'(η) × X
                g_prime = _dmu_deta(engine.link, η)
                @inbounds for j in eachindex(engine.β)
                    grad_sum[j] += g_prime * xrow_buf[j]
                end
            else
                # η scale: prediction is η
                pred_sum += η

                # Gradient: ∂η/∂β = X
                grad_sum .+= xrow_buf
            end
        end

        # Average over observations
        n = length(context_indices)
        pred_val = pred_sum / n
        gβ_avg = grad_sum ./ n

    else
        # Weighted case
        pred_sum = 0.0
        grad_sum = zeros(Float64, length(engine.β))
        total_weight = 0.0

        for (i, row) in enumerate(context_indices)
            w = weights[row]
            if w > 0
                # Evaluate model matrix row: X[row, :]
                engine.compiled(xrow_buf, context_data, row)

                # Compute linear predictor: η = Xβ
                η = dot(xrow_buf, engine.β)

                if scale === :response
                    # Apply link inverse: μ = g⁻¹(η)
                    μ = GLM.linkinv(engine.link, η)
                    pred_sum += w * μ

                    # Weighted gradient: ∂μ/∂β = g'(η) × X
                    g_prime = _dmu_deta(engine.link, η)
                    @inbounds for j in eachindex(engine.β)
                        grad_sum[j] += w * g_prime * xrow_buf[j]
                    end
                else
                    # η scale: prediction is η
                    pred_sum += w * η

                    # Weighted gradient: ∂η/∂β = X
                    @inbounds for j in eachindex(engine.β)
                        grad_sum[j] += w * xrow_buf[j]
                    end
                end

                total_weight += w
            end
        end

        if total_weight <= 0
            error("All weights are zero in this context")
        end

        # Weighted average
        pred_val = pred_sum / total_weight
        gβ_avg = grad_sum ./ total_weight
    end

    # Compute standard error
    se = compute_se_only(gβ_avg, engine.Σ)

    # Build result DataFrame
    result_df = DataFrame(
        estimate = [pred_val],
        se = [se],
        n = [length(context_indices)]
    )

    return (result_df, reshape(gβ_avg, 1, :))
end

# Delegate helper functions that depend on other modules

"""
    _parse_at_specification(scenarios) -> Vector{Dict{Symbol, Any}}

Normalize the user-provided `scenarios` specification into a vector of dictionaries.
Each dictionary represents a single counterfactual context and multi-valued inputs
produce a full Cartesian expansion across all supplied values.

- Accepts a `NamedTuple` where each field is either a scalar or a collection.
- Scalars are promoted to length-1 vectors to unify the expansion logic.
- The return order matches `Iterators.product`, guaranteeing deterministic ordering.
"""
function _parse_at_specification(scenarios)
    if scenarios isa NamedTuple
        var_names = collect(keys(scenarios))
        var_values = [
            begin
                val = getproperty(scenarios, name)
                val isa AbstractVector ? collect(val) : [val]
            end for name in var_names
        ]

        contexts = Vector{Dict{Symbol, Any}}()
        for combo in Iterators.product(var_values...)
            push!(contexts, Dict(zip(var_names, combo)))
        end
        return contexts
    else
        error("Scenarios specification not yet implemented for type $(typeof(scenarios))")
    end
end

function _parse_groups_specification(groups, data_nt)
    # Simple categorical grouping: :education
    if groups isa Symbol
        return _parse_over_specification([groups], data_nt)
    end

    # Vector grouping: may be all symbols or mixed specs
    if groups isa AbstractVector
        if all(x -> x isa Symbol, groups)
            # All symbols - simple categorical cross-tabulation
            return _parse_over_specification(groups, data_nt)
        else
            # Mixed vector - handle each spec and create cross-product
            all_groups = []
            for spec in groups
                spec_groups = _parse_groups_specification(spec, data_nt)
                if isempty(all_groups)
                    all_groups = spec_groups
                else
                    # Cross-product with existing groups
                    new_groups = []
                    for existing_group in all_groups
                        for spec_group in spec_groups
                            combined_group = merge(existing_group, spec_group)
                            push!(new_groups, combined_group)
                        end
                    end
                    all_groups = new_groups
                end
            end
            return all_groups
        end
    end

    # Continuous grouping: (:income, 4) or (:age, [25000, 50000])
    if groups isa Tuple && length(groups) == 2
        var, spec = groups
        if var isa Symbol
            # Quantile specification: (:income, 4)
            if spec isa Integer && spec > 0
                return _create_quantile_groups(data_nt[var], var, spec)
            end
            # Threshold specification: (:income, [25000, 50000])
            if spec isa AbstractVector && all(x -> x isa Real, spec)
                return _create_threshold_groups(data_nt[var], var, spec)
            end
        end
    end

    # Syntax for nested grouping: :region => :education
    if groups isa Pair
        outer_spec = groups.first
        inner_spec = groups.second
        return _create_nested_strata(outer_spec, inner_spec, data_nt)
    end

    error("Invalid groups specification. Supported syntax: Symbol, Vector{Symbol}, (Symbol, Int), (Symbol, Vector), or outer => inner")
end

function _parse_over_specification(over, data_nt)
    if over isa NamedTuple
        # Enhanced flexible syntax
        contexts = []

        for (var, vals) in pairs(over)
            if isnothing(vals)
                # Unspecified - must be categorical, use all levels
                if _is_continuous_variable(data_nt[var])
                    error("Continuous variable $var in over() must specify values")
                end
                contexts = isempty(contexts) ? [Dict(var => v) for v in unique(data_nt[var])] :
                          [merge(ctx, Dict(var => v)) for ctx in contexts for v in unique(data_nt[var])]
            else
                # Specified values
                if _is_continuous_variable(data_nt[var])
                    # For continuous: create subgroups around specified values
                    subgroups = _create_continuous_subgroups(data_nt[var], vals)
                    contexts = isempty(contexts) ? [Dict(var => sg) for sg in subgroups] :
                              [merge(ctx, Dict(var => sg)) for ctx in contexts for sg in subgroups]
                else
                    # For categorical: use specified subset
                    val_list = vals isa Vector ? vals : [vals]
                    contexts = isempty(contexts) ? [Dict(var => v) for v in val_list] :
                              [merge(ctx, Dict(var => v)) for ctx in contexts for v in val_list]
                end
            end
        end

        return contexts
    elseif over isa Vector
        # Mixed vector syntax - handle both categorical and continuous variables
        contexts = [Dict()]
        for var in over
            if _is_continuous_variable(data_nt[var])
                # For continuous variables in vector syntax, use automatic quartile binning
                quantile_groups = _create_quantile_groups(data_nt[var], var, 4)  # Default to quartiles
                new_contexts = []
                for ctx in contexts, qg in quantile_groups
                    push!(new_contexts, merge(ctx, qg))
                end
                contexts = new_contexts
            else
                # For categorical variables, use all unique levels
                new_contexts = []
                for ctx in contexts, val in unique(data_nt[var])
                    push!(new_contexts, merge(ctx, Dict(var => val)))
                end
                contexts = new_contexts
            end
        end
        return contexts
    elseif over isa Symbol
        # Single variable syntax - convert to vector and recurse
        return _parse_over_specification([over], data_nt)
    else
        error("over parameter must be a NamedTuple, Vector, or Symbol")
    end
end

function _create_nested_strata(outer_spec, inner_spec, data_nt)
    # Parse outer specification
    outer_groups = _parse_groups_specification(outer_spec, data_nt)

    # Handle inner specification - can be single spec or Vector of specs
    if inner_spec isa AbstractVector
        # Multiple inner specifications - create parallel groups within each outer
        nested_groups = []
        for outer_group in outer_groups
            for inner_single_spec in inner_spec
                inner_groups = _parse_groups_specification(inner_single_spec, data_nt)
                for inner_group in inner_groups
                    # Merge outer and inner groups
                    combined_group = merge(outer_group, inner_group)
                    push!(nested_groups, combined_group)
                end
            end
        end
        return nested_groups
    else
        # Single inner specification
        inner_groups = _parse_groups_specification(inner_spec, data_nt)

        # Create nested combinations: outer => inner means inner within each outer
        nested_groups = []
        for outer_group in outer_groups
            for inner_group in inner_groups
                # Merge outer and inner groups
                combined_group = merge(outer_group, inner_group)
                push!(nested_groups, combined_group)
            end
        end

        return nested_groups
    end
end

function _create_quantile_groups(col, var, n_quantiles)
    quantiles = [quantile(col, i/n_quantiles) for i in 0:n_quantiles]
    groups = []
    for i in 1:(n_quantiles)
        lower, upper = quantiles[i], quantiles[i+1]

        # Generate descriptive labels based on n_quantiles
        if n_quantiles == 4
            label = ["Q1", "Q2", "Q3", "Q4"][i]
        elseif n_quantiles == 3
            label = ["T1", "T2", "T3"][i]
        elseif n_quantiles == 5
            label = ["P1", "P2", "P3", "P4", "P5"][i]  # Quintiles
        else
            label = "Bin$i"
        end

        # Create group with range information for filtering
        group = Dict(
            var => (lower=lower, upper=upper, label=label, bin=i),
            Symbol("$(var)_bin") => i
        )
        push!(groups, group)
    end
    return groups
end

function _create_threshold_groups(col, var, thresholds)
    groups = []
    n_groups = length(thresholds) + 1

    for i in 1:n_groups
        if i == 1
            # First group: [min, threshold1)
            lower = minimum(col)
            upper = thresholds[1]
            label = "< $(thresholds[1])"
        elseif i == n_groups
            # Last group: [threshold_last, max]
            lower = thresholds[end]
            upper = maximum(col)
            label = ">= $(thresholds[end])"
        else
            # Middle groups: [threshold_{i-1}, threshold_i)
            lower = thresholds[i-1]
            upper = thresholds[i]
            label = "[$(lower), $(upper))"
        end

        group = Dict(
            var => (lower=lower, upper=upper, label=label, bin=i),
            Symbol("$(var)_bin") => i
        )
        push!(groups, group)
    end

    return groups
end

function _create_continuous_subgroups(col, specified_values)
    val_list = specified_values isa Vector ? specified_values : [specified_values]
    subgroups = []
    for val in val_list
        # Create subgroup of indices within range of this value
        indices = findall(x -> abs(x - val) <= 2.5, col)  # ±2.5 units
        push!(subgroups, (center=val, indices=indices, label="$(val)±2.5"))
    end
    return subgroups
end

function _coerce_override_scalar(col, raw_value)
    T = eltype(col)
    value = _normalize_override_value(raw_value)
    if T <: Bool
        if value isa Bool
            return value
        elseif value isa Integer
            return value != 0
        elseif value isa AbstractString
            lowered = lowercase(value)
            if lowered in ("true", "t", "1")
                return true
            elseif lowered in ("false", "f", "0")
                return false
            end
        end
        return convert(Bool, value)
    elseif T <: Real
        return value isa T ? value : convert(T, value)
    else
        return value isa T ? value : convert(T, value)
    end
end

function _resolve_categorical_override(col, raw_value, var::Symbol)
    T = eltype(col)
    if raw_value isa T
        return raw_value
    end

    level_candidates = levels(col)
    match_idx = findfirst(level -> level == raw_value, level_candidates)
    if match_idx === nothing
        raw_str = string(raw_value)
        match_idx = findfirst(level -> string(level) == raw_str, level_candidates)
    end

    match_idx === nothing && error("Scenario override $raw_value not found among levels for categorical variable :$var")

    matched_level = level_candidates[match_idx]
    matched_str = string(matched_level)
    existing_idx = findfirst(x -> string(x) == matched_str, col)
    if existing_idx !== nothing
        return col[existing_idx]
    end

    pool_ref_type = T.parameters[2]
    ref_value = convert(pool_ref_type, match_idx)
    return T(pool(col), ref_value)
end

function _build_override_column(col, raw_value, var::Symbol)
    if eltype(col) <: CategoricalValue
        categorical_value = _resolve_categorical_override(col, raw_value, var)
        override_col = copy(col)
        fill!(override_col, categorical_value)
        return override_col
    else
        coerced_value = _coerce_override_scalar(col, raw_value)
        return fill(coerced_value, length(col))
    end
end

"""
    _create_context_data_with_weights(data_nt, scenario_spec, group_spec, weights)

Apply group filters and scenario overrides while preserving column storage types.
Categorical overrides reuse the original categorical pool so downstream evaluators
continue to see `CategoricalValue` levels; Boolean overrides stay `Bool` for the
zero-allocation counterfactual pathway.
"""
function _create_context_data_with_weights(data_nt, scenario_spec, group_spec, weights)
    # Create context indices based on group specification with sophisticated filtering
    if isempty(group_spec)
        context_indices = collect(1:length(first(data_nt)))
    else
        # Apply subgroup filtering (groups) - enhanced for continuous binning
        context_indices = collect(1:length(first(data_nt)))
        for (var, spec) in group_spec
            if haskey(data_nt, var)
                if spec isa NamedTuple && haskey(spec, :lower) && haskey(spec, :upper)
                    # Range-based continuous filtering
                    col = data_nt[var]
                    if haskey(spec, :bin) && spec.bin == 1
                        # First bin: include lower boundary
                        range_indices = findall(x -> spec.lower <= x < spec.upper, col)
                    elseif haskey(spec, :bin) && spec.bin > 1
                        # Other bins: exclude lower boundary to avoid overlap
                        range_indices = findall(x -> spec.lower < x <= spec.upper, col)
                    else
                        # Fallback: include both boundaries
                        range_indices = findall(x -> spec.lower <= x <= spec.upper, col)
                    end
                    context_indices = intersect(context_indices, range_indices)
                elseif spec isa NamedTuple && haskey(spec, :indices)
                    # Index-based subgroup (support for around-value subgroups)
                    context_indices = intersect(context_indices, collect(spec.indices))
                else
                    # Categorical - filter by value
                    var_indices = findall(==(spec), data_nt[var])
                    context_indices = intersect(context_indices, var_indices)
                end
            elseif occursin("_bin", String(var)) && haskey(group_spec, var)
                # Skip bin columns - they're handled by their parent variable
                continue
            end
        end
    end

    # Apply scenario overrides to data (if any)
    context_data = if isempty(scenario_spec)
        data_nt
    else
        # Create modified data with scenario values
        modified_data = NamedTuple()
        for (var, col) in pairs(data_nt)
            if haskey(scenario_spec, var)
                override_val = scenario_spec[var]
                override_col = _build_override_column(col, override_val, var)
                modified_data = merge(modified_data, NamedTuple{(var,)}((override_col,)))
            else
                modified_data = merge(modified_data, NamedTuple{(var,)}((col,)))
            end
        end
        modified_data
    end

    # Extract context weights if provided
    context_weights = if isnothing(weights)
        nothing
    else
        weights[context_indices]
    end

    return (context_data, context_indices, context_weights)
end

function _append_results_with_missing_columns(results, new_result)
    if nrow(results) == 0
        return new_result
    end

    # Get all columns from both DataFrames
    all_columns = union(names(results), names(new_result))

    # Add missing columns to both DataFrames
    for col in all_columns
        if !(col in names(results))
            results[!, col] = fill(missing, nrow(results))
        end
        if !(col in names(new_result))
            new_result[!, col] = fill(missing, nrow(new_result))
        end
    end

    # Concatenate the results
    return vcat(results, new_result)
end
