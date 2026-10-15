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
    # Variables dropped by the skip rule below, for the empty-result diagnostic
    skipped_vars = Set{Symbol}()

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
        # NOTE: the third return value (weights subset) is deliberately discarded. Every
        # context compute function below takes the FULL weights vector and indexes it by
        # absolute row index, matching `context_indices`.
        context_data, context_indices, _ = _create_context_data_with_weights(data_nt, scenario_spec, group_spec, weights)

        # A context with no rows has nothing to average over. Emitting a row for it would
        # report 0.0 +/- 0.0, which reads as a precisely estimated null rather than "no
        # data" -- and empty cells are routine in categorical cross-tabulations, where not
        # every level combination is observed. Skip the context instead; if every context
        # is empty the guard after the loop reports that.
        if isempty(context_indices)
            continue
        end

        if type === :effects
            # PERFORMANCE OPTIMIZATION Phase 1: Build evaluators ONCE per context
            # This prevents rebuilding evaluators n_vars times (was causing 300× overhead)
            continuous_vars = continuous_variables(engine.compiled, context_data)
            context_de = derivativeevaluator(backend, engine.compiled, context_data, continuous_vars)

            # Separate continuous and categorical variables for batch vs per-variable processing
            continuous_vars_to_compute = Symbol[]
            categorical_vars_to_compute = Symbol[]
            for var in vars
                # Skip a variable only when the context holds it at a single value: any
                # scenario override (which fills the whole column), or a group that pins it
                # to one level. A group that BINS it leaves it varying within the subgroup,
                # so its effect there is well defined and is computed as usual.
                if haskey(scenario_spec, var) ||
                   (haskey(group_spec, var) && _group_spec_pins(group_spec[var]))
                    push!(skipped_vars, var)
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
                    scale, weights, context_de, measure
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
                engine, context_data, context_indices, scale, weights
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
    # Use _normalize_scenarios_to_namedtuple to handle both NamedTuple and Pair syntax
    scenarios_vars = if isnothing(scenarios)
        Symbol[]
    else
        scenarios_nt = _normalize_scenarios_to_namedtuple(scenarios)
        collect(keys(scenarios_nt))
    end
    groups_vars = isnothing(groups) ? Symbol[] : _extract_group_variables(groups)
    metadata[:scenarios_vars] = scenarios_vars
    metadata[:groups_vars] = groups_vars

    # Add analysis_type for format auto-detection
    metadata[:analysis_type] = :population

    # Guard the empty frame: without this, `results.estimate` below throws an opaque
    # `ArgumentError: column name "estimate" not found ...` from DataFrames internals.
    # Explicit `vars` overlapping `scenarios`/`groups` is rejected up front in
    # `population_margins`; this catches what is left, chiefly auto-detected `vars` that
    # the skip rule consumes entirely (e.g. a model whose only continuous variable is `x`,
    # called with `groups=(:x, 4)`).
    if nrow(results) == 0
        context_vars = union(Set(scenarios_vars), Set(groups_vars))
        if !isempty(skipped_vars)
            skipped_list = join(sort!(collect(skipped_vars)), ", ")
            context_list = join(sort!(collect(context_vars)), ", ")
            throw(MarginsError(
                "No effects could be computed: every requested variable ($(skipped_list)) is " *
                "also used by 'scenarios'/'groups' ($(context_list)), so each was skipped in " *
                "every context. Computing the effect of a variable while holding it fixed, or " *
                "within subgroups defined by it, is contradictory. Request the effect of a " *
                "different variable, use profile_margins() for effects at specific values of " *
                "$(skipped_list), or group by a derived bin column instead."
            ))
        else
            throw(MarginsError(
                "No results could be computed: the 'scenarios'/'groups' specification produced " *
                "no contexts with data. Check that the grouping variables have observed values " *
                "and that any scenario values are attainable."
            ))
        end
    end

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
    _group_spec_pins(spec) -> Bool

Does this group specification hold its variable at a single value?

A categorical level pins: every row in the subgroup carries that one level, so the variable
has no variation left and its own marginal effect is undefined there. A quantile or threshold
bin does not pin: `groups` only ever selects rows (`_create_context_data_with_weights`), never
overwrites them, so within a bin the variable still varies across its range and the average
derivative over those rows is well defined -- Stata's `margins, dydx(x) over(xbin)`.

Bins are marked with `kind=:bin` by `_create_quantile_groups` and `_create_threshold_groups`;
anything else is a pinned level.
"""
_group_spec_pins(spec) = !(spec isa NamedTuple && get(spec, :kind, nothing) === :bin)

"""
    _classify_group_variables(groups) -> (pinned::Vector{Symbol}, binned::Vector{Symbol})

Split the grouping variables into those a spec PINS to a single value and those it BINS into
ranges, walking the raw `groups` syntax.

This is purely syntactic, which is possible because a bare `Symbol` is accepted only for a
categorical variable (`_parse_over_specification` rejects a numeric one), so a bare `Symbol`
always pins and only the `(Symbol, spec)` tuple forms bin. It therefore agrees with
`_group_spec_pins` on the parsed specs by construction.

Only the grammar `_validate_groups_parameter` accepts is handled; it runs first, so anything
else has already been rejected. A variable reachable both ways -- `groups=[(:x, 4), :x]` --
appears in both vectors, and callers should treat pinned as the stronger claim.
"""
function _classify_group_variables(groups)
    pinned = Symbol[]
    binned = Symbol[]

    function walk(g)
        if g isa Symbol
            push!(pinned, g)
        elseif g isa AbstractVector
            foreach(walk, g)
        elseif g isa Tuple && length(g) == 2 && g[1] isa Symbol
            # (:x, 4) quantiles or (:x, [cut1, cut2]) thresholds -- both bin
            push!(binned, g[1])
        elseif g isa Pair
            walk(g.first)
            walk(g.second)
        end
        return nothing
    end
    walk(groups)

    return (unique(pinned), unique(binned))
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
        scale, weights, context_de, measure
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
- `weights::Union{Vector{Float64}, Nothing}`: FULL-length observation weights vector,
  indexed by absolute row index (i.e. the same indices as `context_indices`). Do **not**
  pass a per-context subset — the accumulation loops and every measure helper index it
  as `weights[row]`.
- `context_de`: Pre-built derivative evaluator for all continuous variables
- `measure::Symbol`: Econometric measure (`:effect`, `:elasticity`,
  `:semielasticity_dyex`, `:semielasticity_eydx`)

# Returns
- `(results_df, gradients_matrix)`: Combined results and gradients for ALL continuous variables

# Measure Handling
Measure transformations use the CONTEXT's own normalizers: x̄, ȳ and ∂ȳ/∂β are computed
over `context_indices` with scenario overrides applied (`context_data`). A group's
elasticity therefore uses that group's means, and a scenario's elasticity uses the
overridden x. The gradient gets both the scale factor and the quotient-rule correction,
identically to the no-context path in `_process_continuous_variables!`.

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
    weights::Union{Vector{Float64}, Nothing},
    context_de,
    measure::Symbol=:effect
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

    # Compute the context's mean response and total weight in one pass.
    # For measure === :effect this returns (1.0, Σw) without touching the data.
    (ȳ, total_weight) = _compute_response_mean_if_needed(
        engine, context_data, context_indices, scale, measure, weights
    )

    # For measures that divide by the mean response, the delta-method gradient needs the
    # quotient-rule term ∂ȳ/∂β, evaluated over THIS context's rows.
    ∂ȳ_∂β = if measure === :elasticity || measure === :semielasticity_eydx
        _gradient_of_mean_response(
            engine.compiled, engine.row_buf, engine.β, engine.link,
            context_data, context_indices, scale, weights,
            Vector{Float64}(undef, n_params)
        )
    else
        nothing
    end

    # Call SHARED batch computation function (same code as non-scenarios path)
    # This is the core optimization - single pass through rows extracts ALL variables
    _batch_compute_continuous_effects!(
        batch_ame_values, batch_gradients,
        context_de, engine.β, engine.link, scale,
        var_indices, context_indices, weights, total_weight
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

        # Apply measure transformation (no-op for :effect) using this context's rows,
        # weights and overridden data
        (final_val, transform_factor) = apply_measure_transformations(
            ame_val, measure, var,
            context_data, context_indices, weights, ȳ, total_weight
        )

        if transform_factor != 1.0
            # First term: k × ∂AME/∂β
            gβ_avg .*= transform_factor

            # Second term: quotient-rule correction for measures that divide by ȳ
            # ∂ε/∂β = (x̄/ȳ) × ∂AME/∂β - (ε/ȳ) × ∂ȳ/∂β
            if !isnothing(∂ȳ_∂β)
                quotient_correction = final_val / ȳ
                @inbounds for j in eachindex(gβ_avg)
                    gβ_avg[j] -= quotient_correction * ∂ȳ_∂β[j]
                end
            end
        end

        se = compute_se_only(gβ_avg, engine.Σ)

        push!(results, (
            variable = string(var),
            contrast = "derivative",
            estimate = final_val,
            se = se,
            n = n_obs
        ))

        push!(gradients_list, reshape(collect(gβ_avg), 1, :))
    end

    gradients_matrix = vcat(gradients_list...)

    return (results, gradients_matrix)
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
- `scale::Symbol`: Either `:link` or `:response`
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
    # The kernel selects the computation scale from its link argument: `nothing` ⇒ link
    # scale (Δη), a link object ⇒ response scale (Δμ). Resolve it once, concretely.
    for (i, pair) in enumerate(contrast_pairs)
        # Use the zero-allocation categorical kernel
        ame, se = if scale === :link
            categorical_contrast_ame!(
                contrast_buf, gradient_buf, gradient_accum,
                context_evaluator, var, pair.level1, pair.level2,
                engine.β, engine.Σ, nothing,
                context_indices, weights
            )
        else
            categorical_contrast_ame!(
                contrast_buf, gradient_buf, gradient_accum,
                context_evaluator, var, pair.level1, pair.level2,
                engine.β, engine.Σ, engine.link,
                context_indices, weights
            )
        end

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
- `scale::Symbol`: Either `:link` (linear predictor) or `:response` (mean scale)
- `weights::Union{Vector{Float64}, Nothing}`: Optional FULL-length observation weights
  vector, indexed by absolute row index (the loops below read `weights[row]` for
  `row in context_indices`). Passing a per-context subset is a bounds error.

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
- Accepts a `Tuple` of `Pair{Symbol, Any}` for alternative syntax.
- Scalars are promoted to length-1 vectors to unify the expansion logic.
- The return order matches `Iterators.product`, guaranteeing deterministic ordering.
"""
function _parse_at_specification(scenarios)
    # Convert to NamedTuple for uniform processing
    scenarios_nt = _normalize_scenarios_to_namedtuple(scenarios)

    var_names = collect(keys(scenarios_nt))
    var_values = [
        begin
            val = getproperty(scenarios_nt, name)
            val isa AbstractVector ? collect(val) : [val]
        end for name in var_names
    ]

    contexts = Vector{Dict{Symbol, Any}}()
    for combo in Iterators.product(var_values...)
        push!(contexts, Dict(zip(var_names, combo)))
    end
    return contexts
end

"""
    _normalize_scenarios_to_namedtuple(scenarios) -> NamedTuple

Convert various scenarios specifications to a standardized NamedTuple format.

Accepts:
- `NamedTuple`: Pass through unchanged
- `Tuple` of `Pair{Symbol, Any}`: Convert to NamedTuple

Returns:
- `NamedTuple`: Normalized scenarios specification
"""
function _normalize_scenarios_to_namedtuple(scenarios)
    if scenarios isa NamedTuple
        return scenarios
    elseif scenarios isa Tuple && !(scenarios isa NamedTuple)
        # Convert Tuple of Pairs to NamedTuple
        # Example: (:x => 1, :y => 2) -> (x=1, y=2)
        pairs_vec = collect(scenarios)
        keys_vec = Symbol[p.first for p in pairs_vec]
        vals_vec = [p.second for p in pairs_vec]
        return NamedTuple{tuple(keys_vec...)}(vals_vec)
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
                # A bare Symbol does not say how to group a numeric column. This used to
                # bin into quartiles silently, which meant `groups=:nkids` on an integer
                # count returned quartiles of it rather than its levels -- an analysis the
                # caller never asked for, with cutpoints invisible at the call site. Stata's
                # over() and R marginaleffects' by() both require a discrete grouping
                # variable for the same reason.
                throw(MarginsError(
                    "Grouping variable '$(var)' is numeric, and `groups=$(repr(var))` does not " *
                    "say how to group it. Specify the binning explicitly -- " *
                    "groups=($(repr(var)), 4) for quartiles, or groups=($(repr(var)), [cut1, cut2]) " *
                    "for custom thresholds -- or convert '$(var)' to a categorical column if its " *
                    "distinct values are the groups you want."
                ))
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

    # Tied values can make consecutive quantile boundaries identical, which leaves the bin
    # between them with no rows. Reporting such a bin as 0.0 +/- 0.0 reads as a precisely
    # estimated null, so refuse the request instead and say what to do about it.
    for i in 1:n_quantiles
        if quantiles[i] == quantiles[i+1]
            throw(MarginsError(
                "Cannot form $(n_quantiles) quantile bins for '$(var)': quantile boundaries " *
                "$(i-1) and $(i) are both $(quantiles[i]), so bin $(i) would contain no rows. " *
                "'$(var)' has too many tied values at that point. Use fewer bins " *
                "(e.g. groups=($(repr(var)), 2)), give explicit thresholds " *
                "(groups=($(repr(var)), [cut1, cut2])), or group by a categorical column."
            ))
        end
    end

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
        group = Dict(var => (kind=:bin, lower=lower, upper=upper, label=label, bin=i, nbins=n_quantiles))
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

        group = Dict(var => (kind=:bin, lower=lower, upper=upper, label=label, bin=i, nbins=n_groups))
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
                    # Range-based continuous filtering.
                    #
                    # One convention for every bin: half-open [lower, upper), with the last
                    # bin closed at the top so the maximum is included. This makes the bins
                    # partition the column exactly -- the previous mix of [lo, hi) for bin 1
                    # and (lo, hi] for the rest dropped every row sitting exactly on an
                    # interior boundary, and it contradicted the labels that
                    # _create_threshold_groups assigns ("[25000, 50000)").
                    col = data_nt[var]
                    is_last_bin = !haskey(spec, :nbins) || spec.bin == spec.nbins
                    range_indices = if is_last_bin
                        findall(x -> spec.lower <= x <= spec.upper, col)
                    else
                        findall(x -> spec.lower <= x < spec.upper, col)
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
