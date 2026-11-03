# core/request.jl
# One request-normalization layer for every entry point (Margins 3.0, step 4).
#
# `population_margins`, `profile_margins` and the model-level `second_differences` turn their
# keywords into one `MarginsRequest` with `_normalize_request`, which runs every validator once
# and evaluates `vcov` once, and then compute from the request (`_run`). Nested computations
# (second differences run one AME per profile point or shift) derive requests from it with
# `_with_scenarios`/`_with_data` instead of re-entering keyword parsing.

"""
    MarginsRequest{K,M}

A normalized, validated request (internal). `K` is `:population` or `:profile` and `M` the
model type. The prepared data is not a type parameter: its column types are only known at run
time (data preparation converts columns), so a parameter would make the request's type
uninferable at every entry point and inference would analyse `_run` for an abstract request
before compiling it for the concrete one (first-call latency). Code that loops over the data
is behind function barriers (`build_engine`, `_ame_calculate`, ...) anyway.

| field | meaning |
|:------|:--------|
| `model` | the fitted model |
| `type` | `:effects` or `:predictions` |
| `data_nt` | the data as a column table, prepared by `_prepare_margins_data` (any Tables.jl source) |
| `vars` | resolved focal variables (`nothing` for predictions) |
| `vars_explicit` | whether the caller named `vars` (otherwise auto-detected) |
| `Σ` | the covariance matrix, the one evaluation of `vcov` for the call |
| `vcov_source` | the `vcov` the caller passed, which the t/z df rule keys on |
| `df` | inference degrees of freedom, `_inference_df(model, vcov_source)` |
| `alpha` | significance level, in (0, 1) |
| `weights`, `weights_source` | averaging weights (`nothing` or one per row) and where they came from (`:model`, `:user`, `:none`) |
| `scenarios` | the `scenarios` keyword as a `NamedTuple` (`nothing` if none) |
| `scenario_specs` | its parsed combinations, one `Dict` per scenario (`[Dict()]` if none) |
| `groups` | the validated `groups` syntax (`nothing` if none) |
| `group_specs` | its parsed subgroups (`[Dict()]` if none) |
| `reference_grid` | profile requests: the user's grid with string levels resolved (`at_spec`) |
| `completed_grid` | profile requests: that grid completed with typical values |
| `grid_vars` | profile requests: the grid columns shown in the output |
| `changed_vars` | every variable the request sets counterfactually: scenario keys, grid columns, and the second-differences modifier and `profile` variables |
| `contrasts`, `measure`, `scale`, `backend`, `interval` | validated options |

`vars ∪ changed_vars` is the set step 7's derived-column detector runs over.
"""
struct MarginsRequest{K,M}
    model::M
    type::Symbol
    data_nt::NamedTuple
    vars::Union{Nothing,Vector{Symbol}}
    vars_explicit::Bool
    Σ::Matrix{Float64}
    vcov_source::Any
    df::Float64
    alpha::Float64
    weights::Union{Nothing,Vector{Float64}}
    weights_source::Symbol
    scenarios::Union{Nothing,NamedTuple}
    scenario_specs::Vector
    groups::Any
    group_specs::Vector
    reference_grid::Union{Nothing,DataFrame}
    completed_grid::Union{Nothing,DataFrame}
    grid_vars::Set{Symbol}
    changed_vars::Vector{Symbol}
    contrasts::Symbol
    measure::Symbol
    scale::Symbol
    backend::Symbol
    interval::Symbol
end

_has_contexts(req::MarginsRequest) = !isnothing(req.scenarios) || !isnothing(req.groups)

"""
    _validate_request(model, req::MarginsRequest)

Model-class validation, called once at the end of `_normalize_request` (no-op by default).
FixedEffectModels adds its absorbed-fixed-effect checks here.
"""
_validate_request(model, req::MarginsRequest) = nothing

"""
    _validate_focal_vars(model, vars)

Model-class check of the `vars` a caller named, before they are resolved against the data
(no-op by default). FixedEffectModels rejects absorbed fixed effects here, so that the reason
is given before any data-type check on the column.
"""
_validate_focal_vars(model, vars) = nothing

"""
    _normalize_request(model, data; type=:effects, vars=nothing, scale=:response, backend=:ad,
                       scenarios=nothing, groups=nothing, reference_grid=nothing, measure=:effect,
                       contrasts=:baseline, alpha=0.05, vcov=GLM.vcov, weights=:none,
                       interval=:confidence, scenario_vars=nothing, changed_vars=Symbol[]) -> MarginsRequest

Validate every option and prepare everything a computation needs, once per call: the data
(any Tables.jl source), the focal variables, Σ (`vcov` is evaluated here and nowhere else), the
inference df, the averaging weights, the parsed scenarios and groups and, with a
`reference_grid`, the completed grid. A `reference_grid` makes a `:profile` request, otherwise it
is `:population`.

Two keywords serve callers that run several computations from one request (the model-level
`second_differences`): `scenario_vars` names the variables later requests set with
`_with_scenarios` (one per profile point), so they are validated here as `scenarios` keys
would be, once; `changed_vars` adds variables set counterfactually by other means (the
modifier shifted by `at=:observed`).

The validators run in a fixed order: the required arguments and the scalar options (`type`,
`scale`, `backend`, `measure`, `alpha`, `contrasts`, `interval`), then the `scenarios` and
`groups` syntax and their conflicts, the data, the weights, the variables (and the measure
filter, preceded by the model class's `_validate_focal_vars`), the group columns, the reference
grid, `vcov`, and last the model class (`_validate_request`).
"""
function _normalize_request(model, data;
        type::Symbol=:effects, vars=nothing, scale::Symbol=:response, backend::Symbol=:ad,
        scenarios=nothing, groups=nothing, reference_grid::Union{Nothing,DataFrame}=nothing,
        measure::Symbol=:effect, contrasts::Symbol=:baseline, alpha=0.05, vcov=GLM.vcov,
        weights=:none, interval::Symbol=:confidence,
        scenario_vars::Union{Nothing,Vector{Symbol}}=nothing, changed_vars::Vector{Symbol}=Symbol[])
    validate_required_args(model, data)
    validate_model_methods(model)

    # Scalar options
    validate_population_parameters(type, scale, backend, measure, vars)
    α = _validate_alpha(alpha)
    _validate_contrasts(contrasts)
    _validate_interval(interval, type)

    # `scenarios` and `groups` syntax, and the variables they name
    scenarios_nt = nothing
    if !isnothing(scenarios)
        _validate_scenarios_specific(scenarios, vars, type)
        scenarios_nt = _normalize_scenarios_to_namedtuple(scenarios)
        scenario_vars = collect(Symbol, keys(scenarios_nt))
    elseif !isnothing(scenario_vars) && !isnothing(vars) && type === :effects
        _validate_vars_scenarios_overlap(vars, scenario_vars)
    end
    isnothing(scenario_vars) || _validate_scenario_vars(model, scenario_vars, groups)
    if !isnothing(groups)
        _validate_groups_parameter(groups)
        # Teaching validation: check for vars/groups overlap (mirrors the scenarios check
        # in _validate_scenarios_specific). Only fires for explicit vars; auto-detected
        # vars are still skipped silently, and the empty-result guard in
        # _population_margins_with_contexts catches the case where nothing survives.
        if !isnothing(vars) && type === :effects
            _validate_vars_groups_overlap(vars, groups)
        end
    end

    # Single data conversion: a column table with Float64 numeric columns (type stability)
    data_nt = _prepare_margins_data(Tables.columntable(data), model)

    # Averaging weights (:model default on population requests), where they came from, and the
    # weight column (if a name) to exclude from auto-detected variables
    (weights_vec, weights_source, weight_col) = _resolve_weights(weights, model, data, data_nt)

    # Focal variables. Elasticity measures are only defined for continuous variables; filter or
    # error BEFORE the engine is built so the discrete variables never reach computation
    vars_explicit = !isnothing(vars)
    type === :effects && vars_explicit && _validate_focal_vars(model, vars)
    resolved_vars = if type === :effects
        _filter_vars_for_measure(_process_vars_parameter(model, vars, data_nt, weight_col),
                                 data_nt, measure, vars_explicit)
    else
        nothing  # Not needed for predictions
    end

    # Parsed contexts: each scenario combination and each subgroup is a full pass over the data
    isnothing(groups) || _validate_group_columns_complete(groups, data_nt)
    scenario_specs = _parse_scenario_specs(scenarios_nt)
    group_specs = isnothing(groups) ? [Dict()] : _parse_groups_specification(groups, data_nt; weights=weights_vec)
    _validate_context_count(length(scenario_specs), length(group_specs))

    # Profile requests: resolve, validate and complete the reference grid
    (grid, completed, grid_vars) = _prepare_reference_grid(model, data, data_nt, reference_grid)

    # Σ, the one evaluation of `vcov` for this call, and the df rule on the `vcov` given
    Σ = validate_vcov_parameter(vcov, model)
    df = _inference_df(model, vcov)

    changed = _changed_vars(scenario_vars, grid, changed_vars)
    # (Step 7: the derived-column detector runs here, once, over resolved_vars ∪ changed.)

    req = MarginsRequest{_request_kind(reference_grid), typeof(model)}(
        model, type, data_nt, resolved_vars, vars_explicit, Σ, vcov, df, α, weights_vec,
        weights_source, scenarios_nt, scenario_specs, groups, group_specs, grid, completed,
        grid_vars, changed, contrasts, measure, scale, backend, interval)
    _validate_request(model, req)
    return req
end

_request_kind(::Nothing) = :population
_request_kind(::DataFrame) = :profile

# Variables set counterfactually, in order of first appearance
function _changed_vars(scenario_vars, grid, extra::Vector{Symbol})
    out = Symbol[]
    isnothing(scenario_vars) || append!(out, scenario_vars)
    isnothing(grid) || append!(out, Symbol.(names(grid)))
    append!(out, extra)
    return unique!(out)
end

_parse_scenario_specs(::Nothing) = [Dict()]
_parse_scenario_specs(scenarios_nt::NamedTuple) = _parse_at_specification(scenarios_nt)

"""
    _with_scenarios(req, scenarios) -> MarginsRequest

`req` with its `scenarios` replaced (a `NamedTuple`, or `nothing`) and parsed again. Its
variables are not validated again: they must be among the `scenario_vars` `req` was normalized
with. The second differences validate those once and then run one computation per profile
point (and evaluation point).
"""
function _with_scenarios(req::MarginsRequest{K,M}, scenarios::Union{Nothing,NamedTuple}) where {K,M}
    specs = _parse_scenario_specs(scenarios)
    _validate_context_count(length(specs), length(req.group_specs))
    return MarginsRequest{K,M}(
        req.model, req.type, req.data_nt, req.vars, req.vars_explicit, req.Σ, req.vcov_source,
        req.df, req.alpha, req.weights, req.weights_source, scenarios, specs, req.groups,
        req.group_specs, req.reference_grid, req.completed_grid, req.grid_vars, req.changed_vars,
        req.contrasts, req.measure, req.scale, req.backend, req.interval)
end

"""
    _with_data(req, data_nt) -> MarginsRequest

`req` evaluated on other rows' values: `data_nt` has the same rows and columns as `req.data_nt`
with some values changed (the second differences' `at=:observed` shifts the modifier column).
Nothing is validated again; parsed groups keep their row membership.
"""
function _with_data(req::MarginsRequest{K,M}, data_nt::NamedTuple) where {K,M}
    return MarginsRequest{K,M}(
        req.model, req.type, data_nt, req.vars, req.vars_explicit, req.Σ, req.vcov_source,
        req.df, req.alpha, req.weights, req.weights_source, req.scenarios, req.scenario_specs,
        req.groups, req.group_specs, req.reference_grid, req.completed_grid, req.grid_vars,
        req.changed_vars, req.contrasts, req.measure, req.scale, req.backend, req.interval)
end

# ---- validators of single options ---------------------------------------------------------

"""
    _validate_alpha(alpha) -> Float64

The significance level as a `Float64`. Any `Real` in (0, 1) is accepted (`0.1`, `1//10`,
`0.1f0`); anything else is an `ArgumentError`. `alpha ≥ 1` used to be accepted and gave an
inverted confidence interval.
"""
function _validate_alpha(alpha)::Float64
    alpha isa Real || throw(ArgumentError("alpha must be a real number in (0, 1), got $(repr(alpha))"))
    a = Float64(alpha)
    0 < a < 1 || throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    return a
end

"""
    _validate_contrasts(contrasts::Symbol)

`contrasts` must be `:baseline` or `:pairwise`. Checked on every call, whether or not a
categorical variable is requested.
"""
function _validate_contrasts(contrasts::Symbol)
    contrasts ∉ (:baseline, :pairwise) &&
        throw(ArgumentError("contrasts must be :baseline or :pairwise, got :$(contrasts)"))
    return nothing
end

# Interval kind. Prediction intervals add observation-level residual variance and are only
# meaningful for predictions (a derivative of a mean has no "new observation" analogue), so
# they are rejected for type=:effects.
function _validate_interval(interval::Symbol, type::Symbol)
    if !(interval in (:confidence, :prediction))
        throw(ArgumentError("interval must be :confidence or :prediction, got $(repr(interval))"))
    end
    if interval === :prediction && type === :effects
        throw(ArgumentError("interval=:prediction is only valid with type=:predictions; " *
                            "marginal effects do not have prediction intervals."))
    end
    return nothing
end

"""
    _validate_scenario_vars(model, scenario_vars, groups)

Checks on the variables `scenarios` sets: each must be a predictor of the model, and none may
also be used by `groups`.
"""
function _validate_scenario_vars(model, scenario_vars, groups)
    _validate_scenarios_are_predictors(model, scenario_vars)
    # A variable cannot be binned by groups and overridden by scenarios at once: the
    # bins come from its observed values, the override then discards them.
    isnothing(groups) || _validate_scenarios_groups_conflict(scenario_vars, groups)
    return nothing
end

# Enforce the hard limit (error-first policy): every combination is a full pass over the data
function _validate_context_count(n_scenarios::Int, n_groups::Int)
    total_combinations = n_scenarios * n_groups
    if total_combinations > _MAX_CONTEXT_COMBINATIONS
        throw(MarginsError(
            "Too many scenario/group combinations: $(n_scenarios) scenarios × " *
            "$(n_groups) groups = $(total_combinations), above the limit of " *
            "$(_MAX_CONTEXT_COMBINATIONS). Use fewer scenario values or coarser groups, or " *
            "split the request into several calls."
        ))
    end
    return nothing
end

# ---- reference grid (profile requests) -----------------------------------------------------

_prepare_reference_grid(model, data, data_nt, ::Nothing) = (nothing, nothing, Set{Symbol}())

"""
    _prepare_reference_grid(model, data, data_nt, reference_grid) -> (grid, completed_grid, grid_vars)

Resolve string levels of categorical columns (`process_reference_grid`), validate the grid, and
complete it with typical values for the model variables it does not give. `grid_vars` are the
grid columns shown in the output: the user's model variables, without the response.
"""
function _prepare_reference_grid(model, data, data_nt::NamedTuple, reference_grid::DataFrame)
    # Process reference grid to convert string categorical specifications to proper categorical values
    # This enables users to write cartesian_grid(education=["High School", "College"]) naturally
    grid = process_reference_grid(data, reference_grid)

    # Reference grid specific validation (use processed grid for actual validation)
    if nrow(grid) == 0
        throw(ArgumentError("reference_grid cannot be empty"))
    end

    # Validate data types in reference grid - explicit error policy (use processed grid)
    for col_name in names(grid)
        col = grid[!, col_name]
        _validate_reference_grid_column_type(col, col_name)
    end

    if ncol(grid) == 0
        throw(ArgumentError("reference_grid must have at least one column"))
    end

    # Validate that reference grid variables exist in the data or model
    data_vars = Set(keys(data_nt))
    # Get model variables by compiling the formula and extracting variable references (using cache)
    try
        compiled = _shared_compiled(model, data_nt)
        model_vars = extract_variable_names(compiled)   # every data column the formula reads

        # Check reference grid variables against both data and model variables
        for col_name in names(grid)
            col_symbol = Symbol(col_name)
            if !(col_symbol in data_vars) && !(col_symbol in model_vars)
                throw(ArgumentError("Reference grid variable '$col_name' not found in model data. Available variables: $(sort(collect(data_vars)))"))
            end
        end
    catch e
        # If we can't compile the formula (which might happen if the reference grid is invalid),
        # just check against data variables for basic validation
        for col_name in names(grid)
            col_symbol = Symbol(col_name)
            if !(col_symbol in data_vars)
                throw(ArgumentError("Reference grid variable '$col_name' not found in data. Available variables: $(sort(collect(data_vars)))"))
            end
        end
    end

    # A grid column that is not a model predictor and VARIES across the grid (a typo that
    # happens to match a data column, the response, an unused column) changes nothing, so
    # every profile would silently report the same value. Constant extra columns are
    # harmless and common: means_grid(data) carries every data column.
    varying = [Symbol(c) for c in names(grid) if length(unique(grid[!, c])) > 1]
    bad_cols = _nonpredictor_vars(model, varying)
    if !isempty(bad_cols)
        throw(ArgumentError(
            "Reference grid variable(s) $(join(bad_cols, ", ")) vary across the grid but are " *
            "not predictors in the model, so they would not change any estimate. Check for " *
            "typos; model predictors are: $(join(_predictor_names(model), ", "))."
        ))
    end

    # Store original reference grid variable names for output filtering
    grid_vars = Set(Symbol.(names(grid)))

    # Attempt to remove response variable from display if it appears in the provided grid.
    # Be defensive: formulas with transformed responses (e.g., log(y)) may not expose `.sym`.
    try
        response_var = Symbol(model.mf.f.lhs.sym)
        delete!(grid_vars, response_var)
    catch
        # If we cannot determine a simple response symbol, do nothing.
        # Reference grids almost never include the response variable anyway.
    end

    # Complete reference grid with typical values for missing model variables
    # Note: complete_reference_grid now filters to model variables internally
    completed = complete_reference_grid(grid, model, data_nt)

    # Filter grid_vars to only model variables for proper output display
    # Get model variables from the completed grid to know what was actually kept
    grid_vars = intersect(grid_vars, Set(Symbol.(names(completed))))
    return (grid, completed, grid_vars)
end
