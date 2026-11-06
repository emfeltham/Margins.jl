# core/request.jl
# One request-normalization layer for every entry point (Margins 3.0, step 4).
#
# `population_margins`, `profile_margins` and the model-level `second_differences` turn their
# keywords into one `MarginsRequest` with `_normalize_request`, which runs every validator once
# and evaluates `vcov` once, and then compute from the request (`_run`). Nested computations
# (second differences run one AME per profile point or shift) derive requests from it with
# `_with_scenarios` instead of re-entering keyword parsing.

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
| `Σ` | the covariance matrix, the one evaluation of `vcov` for the call, with the rows and columns of dropped coefficients set to 0 |
| `dropped` | the coefficients the fit dropped as collinear (empty for a full-rank fit; see `_vcov_and_dropped`) |
| `vcov_source` | the `vcov` the caller passed (provenance) |
| `df` | inference degrees of freedom: the `df` keyword, else `_inference_df(model)` |
| `alpha` | significance level, in (0, 1) |
| `weights`, `weights_source` | averaging weights (`nothing` or one per row) and where they came from (`:model`, `:user`, `:none`) |
| `offset` | the η offset (`nothing`, or a `Vector{Float64}`): population requests, one per row of `data_nt` (the model's stored offset); profile requests, one per row of `completed_grid`. `nothing` also when the model has an offset that cannot change the requested quantity (`_offset_matters`) |
| `scenarios` | the `scenarios` keyword as a `NamedTuple` (`nothing` if none) |
| `scenario_specs` | its parsed combinations, one `Dict` per scenario (`[Dict()]` if none) |
| `groups` | the validated `groups` syntax (`nothing` if none) |
| `group_specs` | its parsed subgroups (`[Dict()]` if none) |
| `reference_grid` | profile requests: the user's grid with string levels resolved (`at_spec`) |
| `completed_grid` | profile requests: that grid completed with typical values |
| `grid_vars` | profile requests: the grid columns shown in the output |
| `changed_vars` | every variable the request sets counterfactually: scenario keys, grid columns, and the second-differences modifier and `profile` variables |
| `contrasts`, `measure`, `scale`, `backend`, `interval` | validated options |
| `random_effects` | mixed models: `nothing` for `random_effects = :zero` (every random effect 0), else a `RandomEffectsSpec`; with `:conditional` the predicted random effects z'b̂ are in `offset`, with `:marginal` its `average` is the engine's `re_average` |

`vars ∪ changed_vars` is the set the derived-column detector (`_warn_derived_columns`) runs over.

`data_nt` holds the model's estimation rows only (`_estimation_rows`; FixedEffectModels drops
rows), so weights, offsets and groups all refer to those rows.
"""
struct MarginsRequest{K,M}
    model::M
    type::Symbol
    data_nt::NamedTuple
    vars::Union{Nothing,Vector{Symbol}}
    vars_explicit::Bool
    Σ::Matrix{Float64}
    dropped::Vector{Int}
    vcov_source::Any
    df::Float64
    alpha::Float64
    weights::Union{Nothing,Vector{Float64}}
    weights_source::Symbol
    offset::AnyOffset
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
    random_effects::Union{Nothing,RandomEffectsSpec}
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
                       interval=:confidence, df=nothing, offset=nothing, scenario_vars=nothing,
                       changed_vars=Symbol[], random_effects=:zero, re_new=:zero, re_terms=nothing,
                       re_nodes=:auto, re_se=:auto, warn=true) -> MarginsRequest

Validate every option and prepare everything a computation needs, once per call: the data
(any Tables.jl source, subset to the model's estimation rows), the focal variables, Σ (`vcov` is
evaluated here and nowhere else), the inference df (`df`, else the model class's), the averaging
weights, the η offset, the parsed scenarios and groups and, with a `reference_grid`, the
completed grid. A `reference_grid` makes a `:profile` request, otherwise it is `:population`.
`offset` (profile requests only) gives the offset at the grid rows; see `_profile_offset`.

Two keywords serve callers that run several computations from one request (the model-level
`second_differences`): `scenario_vars` names the variables later requests set with
`_with_scenarios` (one per profile point), so they are validated here as `scenarios` keys
would be, once; `changed_vars` adds variables set counterfactually by other means (the
modifier shifted by `at=:observed`).

`warn=true` runs the derived-column detector (`_warn_derived_columns`) once, over the focal
variables and every changed variable (`changed_vars` of the request).

The validators run in a fixed order: the required arguments and the scalar options (`type`,
`scale`, `backend`, `measure`, `alpha`, `contrasts`, `interval`), then the `scenarios` and
`groups` syntax and their conflicts, the data, the weights, the variables (and the measure
filter, preceded by the model class's `_validate_focal_vars`), the group columns, the reference
grid and its offset, `vcov`, and last the model class (`_validate_request`).
"""
function _normalize_request(model, data;
        type::Symbol=:effects, vars=nothing, scale::Symbol=:response, backend::Symbol=:ad,
        scenarios=nothing, groups=nothing, reference_grid::Union{Nothing,DataFrame}=nothing,
        measure::Symbol=:effect, contrasts::Symbol=:baseline, alpha=0.05, vcov=GLM.vcov,
        weights=:none, interval::Symbol=:confidence, df=nothing, offset=nothing,
        scenario_vars::Union{Nothing,Vector{Symbol}}=nothing, changed_vars::Vector{Symbol}=Symbol[],
        random_effects=:zero, re_new=:zero, re_terms=nothing, re_nodes=:auto, re_se=:auto,
        warn::Bool=true)
    validate_required_args(model, data)
    validate_model_methods(model)
    # The formula's variables, walked once per call and passed to every check that needs them
    # (on a 55-term formula a walk costs about 6 μs and 32 KB, and a call needed up to five)
    fvars = _get_model_formula_variables(model)

    # Scalar options
    validate_population_parameters(type, scale, backend, measure, vars)
    α = _validate_alpha(alpha)
    df_override = isnothing(df) ? nothing : _validate_df(df)
    isnothing(offset) || isnothing(reference_grid) || offset isa Real || offset isa AbstractVector ||
        throw(ArgumentError("offset must be a number or a vector with one entry per reference-grid row"))
    isnothing(offset) || !isnothing(reference_grid) || throw(ArgumentError(
        "offset= is a profile_margins keyword (the offset at the reference-grid rows); population " *
        "margins use the model's own offset at each data row"))
    _validate_contrasts(contrasts)
    _validate_interval(interval, type)
    re_spec = _validate_random_effects(model, random_effects, re_new, re_terms, re_nodes, re_se)
    # The mixed model's wrapper: its formula adds the random-effects columns, so every computation
    # below evaluates z from the compiled row. `:conditional`: z'b̂ as coefficients (over hidden
    # grouping columns added to the data), the parameter space (β, b); `:marginal` with random
    # slopes: z's columns with coefficient 0 (for τ, see `RowSpreadOffset`). Its covariance already
    # holds Σ_β (from `vcov` unless `re_se = :joint`), so `vcov` is the wrapper's own from here on
    if _is_wrapped(re_spec)
        raw = Tables.columntable(data)
        # Checked on the mixed model itself: the wrapper's formula names no grouping columns, so
        # the data checks below would not see a missing grouping value (a row the fit dropped)
        _check_model_vars_not_missing(raw, model)
        # A population request evaluates the conditioned factors at every data row
        isnothing(reference_grid) && _conditions(re_spec) &&
            _check_re_grouping_present(model, raw, _conditioned(re_spec))
        (model, data) = _re_wrapper_model(model, raw, re_spec, vcov)
        fvars = _get_model_formula_variables(model)
        vcov = GLM.vcov
    elseif !isnothing(re_spec) && re_spec.mode === :conditional
        # Above the dummy-column limit, keep z'b in the offset. A slope variable absent
        # from X still needs a zero-variance column for derivative and grid machinery.
        original = model
        (model, data) = _re_offset_model(model, Tables.columntable(data), re_spec, vcov)
        if model !== original
            raw = Tables.columntable(data)
            _check_model_vars_not_missing(raw, original)
            isnothing(reference_grid) && _check_re_grouping_present(original, raw, re_spec.factors)
            fvars = _get_model_formula_variables(model)
            vcov = GLM.vcov
        end
    end

    # `scenarios` and `groups` syntax, and the variables they name
    scenarios_nt = nothing
    if !isnothing(scenarios)
        _validate_scenarios_specific(scenarios, vars, type)
        scenarios_nt = _normalize_scenarios_to_namedtuple(scenarios)
        scenario_vars = collect(Symbol, keys(scenarios_nt))
    elseif !isnothing(scenario_vars) && !isnothing(vars) && type === :effects
        _validate_vars_scenarios_overlap(vars, scenario_vars)
    end
    isnothing(scenario_vars) || _validate_scenario_vars(model, scenario_vars, groups, fvars)
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

    # The model's estimation rows only (FixedEffectModels' `esample`): everything below reads
    # this subset, so weights, offsets and groups refer to the rows the model was fit to
    cols = Tables.columntable(data)
    n_full = length(first(cols))
    est_rows = _estimation_rows(model, n_full)
    if !isnothing(est_rows)
        cols = map(c -> c[est_rows], cols)
        data = cols
    end

    # Single data conversion: a column table with Float64 numeric columns (type stability)
    data_nt = _prepare_margins_data(cols, model, fvars)

    # Averaging weights (:model default on population requests), where they came from, and the
    # weight column (if a name) to exclude from auto-detected variables
    (weights_vec, weights_source, weight_col) = _resolve_weights(weights, model, data, data_nt;
                                                                 rows=est_rows, n_full)

    # Focal variables. Elasticity measures are only defined for continuous variables; filter or
    # error BEFORE the engine is built so the discrete variables never reach computation
    vars_explicit = !isnothing(vars)
    type === :effects && vars_explicit && _validate_focal_vars(model, vars)
    resolved_vars = if type === :effects
        _filter_vars_for_measure(_process_vars_parameter(model, vars, data_nt, weight_col, fvars),
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
    # (an `offset` column of the grid is the offset at its rows, not a variable: split it off)
    (grid_in, grid_offset) = _split_offset_column(model, reference_grid)
    (grid, completed, grid_vars, fill_source) = _prepare_reference_grid(model, data, data_nt, grid_in, fvars;
                                                                        extra_vars=_re_grid_vars(model, re_spec),
                                                                        re_new)
    # A profile request averages nothing; its weights are those of the values it fills in for model
    # variables the grid leaves out (the model's estimation weights by default)
    isnothing(completed) || (weights_source = fill_source)

    # The η offset at the rows the request evaluates, when the model has one that matters
    offset_vec = if _has_offset(model) && _offset_matters(_link(model), type, scale, measure)
        isnothing(completed) ? _offset(model, data_nt) :
            _profile_offset(model, grid, nrow(completed), offset, grid_offset)
    else
        isnothing(completed) || _check_unused_offset(model, offset, grid_offset)
        nothing
    end
    # Mixed models with random_effects = :conditional: the predicted random effects z'b̂ join the
    # offset (when it can change the requested quantity, as for any offset)
    re_tbl = nothing
    if _is_wrapped(re_spec) && _conditions(re_spec)
        isnothing(completed) || (re_spec = _with_used(re_spec, _joint_used(model, grid)))
    elseif !isnothing(re_spec) && re_spec.mode === :conditional
        if isnothing(completed)
            (re_tbl, used) = (data_nt, re_spec.factors)
        else
            (re_tbl, used) = _profile_re_table(model, completed, grid, re_spec)
        end
        re_spec = _with_used(re_spec, used)
        if !isempty(used) && _offset_matters(_link(model), type, scale, measure)
            re_off = _re_offset(model, re_tbl, re_spec)
            offset_vec = isnothing(offset_vec) ? re_off : offset_vec .+ re_off
        end
    end
    # random_effects = :marginal: the quadrature, checked at the largest SD of z'b the request
    # evaluates; with random slopes each row's SD comes from its model row (`RowSpreadOffset`)
    if _integrates(re_spec)
        re_tbl = isnothing(completed) ? data_nt : Tables.columntable(completed)
        settings = _re_has_slopes(model, _integrated(re_spec)) ?
            _marginal_settings(model, re_spec, scenario_specs, resolved_vars, re_tbl, data_nt) : ()
        (re_spec, offset_vec) = _finalize_marginal(model, re_spec, re_tbl, settings, offset_vec)
    end

    # Σ, the one evaluation of `vcov` for this call (dropped coefficients' rows and columns zeroed);
    # df from the model class unless given
    (Σ, dropped) = _vcov_and_dropped(vcov, model)
    isempty(dropped) || _warn_dropped_coefficients(model, dropped)
    df_value = isnothing(df_override) ? _inference_df(model) : df_override

    changed = _changed_vars(scenario_vars, grid, changed_vars)
    isnothing(re_spec) || _validate_random_effects_request(model, re_spec, type, resolved_vars,
                                                           vars_explicit, changed, grid, data_nt)
    # The effect of a variable with a random slope: its per-row slope terms (`SlopedOffset`)
    if type === :effects && !isnothing(re_spec) && !isnothing(resolved_vars)
        slope_vars = intersect(resolved_vars, _re_active_slope_vars(model, re_spec))
        if !isempty(slope_vars)
            doff = _re_slope_derivatives(model, re_tbl, re_spec, slope_vars)
            offset_vec = SlopedOffset(offset_vec, slope_vars, doff)
        end
    end
    # The derived-column detector, once per call, over the focal and the changed variables. The
    # values the call sets are known here for `scenarios` and a reference grid; the model-level
    # second differences set theirs later (`scenario_vars`), so they are not known (`nothing`).
    # It is called without inference (`invokelatest`), as the contexts path is, so that its code is
    # compiled only when a call changes a variable, not into every entry point (first-call latency).
    if warn && !(isempty(changed) && (isnothing(resolved_vars) || isempty(resolved_vars)))
        settings = !isnothing(completed) ? completed : !isnothing(scenarios_nt) ? scenario_specs : nothing
        Base.invokelatest(_warn_derived_columns, model, data_nt,
                          isnothing(resolved_vars) ? Symbol[] : resolved_vars, changed, settings, fvars)
    end

    req = MarginsRequest{_request_kind(reference_grid), typeof(model)}(
        model, type, data_nt, resolved_vars, vars_explicit, Σ, dropped, vcov, df_value, α, weights_vec,
        weights_source, offset_vec, scenarios_nt, scenario_specs, groups, group_specs, grid, completed,
        grid_vars, changed, contrasts, measure, scale, backend, interval, re_spec)
    _validate_request(model, req)
    return req
end

# One warning per call on a rank-deficient fit (decision for ISSUES #10: R's behaviour, with a NaN
# standard error for quantities that are not identified)
function _warn_dropped_coefficients(model, dropped::Vector{Int})
    @warn "The model is rank deficient: the fit dropped coefficient(s) $(_coef_list(model, dropped)) " *
          "as collinear (reported as 0 with NaN variance). Margins treats them as fixed at 0, as R " *
          "does: their rows and columns of the covariance matrix are set to 0 (result.inference.vcov), " *
          "and estimates and standard errors that depend on them are computed with those coefficients " *
          "at 0. A quantity that depends only on dropped coefficients (e.g. the effect of a variable " *
          "whose every coefficient was dropped) is not identified: its estimate is 0 and its standard " *
          "error NaN. Refit without the redundant term(s) to remove this warning."
    return nothing
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
variables are not validated again: they must be among the `scenario_vars` (or, for a `Shift`,
the `changed_vars`) `req` was normalized with. The second differences validate those once and
then run one computation per profile point (and evaluation point). A value may be a
`Counterfactual` (`at=:observed` passes the modifier as `[Shift(-δ), Shift(δ)]`).
"""
function _with_scenarios(req::MarginsRequest{K,M}, scenarios::Union{Nothing,NamedTuple}) where {K,M}
    specs = _parse_scenario_specs(scenarios)
    _validate_context_count(length(specs), length(req.group_specs))
    (re_spec, offset) = isnothing(req.random_effects) ? (nothing, req.offset) :
        _marginal_with_scenarios(req.model, req.random_effects, req.offset, specs, req.vars, req.data_nt)
    return MarginsRequest{K,M}(
        req.model, req.type, req.data_nt, req.vars, req.vars_explicit, req.Σ, req.dropped, req.vcov_source,
        req.df, req.alpha, req.weights, req.weights_source, offset, scenarios, specs, req.groups,
        req.group_specs, req.reference_grid, req.completed_grid, req.grid_vars, req.changed_vars,
        req.contrasts, req.measure, req.scale, req.backend, req.interval, re_spec)
end

"""
    _with_group_specs(req, group_specs) -> MarginsRequest

`req` restricted to the given parsed subgroups (entries of `req.group_specs`). The `groups`
syntax is kept, so the result's context columns are named as for the full request. The second
differences use it to compute one stratum at a time when the evaluation point depends on the
stratum (`at=:mean` / `:median` with `groups=`).
"""
function _with_group_specs(req::MarginsRequest{K,M}, group_specs::Vector) where {K,M}
    return MarginsRequest{K,M}(
        req.model, req.type, req.data_nt, req.vars, req.vars_explicit, req.Σ, req.dropped, req.vcov_source,
        req.df, req.alpha, req.weights, req.weights_source, req.offset, req.scenarios,
        req.scenario_specs, req.groups, group_specs, req.reference_grid, req.completed_grid,
        req.grid_vars, req.changed_vars, req.contrasts, req.measure, req.scale, req.backend,
        req.interval, req.random_effects)
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
function _validate_scenario_vars(model, scenario_vars, groups, fvars=_get_model_formula_variables(model))
    _validate_scenarios_are_predictors(model, scenario_vars, fvars)
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

_prepare_reference_grid(model, data, data_nt, ::Nothing, fvars=nothing; extra_vars=Symbol[], re_new=:zero) = (nothing, nothing, Set{Symbol}(), :none)

"""
    _prepare_reference_grid(model, data, data_nt, reference_grid) -> (grid, completed_grid, grid_vars, fill_source)

Resolve string levels of categorical columns (`process_reference_grid`), validate the grid, and
complete it with typical values for the model variables it does not give. `grid_vars` are the
grid columns shown in the output: the user's model variables, without the response.
"""
function _prepare_reference_grid(model, data, data_nt::NamedTuple, reference_grid::DataFrame,
                                 fvars=_get_model_formula_variables(model); extra_vars=Symbol[], re_new=:zero)
    # Process reference grid to convert string categorical specifications to proper categorical values
    # This enables users to write cartesian_grid(education=["High School", "College"]) naturally
    grid = process_reference_grid(data, reference_grid)
    # (the hidden grouping columns of the joint random-effects wrapper, from the grid's grouping
    # columns or the zero level)
    grid = _augment_grid(model, grid, re_new)

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
    # (`extra_vars`: other columns the request reads at the grid rows, the grouping columns of
    # `random_effects = :conditional`)
    bad_cols = setdiff(_nonpredictor_vars(model, varying, fvars), extra_vars)
    if !isempty(bad_cols)
        throw(ArgumentError(
            "Reference grid variable(s) $(join(bad_cols, ", ")) vary across the grid but are " *
            "not predictors in the model, so they would not change any estimate. Check for " *
            "typos; model predictors are: $(join(_predictor_names(model), ", "))."
        ))
    end

    # Store original reference grid variable names for output filtering
    grid_vars = setdiff!(Set(Symbol.(names(grid))), _hidden_columns(model))

    # Attempt to remove response variable from display if it appears in the provided grid.
    # Be defensive: formulas with transformed responses (e.g., log(y)) may not expose `.sym`.
    response_var = _response_name(model)
    isnothing(response_var) || delete!(grid_vars, response_var)

    # Complete reference grid with typical values for missing model variables
    # Note: complete_reference_grid now filters to model variables internally
    # (filled-in values weighted by the model's estimation weights, as `complete_reference_grid`'s
    # default; `fill_source` records which weights they used)
    # (`data_nt` already holds the model's estimation rows)
    (completed, fill_source) = _complete_reference_grid(grid, model, data_nt; subset=false)
    # Predictors outside the formula Margins evaluates (FixedEffectModels' absorbed fixed effects,
    # which a grid may set to choose their estimates, and the grouping columns of
    # `random_effects = :conditional`) are kept, so they label the output rows
    if nrow(completed) == nrow(grid)
        formula_vars = Set(_formula_predictor_names(model, fvars))
        predictors = _predictor_names(model, fvars)
        for c in names(grid)
            s = Symbol(c)
            if !(c in names(completed)) && !(s in formula_vars) && (s in predictors || s in extra_vars)
                completed[!, c] = grid[!, c]
            end
        end
    end

    # Filter grid_vars to only model variables for proper output display
    # Get model variables from the completed grid to know what was actually kept
    grid_vars = intersect(grid_vars, Set(Symbol.(names(completed))))
    return (grid, completed, grid_vars, fill_source)
end

# ---- offsets at reference-grid rows (decision D3) -------------------------------------------

# The grid column that gives the offset at each grid row. Only for a model with an offset, and
# only when `offset` is not itself a predictor of the model.
_is_offset_column(model, grid::DataFrame) =
    _has_offset(model) && "offset" in names(grid) && !(:offset in _predictor_names(model))

"""
    _split_offset_column(model, reference_grid) -> (grid, offset_column)

The reference grid without its `offset` column, and that column (`nothing` if there is none).
"""
_split_offset_column(model, ::Nothing) = (nothing, nothing)
function _split_offset_column(model, grid::DataFrame)
    _is_offset_column(model, grid) || return (grid, nothing)
    return (select(grid, Not("offset")), grid[!, "offset"])
end

"""
    _profile_offset(model, grid, n, offset_kw, offset_col) -> Vector{Float64}

The η offset at each of the `n` reference-grid rows of a profile request on a model with an
offset (decision D3 of the 3.0 plan; GLM's `predict(m, newdata; offset=)` convention):

1. the `offset=` keyword (a number for every row, or one value per row);
2. else an `offset` column in the grid;
3. else what the model class derives from the grid (`_grid_offset`; FixedEffectModels looks up
   the estimated fixed effects of the levels the grid sets);
4. else an `ArgumentError`. The offset is never silently 0: for a Poisson exposure model that
   would predict the rate per unit exposure.
"""
function _profile_offset(model, grid::DataFrame, n::Int, offset_kw, offset_col)
    !isnothing(offset_kw) && !isnothing(offset_col) && throw(ArgumentError(
        "The offset is given twice, as offset= and as an `offset` column of the reference grid; give one"))
    raw = !isnothing(offset_kw) ? offset_kw : offset_col
    if isnothing(raw)
        derived = _grid_offset(model, grid)
        isnothing(derived) || return derived
        throw(ArgumentError(
            "The model has an offset (η = x'β + offset), and the requested quantity depends on it, so " *
            "profile_margins needs the offset at each reference-grid row: pass offset=<number or one " *
            "value per grid row> (as in GLM's predict(m, newdata; offset=)), or add an `offset` column to " *
            "the grid. For a log-exposure offset, offset=log(exposure); offset=0 gives the rate per unit " *
            "of exposure."))
    end
    v = raw isa Real ? fill(Float64(raw), n) : raw
    length(v) == n || throw(ArgumentError(
        "offset has $(length(v)) values but the reference grid has $n rows"))
    any(ismissing, v) && throw(ArgumentError("offset contains missing values"))
    out = Vector{Float64}(v)
    all(isfinite, out) || throw(ArgumentError("offset contains NaN or Inf values"))
    return out
end

# An offset given for a quantity it cannot change is accepted (and unused), but an offset given
# to a model without one is a mistake
function _check_unused_offset(model, offset_kw, offset_col)
    if !isnothing(offset_kw) && !_has_offset(model)
        throw(ArgumentError("offset= was given, but the model has no offset"))
    end
    return nothing
end

"""
    _estimation_subset(model, cols, weights) -> (cols, weights)

`cols` restricted to the model's estimation rows (`_estimation_rows`), and a weight vector given
for every row of `cols` restricted the same way. For callers that read the data themselves
before building a request (the model-level `second_differences`); the request is still built
from the data as given (`_normalize_request` subsets it the same way).
"""
function _estimation_subset(model, cols::NamedTuple, weights)
    n = length(first(cols))
    rows = _estimation_rows(model, n)
    isnothing(rows) && return (cols, weights)
    weights isa AbstractVector && length(weights) == n && (weights = weights[rows])
    return (map(c -> c[rows], cols), weights)
end
