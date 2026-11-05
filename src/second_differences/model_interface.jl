# model_interface.jl
# One entry point for second differences: second_differences(model, data, variables, modifier; ...)
#
# It builds the scenarios, computes the AMEs and applies the delta method itself, so the user
# does not assemble scenarios, run population_margins and pass a covariance matrix by hand.
# The result-based methods (contrasts.jl) remain for users who already hold an EffectsResult.

# Levels of a discrete modifier in data order: CategoricalArray levels (present ones) in level
# order, otherwise the sorted unique values (false < true, 0 < 1, "a" < "b").
function _discrete_levels(col)
    vals = collect(skipmissing(col))
    if col isa CategoricalArray
        present = Set(CategoricalArrays.unwrap.(vals))
        return [l for l in levels(col) if l in present]
    end
    return sort(unique(vals))
end

# :continuous for numeric modifiers with more than two distinct values, :categorical otherwise
# (Bool, 0/1, two-valued, strings, categorical arrays)
function _detect_modifier_type(col)::Symbol
    vals = collect(skipmissing(col))
    (all(x -> x isa Number, vals) && !(eltype(vals) <: Bool) && length(unique(vals)) > 2) ?
        :continuous : :categorical
end

# A numeric modifier with few distinct integer values is more often a coded category (1/2/3)
# than a continuous variable; say how it was treated and how to change that.
const INTEGER_CODED_MAX_LEVELS = 5

function _warn_if_integer_coded(col, modifier::Symbol)
    vals = collect(skipmissing(col))
    all(x -> x isa Integer || (x isa AbstractFloat && isinteger(x)), vals) || return nothing
    lv = sort(unique(vals))
    length(lv) <= INTEGER_CODED_MAX_LEVELS || return nothing
    @warn "Modifier :$modifier takes only $(length(lv)) integer values ($(join(lv, ", "))) and is treated as " *
          "continuous, so the result is the average derivative ∂AME/∂$modifier over the observed values. If these are " *
          "category codes, pass modifier_type=:categorical to compare levels; to keep the derivative " *
          "and silence this warning, pass `at=` (e.g. at=:observed)."
    return nothing
end

# Does the fitted model's formula mention `modifier`? `nothing` if that cannot be determined.
function _modifier_in_model(model, modifier::Symbol)
    try
        return modifier in StatsModels.termvars(StatsModels.formula(model))
    catch
        return nothing
    end
end

"""
    second_differences(model, data, variables, modifier;
                       levels=nothing, at=nothing, modifier_type=:auto, profile=NamedTuple(), groups=nothing,
                       delta=:auto, scale=:response, measure=:effect, contrasts=:baseline, contrast=nothing,
                       vcov=GLM.vcov, df=nothing, weights=:model, alpha=0.05, p_adjust=:none,
                       include_gradients=false, warn=true, backend=:ad, random_effects=:zero,
                       re_new=:zero, re_terms=nothing, re_nodes=:auto, re_se=:auto)

How does the marginal effect of `variables` change with `modifier`? Computes the second
difference (interaction effect on the response scale) with delta-method standard errors
directly from a fitted model; no scenarios or covariance matrix to assemble.

What is computed depends on the modifier:

- **Discrete modifier** (Bool, 0/1, strings, categorical, or numeric with ≤ 2 values):
  AME(`variables`) at every pair of modifier levels, and the difference between them.
- **Continuous modifier** (numeric with more than 2 distinct values):
  the derivative ∂AME/∂modifier by symmetric finite differences, averaged over the observed
  values (`at=:observed`, the default) or taken at the points `at` gives; or, with `levels=[…]`, the slope between those modifier values.

# Arguments
- `model`, `data`: fitted model and the data it was fit to (a `DataFrame` or any Tables.jl table).
- `variables::Union{Symbol,Vector{Symbol}}`: focal variable(s); each continuous variable gives
  one row, each categorical variable one row per contrast (see `contrasts`).
- `modifier::Symbol`: the moderating variable (must not also be in `variables`).

# Keyword Arguments
- `levels`: modifier values to compare. Default for a discrete modifier: all its levels.
  For a continuous modifier, giving `levels` returns slopes `(AME₂ − AME₁)/(m₂ − m₁)` for every
  pair of those values instead of a local derivative.
- `at`: continuous modifier only; where to take the derivative. `:observed` (default) is the **average
  derivative over the observed values**: the modifier is shifted by ±δ for every row (each row keeps its
  own other values) rather than set to one value for everyone, so the result is the mean over observations
  of ∂²ŷᵢ/∂x∂m at each row's own (xᵢ, mᵢ, …), the analogue of an AME for the interaction (Ai & Norton's
  average interaction effect). It needs no evaluation point (`eval_point` and the levels are `NaN`). The
  alternatives set the modifier to a value for everyone: `:mean`, `:median`, a number, a vector of
  numbers, or a `NamedTuple` that includes the modifier together with other variables to hold fixed. They
  differ from `:observed` in a nonlinear model, where the derivative changes along the modifier. Cannot be
  combined with `levels`.
- `modifier_type`: `:auto` (default; decided from the data as above), `:categorical` or
  `:continuous`. Use it to treat an integer-coded categorical modifier as categorical (a numeric
  modifier with at most 5 distinct integer values is treated as continuous with a warning, unless
  `at` or `levels` is given).
- `profile::NamedTuple`: other variables to hold fixed, e.g. `(age=40, region="north")`; vector
  values give one profile point each (Cartesian product), reported as extra columns. These are
  counterfactuals (everyone's value is set), not subgroups; for strata use `groups`. Must not contain
  the modifier.
- `groups`: strata, with the syntax of `population_margins` (`:g`, `[:g, :h]`, `(:x, 4)` quartiles,
  `(:x, [cuts…])` thresholds, `:g => :h` nested). The second difference is computed **within each
  stratum**: the same counterfactual, averaged over that stratum's rows only, so each stratum's row equals
  the ungrouped call on that stratum's data with the same `vcov`. Strata are blocks of rows with
  `over_<var>` columns (in the order of `groups`). With `at=:mean` / `:median` the evaluation point is the
  stratum's own (weighted) mean / median; a number, a vector or an `at` NamedTuple is the same point in
  every stratum; `at=:observed` averages the cross-derivative over the stratum's rows; `levels` and
  `profile` apply in every stratum. δ is the call's (from all rows). The modifier, a `profile` variable and
  a held variable of `at` cannot be in `groups` (a stratum is selected by observed values the calculation
  overwrites), nor can a focal variable unless binned. To compare the effect across the *observed*
  subgroups of the modifier (different rows, a descriptive comparison) use the result-level form:
  `second_differences(population_margins(model, data; vars=…, groups=modifier), variables, modifier)`.
  All strata share one Σ, so the covariance of two strata's second differences is g₁'Σg₂
  (`include_gradients=true`).
- `delta`: finite-difference step for a local derivative (`:auto` = 1% of the modifier's SD).
- `measure`: `:effect` (default), `:elasticity`, `:semielasticity_dyex` or `:semielasticity_eydx`, as in
  `population_margins`: the second difference is then the change in that measure across the modifier.
  The measures are defined for continuous focal variables only, so request categorical ones separately.
- `scale`: `:response` (default) or `:link`. `backend`: `:ad` (default) or `:fd`, as in
  `population_margins`. `contrasts`: how contrasts of a categorical focal variable
  are built, `:baseline` (K−1) or `:pairwise` (K(K−1)/2), as in `population_margins`.
- `contrast`: restrict the result to one contrast label, such as `"college - hs"` (or `"dy/dx"` for a
  continuous variable). `nothing` (default) keeps every contrast. It applies to every focal variable, so
  each must have it, and it must exist under the chosen `contrasts`. Same keyword as the result-level
  form; `contrasts` (plural) builds the contrasts, `contrast` selects one.
- `vcov`: covariance estimator, as in `population_margins` (default `GLM.vcov`; a function, a
  matrix, or a CovarianceMatrices estimator). The t/z choice follows the same rule as the rest of
  the package, whatever `vcov` is: t with the residual df for a linear model (and
  FixedEffectModels), z otherwise.
- `df`: degrees of freedom of the t reference distribution, overriding that rule (`Inf` for z), as
  in `population_margins`.
- `random_effects`, `re_new`, `re_terms`, `re_nodes`, `re_se`: mixed models only, as in `population_margins`. With
  `random_effects=:conditional` and `re_se=:fixed` above 2000 random-effect coefficients, the
  modifier must not have a random slope in a conditioned factor (it is set counterfactually).
  With `:marginal` the quadrature rule is checked at the SD of z'b of the modifier's evaluation
  points too.
- `weights`: observation weights, as in `population_margins`: `:model` (default; the fit's estimation
  weights if it has any, otherwise none), `:none` (or `nothing`) for unweighted, a column name or a vector.
  Every average marginal effect is then weighted, and the evaluation point `at=:mean` / `:median` of a
  continuous modifier is the weighted mean / median. The source (`:model`, `:user` or `:none`) is recorded
  as table metadata: `metadata(result, "weights_source")`. The standard errors come from `vcov`, as in
  `population_margins` (not design-based).
- `alpha`: as in `population_margins`; `ci_lower`/`ci_upper` give the (1 − `alpha`) interval
  (default 0.05, a 95% interval) and `significant` is `p_value < alpha`.
- `p_adjust`: multiple-testing adjustment over **all rows returned** (every variable, contrast, level
  pair, evaluation point, `profile` point and `groups` stratum): `:none` (default), `:bonferroni`, `:holm` (family-wise
  error rate) or `:bh` (Benjamini–Hochberg false discovery rate). Adds a `p_adjusted` column after
  `p_value` and bases `significant` on it. Rows that are zero by construction (see `warn`) are not
  tests and are left out of the family.
- `include_gradients`: add a `gradient` column (the vector ∂`second_diff`/∂β for each row, for custom
  variance calculations). Omitted by default, as in `DataFrame(result)`, so the table has only scalar columns.
- `warn`: `true` (default) warns, once per call, about (1) second differences that are **zero by
  construction**: the model cannot make the effect depend on the modifier (no interaction term in an
  identity-link model, or the modifier is not in the model). Those rows have `NaN` `statistic`,
  `p_value`, `ci_lower` and `ci_upper`, `significant = false`, and are left out of the `p_adjust` family.
  In a nonlinear model the effect can vary with the modifier through the link without an interaction
  term, so on the response scale such rows are reported normally (with `scale=:link` the model is
  additive, so they are flagged). And (2) modifier or `profile` values outside the observed
  range of that variable, where the result extrapolates the fitted model. And (3) model columns that
  look like precomputed transforms of the modifier, a `profile` variable or a focal variable (`m2`
  holding `m.^2`, `log_m`, a standardized copy, a precomputed product `x .* m`, or a column constant
  within each value of a discrete variable): changing the variable re-evaluates the formula, so such a
  column is held fixed rather than following it. Write the transform in the formula instead (`m^2`,
  `log(m)`, `x & m`); see `population_margins`, *Derived columns*. `warn=false` silences all three.

# Returns
A `DataFrame` with the columns listed under "Output Schema" in the second-differences
documentation (`variable, contrast, modifier, modifier_type, modifier_level1, modifier_level2,
ame_at_level1, ame_at_level2, second_diff, se, statistic, df, p_value, ci_lower, ci_upper, significant`);
a local derivative adds `eval_point` and `delta_used`, `groups` adds `over_<var>` columns, and `profile`
variables add one column each.
For a discrete or continuous-`levels` comparison, `second_diff` is `AME(level2) − AME(level1)` (or
that difference per unit of the modifier for a continuous modifier), with numeric levels ascending.

Every invalid input raises an `ArgumentError` (as in `population_margins`). Everything shared with
`population_margins` is validated, and `vcov` evaluated, once per call (one request).

# Examples
```julia
m = glm(@formula(y ~ age * treated + income), df, Binomial(), LogitLink())

second_differences(m, df, :age, :treated)              # effect of age: treated vs not
second_differences(m, df, [:age, :income], :treated)   # several focal variables
second_differences(m, df, :age, :income)               # average ∂AME(age)/∂income over the observed incomes
second_differences(m, df, :age, :income; at=[20_000, 50_000])
second_differences(m, df, :age, :income; levels=[20_000, 50_000])   # slope between two values
second_differences(m, df, :age, :treated; profile=(female=[0, 1],)) # everyone's `female` set to 0, then 1
second_differences(m, df, :age, :treated; groups=:region)          # the same comparison within each region
second_differences(m, df, :age, :treated; vcov=vcov(m))
second_differences(m, df, :age, :education; p_adjust=:holm, alpha=0.1)    # adjusted tests, 90% intervals
```

For an `EffectsResult` you have already computed, `second_differences(result, variables, modifier)`
does the same comparison on its rows (see its docstring).
"""
function second_differences(
    model::RegressionModel, data, variables::Union{Symbol,Vector{Symbol}}, modifier::Symbol;
    levels=nothing, at=nothing, modifier_type::Symbol=:auto, profile::NamedTuple=NamedTuple(),
    groups=nothing, delta::Union{Symbol,Real}=:auto, scale::Symbol=:response, measure::Symbol=:effect,
    contrasts::Symbol=:baseline, contrast::Union{Nothing,AbstractString}=nothing, vcov=GLM.vcov,
    weights=:model, alpha=0.05, p_adjust::Symbol=:none, include_gradients::Bool=false,
    warn::Bool=true, backend::Symbol=:ad, df=nothing,
    random_effects::Union{Symbol,NamedTuple}=:zero, re_new::Symbol=:zero, re_terms=nothing, re_nodes=:auto, re_se::Symbol=:auto,
)
    # The checks here are the ones specific to second differences; everything shared with
    # population_margins (alpha, df, contrasts, measure, scale, backend, weights, vcov, the groups
    # syntax and its conflicts, the focal variables) is validated once, by the request below.
    _check_p_adjust(p_adjust)
    # Any Tables.jl source: the checks below read the columns restricted to the model's estimation
    # rows (FixedEffectModels' `esample`), as every computation from the request is. The request
    # itself gets the data and weights as given, and subsets them the same way.
    data_cols = Tables.columntable(data)
    (cols, _) = _estimation_subset(model, data_cols, weights)
    hasproperty(cols, modifier) || throw(ArgumentError("Modifier variable :$modifier not found in data. " *
        "Available columns: $(_column_list(cols))"))
    modifier_type ∈ (:auto, :categorical, :continuous) ||
        throw(ArgumentError("modifier_type must be :auto, :categorical or :continuous, got :$modifier_type"))
    !isnothing(levels) && !isnothing(at) && throw(ArgumentError(
        "Give either `levels` (compare these modifier values) or `at` (local derivative), not both."))

    col = _column(cols, modifier)
    modifier_in_model = _modifier_in_model(model, modifier)
    mtype = modifier_type === :auto ? _detect_modifier_type(col) : modifier_type
    if warn && modifier_type === :auto && mtype === :continuous && isnothing(at) && isnothing(levels)
        _warn_if_integer_coded(col, modifier)
    end
    vars_vec = variables isa Symbol ? [variables] : variables
    modifier in keys(profile) && throw(ArgumentError("`profile` contains the modifier :$modifier; the " *
        "modifier is varied by the calculation (use `levels` or `at` to choose where). Remove it from `profile`."))
    mtype === :continuous && any(x -> x isa Bool, skipmissing(col)) &&
        throw(ArgumentError("Modifier :$modifier is Bool, so modifier_type=:continuous is not defined. " *
              "Leave modifier_type=:auto to compare false and true."))
    isnothing(groups) || _check_groups_for_second_differences(groups, modifier,
                                                               keys(_local_profile(at, profile, modifier)))

    if warn
        # modifier / profile values the data does not cover
        if !isnothing(levels)
            _warn_extrapolation(cols, modifier, levels)
        elseif at isa Real || at isa AbstractVector
            _warn_extrapolation(cols, modifier, at isa Real ? [at] : at)
        elseif at isa NamedTuple
            for (k, v) in pairs(at)
                _warn_extrapolation(cols, k, v isa AbstractVector ? v : [v])
            end
        end
        for (k, v) in pairs(profile)
            _warn_extrapolation(cols, k, v isa AbstractVector ? v : [v])
        end
    end

    if mtype === :continuous && isnothing(levels)
        # Derivative of the AME with respect to the modifier (default `at=:observed`)
        at = isnothing(at) ? :observed : at
        _check_local_derivative_inputs(cols, modifier, at)
        _check_focal_and_profile_vars(cols, vars_vec, keys(_local_profile(at, profile, modifier)))
        # One request for the whole call: every validator and the one evaluation of `vcov`
        req = _local_derivative_request(model, data_cols, vars_vec, modifier, at, profile;
                                        scale, backend, measure, contrasts, vcov, weights, alpha, df,
                                        groups, warn, random_effects, re_new, re_terms, re_nodes, re_se)
        out = _local_derivatives(req, cols, vars_vec, modifier; at, profile, delta, p_adjust,
                                 include_gradients, contrast)
        warn && _warn_structural_zeros(out, modifier_in_model)
        return out
    end

    !isnothing(at) && throw(ArgumentError("`at` (local derivative) applies to a continuous modifier; " *
        ":$modifier is treated as $mtype. Use `levels` to compare specific values, or modifier_type=:continuous."))
    warn && !(isnothing(delta) || delta === :auto) && @warn "`delta` is ignored when comparing modifier levels"

    _check_focal_and_profile_vars(cols, vars_vec, keys(profile))

    lv = isnothing(levels) ? _discrete_levels(col) : collect(levels)
    length(lv) >= 2 || throw(ArgumentError("Need at least 2 modifier levels to compare, got $(length(lv))."))
    result_type = mtype === :continuous ? :continuous : (length(lv) == 2 ? :binary : :categorical)
    mtype === :continuous && !all(x -> x isa Number && !(x isa Bool), lv) &&
        throw(ArgumentError("A continuous modifier needs numeric `levels`, got $(eltype(lv))."))

    # One request for the whole call: every validator (the scenario variables, the modifier and the
    # `profile` variables, included; `groups` and its conflicts) and the one evaluation of `vcov` and of
    # the weights. Each profile point is then one AME computation from it, with the scenarios set to that
    # point and, with `groups=`, every stratum in the same computation; the result-level form pairs the
    # modifier levels within each stratum.
    req = _normalize_request(model, data_cols; type=:effects, vars=vars_vec, scale, backend, measure,
                             contrasts, alpha, vcov, weights, df, groups, warn,
                             random_effects, re_new, re_terms, re_nodes, re_se,
                             scenario_vars=Symbol[modifier, keys(profile)...])
    pieces = DataFrame[]
    for profile_point in _expand_profile_to_cartesian(profile)
        scenarios = merge(NamedTuple{(modifier,)}((lv,)), profile_point)
        ames = _run(_with_scenarios(req, scenarios))
        sd = second_differences(
            ames, vars_vec, modifier; modifier_type=result_type, contrast, alpha=req.alpha,
            include_gradients=true, warn=false,
        )
        for (k, v) in pairs(profile_point)
            sd[!, k] = fill(v, nrow(sd))
        end
        push!(pieces, sd)
    end
    out = length(pieces) == 1 ? only(pieces) : vcat(pieces...; cols=:union)
    return _finalize_frame!(out, p_adjust, req.alpha, include_gradients; warn, modifier_in_model,
                            weights_source=req.weights_source)
end

# `groups=` on the model-level second differences: the strata must not be defined by a variable the
# calculation sets. The focal variables and the scenario variables are checked again by the request
# (`_validate_vars_groups_overlap`, `_validate_scenarios_groups_conflict`); these messages say what each
# combination would mean for a second difference.
function _check_groups_for_second_differences(groups, modifier::Symbol, held)
    gvars = _extract_group_variables(groups)
    if modifier in gvars
        throw(ArgumentError(
            "`groups` contains the modifier :$modifier. second_differences(model, data, …; groups=…) " *
            "computes the counterfactual second difference within each stratum: every row of the " *
            "stratum is set to each level of :$modifier (or shifted along it), so a stratum defined by " *
            ":$modifier would be labelled by values its rows no longer carry. To stratify by another " *
            "variable, put that variable in `groups`. To compare the effect across the observed " *
            "subgroups of :$modifier instead (the effect among the rows with one value vs among the " *
            "rows with another; a descriptive comparison of different rows, not a counterfactual one), " *
            "use the result-level form: r = population_margins(model, data; vars=…, groups=" *
            "$(repr(modifier))) and then second_differences(r, variables, $(repr(modifier)))."))
    end
    clash = [v for v in held if v in gvars]
    isempty(clash) || throw(ArgumentError(
        "`groups` contains $(join(":" .* string.(clash), ", ")), which `profile` (or `at`) also sets for " *
        "every row. A stratum is selected by the observed values of its variables; setting the same " *
        "variable counterfactually would leave each stratum labelled by values its rows no longer carry. " *
        "Use the variable in one of the two."))
    return nothing
end

# A column of a column table, by a name known only at run time. The barrier hides the column's
# type from inference, so the code reading it is compiled for the column it gets, not for every
# column type of the table (first-call latency).
_column(cols::NamedTuple, name::Symbol)::AbstractVector = Base.inferencebarrier(getproperty(cols, name))

# Column names for messages, as `names(df)` lists them
_column_list(cols::NamedTuple) = join(string.(keys(cols)), ", ")

# The focal and profile variables must be columns of the data
function _check_focal_and_profile_vars(cols::NamedTuple, vars_vec, profile_vars)
    for v in vars_vec
        hasproperty(cols, v) || throw(ArgumentError("Focal variable :$v not found in data. " *
            "Available columns: $(_column_list(cols))"))
    end
    for k in profile_vars
        hasproperty(cols, k) || throw(ArgumentError("Profile variable :$k not found in data. " *
            "Available columns: $(_column_list(cols))"))
    end
    return nothing
end
