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
          "continuous, so the result is the local derivative ∂AME/∂$modifier at its mean. If these are " *
          "category codes, pass modifier_type=:categorical to compare levels; to keep the derivative " *
          "and silence this warning, pass `at=` (e.g. at=:mean)."
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
                       levels=nothing, at=nothing, modifier_type=:auto, profile=NamedTuple(),
                       delta=:auto, scale=:response, measure=:effect, contrasts=:baseline, contrast=nothing, vcov=GLM.vcov,
                       weights=nothing, alpha=0.05, p_adjust=:none, include_gradients=false, warn=true)

How does the marginal effect of `variables` change with `modifier`? Computes the second
difference (interaction effect on the response scale) with delta-method standard errors
directly from a fitted model; no scenarios or covariance matrix to assemble.

What is computed depends on the modifier:

- **Discrete modifier** (Bool, 0/1, strings, categorical, or numeric with ≤ 2 values):
  AME(`variables`) at every pair of modifier levels, and the difference between them.
- **Continuous modifier** (numeric with more than 2 distinct values):
  the local derivative ∂AME/∂modifier at `at` (default the mean), by symmetric finite
  differences; or, with `levels=[…]`, the slope between those modifier values.

# Arguments
- `model`, `data`: fitted model and the data frame it was fit to.
- `variables::Union{Symbol,Vector{Symbol}}`: focal variable(s); each continuous variable gives
  one row, each categorical variable one row per contrast (see `contrasts`).
- `modifier::Symbol`: the moderating variable (must not also be in `variables`).

# Keyword Arguments
- `levels`: modifier values to compare. Default for a discrete modifier: all its levels.
  For a continuous modifier, giving `levels` returns slopes `(AME₂ − AME₁)/(m₂ − m₁)` for every
  pair of those values instead of a local derivative.
- `at`: continuous modifier only; where to take the derivative: `:mean` (default), `:median`,
  a number, a vector of numbers, or a `NamedTuple` that includes the modifier together with other
  variables to hold fixed. Cannot be combined with `levels`.
- `modifier_type`: `:auto` (default; decided from the data as above), `:categorical` or
  `:continuous`. Use it to treat an integer-coded categorical modifier as categorical (a numeric
  modifier with at most 5 distinct integer values is treated as continuous with a warning, unless
  `at` or `levels` is given).
- `profile::NamedTuple`: other variables to hold fixed, e.g. `(age=40, region="north")`; vector
  values give one profile point each (Cartesian product), reported as extra columns. These are
  counterfactuals (everyone's value is set), not subgroups; for subgroups use `groups=` on
  `population_margins` and the result-level form. Must not contain the modifier.
- `delta`: finite-difference step for a local derivative (`:auto` = 1% of the modifier's SD).
- `measure`: `:effect` (default), `:elasticity`, `:semielasticity_dyex` or `:semielasticity_eydx`, as in
  `population_margins`: the second difference is then the change in that measure across the modifier.
  The measures are defined for continuous focal variables only, so request categorical ones separately.
- `scale`: `:response` (default) or `:link`. `contrasts`: how contrasts of a categorical focal variable
  are built, `:baseline` (K−1) or `:pairwise` (K(K−1)/2), as in `population_margins`.
- `contrast`: restrict the result to one contrast label, such as `"college - hs"` (or `"dy/dx"` for a
  continuous variable). `nothing` (default) keeps every contrast. It applies to every focal variable, so
  each must have it, and it must exist under the chosen `contrasts`. Same keyword as the result-level
  form; `contrasts` (plural) builds the contrasts, `contrast` selects one.
- `vcov`: covariance estimator, as in `population_margins` (default `GLM.vcov`; a function, a
  matrix, or a CovarianceMatrices estimator). The t/z choice follows the same rule as the rest of
  the package: t with the residual df for a linear model's own covariance, z otherwise.
- `weights`: observation weights, as in `population_margins` (a column name or a vector). Every average
  marginal effect is then weighted, and the default evaluation point `at=:mean` / `:median` of a continuous
  modifier is the weighted mean / median. Without it the numbers are unweighted, even if you computed
  weighted effects elsewhere.
- `alpha`: significance level; `significant` is `p_value < alpha` and `ci_lower`/`ci_upper` give the
  (1 − `alpha`) interval (default 0.05, a 95% interval).
- `p_adjust`: multiple-testing adjustment over **all rows returned** (every variable, contrast, level
  pair, evaluation point and `profile` stratum): `:none` (default), `:bonferroni`, `:holm` (family-wise
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
  range of that variable, where the result extrapolates the fitted model.

# Returns
A `DataFrame` with the columns listed under "Output Schema" in the second-differences
documentation (`variable, contrast, modifier, modifier_type, modifier_level1, modifier_level2,
ame_at_level1, ame_at_level2, second_diff, se, statistic, df, p_value, ci_lower, ci_upper, significant`);
a local derivative adds `eval_point` and `delta_used`, and `profile` variables add one column each.
For a discrete or continuous-`levels` comparison, `second_diff` is `AME(level2) − AME(level1)` (or
that difference per unit of the modifier for a continuous modifier), with numeric levels ascending.

# Examples
```julia
m = glm(@formula(y ~ age * treated + income), df, Binomial(), LogitLink())

second_differences(m, df, :age, :treated)              # effect of age: treated vs not
second_differences(m, df, [:age, :income], :treated)   # several focal variables
second_differences(m, df, :age, :income)               # ∂AME(age)/∂income at mean(income)
second_differences(m, df, :age, :income; at=[20_000, 50_000])
second_differences(m, df, :age, :income; levels=[20_000, 50_000])   # slope between two values
second_differences(m, df, :age, :treated; profile=(female=[0, 1],)) # everyone's `female` set to 0, then 1
second_differences(m, df, :age, :treated; vcov=vcov(m))
second_differences(m, df, :age, :education; p_adjust=:holm, alpha=0.1)    # adjusted tests, 90% intervals
```

For an `EffectsResult` you have already computed, `second_differences(result, variables, modifier[, vcov])`
does the same comparison on its rows (see its docstring).
"""
function second_differences(
    model::RegressionModel, data, variables::Union{Symbol,Vector{Symbol}}, modifier::Symbol;
    levels=nothing, at=nothing, modifier_type::Symbol=:auto, profile::NamedTuple=NamedTuple(),
    delta::Union{Symbol,Real}=:auto, scale::Symbol=:response, measure::Symbol=:effect,
    contrasts::Symbol=:baseline, contrast::Union{Nothing,AbstractString}=nothing, vcov=GLM.vcov, weights=nothing, alpha::Real=0.05, p_adjust::Symbol=:none, include_gradients::Bool=false,
    warn::Bool=true,
)
    _check_p_adjust(p_adjust)
    0 < alpha < 1 || error("alpha must be in (0, 1), got $alpha")
    hasproperty(data, modifier) || error("Modifier variable :$modifier not found in data. " *
        "Available columns: $(join(string.(names(data)), ", "))")
    modifier_type ∈ (:auto, :categorical, :continuous) ||
        error("modifier_type must be :auto, :categorical or :continuous, got :$modifier_type")
    !isnothing(levels) && !isnothing(at) &&
        error("Give either `levels` (compare these modifier values) or `at` (local derivative), not both.")

    col = data[!, modifier]
    modifier_in_model = _modifier_in_model(model, modifier)
    mtype = modifier_type === :auto ? _detect_modifier_type(col) : modifier_type
    if warn && modifier_type === :auto && mtype === :continuous && isnothing(at) && isnothing(levels)
        _warn_if_integer_coded(col, modifier)
    end
    vars_vec = variables isa Symbol ? [variables] : variables
    modifier in keys(profile) && error("`profile` contains the modifier :$modifier; the modifier is " *
        "varied by the calculation (use `levels` or `at` to choose where). Remove it from `profile`.")
    mtype === :continuous && any(x -> x isa Bool, skipmissing(col)) &&
        error("Modifier :$modifier is Bool, so modifier_type=:continuous is not defined. " *
              "Leave modifier_type=:auto to compare false and true.")

    if warn
        # modifier / profile values the data does not cover
        if !isnothing(levels)
            _warn_extrapolation(data, modifier, levels)
        elseif at isa Real || at isa AbstractVector
            _warn_extrapolation(data, modifier, at isa Real ? [at] : at)
        elseif at isa NamedTuple
            for (k, v) in pairs(at)
                _warn_extrapolation(data, k, v isa AbstractVector ? v : [v])
            end
        end
        for (k, v) in pairs(profile)
            _warn_extrapolation(data, k, v isa AbstractVector ? v : [v])
        end
    end

    if mtype === :continuous && isnothing(levels)
        # Local derivative at `at` (default: the mean)
        out = _second_differences_local(
            model, data, variables, modifier, vcov;
            at = isnothing(at) ? :mean : at, profile, delta, scale, measure, contrasts, alpha, p_adjust, include_gradients, weights, contrast,
        )
        warn && _warn_structural_zeros(out, modifier_in_model)
        return out
    end

    !isnothing(at) && error("`at` (local derivative) applies to a continuous modifier; :$modifier is " *
        "treated as $mtype. Use `levels` to compare specific values, or modifier_type=:continuous.")
    warn && !(isnothing(delta) || delta === :auto) && @warn "`delta` is ignored when comparing modifier levels"

    for v in vars_vec
        hasproperty(data, v) || error("Focal variable :$v not found in data. " *
            "Available columns: $(join(string.(names(data)), ", "))")
    end
    for k in keys(profile)
        hasproperty(data, k) || error("Profile variable :$k not found in data. " *
            "Available columns: $(join(string.(names(data)), ", "))")
    end

    lv = isnothing(levels) ? _discrete_levels(col) : collect(levels)
    length(lv) >= 2 || error("Need at least 2 modifier levels to compare, got $(length(lv)).")
    result_type = mtype === :continuous ? :continuous : (length(lv) == 2 ? :binary : :categorical)
    mtype === :continuous && !all(x -> x isa Number && !(x isa Bool), lv) &&
        error("A continuous modifier needs numeric `levels`, got $(eltype(lv)).")

    Σ = _covariance_matrix(model, vcov)   # computed here, not read back from the cached engine
    pieces = DataFrame[]
    for profile_point in _expand_profile_to_cartesian(profile)
        scenarios = merge(NamedTuple{(modifier,)}((lv,)), profile_point)
        ames = population_margins(
            model, data; scenarios, vars=vars_vec, type=:effects, scale, measure, vcov, contrasts, weights,
            ci_alpha=Float64(alpha),
        )
        _align_second_difference_df!(ames, model, vcov)
        sd = second_differences(
            ames, vars_vec, modifier, Σ; modifier_type=result_type, contrast, alpha, include_gradients=true, warn=false,
        )
        for (k, v) in pairs(profile_point)
            sd[!, k] = fill(v, nrow(sd))
        end
        push!(pieces, sd)
    end
    out = length(pieces) == 1 ? only(pieces) : vcat(pieces...; cols=:union)
    return _finalize_frame!(out, p_adjust, alpha, include_gradients; warn, modifier_in_model)
end
