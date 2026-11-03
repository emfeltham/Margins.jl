# utilities.jl
# Shared utilities for second differences computations

"""
    _expand_profile_to_cartesian(profile::NamedTuple) -> Vector{NamedTuple}

Expand a profile NamedTuple with vector-valued entries into a vector of NamedTuples
representing the Cartesian product of all values.

# Arguments
- `profile::NamedTuple`: Profile specification where values can be scalars or vectors

# Returns
- `Vector{NamedTuple}`: Vector of profile points (Cartesian product)

# Examples
```julia
# Single scalar
_expand_profile_to_cartesian((age=40,))
# → [(age=40,)]

# Single vector
_expand_profile_to_cartesian((socio4=[false, true],))
# → [(socio4=false,), (socio4=true,)]

# Multiple vectors
_expand_profile_to_cartesian((socio4=[false, true], region=["north", "south"]))
# → [(socio4=false, region="north"),
#    (socio4=true,  region="north"),
#    (socio4=false, region="south"),
#    (socio4=true,  region="south")]
```
"""
function _expand_profile_to_cartesian(profile::NamedTuple)
    # If empty, return single empty NamedTuple
    if length(profile) == 0
        return [NamedTuple()]
    end

    # Get keys and values
    keys_vec = keys(profile)

    # Normalize all values to vectors
    value_vecs = []
    for k in keys_vec
        val = profile[k]
        if val isa AbstractVector
            push!(value_vecs, collect(val))
        else
            # Scalar - wrap in vector
            push!(value_vecs, [val])
        end
    end

    # Compute Cartesian product
    # For example: [[:a, :b], [1, 2]] → [(:a, 1), (:b, 1), (:a, 2), (:b, 2)]
    cartesian_indices = Iterators.product(value_vecs...)

    # Build NamedTuples from the product
    result = NamedTuple[]
    for combo in cartesian_indices
        nt = NamedTuple{keys_vec}(combo)
        push!(result, nt)
    end

    return result
end

# ---- observation weights --------------------------------------------------------------------

# `weights` as `population_margins` takes it (nothing, a column name, or a vector), as a Float64 vector.
function _weights_vector(weights, data)
    isnothing(weights) && return nothing
    _validate_weights_parameter(weights, data)
    return weights isa Symbol ? Float64.(data[!, weights]) : Float64.(weights)
end

_weighted_mean(x, w) = sum(w .* x) / sum(w)

# Smallest x whose cumulative weight reaches half the total (the midpoint of two x when it lands exactly on half).
function _weighted_median(x, w)
    p = sortperm(x)
    xs, cw = x[p], cumsum(w[p])
    half = cw[end] / 2
    i = findfirst(c -> c >= half, cw)
    return (cw[i] == half && i < length(xs)) ? (xs[i] + xs[i + 1]) / 2 : float(xs[i])
end

# Continuous effects are labelled "dy/dx" (v2.3.0; the scenarios/groups path used to say
# "derivative"). "derivative" is still accepted wherever a contrast label is passed in.
_contrast_label(c::AbstractString) = c == "derivative" ? "dy/dx" : String(c)

# =============================================================================
# Unified output schema
# =============================================================================

"""
Columns of every second-difference DataFrame returned by `second_differences`, in order:

| column            | meaning                                                                  |
|:------------------|:-------------------------------------------------------------------------|
| `variable`        | focal variable (`Symbol`)                                                |
| `contrast`        | focal contrast (`"dy/dx"` for a continuous variable, `"B - A"` levels)   |
| `modifier`        | modifier variable (`Symbol`)                                             |
| `modifier_type`   | `:binary`, `:categorical` or `:continuous`                               |
| `modifier_level1` | first modifier value compared                                            |
| `modifier_level2` | second modifier value compared                                           |
| `ame_at_level1`   | focal effect at `modifier_level1`                                        |
| `ame_at_level2`   | focal effect at `modifier_level2`                                        |
| `second_diff`     | `ame_at_level2 - ame_at_level1`; for `:continuous`, divided by the modifier distance (a slope) |
| `se`              | delta-method standard error                                              |
| `statistic`       | `second_diff / se`                                                       |
| `df`              | inference degrees of freedom (t when finite, z when `Inf`)               |
| `p_value`         | two-sided p-value for H₀: `second_diff = 0`                              |
| `ci_lower`, `ci_upper` | (1 − `alpha`) confidence interval for `second_diff`, from the same t/z reference as the test |
| `significant`     | `p_value < alpha` (`p_adjusted < alpha` when `p_adjust` is set); `alpha` defaults to 0.05 |

A local derivative (a continuous modifier with `at`) returns the same columns, with
`modifier_type = :continuous`, `modifier_level1/2 = eval_point ∓ delta_used` and `second_diff` the
derivative ∂AME/∂modifier, followed by `eval_point`, `delta_used` and one column per `profile`
variable. With `p_adjust` set, a `p_adjusted` column follows `p_value`. With `include_gradients=true`, a `gradient`
column (the vector ∂`second_diff`/∂β, for custom variance calculations) follows the schema columns; it is
omitted by default, as in `DataFrame(result)`.

`df` follows the same rule as the rest of the package, applied to the covariance matrix you pass:
the residual df for a linear model when `vcov` is the model's own covariance, `Inf` (z) for any
other matrix (robust, clustered, custom) and for non-linear models.
"""
const SECOND_DIFFERENCE_COLUMNS = (
    :variable, :contrast, :modifier, :modifier_type, :modifier_level1, :modifier_level2,
    :ame_at_level1, :ame_at_level2, :second_diff, :se, :statistic, :df, :p_value,
    :ci_lower, :ci_upper, :significant,
)

# One row of the unified second-difference schema. A *structurally* zero second difference (the
# gradient of the difference is zero: no interaction in an identity-link model, or the modifier is not
# in the model) has NaN statistic, p-value and interval. Otherwise a zero standard error gives
# statistic 0 / p 1 for a zero estimate and Inf / 0 for a non-zero one.
# Relative size below which the gradient of a difference counts as zero. Rounding noise in the
# gradients is ~1e-16 of their norm; a real dependence on the modifier is many orders larger.
const STRUCTURAL_ZERO_RTOL = 1e-12

# Is the difference of two effects with gradients g1, g2 identically zero as a function of the
# parameters? Then no parameter uncertainty can move it, whatever Σ is.
_is_structural_zero(g_diff, g1, g2) = norm(g_diff) <= STRUCTURAL_ZERO_RTOL * max(norm(g1), norm(g2))

function _second_difference_row(;
    variable::Symbol, contrast::AbstractString, modifier::Symbol, modifier_type::Symbol,
    level1, level2, est1::Real, est2::Real, second_diff::Real, se::Real,
    gradient::AbstractVector, dof::Real, alpha::Real=0.05, structural_zero::Bool=false,
)
    if structural_zero
        # The fitted model does not let the focal effect depend on the modifier, so the difference is
        # zero by construction, not estimated: no test statistic, p-value or interval exists.
        statistic = NaN
        p_value = NaN
    elseif se ≈ 0.0
        statistic = second_diff ≈ 0.0 ? 0.0 : Inf
        p_value = second_diff ≈ 0.0 ? 1.0 : 0.0
    else
        statistic = second_diff / se
        p_value = compute_p_value(Float64(statistic), dof)
    end
    half = structural_zero ? NaN : _critical_value(alpha, dof) * se
    return (
        variable = variable,
        contrast = _contrast_label(contrast),
        modifier = modifier,
        modifier_type = modifier_type,
        modifier_level1 = level1,
        modifier_level2 = level2,
        ame_at_level1 = Float64(est1),
        ame_at_level2 = Float64(est2),
        second_diff = Float64(second_diff),
        se = Float64(se),
        statistic = Float64(statistic),
        df = Float64(dof),
        p_value = Float64(p_value),
        ci_lower = Float64(second_diff - half),
        ci_upper = Float64(second_diff + half),
        significant = !isnan(p_value) && p_value < alpha,
        gradient = gradient,
    )
end

# ---- significance level and multiple-testing adjustment ----------------------------------

const P_ADJUST_METHODS = (:none, :bonferroni, :holm, :bh)

# Significance level / confidence level: an explicit `alpha`, else the one the result was
# computed with (population_margins `ci_alpha`), else 0.05.
function _resolve_alpha(alpha, ame_result)::Float64
    a = isnothing(alpha) ? _result_alpha(ame_result) : Float64(alpha)
    0 < a < 1 || error("alpha must be in (0, 1), got $a")
    return a
end

function _check_p_adjust(method::Symbol)
    method in P_ADJUST_METHODS ||
        error("p_adjust must be one of $(join(repr.(P_ADJUST_METHODS), ", ")), got $(repr(method))")
    return method
end

"""
    _adjust_pvalues(p, method) -> Vector{Float64}

Adjusted p-values for the family `p`: `:bonferroni` (min(1, m·p)), `:holm` (step-down,
controls the family-wise error rate) or `:bh` (Benjamini–Hochberg, controls the false
discovery rate). `:none` returns `p` unchanged.
"""
function _adjust_pvalues(p::AbstractVector{<:Real}, method::Symbol)::Vector{Float64}
    _check_p_adjust(method)
    q = Float64.(p)
    m = length(q)
    (method === :none || m == 0) && return q
    if method === :bonferroni
        return min.(1.0, m .* q)
    end
    ord = sortperm(q)
    adj = similar(q)
    if method === :holm
        running = 0.0
        for (rank, i) in enumerate(ord)
            running = max(running, (m - rank + 1) * q[i])
            adj[i] = min(1.0, running)
        end
    else  # :bh
        running = 1.0
        for rank in m:-1:1
            i = ord[rank]
            running = min(running, q[i] * m / rank)
            adj[i] = min(1.0, running)
        end
    end
    return adj
end

# Add `p_adjusted` after `p_value` and base `significant` on it. The family is every row of `df` that
# has a p-value; structurally zero rows (NaN p-value) are not tests, so they are left out of the family
# size and keep a NaN `p_adjusted`.
function _apply_p_adjust!(df::DataFrame, method::Symbol, alpha::Real)
    _check_p_adjust(method)
    method === :none && return df
    p = Float64.(df.p_value)
    tested = .!isnan.(p)
    padj = fill(NaN, length(p))
    padj[tested] = _adjust_pvalues(p[tested], method)
    insertcols!(df, findfirst(==("p_value"), names(df)) + 1, :p_adjusted => padj)
    df.significant = tested .& (padj .< alpha)
    return df
end

# (the `gradient` column, when present, follows the schema columns)
function _second_difference_frame(rows)
    df = DataFrame(rows)
    grad = :gradient in propertynames(df) ? [:gradient] : Symbol[]
    extras = [c for c in propertynames(df) if !(c in SECOND_DIFFERENCE_COLUMNS) && c !== :gradient]
    return select(df, [SECOND_DIFFERENCE_COLUMNS..., grad..., extras...])
end

# ---- warnings for calculations the model cannot support ------------------------------------

# One aggregated warning per call for rows whose second difference is zero by construction. Such rows
# carry NaN inference (see `_second_difference_row`).
function _warn_structural_zeros(df::DataFrame, modifier_in_model::Union{Nothing,Bool}=nothing)
    :p_value in propertynames(df) || return nothing
    bad = findall(isnan, df.p_value)
    isempty(bad) && return nothing
    pairs = unique(["$(df.variable[i]) × $(df.modifier[i])" for i in bad])
    shown = length(pairs) <= 5 ? join(pairs, ", ") : join(pairs[1:5], ", ") * ", …"
    modifiers = unique(df.modifier[bad])
    why = if modifier_in_model === false
        "the modifier $(join(":" .* string.(modifiers), ", ")) does not enter the model"
    else
        "the fitted model does not let the effect of the focal variable depend on the modifier (for " *
        "example there is no interaction term between them in a model with an identity link, or the " *
        "modifier is not in the model)"
    end
    @warn "$(length(bad)) second difference(s) ($shown) are zero by construction: $why. " *
          "Their statistic, p-value and confidence interval are NaN and they are left out of `p_adjust`. " *
          "In a nonlinear model (e.g. logit) the effect can still vary with the modifier through the " *
          "link without an interaction term; a structural zero means it cannot."
    return nothing
end

# Warn when numeric modifier or profile values lie outside the range seen in the data: the
# counterfactual there extrapolates the fitted model.
function _warn_extrapolation(data, name::Symbol, values)
    hasproperty(data, name) || return nothing
    col = collect(skipmissing(data[!, name]))
    isempty(col) && return nothing
    all(x -> x isa Real, col) || return nothing
    vals = [v for v in values if v isa Real]
    isempty(vals) && return nothing
    lo, hi = extrema(col)
    outside = unique(filter(v -> v < lo || v > hi, vals))
    isempty(outside) && return nothing
    @warn "Value(s) $(join(outside, ", ")) of :$name lie outside the observed range [$lo, $hi]; the " *
          "counterfactual there extrapolates the fitted model, and in a nonlinear model the result " *
          "can be badly wrong."
    return nothing
end

# Last step of every public function that returns a table: multiple-testing adjustment, then drop the
# per-row gradient vectors unless `include_gradients` (as `DataFrame(result)` does). Nested calls
# between these functions pass `include_gradients=true` so the gradients survive until this step.
function _finalize_frame!(df::DataFrame, p_adjust::Symbol, alpha::Real, include_gradients::Bool;
                          warn::Bool=false, modifier_in_model::Union{Nothing,Bool}=nothing)
    warn && _warn_structural_zeros(df, modifier_in_model)
    _apply_p_adjust!(df, p_adjust, alpha)
    include_gradients || :gradient ∉ propertynames(df) || select!(df, Not(:gradient))
    return df
end

# ============================================================================
# Row selection shared by the second-difference engines
# ============================================================================

"""
    EffectIndex

Typed view of an `EffectsResult` for the second-difference code: the focal variable of every row as a
`Symbol`, its contrast as a canonical label (`"derivative"` → `"dy/dx"`), and the scenario/profile and
group columns as one `NamedTuple`. Built once per call by `_effect_index`, it is the only place that
reads the result's string labels; every lookup (a variable's rows and contrasts, the rows of one
contrast, whether a column exists) goes through it.
"""
struct EffectIndex
    n::Int
    variables::Vector{Symbol}                 # focal variable of each row
    contrasts::Vector{String}                 # canonical contrast label of each row
    cells::NamedTuple                         # scenario/profile + group columns, one entry per row
    sources::Dict{Symbol,String}              # "scenario" or "group", for messages
    by_variable::Dict{Symbol,Vector{Int}}     # row numbers per focal variable, in row order
end

function _effect_index(r::EffectsResult)
    n = length(r.estimates)
    (length(r.variables) == n && length(r.terms) == n && size(r.gradients, 1) == n) ||
        error("Malformed EffectsResult: $n estimates but $(length(r.variables)) variables, " *
              "$(length(r.terms)) terms and $(size(r.gradients, 1)) gradient rows.")
    variables = Symbol.(r.variables)
    contrasts = String[_contrast_label(t) for t in r.terms]
    cells = NamedTuple()
    sources = Dict{Symbol,String}()
    # population_margins stores group columns in `profile_values`, and records which columns are
    # groups and which are scenario variables in the metadata
    group_vars = Set(_result_groups_vars(r))
    for (source, nt) in (("scenario", r.profile_values), ("group", r.group_values))
        isnothing(nt) && continue
        for k in keys(nt)
            length(nt[k]) == n || error("Malformed EffectsResult: $source column :$k has " *
                                        "$(length(nt[k])) values for $n rows.")
            sources[k] = (source == "group" || k in group_vars) ? "group" : "scenario"
        end
        cells = merge(cells, nt)
    end
    by_variable = Dict{Symbol,Vector{Int}}()
    for i in 1:n
        push!(get!(by_variable, variables[i], Int[]), i)
    end
    return EffectIndex(n, variables, contrasts, cells, sources, by_variable)
end

_variable_rows(idx::EffectIndex, variable::Symbol) = get(idx.by_variable, variable, Int[])

# Contrast labels of a variable, in row order (`["dy/dx"]` for a continuous variable)
_variable_contrasts(idx::EffectIndex, variable::Symbol) =
    unique(idx.contrasts[_variable_rows(idx, variable)])

# Error unless the result has rows for `variable`
function _require_variable(idx::EffectIndex, variable::Symbol)
    haskey(idx.by_variable, variable) && return nothing
    present = unique(idx.variables)
    error("Variable :$variable not found in the effects result (variables: $(join(present, ", "))). " *
          "Include it in `vars` of the population_margins/profile_margins call.")
end

# Error unless the modifier is a scenario/profile column of the result
function _require_modifier(idx::EffectIndex, modifier::Symbol)
    haskey(idx.cells, modifier) && return nothing
    have = isempty(idx.cells) ? "none" : join(keys(idx.cells), ", ")
    error("The effects result does not have modifier :$modifier among its scenario/profile columns " *
          "(it has: $have). Compute it with `scenarios=($modifier=levels,)`, or with a reference grid " *
          "that varies :$modifier.")
end

"""
    _modifier_rows(idx, variable, modifier, contrast) -> (rows, levels)
    _modifier_rows(ame_result, variable, modifier, contrast)

Rows of the result for `variable` and `contrast`, with their `modifier` levels, in a canonical order:
numeric (incl. Bool) levels ascending, other levels in scenario order.

Throws if any other scenario or group column varies across the selected rows. Without that check, a
result built with several scenario variables (or `groups=`) would pair rows from different cells and
report meaningless differences. Returns empty vectors when no row matches; callers decide whether that
is an error.
"""
function _modifier_rows(idx::EffectIndex, variable::Symbol, modifier::Symbol, contrast::AbstractString)
    label = _contrast_label(contrast)
    rows = [i for i in _variable_rows(idx, variable) if idx.contrasts[i] == label]
    levels = [idx.cells[modifier][i] for i in rows]

    for k in keys(idx.cells)
        k == modifier && continue
        if length(unique(idx.cells[k][rows])) > 1
            error("The effects result varies over $(idx.sources[k]) variable :$k as well as the modifier " *
                  ":$modifier, so its rows cannot be paired as modifier levels. Compute the " *
                  "result over :$modifier alone, or subset the result to one :$k value first.")
        end
    end

    if length(unique(levels)) != length(levels)
        error("Modifier :$modifier repeats a level among the rows for variable :$variable " *
              "(contrast '$label'); expected one row per modifier level.")
    end

    if all(x -> x isa Number, levels)
        perm = sortperm(levels)
        rows, levels = rows[perm], levels[perm]
    end
    return rows, levels
end
_modifier_rows(r::EffectsResult, variable::Symbol, modifier::Symbol, contrast::AbstractString) =
    _modifier_rows(_effect_index(r), variable, modifier, contrast)

# ============================================================================
# `vcov` defaults to the covariance the result was computed with
# ============================================================================

# Without an explicit `vcov`, a `population_margins` result supplies its own covariance matrix
# (`metadata[:vcov]`), so the standard errors of the second differences cannot disagree with
# those of the AMEs they are built from. Pass `vcov` to use a different estimator.
second_differences(ame_result::EffectsResult, variables::Union{Symbol,Vector{Symbol}}, modifier::Symbol;
                   vcov=nothing, kwargs...) =
    second_differences(ame_result, variables, modifier, isnothing(vcov) ? _stored_vcov(ame_result) : vcov; kwargs...)
_second_difference(ame_result::EffectsResult, variable::Symbol, modifier::Symbol; kwargs...) =
    _second_difference(ame_result, variable, modifier, _stored_vcov(ame_result); kwargs...)
_second_differences_pairwise(ame_result::EffectsResult, variable::Symbol, modifier::Symbol; kwargs...) =
    _second_differences_pairwise(ame_result, variable, modifier, _stored_vcov(ame_result); kwargs...)
_second_differences_all_contrasts(ame_result::EffectsResult, variable::Symbol, modifier::Symbol; kwargs...) =
    _second_differences_all_contrasts(ame_result, variable, modifier, _stored_vcov(ame_result); kwargs...)

# ============================================================================
# Predictions results: say what to do instead of a bare MethodError
# ============================================================================

# Second differences compare marginal *effects*; a PredictionsResult (type=:predictions) holds
# predicted values, so point to the right call.
function _predictions_error(fname)
    error("$fname compares marginal effects across modifier levels, but this is a PredictionsResult " *
          "(type=:predictions). Recompute with type=:effects (population_margins or profile_margins), " *
          "or call second_differences(model, data, variables, modifier), which does it for you.")
end
second_differences(::PredictionsResult, args...; kwargs...) = _predictions_error("second_differences")
