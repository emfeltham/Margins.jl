# contrasts.jl
# Second differences from a precomputed EffectsResult: the public `second_differences(result, ...)`
# and the internal engines behind it (_second_difference, _second_differences_pairwise,
# _second_differences_all_contrasts). The local-derivative engine is in at_point.jl and the
# model-level entry point in model_interface.jl.

using LinearAlgebra: dot

"""
    second_differences(result, variables, modifier; contrast=nothing, modifier_type=:auto,
                       alpha=nothing, p_adjust=:none, include_gradients=false, warn=true)

Second differences (interaction effects) from an `EffectsResult` you already hold, from
`population_margins(...; scenarios=(modifier=levels,), type=:effects)` or
`profile_margins(model, data, cartesian_grid(modifier=levels); type=:effects)`. To go straight from a
fitted model use `second_differences(model, data, variables, modifier; …)`, which builds the scenarios
for you.

For every focal variable and contrast it compares the effect at every pair of modifier levels:
`second_diff = effect(level2) − effect(level1)`, with delta-method standard errors that use the
covariance between the two effects. Numeric (and Bool) levels are compared in ascending order; other
levels in scenario order. A continuous modifier gives slopes (the difference per unit of the modifier).

# Arguments
- `result::EffectsResult`: effects computed over the modifier. Its `groups=` variables (other than the
  modifier) define strata: the modifier levels are paired within each stratum, which gets its own rows
  with `over_<var>` columns. If another *scenario* variable also varies across its rows, an
  `ArgumentError` is raised instead of pairing rows from different cells. With `groups=modifier` (from
  `population_margins(…; groups=modifier)`), the comparison is across the observed subgroups of the
  modifier. A `PredictionsResult` is rejected.
- `variables::Union{Symbol,Vector{Symbol}}`: focal variable(s)
- `modifier::Symbol`: the moderating variable

The standard errors use the covariance matrix the result was computed with, and the `df` column
its degrees of freedom (`result.inference`), so they agree with those of the effects. For a
different covariance estimator, recompute the result with `vcov=…`, or call the model-level
form with `vcov=…`.

# Keyword Arguments
- `contrast=nothing`: `nothing` uses every contrast of each focal variable (a categorical variable has
  several, a continuous one has `"dy/dx"`). A label such as `"college - hs"` (or `"dy/dx"`; `"derivative"`
  is an alias) restricts the result to that contrast, and is an error for a variable that has no such
  contrast. Not to be confused with `contrasts=` of the model-level form, which chooses how the
  contrasts are built (`:baseline` or `:pairwise`); a result already holds its contrasts
- `modifier_type::Symbol=:auto`: `:binary`, `:categorical`, `:continuous` or `:auto`. Under `:auto`, two
  levels are `:binary` (with a warning when they are numeric and not 0/1, because the result is then the
  raw difference rather than a slope), three or more numeric levels are `:continuous`, anything else is
  `:categorical`
- `alpha=nothing`: as in `population_margins`: the (1 − alpha) intervals and the level for
  `significant`; defaults to the `alpha` the result was computed with (0.05 unless changed)
- `p_adjust::Symbol=:none`: multiple-testing adjustment over all rows returned: `:bonferroni`, `:holm`
  or `:bh` (Benjamini–Hochberg). Adds a `p_adjusted` column and bases `significant` on it
- `include_gradients::Bool=false`: add a `gradient` column (∂`second_diff`/∂β per row), as
  `DataFrame(result; include_gradients=true)` does
- `warn::Bool=true`: warn (once per call) about second differences that are zero by construction

# Structurally zero rows
When the model cannot make the effect depend on the modifier (no interaction term in an identity-link
model, or the modifier is not in the model), the gradient of the difference is zero and the second
difference is zero by construction, not estimated. Such rows report `second_diff = 0`, `se = 0`,
`significant = false` and `NaN` for `statistic`, `p_value`, `ci_lower` and `ci_upper`; they are left out
of the `p_adjust` family, and one warning per call lists them. In a nonlinear model (e.g. logit) the
effect can vary with the modifier through the link without an interaction term, so such a row is not a
structural zero on the response scale and is reported normally (with `scale=:link` it is flagged).

# Returns
A `DataFrame` with one row per focal variable × contrast × modifier pair and the columns listed under
"Output Schema" in the second-differences documentation: `variable, contrast, modifier, modifier_type,
modifier_level1, modifier_level2, ame_at_level1, ame_at_level2, second_diff, se, statistic, df,
p_value, ci_lower, ci_upper, significant`.

# Examples
```julia
ames = population_margins(model, data; scenarios=(treated=[0,1],), type=:effects)
second_differences(ames, :age, :treated)                       # binary modifier, 1 row
second_differences(ames, [:age, :education], :treated)         # several focal variables

ames = population_margins(model, data; vars=[:education], contrasts=:pairwise,
                          scenarios=(region=[:north, :south, :west],), type=:effects)
second_differences(ames, :education, :region)                  # 3 contrasts × 3 region pairs = 9 rows
second_differences(ames, :education, :region; contrast="college - hs")
```
"""
function second_differences(
    ame_result::EffectsResult,
    variables::Union{Symbol,Vector{Symbol}},
    modifier::Symbol;
    contrast::Union{Nothing,AbstractString}=nothing,
    modifier_type::Symbol=:auto,
    all_contrasts=nothing,
    vcov=nothing,
    alpha=nothing,
    p_adjust::Symbol=:none,
    include_gradients::Bool=false,
    warn::Bool=true
)
    isnothing(all_contrasts) || throw(ArgumentError("`all_contrasts` was removed: leave `contrast` unset " *
        "for every contrast (the default), or pass `contrast=\"label\"` for one."))
    isnothing(vcov) || throw(ArgumentError("`vcov` was removed from second_differences(result, …) in " *
        "Margins 3.0: the result's own covariance matrix is used (result.inference.vcov). Recompute " *
        "the result with vcov=…, or call second_differences(model, data, variables, modifier; vcov=…)."))
    _check_p_adjust(p_adjust)
    alpha = _resolve_alpha(alpha, ame_result)

    # Normalize variables to Vector{Symbol}
    vars_vec = variables isa Symbol ? [variables] : variables

    # One block of rows per `groups=` stratum (the whole result when it has no groups), each with
    # its over_<var> columns
    frames = DataFrame[]
    for (stratum, over) in _result_strata(ame_result, modifier)
        frame = _second_differences_of(stratum, vars_vec, modifier, contrast, modifier_type, alpha, warn)
        for (k, v) in pairs(over)
            frame[!, k] = fill(v, nrow(frame))
        end
        push!(frames, frame)
    end
    out = length(frames) == 1 ? only(frames) : vcat(frames...; cols=:union)
    return _finalize_frame!(out, p_adjust, alpha, include_gradients; warn)
end

# The second differences of one stratum (a result that varies over the modifier only), every focal
# variable, gradients included
function _second_differences_of(ame_result::EffectsResult, vars_vec::Vector{Symbol}, modifier::Symbol,
                                contrast, modifier_type::Symbol, alpha::Float64, warn::Bool)
    idx = _effect_index(ame_result)
    results = DataFrame[]
    for variable in vars_vec
        # Get information about the variable's contrasts
        variable_contrasts = _variable_contrasts(idx, variable)

        # Every contrast (the default) or the one asked for
        if isnothing(contrast) && length(variable_contrasts) > 1
            df = _second_differences_all_contrasts(
                ame_result, variable, modifier;
                modifier_type=modifier_type, alpha=alpha, include_gradients=true, warn=warn
            )
        else
            # One contrast: the one asked for, or the variable's only one
            if isnothing(contrast)
                _require_variable(idx, variable)
                contrast_to_use = only(variable_contrasts)
            else
                contrast_to_use = String(contrast)
            end
            df = _second_differences_pairwise(
                ame_result, variable, modifier;
                contrast=contrast_to_use,
                modifier_type=modifier_type, alpha=alpha, include_gradients=true, warn=warn
            )
        end
        push!(results, df)
    end
    return length(results) == 1 ? only(results) : vcat(results..., cols=:union)
end

# The covariance matrix of a result's parameters (`result.inference.vcov`), checked against its
# gradients (a hand-built result could disagree)
function _result_vcov(r::MarginsResult)::Matrix{Float64}
    V = r.inference.vcov
    size(V, 1) == size(r.gradients, 2) || error("Malformed result: its covariance matrix has " *
        "$(size(V, 1)) parameters but its gradients have $(size(r.gradients, 2)).")
    return V
end

# Two numeric levels other than 0/1 are auto-typed :binary, i.e. a raw difference of AMEs, not a slope.
function _warn_numeric_binary(modifier::Symbol, levels)
    all(x -> x isa Number && !(x isa Bool), levels) && !all(x -> x == 0 || x == 1, levels) || return nothing
    @warn "Modifier :$modifier has two numeric levels ($(levels[1]), $(levels[2])) " *
          "and is treated as :binary, so second_diff is the raw difference in AMEs. " *
          "Pass modifier_type=:continuous for the slope per unit of :$modifier."
    return nothing
end

# Engine: exactly two modifier levels for one variable and contrast. Returns one row of the unified
# schema as a NamedTuple, gradient included. `second_differences` is the public entry point.
function _second_difference(
    ame_result::EffectsResult, variable::Symbol, modifier::Symbol;
    contrast="dy/dx", alpha=nothing
)
    idx = _effect_index(ame_result)
    _require_modifier(idx, modifier)
    vcov = _result_vcov(ame_result)

    # Rows for this variable and contrast, one per modifier level, in canonical order
    indices, modifier_levels = _modifier_rows(idx, variable, modifier, contrast)

    if length(indices) != 2
        throw(ArgumentError("Expected 2 rows (one for each $modifier level), got $(length(indices))"))
    end

    idx1, idx2 = indices
    level1, level2 = modifier_levels

    # Get estimates
    est_level1 = ame_result.estimates[idx1]
    est_level2 = ame_result.estimates[idx2]
    second_diff = est_level2 - est_level1

    # Get gradients for each AME
    g1 = ame_result.gradients[idx1, :]
    g2 = ame_result.gradients[idx2, :]

    # Gradient of the difference: g_diff = g2 - g1
    g_diff = g2 .- g1

    # Delta method SE: sqrt((g2-g1)' * Σ * (g2-g1))
    # This properly accounts for the covariance between the two AMEs
    variance = dot(g_diff, vcov, g_diff)
    se_diff = sqrt(max(0.0, variance))  # Ensure non-negative due to numerical precision

    return _second_difference_row(;
        variable, contrast = contrast, modifier, modifier_type = :binary,
        level1, level2, est1 = est_level1, est2 = est_level2,
        second_diff, se = se_diff, gradient = g_diff, dof = ame_result.inference.df,
        alpha = _resolve_alpha(alpha, ame_result), structural_zero = _is_structural_zero(g_diff, g1, g2),
    )
end

# Engine: all focal contrasts of one variable × all modifier pairs. A binary modifier goes through
# `_second_difference` per contrast, anything else through `_second_differences_pairwise`.
# Contrasts with fewer than two modifier levels are skipped with a warning. Public entry point:
# `second_differences`.
function _second_differences_all_contrasts(
    ame_result::EffectsResult, variable::Symbol, modifier::Symbol;
    modifier_type::Symbol=:auto,
    alpha=nothing,
    p_adjust::Symbol=:none,
    include_gradients::Bool=false,
    warn::Bool=true
)
    _check_p_adjust(p_adjust)
    alpha = _resolve_alpha(alpha, ame_result)

    idx = _effect_index(ame_result)
    contrasts = _variable_contrasts(idx, variable)

    # Number of modifier levels (from the first contrast) decides which approach to use
    n_modifier_levels = isempty(contrasts) ? 0 :
        length(_modifier_rows(idx, variable, modifier, first(contrasts))[1])

    # Contrasts without usable rows are skipped with a warning
    skipped = [
        c for c in contrasts
        if length(_modifier_rows(idx, variable, modifier, c)[1]) < 2
    ]
    if warn && !isempty(skipped)
        @warn "Skipped $(length(skipped)) contrast(s) for variable $variable with too few modifier levels: $(join(skipped, ", "))"
    end
    kept = [c for c in contrasts if c ∉ skipped]
    isempty(kept) && throw(ArgumentError("No valid contrasts found for variable $variable with modifier $modifier"))

    if n_modifier_levels == 2 && modifier_type ∈ (:auto, :binary)
        # Binary modifier: one comparison per contrast
        rows = NamedTuple[]
        warn && modifier_type === :auto && _warn_numeric_binary(modifier, _modifier_rows(idx, variable, modifier, first(kept))[2])
        for contrast in kept
            if length(_modifier_rows(idx, variable, modifier, contrast)[1]) == 2
                push!(rows, _second_difference(ame_result, variable, modifier; contrast, alpha))
            elseif warn
                @warn "Skipped contrast '$contrast' for variable $variable: not exactly 2 modifier levels"
            end
        end
        isempty(rows) && throw(ArgumentError("No valid contrasts found for variable $variable with modifier $modifier"))
        return _finalize_frame!(_second_difference_frame(rows), p_adjust, alpha, include_gradients)
    else
        # >2 modifier levels (or explicit categorical/continuous): all pairs per contrast
        frames = [
            _second_differences_pairwise(
                ame_result, variable, modifier; contrast, modifier_type, alpha, include_gradients=true, warn
            ) for contrast in kept
        ]
        return _finalize_frame!(vcat(frames..., cols=:union), p_adjust, alpha, include_gradients)
    end
end

# Engine: all pairs of modifier levels for one variable and contrast. Rows are selected by
# `_modifier_rows` (numeric levels ascending; error if other scenario/group columns vary), the
# difference of the two effects has variance (g₂−g₁)'Σ(g₂−g₁), and a continuous modifier divides
# estimate, SE and gradient by the modifier distance. Public entry point: `second_differences`.
function _second_differences_pairwise(
    ame_result::EffectsResult,
    variable::Symbol,
    modifier::Symbol;
    contrast::String="dy/dx",
    modifier_type::Symbol=:auto,
    alpha=nothing,
    p_adjust::Symbol=:none,
    include_gradients::Bool=false,
    warn::Bool=true
)
    _check_p_adjust(p_adjust)
    alpha = _resolve_alpha(alpha, ame_result)

    idx = _effect_index(ame_result)
    _require_modifier(idx, modifier)
    vcov = _result_vcov(ame_result)

    # Rows for this variable and contrast, one per modifier level, in canonical order
    # (numeric levels ascending, others in scenario order), so level1 < level2 for numeric modifiers
    indices, modifier_levels = _modifier_rows(idx, variable, modifier, contrast)

    if isempty(indices)
        _require_variable(idx, variable)
        throw(ArgumentError("No rows for variable :$variable with contrast '$contrast'. Contrasts available: " *
              join(_variable_contrasts(idx, variable), ", ")))
    end
    if length(indices) < 2
        throw(ArgumentError("Need at least 2 modifier levels, found $(length(indices)) for variable $variable with contrast $contrast"))
    end

    # Auto-detect modifier type if requested
    n_levels = length(modifier_levels)

    if modifier_type == :auto
        if n_levels == 2
            modifier_type = :binary
            warn && _warn_numeric_binary(modifier, modifier_levels)
        elseif all(x -> x isa Number, modifier_levels)
            modifier_type = :continuous
        else
            modifier_type = :categorical
        end
    end

    # Generate all pairwise combinations (i < j to avoid duplicates)
    level_pairs = [(i, j) for i in 1:length(indices), j in 1:length(indices) if i < j]

    if isempty(level_pairs)
        error("No pairwise combinations found for modifier $modifier")
    end

    # Compute second difference for each pair
    results = []

    for (i, j) in level_pairs
        idx1, idx2 = indices[i], indices[j]
        level1, level2 = modifier_levels[i], modifier_levels[j]

        # Get estimates
        est_level1 = ame_result.estimates[idx1]
        est_level2 = ame_result.estimates[idx2]

        # Get gradients
        g1 = ame_result.gradients[idx1, :]
        g2 = ame_result.gradients[idx2, :]
        g_diff = g2 .- g1

        # Delta method variance
        variance = dot(g_diff, vcov, g_diff)
        se_raw = sqrt(max(0.0, variance))

        # For continuous modifiers, compute slope and scale SE
        if modifier_type == :continuous
            if !isa(level1, Number) || !isa(level2, Number)
                throw(ArgumentError("Continuous modifier requires numeric levels, got $(typeof(level1)) and $(typeof(level2))"))
            end

            modifier_diff = abs(level2 - level1)
            if modifier_diff ≈ 0.0
                @warn "Modifier levels are nearly identical ($level1 ≈ $level2), skipping"
                continue
            end

            # Slope: change in AME per unit change in modifier
            second_diff = (est_level2 - est_level1) / modifier_diff
            se_diff = se_raw / modifier_diff
            # Scale gradient to match scaled second_diff
            g_diff_scaled = g_diff ./ modifier_diff
        else
            # Categorical/binary: simple difference
            second_diff = est_level2 - est_level1
            se_diff = se_raw
            g_diff_scaled = g_diff
        end

        push!(results, _second_difference_row(;
            variable, contrast, modifier, modifier_type,
            level1, level2, est1 = est_level1, est2 = est_level2,
            second_diff, se = se_diff, gradient = g_diff_scaled, dof = ame_result.inference.df,
            alpha, structural_zero = _is_structural_zero(g_diff, g1, g2),
        ))
    end

    if isempty(results)
        throw(ArgumentError("No valid pairwise contrasts computed for variable $variable with modifier $modifier"))
    end

    return _finalize_frame!(_second_difference_frame(results), p_adjust, alpha, include_gradients)
end
