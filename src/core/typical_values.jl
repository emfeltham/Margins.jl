# core/typical_values.jl
#
# The one typical-value computation. Every place that needs "the typical value of a column" calls
# `typical_value`: `get_typical_values` (and through it `means_grid`, `balanced_grid`,
# `quantile_grid`'s fill-ins and `complete_reference_grid`), the `hierarchical_grid` fill-ins and
# its `(:x, :mean)` / `(:x, :median)` / `(:g, :mix_proportional)` representatives, the categorical
# frequency mixtures of reference-grid completion, and the second differences' `at=:mean` /
# `:median` (overall and per `groups` stratum).

"""
    typical_value(col, typical=mean; weights=nothing)

The typical value of one data column, optionally weighted (internal; not exported).

| column | value |
|:-------|:------|
| numeric (not `Bool`) | `typical(col)`: the (weighted) mean for `mean`, the (weighted) median for `median` |
| `Bool` | the (weighted) share of `true`, a `Float64` |
| `CategoricalArray`, strings | a `CategoricalMixture` of the observed values with their (weighted) shares |

`weights` is `nothing` (unweighted) or a vector of real, finite, non-negative weights, one per row,
not all zero; anything else is an `ArgumentError`. A row with weight 0 contributes nothing: the
result is the one computed with that row dropped. When all positive weights are equal the result
is the unweighted one on the positively weighted rows, bit for bit.

- **Weighted mean:** `Σ wᵢxᵢ / Σ wᵢ`.
- **Weighted median** (`_weighted_median`): sort the positively weighted values,
  x₍₁₎ ≤ … ≤ x₍ₘ₎, with cumulative weights Cₖ and total W. The median is the smallest x₍ₖ₎ with
  Cₖ ≥ W/2, except that when Cₖ = W/2 exactly it is the midpoint `middle(x₍ₖ₎, x₍ₖ₊₁₎)`. With equal
  weights this is `Statistics.median` (including the average of the two middle values for even m),
  for integer weights it is the median of the sample with each row repeated wᵢ times, and it is
  invariant to rescaling the weights. "Exactly" is decided up to the rounding bound of the
  cumulative sums (`m·eps·W`), so the tie, and the result, do not depend on the weights' scale.
- **Shares and mixtures:** the share of each value is the sum of its rows' weights over W.
  Mixture levels are listed in the same order as without weights.

A custom `typical` function cannot be weighted (Margins cannot know how to weight an arbitrary
function), so `weights` with a `typical` other than `mean` or `median` is an `ArgumentError`, also
for a column where `typical` would not be used. Compute such a value yourself and put it in the grid.

A column with missing values has no typical value (`MarginsError`).
"""
function typical_value(col, typical=mean; weights=nothing)
    isnothing(weights) || _check_weighted_typical(typical)
    col = _narrow_missing_eltype(col)
    if col isa AbstractVector && Missing <: eltype(col) && any(ismissing, col)
        throw(MarginsError("A column used to build the reference grid contains " *
                           "$(count(ismissing, col)) missing values, so it has no typical value. " *
                           "Build the grid from the estimation sample, e.g. " *
                           "`dropmissing(data, <model variables>)`."))
    end
    if !isnothing(weights)
        length(weights) == length(col) || throw(ArgumentError(
            "weights has $(length(weights)) entries but the column has $(length(col)) rows"))
        _check_weight_values(weights, "weights")
        pos = findall(>(0), weights)
        # Rows with weight 0 are dropped (a copy only when there are any)
        if length(pos) < length(col)
            col, weights = col[pos], weights[pos]
        end
        # Equal positive weights: the unweighted value of those rows, bit for bit
        allequal(weights) && return _typical_value(col, typical, nothing)
        return _typical_value(col, typical, Float64.(weights))
    end
    return _typical_value(col, typical, nothing)
end

# A custom `typical` cannot be weighted: say so instead of ignoring the weights
function _check_weighted_typical(typical)
    (typical === mean || typical === median) && return nothing
    throw(ArgumentError(
        "weights were given with typical=$(typical), a custom function, which cannot be weighted " *
        "(only `mean` and `median` have weighted versions here). Use typical=mean or typical=median, " *
        "drop `weights` for an unweighted $(typical), or compute the value yourself and put it in the grid."))
end

# `w` is `nothing` or the positive weights of `col`'s rows
function _typical_value(col, typical, w)
    if _is_continuous_variable(col)
        isnothing(w) && return typical(col)
        return typical === mean ? _weighted_mean(col, w) : _weighted_median(col, w)
    elseif eltype(col) <: Bool
        # The share of `true`, a Float64
        return isnothing(w) ? mean(col) : _weighted_mean(col, w)
    elseif col isa CategoricalArray || eltype(col) <: AbstractString
        return _create_frequency_mixture(col, w)
    else
        throw(MarginsError("Unsupported data type $(eltype(col)) for variable. " *
                          "Statistical correctness cannot be guaranteed for unknown data types. " *
                          "Supported types: numeric (Int64, Float64), Bool, CategoricalArray, AbstractString."))
    end
end

# The 2.x name, kept for internal callers and tests
_get_typical_value(col, typical) = typical_value(col, typical)

_weighted_mean(x, w) = sum(w .* x) / sum(w)

"""
    _weighted_median(x, w) -> Float64

The weighted median of `x` with non-negative weights `w` (see `typical_value`): the smallest value
whose cumulative weight reaches half the total, or the midpoint of it and the next value when the
cumulative weight equals half the total (up to the rounding bound `m·eps·W` of the cumulative sums).
Zero-weight rows are ignored.
"""
function _weighted_median(x, w)
    pos = findall(>(0), w)
    isempty(pos) && throw(ArgumentError("weighted median: every weight is zero"))
    p = pos[sortperm(view(x, pos))]
    xs, cw = x[p], cumsum(w[p])
    W = cw[end]
    half = W / 2
    tol = length(cw) * eps(Float64) * W
    i = something(findfirst(c -> c >= half - tol, cw), length(cw))
    return (abs(cw[i] - half) <= tol && i < length(xs)) ? Float64(middle(xs[i], xs[i + 1])) : float(xs[i])
end

"""
    _typical_quantiles(col, probs; weights=nothing) -> Vector

Quantiles of a numeric column at probabilities `probs` for the reference-grid builders
(`quantile_grid`, `hierarchical_grid`'s `:quartiles`, `(:percentiles, …)`, …). Unweighted:
`Statistics.quantile` (type 7). Weighted: the weighted analogue of type 7 that `groups=(:x, n)`
uses for weighted quantile bins (`_weighted_quantiles`), on the positively weighted rows; with
equal positive weights it is `Statistics.quantile` of those rows, bit for bit. Weights are
validated as for `typical_value`.
"""
function _typical_quantiles(col, probs; weights=nothing)
    isnothing(weights) && return [quantile(col, p) for p in probs]
    length(weights) == length(col) || throw(ArgumentError(
        "weights has $(length(weights)) entries but the column has $(length(col)) rows"))
    _check_weight_values(weights, "weights")
    pos = findall(>(0), weights)
    sub = length(pos) < length(col) ? col[pos] : col
    allequal(view(weights, pos)) && return [quantile(sub, p) for p in probs]
    return _weighted_quantiles(sub, Float64.(weights[pos]), probs)
end

"""
    _create_frequency_mixture(col, w=nothing) -> Union{Float64, CategoricalMixture}

The frequency mixture of a categorical or string column: a `CategoricalMixture` of its observed
values with their shares (weighted shares `Σ_{rows of the value} w / Σ w` when `w`, the positive
weights of the rows, is given). For a `Bool` column, the share of `true` as a `Float64`.

The mixture lists its levels in the order of an internal count table (a `Dict`, filled in row
order), which does not depend on the weights. That order is not the categorical level order; a
mixture's value does not depend on it.
"""
function _create_frequency_mixture(col, w=nothing)
    if eltype(col) <: Bool
        return isnothing(w) ? mean(col) : _weighted_mean(col, w)
    end

    level_counts = Dict()
    if isnothing(w)
        total_count = length(col)
        for value in col
            level_counts[value] = get(level_counts, value, 0) + 1
        end
    else
        total_count = sum(w)
        for (value, wi) in zip(col, w)
            level_counts[value] = get(level_counts, value, 0.0) + wi
        end
    end

    levels = collect(keys(level_counts))
    weights = [level_counts[level] / total_count for level in levels]
    return CategoricalMixture(levels, weights)
end

"""
    get_typical_values(data; typical=mean, weights=nothing) -> Dict{Symbol, Any}

`typical_value` of every column of `data` (internal). `weights` is `nothing` or one weight per row
(already resolved: a vector, not a column name). The `Dict` has no order; callers that build
columns iterate the data's column order.
"""
function get_typical_values(data; typical = mean, weights = nothing)
    isnothing(weights) || _check_weighted_typical(typical)
    data_nt = _narrow_missing_columns(data isa NamedTuple ? data : Tables.columntable(data))
    typical_values = Dict{Symbol, Any}()

    for (name, col) in pairs(data_nt)
        typical_values[name] = typical_value(col, typical; weights)
    end

    return typical_values
end
