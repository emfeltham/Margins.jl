# core/counterfactuals.jl
# One counterfactual mechanism (Margins 3.0, step 7).
#
# Every place that evaluates the model on data other than the observed values says what it
# changes with a counterfactual spec, `variable => SetTo(value)` or `variable => Shift(δ)`, and
# applies it with `_apply_counterfactuals` (a whole table) or `_counterfactual_column` (one
# column):
#
# - `scenarios=` (population contexts): `SetTo` for every row, one context per combination;
# - second differences: the levels and evaluation points of the modifier (`SetTo`, through
#   `scenarios`), and `at=:observed`, where each row's modifier is shifted by ±δ (`Shift`, two
#   contexts of one computation);
# - reference-grid completion: the model variables a grid does not give are set to their typical
#   values (`SetTo`, a column of the grid's length).
#
# The override is column-level: a changed column is one new vector, every other column of the
# table is shared, and the computation kernels read the resulting table as they read the data.
# FormulaCompiler's own row-level overrides inside the derivative and contrast evaluators (its
# counterfactual vectors for the focal variable) are a kernel mechanism, not a data
# counterfactual, and are not routed here.
#
# Anything the formula computes from a changed variable (`x^2`, `log(x)`, `x & z`) follows it,
# because the formula is re-evaluated on the changed table. A column computed outside the formula
# is a separate column and is held fixed; see `_warn_derived_columns`.

"""
    Counterfactual

What a counterfactual does to one variable (internal): [`SetTo`](@ref Margins.SetTo) sets every
row to one value, [`Shift`](@ref Margins.Shift) adds a constant to each row's own value. Applied
by `_apply_counterfactuals` / `_counterfactual_column`. A plain value in a spec means `SetTo`.
"""
abstract type Counterfactual end

"""
    SetTo(value)

Set the variable to `value` on every row (internal). `value` is coerced to the column's type: a
categorical column keeps its pool (`value` may be a level or its string), a `Bool` column accepts
`true`/`false`, `0`/`1` and their strings, and a numeric column converts it.
"""
struct SetTo{T} <: Counterfactual
    value::T
end

"""
    Shift(δ)

Add `δ` to each row's own value of a numeric variable (internal); the other variables of the
row keep their values. Used by the second differences' `at=:observed`.
"""
struct Shift <: Counterfactual
    δ::Float64
end

_as_counterfactual(cf::Counterfactual) = cf
_as_counterfactual(value) = SetTo(value)

"""
    _context_label(value)

How a scenario or group value is reported in the result's context column: a `SetTo` by its
value, a `Shift` by its δ (the second differences pair the two shifted contexts by it), a group
bin by its label, anything else as given.
"""
_context_label(cf::SetTo) = cf.value
_context_label(cf::Shift) = cf.δ
_context_label(v::NamedTuple) = haskey(v, :label) ? v.label : v
_context_label(v) = v

"""
    _counterfactual_column(cf, col, var) -> AbstractVector
    _counterfactual_column(cf::SetTo, n::Int) -> Vector

The column `col` (variable `var`) under the counterfactual `cf`: a new vector, `col` itself is
not modified. The second form builds a column of length `n` with no data column to start from
(reference-grid completion): a real, non-`Bool` value is stored as `Float64`.
"""
_counterfactual_column(cf::SetTo, col, var::Symbol) = _build_override_column(col, cf.value, var)

function _counterfactual_column(cf::Shift, col, var::Symbol)
    (col isa CategoricalArray || !(eltype(col) <: Union{Missing,Real}) || eltype(col) <: Union{Missing,Bool}) &&
        throw(ArgumentError("Cannot shift :$var by $(cf.δ): it is not a continuous numeric variable."))
    return _narrow_missing_eltype([ismissing(v) ? missing : Float64(v) + cf.δ for v in col])
end

function _counterfactual_column(cf::SetTo, n::Int)
    v = cf.value
    return v isa Real && !(v isa Bool) ? fill(Float64(v), n) : fill(v, n)
end

"""
    _apply_counterfactuals(data_nt::NamedTuple, spec) -> NamedTuple

`data_nt` with the counterfactuals of `spec` applied: `spec` maps variable names to a
[`Counterfactual`](@ref Margins.Counterfactual) or a plain value (meaning `SetTo`), as a `Dict` or
a `NamedTuple`. Columns keep their order; unchanged columns are the same vectors as in `data_nt`.
Every variable of `spec` must be a column of `data_nt`.
"""
function _apply_counterfactuals(data_nt::NamedTuple{names}, spec) where {names}
    isempty(spec) && return data_nt
    for k in keys(spec)
        Symbol(k) in names || throw(ArgumentError(
            "Counterfactual variable :$k is not a column of the data. Columns: $(join(names, ", "))"))
    end
    cols = map(names) do name
        col = getproperty(data_nt, name)
        haskey(spec, name) ? _counterfactual_column(_as_counterfactual(spec[name]), col, name) : col
    end
    return NamedTuple{names}(cols)
end

# ---- SetTo: coercion of the value to the column's type --------------------------------------

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

# Categorical overrides reuse the original pool, so evaluators still see `CategoricalValue`
# levels; Bool overrides stay `Bool`.
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
