# core/data_conversion.jl
# Data type conversion utilities for type stability

"""
    _convert_numeric_to_float64(data_nt::NamedTuple) -> NamedTuple

Convert all numeric columns in a NamedTuple to Float64 for type stability.

This function ensures homogeneous numeric types in the NamedTuple, preventing
type instability that causes allocations in downstream code. Categorical and
other non-numeric columns are preserved as-is.

# Type Stability Issue
Heterogeneous NamedTuples like:
```julia
@NamedTuple{x1::Vector{Float64}, x2::Vector{Int64}, cat::CategoricalVector{...}}
```
cause type instability when accessing columns generically, leading to allocations.

# Solution
Convert all numeric types to Float64:
```julia
@NamedTuple{x1::Vector{Float64}, x2::Vector{Float64}, cat::CategoricalVector{...}}
```
This allows type-stable column access for numeric operations.

# Arguments
- `data_nt::NamedTuple`: Input data as NamedTuple from Tables.columntable()

# Returns
- `NamedTuple`: Data with all numeric columns converted to Float64

# Performance
- Converts Int64, Int32, Float32, etc. to Float64
- Preserves CategoricalArray, String, Bool columns
- One-time conversion cost, eliminates per-row allocations

# Examples
```julia
# Before: heterogeneous types
data = (x1=[1.0, 2.0], x2=[1, 2], cat=categorical(["A", "B"]))
# Type: @NamedTuple{x1::Vector{Float64}, x2::Vector{Int64}, cat::CategoricalVector}

# After: homogeneous numeric types
data_converted = _convert_numeric_to_float64(data)
# Type: @NamedTuple{x1::Vector{Float64}, x2::Vector{Float64}, cat::CategoricalVector}
```
"""
function _convert_numeric_to_float64(data_nt::NamedTuple{names}) where {names}
    # Build new NamedTuple with Float64 numeric columns
    converted_cols = map(names) do name
        col = _narrow_missing_eltype(getproperty(data_nt, name))
        # Convert numeric columns (excluding Bool and CategoricalArray)
        if col isa AbstractVector{<:Real} && !(eltype(col) <: Bool) && !(eltype(col) === Float64)
            # Need to convert: Int64, Int32, Float32, etc. (a per-call copy)
            return Float64.(col)
        else
            # Keep as-is: Float64 (already correct), Bool, CategoricalArray, String, etc.
            return col
        end
    end
    # Reconstruct NamedTuple with same names
    return NamedTuple{names}(converted_cols)
end

# =============================================================================
# Columns whose element type admits `missing`
# =============================================================================
#
# CSV.jl, `allowmissing`, and `dropmissing(df, cols)` (which narrows only `cols`) all
# produce columns typed `Union{Missing, T}` that may hold no missing value at all.
# FormulaCompiler compiles on the column *types*, and a `Union{Missing, T}` column used to
# fail deep inside it with an opaque `MethodError: no method matching fieldindex(...)`.
# Such columns are narrowed here to their non-missing element type; columns that do hold
# missing values and are model variables are rejected with an actionable error by
# `_check_model_vars_not_missing`.

"""
    _narrow_missing_eltype(col) -> AbstractVector

Narrow a `Union{Missing, T}` column that holds no `missing` to a column of element type
`T` (a `Vector{T}`, or a non-missing `CategoricalArray` with the same levels). Columns that
do hold `missing` values, and columns without `Missing` in their element type, are returned
unchanged.

The narrowed copy is a real `Vector`/`CategoricalArray` because FormulaCompiler's
derivative and contrast machinery dispatches on those concrete types. It is made on every
call, so in-place edits of the source are always picked up.
"""
function _narrow_missing_eltype(col::AbstractVector)
    E = eltype(col)
    (E === Missing || !(Missing <: E)) && return col
    any(ismissing, col) && return col
    return _narrow_copy(col)
end
_narrow_missing_eltype(col) = col

"""
    _narrow_missing_columns(data_nt::NamedTuple) -> NamedTuple

Apply `_narrow_missing_eltype` to every column (used by the reference-grid builders).
"""
_narrow_missing_columns(data_nt::NamedTuple) = map(_narrow_missing_eltype, data_nt)

function _narrow_copy(col::AbstractVector)
    T = nonmissingtype(eltype(col))
    return col isa CategoricalArrays.CategoricalArray ?
        convert(CategoricalArrays.CategoricalArray{T}, col) : convert(Vector{T}, col)
end

_unwrap_value(x) = x isa CategoricalArrays.CategoricalValue ? CategoricalArrays.unwrap(x) : x

"""
    _shared_compiled(model, data) -> UnifiedCompiled

FormulaCompiler's process-wide cached compiled formula for `(model, data schema)`. The
cache is FormulaCompiler's, guarded by its own lock (it compiles outside the lock), so
concurrent Margins calls need no lock here. The returned object is SHARED: use it only for
read-only inspection (its `ops`, `continuous_variables`, `categorical_levels`, `length`).
Anything that evaluates rows must use `_private_compiled`, because evaluation writes into
the compiled object's scratch buffer.
"""
_shared_compiled(model, data) = get_or_compile_formula(model, data)

"""
    _private_compiled(model, data) -> UnifiedCompiled

A compiled formula for `(model, data schema)` with its own scratch buffer: the cached
operations (shared; their only mutable state, FormulaCompiler's categorical level cache, is
published atomically since FC 1.5.0) with a freshly allocated scratch vector, so concurrent
Margins calls on the same model never write into the same buffer. Costs one small
allocation (the scratch vector) instead of a recompilation.
"""
_private_compiled(model, data) = _with_private_scratch(_shared_compiled(model, data))
_with_private_scratch(c::FormulaCompiler.UnifiedCompiled{T, Ops, S, O}) where {T, Ops, S, O} =
    with_private_scratch(c)

"""
    _model_variable_names(model) -> Vector{Symbol}

Variables named in the model's formula (response included), or an empty vector when the
formula cannot be obtained.
"""
function _model_variable_names(model)
    # Walks the FULL formula on purpose (not `fixed_effects_form`): the missing-value check
    # must include MixedModels grouping variables, which the fixed-effects form drops.
    try
        f = StatsModels.formula(model)
        return collect(Symbol, StatsModels.termvars(f))
    catch
        return Symbol[]
    end
end

"""
    _check_model_vars_present(data_nt, model)

Throw an `ArgumentError` naming the columns when a predictor of the model's (fixed-effects)
formula is not a column of `data_nt`. Without this check the first evaluation of the formula
fails inside FormulaCompiler with a `FieldError` (Julia ≥ 1.12) or an `ErrorException`. The
response is not required: Margins never evaluates it.
"""
function _check_model_vars_present(data_nt::NamedTuple, model)
    absent = sort!([v for v in _get_model_formula_variables(model) if !haskey(data_nt, v)])
    isempty(absent) && return nothing
    throw(ArgumentError(
        "data is missing the model variable(s) $(join(repr.(absent), ", ")). Margins evaluates " *
        "the fitted formula on `data`, so it needs every predictor column the model was fit " *
        "with (columns in `data`: $(join(repr.(collect(keys(data_nt))), ", ")))."))
end

"""
    _check_model_vars_not_missing(data_nt, model)

Throw an actionable `MarginsError` when a model variable in `data_nt` contains `missing`.

Margins evaluates the model on every row of the data it is given. Rows with a missing
model variable cannot be evaluated, and are not part of the estimation sample (the fit
dropped them), so they must be removed before calling Margins.
"""
function _check_model_vars_not_missing(data_nt::NamedTuple, model)
    # Fast path (the common case): no column holds a missing value, so there is nothing to
    # check and no need to walk the model formula.
    _has_missing_value(col) = col isa AbstractVector && Missing <: eltype(col) && any(ismissing, col)
    any(_has_missing_value, values(data_nt)) || return nothing
    offending = Symbol[]
    for v in _model_variable_names(model)
        haskey(data_nt, v) || continue
        col = getproperty(data_nt, v)
        col isa AbstractVector || continue
        Missing <: eltype(col) && any(ismissing, col) && push!(offending, v)
    end
    isempty(offending) && return nothing
    counts = join(["`:$v` ($(count(ismissing, getproperty(data_nt, v))) missing)" for v in offending], ", ")
    vars_list = join(repr.(_model_variable_names(model)), ", ")
    throw(MarginsError(
        "Model variable(s) $counts contain missing values. Margins evaluates the model " *
        "on every row of `data`, and rows with missing model variables were not part of " *
        "the estimation sample. Pass the estimation sample instead, e.g. " *
        "`dropmissing(data, [$vars_list])`."))
end

"""
    _prepare_margins_data(data_nt_raw::NamedTuple, model) -> NamedTuple

Entry-point preprocessing of the user's data (a columntable): reject model variables that
contain missing values (`_check_model_vars_not_missing`), then narrow `Union{Missing,T}`
columns and convert numeric columns to `Float64` (`_convert_numeric_to_float64`).

Categorical pools that lack some of the model's levels need no handling here:
FormulaCompiler resolves categorical values, and contrast endpoints, against the fitted
model's levels.
"""
function _prepare_margins_data(data_nt_raw::NamedTuple, model)
    _check_model_vars_present(data_nt_raw, model)
    _check_model_vars_not_missing(data_nt_raw, model)
    data_nt = _categorize_model_categoricals(_narrow_missing_columns(data_nt_raw), model)
    return _convert_numeric_to_float64(data_nt)
end

"""
    _model_categorical_levels(model, data_nt) -> Dict{Symbol, Vector}

The variables the fitted model treats as categorical and their levels in model order, from
FormulaCompiler's `categorical_levels` on the compiled formula. Decided by the model's
contrasts, not by column element types, so an integer or String column declared
categorical (e.g. `contrasts = Dict(:k => DummyCoding())`) is included.
"""
function _model_categorical_levels(model, data_nt::NamedTuple)
    # No try/catch: a model Margins accepts always compiles (the engine is built from the
    # same compiled formula), and swallowing an error here would silently disable the
    # model-based classification.
    return categorical_levels(_shared_compiled(model, data_nt))
end

"""
    _categorize_model_categoricals(data_nt, model) -> NamedTuple

Convert every column the MODEL treats as categorical but that is not a `CategoricalArray`
— e.g. an integer, Float or Bool column fitted with `contrasts = Dict(:k => DummyCoding())` — to
a `CategoricalArray` whose levels are the model's levels (followed by any other observed
values). Margins' downstream logic (variable detection, contrast enumeration, `groups`,
typical values) then sees the model's view of the variable instead of treating the integer
codes as a continuous variable. The converted column is a per-call copy.
"""
function _categorize_model_categoricals(data_nt::NamedTuple, model)
    any(c -> c isa AbstractVector && !(c isa CategoricalArrays.CategoricalArray), values(data_nt)) ||
        return data_nt
    cats = _model_categorical_levels(model, data_nt)
    isempty(cats) && return data_nt
    replaced = Dict{Symbol, Any}()
    for (v, levs) in cats
        haskey(data_nt, v) || continue
        col = getproperty(data_nt, v)
        # Integer, Float and Bool columns fitted as categorical are all converted; a Bool
        # used as a continuous regressor (no contrast) is not in `cats` and is untouched.
        col isa CategoricalArrays.CategoricalArray && continue
        col isa AbstractVector || continue
        cc = CategoricalArrays.categorical(col)
        observed = CategoricalArrays.unwrap.(levels(cc))
        extra = [x for x in observed if !any(l -> isequal(l, x), levs)]
        replaced[v] = CategoricalArrays.levels!(cc, vcat(convert(Vector{eltype(observed)}, collect(levs)), extra))
    end
    isempty(replaced) && return data_nt
    return NamedTuple{keys(data_nt)}(map(k -> get(replaced, k, getproperty(data_nt, k)), keys(data_nt)))
end
