# core/derived_columns.jl
# The derived-column detector (Margins 3.0, step 7; ISSUES #2b).
#
# Margins changes a variable by re-evaluating the model's formula on changed data
# (`_apply_counterfactuals`, or a derivative/contrast of a focal variable). Whatever the formula
# computes from the variable (`x^2`, `log(x)`, `x & z`) follows it. A transform computed outside
# the formula and stored as its own column (`x2 = x.^2`) is, to the model, a separate regressor,
# so it is held fixed: the AME of `x` is then ∂ŷ/∂x with `x2` held fixed, and a scenario that sets
# `x` leaves `x2` at its observed values. This is a modelling choice the formula makes, so it is a
# warning, not an error (a column that happens to be such a function can legitimately be held
# fixed).
#
# The detector runs once per call, from `_normalize_request`, over the focal variables and the
# variables the call changes. For each such variable v and each other numeric column c the
# model's formula uses (never the response), it reports c when c is, in the data, an exact
# function of v (up to floating-point rounding):
#
# 1. a common transform of v: v^2, v^3, sqrt(v), log(v), log10(v), log(v + 1), exp(v), 1/v, or
#    a linear rescaling a*v + b (a standardized or rescaled copy);
# 2. a product v * u with another numeric model column u (a precomputed interaction).
#
# There is deliberately no general "c is constant within each value of v" (lookup) rule: it is
# vacuous for a variable whose values are (nearly) all distinct, and on a discrete variable it
# fires on legitimate group-level covariates (a year's GDP beside `year`). The docs carry the
# general rule; the detector reports only columns that are almost certainly transforms. Every
# rule is an exact check over all rows (no correlation threshold): a cheap screen on a few rows
# first, then one pass over the data for the columns that pass it.

const _DERIVED_SAMPLE = 2048   # rows used to fit an affine relation before the full pass
const _DERIVED_SCREEN = 32     # rows checked before a full pass
const _DERIVED_RTOL = 1e-8     # relative tolerance of "equal up to rounding"

# One detected pair: `derived` looks like `relation` of the changed variable `source`
struct _DerivedColumn
    source::Symbol
    derived::Symbol
    relation::String
    kind::Symbol          # :transform, :affine, :product
    eval::Any             # how to compute `derived` at a counterfactual value (see _derived_value)
end

# Evenly spaced row numbers, at most `k` of them
_spaced_rows(n::Int, k::Int) = n <= k ? collect(1:n) : unique!([round(Int, x) for x in range(1, n; length=k)])

_close(a::Float64, b::Float64) = a == b || abs(a - b) <= _DERIVED_RTOL * max(abs(a), abs(b))

_is_numeric_column(col) = col isa AbstractVector && !(col isa CategoricalArray) && eltype(col) <: Real

# The transforms of rule 1, by number: their label and value (`NaN` outside the domain). One
# function with a branch, not one closure each, so the checks compile once per column type pair.
const _DERIVED_TRANSFORM_LABELS = ("%^2", "%^3", "sqrt(%)", "log(%)", "log10(%)", "log(% + 1)", "exp(%)", "1/%")
_transform_label(k::Int, v::Symbol) = replace(_DERIVED_TRANSFORM_LABELS[k], "%" => string(v))

function _transform(k::Int, x::Float64)::Float64
    k == 1 && return x^2
    k == 2 && return x^3
    k == 3 && return x >= 0 ? sqrt(x) : NaN
    k == 4 && return x > 0 ? log(x) : NaN
    k == 5 && return x > 0 ? log10(x) : NaN
    k == 6 && return x > -1 ? log1p(x) : NaN
    k == 7 && return x < 700 ? exp(x) : NaN
    return x != 0 ? 1 / x : NaN
end

# c[i] == transform k of v[i] up to rounding on every row, `screen` rows first (a value outside
# the transform's domain, NaN, never matches)
function _matches_transform(k::Int, v::AbstractVector, c::AbstractVector, screen::Vector{Int})
    for i in screen
        _close(Float64(c[i]), _transform(k, Float64(v[i]))) || return false
    end
    @inbounds for i in eachindex(v, c)
        _close(Float64(c[i]), _transform(k, Float64(v[i]))) || return false
    end
    return true
end

# (a, b) with c == a*v + b on every row (a ≠ 0), or nothing
function _affine_relation(v::AbstractVector, c::AbstractVector, sample)
    lo = hi = first(sample)
    for i in sample
        v[i] < v[lo] && (lo = i)
        v[i] > v[hi] && (hi = i)
    end
    v[lo] == v[hi] && return nothing
    a = (Float64(c[hi]) - Float64(c[lo])) / (Float64(v[hi]) - Float64(v[lo]))
    (a == 0 || !isfinite(a)) && return nothing
    b = Float64(c[lo]) - a * Float64(v[lo])
    fits(i) = (p = a * Float64(v[i]) + b;
               abs(Float64(c[i]) - p) <= _DERIVED_RTOL * (abs(Float64(c[i])) + abs(a * Float64(v[i])) + abs(b)))
    all(fits, sample) || return nothing
    @inbounds for i in eachindex(v, c)
        fits(i) || return nothing
    end
    return (a, b)
end

# c[i] == v[i] * u[i] up to rounding on every row
function _matches_product(v::AbstractVector, u::AbstractVector, c::AbstractVector, screen)
    for i in screen
        _close(Float64(c[i]), Float64(v[i]) * Float64(u[i])) || return false
    end
    @inbounds for i in eachindex(v, u, c)
        _close(Float64(c[i]), Float64(v[i]) * Float64(u[i])) || return false
    end
    return true
end

"""
    _derived_columns(model, data_nt, sources) -> Vector{_DerivedColumn}

The numeric model columns that are, in `data_nt`, an exact function of one of the variables
`sources` (see the rules at the top of `core/derived_columns.jl`). Each unordered pair of columns
is reported once.
"""
function _derived_columns(model, @nospecialize(data_nt::NamedTuple), sources::Vector{Symbol})
    # The columns in an untyped Dict, so nothing below is compiled for the data table's type (only
    # for each column type, behind the calls); a categorical column is read through its codes.
    raw = Dict{Symbol,AbstractVector}()
    for k in keys(data_nt)
        raw[k] = getfield(data_nt, k)
    end
    predictors = try
        _formula_predictor_names(model)
    catch
        return _DerivedColumn[]
    end
    response = _response_name(model)
    return _derived_columns(raw, Vector{Symbol}(predictors), response, sources)
end

# The detector proper, independent of the model's type (precompiled below)
function _derived_columns(raw::Dict{Symbol,AbstractVector}, predictors::Vector{Symbol},
                          response::Union{Nothing,Symbol}, sources::Vector{Symbol})
    found = _DerivedColumn[]
    model_cols = Symbol[s for s in predictors if haskey(raw, s) && s !== response]
    cols = Dict{Symbol,AbstractVector}(s => _detector_column(raw[s]) for s in model_cols)
    numeric = Symbol[s for s in model_cols if _is_numeric_column(raw[s]) && !_is_constant(cols[s])]
    isempty(numeric) && return found
    n = length(raw[first(numeric)])
    n >= 3 || return found
    sample = _spaced_rows(n, _DERIVED_SAMPLE)
    screen = _spaced_rows(n, _DERIVED_SCREEN)
    pair_seen = Set{Tuple{Symbol,Symbol}}()
    for v in sources
        (v in model_cols) || continue
        vcol = cols[v]
        _is_constant(vcol) && continue
        v_numeric = _is_numeric_column(raw[v])
        v_continuous = v_numeric && !(eltype(vcol) <: Bool) && _n_distinct(vcol, screen) >= 3
        for c in numeric
            c == v && continue
            (minmax(v, c) in pair_seen) && continue
            d = _relation(v, c, cols, numeric, sample, screen, v_numeric, v_continuous)
            isnothing(d) && continue
            push!(found, d)
            push!(pair_seen, minmax(v, c))
        end
    end
    return found
end

_detector_column(col::CategoricalArray) = CategoricalArrays.refs(col)
_detector_column(col::AbstractVector) = col

_is_constant(col::AbstractVector) = isempty(col) || all(isequal(first(col)), col)

function _n_distinct(col::AbstractVector, rows::Vector{Int})
    seen = Set{Float64}()
    for i in rows
        push!(seen, Float64(col[i]))
    end
    return length(seen)
end

function _relation(v::Symbol, c::Symbol, cols::Dict{Symbol,AbstractVector}, numeric::Vector{Symbol},
                   sample::Vector{Int}, screen::Vector{Int}, v_numeric::Bool, v_continuous::Bool)
    vcol = cols[v]
    ccol = cols[c]
    if v_continuous
        for k in eachindex(_DERIVED_TRANSFORM_LABELS)
            _matches_transform(k, vcol, ccol, screen) &&
                return _DerivedColumn(v, c, _transform_label(k, v), :transform, k)
        end
        ab = _affine_relation(vcol, ccol, sample)
        if !isnothing(ab)
            (a, b) = ab
            label = a == 1 && b == 0 ? "a copy of $v" : "a linear rescaling of $v (a standardized or rescaled copy)"
            return _DerivedColumn(v, c, label, :affine, ab)
        end
    end
    if v_numeric
        for u in numeric
            (u == v || u == c) && continue
            _matches_product(vcol, cols[u], ccol, screen) &&
                return _DerivedColumn(v, c, "$v * $u (a precomputed interaction)", :product, u)
        end
    end
    return nothing
end

# The value `d.derived` would take where `d.source` is `x` (and the product's other factor `u`),
# or `nothing` if it cannot be computed (a non-number)
function _derived_value(d::_DerivedColumn, x, u)
    x isa Real || return nothing
    if d.kind === :transform
        return _transform(d.eval::Int, Float64(x))
    elseif d.kind === :affine
        (a, b) = d.eval
        return a * Float64(x) + b
    else # :product
        u isa Real || return nothing
        return Float64(x) * Float64(u)
    end
end

# Do the call's settings (one per scenario combination or completed grid row) set `d.derived`
# consistently with `d.source`, so that it is not held fixed? `settings` is `nothing` when the
# values are not known here (the model-level second differences set them later).
function _set_consistently(d::_DerivedColumn, settings, set_vars)
    (d.derived in set_vars && d.source in set_vars) || return false
    isnothing(settings) && return false
    for row in _setting_rows(settings)
        haskey(row, d.source) && haskey(row, d.derived) || return false
        u = d.kind === :product ? get(row, d.eval, nothing) : nothing
        d.kind === :product && isnothing(u) && return false
        want = try
            _derived_value(d, _unwrap_setting(row[d.source]), _unwrap_setting(u))
        catch
            nothing
        end
        have = _unwrap_setting(row[d.derived])
        isnothing(want) && return false
        if want isa Real && have isa Real
            _close(Float64(want), Float64(have)) || return false
        else
            isequal(want, have) || return false
        end
    end
    return true
end

_setting_rows(settings::AbstractVector) = settings
_setting_rows(grid::DataFrame) = (Dict(Symbol(k) => v for (k, v) in pairs(r)) for r in eachrow(grid))
_unwrap_setting(x::SetTo) = x.value
_unwrap_setting(x::CategoricalValue) = CategoricalArrays.unwrap(x)
_unwrap_setting(x) = x

"""
    _warn_derived_columns(model, data_nt, focal, changed, settings)

Warn, once, about model columns that look like functions of a variable the call changes but are
separate columns of the data (`_derived_columns`), so the formula cannot update them. `focal` are
the focal variables (a derivative or contrast moves them; anything derived from them is held
fixed), `changed` the variables set counterfactually, and `settings` the values the call sets
(`scenario_specs`, or the completed reference grid; `nothing` if not known): a derived column that
the call also sets, consistently with its source at every setting, is not reported.
"""
function _warn_derived_columns(model, @nospecialize(data_nt::NamedTuple), focal::Vector{Symbol}, changed::Vector{Symbol}, settings)
    sources = unique!(Symbol[focal; changed])
    isempty(sources) && return nothing
    found = _derived_columns(model, data_nt, sources)
    isempty(found) && return nothing
    return _report_derived_columns(found, focal, changed, settings)
end

function _report_derived_columns(found::Vector{_DerivedColumn}, focal::Vector{Symbol},
                                 changed::Vector{Symbol}, settings)
    set_vars = Set(changed)
    filter!(d -> d.source in focal || !_set_consistently(d, settings, set_vars), found)
    isempty(found) && return nothing
    lines = ["  - `$(d.derived)` is $(d.relation) in the data: changing `$(d.source)` holds `$(d.derived)` fixed"
             for d in found]
    @warn "Some model columns look like functions of a variable this call changes, but they are " *
          "separate columns of the data rather than terms of the formula:\n" * join(lines, "\n") *
          "\nMargins changes a variable by re-evaluating the formula, so a column computed outside the " *
          "formula is held fixed (at its observed values, or at its typical value in a reference grid). " *
          "If it should follow the variable, write the transform in the formula instead (e.g. x^2, " *
          "log(x), x & z) and refit. If it should stay fixed, the results are as intended; pass " *
          "warn=false to silence this warning."
    return nothing
end

# Compile the checks for the column types prepared data has (Float64, Bool, categorical codes) into
# the package image, so the detector adds little to a first call.
for V in (Vector{Float64}, Vector{Bool}, BitVector, Vector{UInt32}), C in (Vector{Float64}, Vector{Bool}, BitVector)
    precompile(_n_distinct, (V, Vector{Int}))
    precompile(_is_constant, (V,))
    precompile(_matches_transform, (Int, V, C, Vector{Int}))
    precompile(_affine_relation, (V, C, Vector{Int}))
    for U in (Vector{Float64}, Vector{Bool}, BitVector)
        precompile(_matches_product, (V, U, C, Vector{Int}))
    end
end
precompile(_relation, (Symbol, Symbol, Dict{Symbol,AbstractVector}, Vector{Symbol}, Vector{Int}, Vector{Int}, Bool, Bool))
precompile(_spaced_rows, (Int, Int))
for S in (Nothing, Vector{Dict{Symbol,Any}}, Vector{Dict}, DataFrame)
    precompile(_report_derived_columns, (Vector{_DerivedColumn}, Vector{Symbol}, Vector{Symbol}, S))
end
precompile(_derived_columns, (Dict{Symbol,AbstractVector}, Vector{Symbol}, Nothing, Vector{Symbol}))
precompile(_derived_columns, (Dict{Symbol,AbstractVector}, Vector{Symbol}, Symbol, Vector{Symbol}))
