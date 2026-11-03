# core/variable_detection.jl - Variable type detection and validation utilities

using CategoricalArrays: CategoricalValue
using FormulaCompiler

"""
    _normalize_override_value(v)

Normalize user/profile override values before constructing scenarios.
Converts CategoricalValue to String for proper scenario handling.
"""
_normalize_override_value(v) = v
_normalize_override_value(v::CategoricalValue) = String(v)

"""
    _model_baseline_level(model, var::Symbol) -> level

The reference level of the categorical variable `var` for its marginal-effect contrasts,
from the fitted model's contrast coding. FormulaCompiler's `baseline_level` reads only a
main-effect term; this also finds `var` when it appears only inside interactions
(`y ~ x + g & w`, `g & x & w`, `g & log(w)`) or function terms.

- A main-effect term is used when there is one, as in `baseline_level` (so
  `DummyCoding(base = …)`, `EffectsCoding(base = …)` and custom level orders are respected).
- Otherwise the first term that contains `var`, in formula order.
- The baseline is the level of that term's contrasts with no coefficient. A full-rank coding
  has none: StatsModels uses `FullDummyCoding` for a categorical variable that appears only
  in interactions (it has no lower-order term to be redundant with), and `y ~ 0 + g` does the
  same. Then the first level of the contrasts (the column's level order, or `levels = […]`)
  is the reference: the contrasts "B − A", "C − A" are the effects of changing every row's
  level, which the model defines whatever its coding.
"""
function _model_baseline_level(model, var::Symbol)
    rhs = fixed_effects_form(model).rhs
    found = _find_categorical_term(rhs, var)
    isnothing(found) && throw(ArgumentError("Could not find categorical variable $var in model formula " *
        "terms. Ensure the variable is categorical and present in the model."))
    contrasts = found.contrasts
    lvls = contrasts.levels
    missing_coef = setdiff(lvls, StatsModels.coefnames(contrasts))
    length(missing_coef) == 1 && return only(missing_coef)
    isempty(missing_coef) && return first(lvls)    # full-rank coding: the first level
    throw(ArgumentError("Multiple baseline levels found for variable $var: $missing_coef"))
end

# The CategoricalTerm of `var` in a formula's right-hand side: a main-effect term when there is
# one, otherwise the first nested one (interactions, function arguments), in formula order.
function _find_categorical_term(rhs, var::Symbol)
    top = rhs isa Tuple ? collect(rhs) : [rhs]   # MixedModels: (MatrixTerm, re terms...)
    terms = Any[]
    for t in top
        t isa StatsModels.MatrixTerm ? append!(terms, t.terms) : push!(terms, t)
    end
    for t in terms
        t isa StatsModels.CategoricalTerm && t.sym == var && return t
    end
    for t in terms
        nested = _find_nested_categorical(t, var)
        isnothing(nested) || return nested
    end
    return nothing
end

_find_nested_categorical(t::StatsModels.CategoricalTerm, var::Symbol) = t.sym == var ? t : nothing
function _find_nested_categorical(t::StatsModels.InteractionTerm, var::Symbol)
    for s in t.terms
        r = _find_nested_categorical(s, var)
        isnothing(r) || return r
    end
    return nothing
end
function _find_nested_categorical(t::StatsModels.FunctionTerm, var::Symbol)
    for s in t.args
        r = _find_nested_categorical(s, var)
        isnothing(r) || return r
    end
    return nothing
end
_find_nested_categorical(t::StatsModels.MatrixTerm, var::Symbol) = _find_categorical_term(t, var)
_find_nested_categorical(t, var::Symbol) = nothing

"""
    _baseline_level(model, var::Symbol, data_nt::NamedTuple)

The model's baseline level (`_model_baseline_level`, from the model's contrasts), with Boolean
variable support: a Boolean variable's baseline is `false`.

# Arguments
- `model`: Statistical model (GLM, etc.)
- `var`: Variable name
- `data_nt`: Named tuple containing data columns

# Returns
- Baseline level for the variable (specific to variable type)
"""
function _baseline_level(model, var::Symbol, data_nt::NamedTuple)
    col = getproperty(data_nt, var)
    if eltype(col) <: Bool
        return false
    elseif eltype(col) <: CategoricalArrays.CategoricalValue
        # For categorical variables, return String for consistent level_map lookup
        # FormulaCompiler returns the actual type (Bool, String, Int, etc.)
        baseline = _model_baseline_level(model, var)
        return string(baseline)
    else
        return _model_baseline_level(model, var)
    end
end

# _validate_variables function is defined in src/engine/core.jl

"""
    _detect_variable_type(data_nt::NamedTuple, var::Symbol) -> Symbol

Detect variable type for unified processing.
Classifies variables into :boolean, :continuous, or :categorical.

# Arguments
- `data_nt`: Named tuple containing data columns
- `var`: Variable name to analyze

# Returns
- `:boolean`: Boolean variables (true/false)
- `:continuous`: Numeric variables (excluding Boolean)
- `:categorical`: All other variables (strings, factors, etc.)

# Examples
```julia
data_nt = (x = [1.0, 2.0, 3.0], flag = [true, false, true], group = ["A", "B", "A"])
_detect_variable_type(data_nt, :x)      # returns :continuous
_detect_variable_type(data_nt, :flag)   # returns :boolean
_detect_variable_type(data_nt, :group)  # returns :categorical
```
"""
function _detect_variable_type(data_nt::NamedTuple, var::Symbol)
    col = getproperty(data_nt, var)

    if eltype(col) <: Bool
        return :boolean
    elseif eltype(col) <: Real
        return :continuous
    else
        return :categorical
    end
end

"""
    _is_continuous_variable(col) -> Bool

Determine if a data column represents a continuous variable.
Returns true for numeric types excluding Boolean.

# Arguments
- `col`: Data column to analyze

# Returns
- `Bool`: True if continuous, false otherwise
"""
function _is_continuous_variable(col)
    return eltype(col) <: Real && !(eltype(col) <: Bool)
end

"""
    count_categorical_rows(categorical_vars::Vector{Symbol}, engine::MarginsEngine, contrasts::Symbol=:baseline) -> Int

Count the result rows needed for the requested categorical variables (used to size the
results table).

# Arguments
- `categorical_vars`: Vector of categorical variable names
- `engine`: MarginsEngine containing model and data
- `contrasts`: Contrast type (:baseline or :pairwise)

# Returns
- `Int`: Total number of contrast rows needed

# Implementation Notes
For baseline contrasts, returns (n_levels - 1) contrasts per categorical variable.
For pairwise contrasts, returns n_levels*(n_levels-1)/2 contrasts per categorical variable.
Missing variables are handled gracefully by returning 0 rows.
"""
function count_categorical_rows(categorical_vars::Vector{Symbol}, engine::MarginsEngine, contrasts::Symbol=:baseline)
    total = 0
    for var in categorical_vars
        if haskey(engine.data_nt, var)
            col = getproperty(engine.data_nt, var)
            # Observed values, without materializing the column (O(levels) allocation)
            observed = eltype(col) <: Bool ? nothing : _observed_level_strings(col)
            n_levels = isnothing(observed) ? (any(col) + !all(col)) : length(observed)

            if contrasts == :baseline
                # One contrast per observed non-baseline level. The baseline is the
                # model's reference level, which need not occur in the data (e.g. a
                # subset without it), so count against it rather than assume n_levels - 1.
                if isnothing(observed)
                    total += max(0, n_levels - 1)
                else
                    baseline_str = string(_baseline_level(engine.model, var, engine.data_nt))
                    total += count(!=(baseline_str), observed)
                end
            elseif contrasts == :pairwise
                # All unique pairs: n*(n-1)/2
                total += div(n_levels * (n_levels - 1), 2)
            else
                throw(ArgumentError("Unsupported contrast type: $contrasts"))
            end
        end
    end
    return total
end