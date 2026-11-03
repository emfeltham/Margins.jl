# Centralized parameter validation for Margins.jl API functions.
#
# This file provides consistent parameter validation across the entire API,
# ensuring uniform error messages and eliminating duplicate validation code.
# All validation functions follow the principle of failing fast with clear,
# actionable error messages.

"""
    validate_type_parameter(type::Symbol)

Validate the `type` parameter for margins functions.
Must be either `:effects` or `:predictions`.
"""
function validate_type_parameter(type::Symbol)
    type ∉ (:effects, :predictions) && 
        throw(ArgumentError("type must be :effects or :predictions, got :$(type)"))
end

"""
    validate_scale_parameter(scale::Symbol)

Validate the `scale` parameter for margins functions.
Must be either `:link` (link scale) or `:response` (response scale).

"""
function validate_scale_parameter(scale::Symbol)  
    scale ∉ (:link, :response) && 
        throw(ArgumentError("scale must be :link or :response, got :$(scale)"))
end

"""
    _validate_reference_grid_column_type(col, col_name::Symbol)

Validate that a column in a reference grid has a supported data type.
Implements explicit error policy - fails fast with clear message for unsupported types.

# Supported types:
- Numeric: Int64, Float64 (continuous variables)
- Bool (categorical via frequency mixture)
- CategoricalArray (categorical variables)
- AbstractString (limited support in non-optimized paths only)

# Throws
- `MarginsError`: For unsupported data types that could produce invalid statistical results
"""
function _validate_reference_grid_column_type(col, col_name)
    eltype_val = eltype(col)
    
    # Check for supported types - order matters, check specific cases first
    if col isa CategoricalArray
        # CategoricalArray - supported (check this first before eltype checks)
        return
    elseif any(x -> x isa CategoricalMixture, col)
        # Column contains CategoricalMixture objects - supported for fractional specifications
        return
    elseif eltype_val <: CategoricalArrays.CategoricalValue
        # Vector of CategoricalValue objects (extracted from CategoricalArray) - supported
        return
    elseif eltype_val <: Real && !(eltype_val <: Bool)
        # Numeric types (Int64, Float64, etc.) - supported
        return
    elseif eltype_val <: Bool
        # Bool - supported as categorical
        return  
    elseif eltype_val <: AbstractString
        # String - limited support but allowed in reference grids
        return
    else
        # Unsupported types - explicit error
        throw(MarginsError("Unsupported data type $(eltype_val) for variable '$col_name' in reference grid. " *
                          "Statistical correctness cannot be guaranteed for unknown data types. " *
                          "Supported types: numeric (Int64, Float64), Bool, CategoricalArray, AbstractString, CategoricalMixture. " *
                          "Consider converting complex types to supported categorical or numeric formats."))
    end
end

"""
    validate_backend_parameter(backend::Symbol)

Validate the `backend` parameter for derivative computation.
Must be `:ad` (automatic differentiation) or `:fd` (finite differences).
"""
function validate_backend_parameter(backend::Symbol)
    backend ∉ (:ad, :fd) && 
        throw(ArgumentError("backend must be :ad or :fd, got :$(backend)"))
end

"""
    validate_measure_parameter(measure::Symbol, type::Symbol)

Validate the `measure` parameter for effects computation.
Must be one of `:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`.
Only applies when `type=:effects`.
"""
function validate_measure_parameter(measure::Symbol, type::Symbol)
    valid_measures = (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
    measure ∉ valid_measures && 
        throw(ArgumentError("measure must be one of $(valid_measures), got :$(measure)"))
    
    # Measure only applies to effects
    if type === :predictions && measure !== :effect
        throw(ArgumentError("measure parameter only applies when type=:effects"))
    end
end

"""
    validate_vars_parameter(vars, type::Symbol)

Validate the `vars` parameter for variable selection.
Throws error if vars is specified for predictions (where it's invalid).
"""
function validate_vars_parameter(vars, type::Symbol)
    if !isnothing(vars) && type === :predictions
        throw(ArgumentError("vars parameter is incompatible with type=:predictions. Remove vars parameter or change type to :effects."))
    end
end

"""
    validate_grouping_parameters(over, within, by)

Validate grouping parameters for consistency.
Ensures no conflicts between different grouping specifications.
"""
function validate_grouping_parameters(over, within, by)
    # Check for mutual exclusivity where appropriate
    if !isnothing(over) && !isnothing(within)
        # This is actually allowed - within creates nested grouping
        # No validation error needed
    end
    
    # Validate that grouping variables are Symbols or Vector{Symbol}
    for (name, param) in [(:over, over), (:within, within), (:by, by)]
        if !isnothing(param)
            if !(param isa Symbol || (param isa AbstractVector && all(x -> x isa Symbol, param)))
                throw(ArgumentError("$(name) must be Symbol or Vector{Symbol}, got $(typeof(param))"))
            end
        end
    end
end

"""
    validate_common_parameters(type, scale, backend, measure=:effect, vars=nothing)

Validate all common API parameters in a single call.
This is the main validation entry point for most margins functions.

# Arguments
- `type::Symbol`: Effect type (`:effects` or `:predictions`)  
- `scale::Symbol`: Target scale (`:link` or `:response`)
- `backend::Symbol`: Computation backend (`:ad` or `:fd`)
- `measure::Symbol`: Effect measure (default `:effect`)
- `vars`: Variable selection (default `nothing`)

# Examples
```julia
# Basic validation
validate_common_parameters(:effects, :response, :ad)

# Full validation with all parameters  
validate_common_parameters(:effects, :link, :ad, :elasticity, [:x1, :x2])
```
"""
function validate_common_parameters(type, scale, backend, measure=:effect, vars=nothing)
    validate_type_parameter(type)
    validate_scale_parameter(scale)  # Use new scale validation
    validate_backend_parameter(backend)
    validate_measure_parameter(measure, type)
    validate_vars_parameter(vars, type)
end

"""
    validate_profile_parameters(at, type, scale, backend, measure=:effect, vars=nothing)

Validate parameters specific to profile margins functions.
Includes all common parameter validation plus profile-specific checks.
"""
function validate_profile_parameters(at, type, scale, backend, measure=:effect, vars=nothing)
    validate_at_parameter(at)
    validate_common_parameters(type, scale, backend, measure, vars)
end

"""
    validate_population_parameters(type, scale, backend, measure=:effect, vars=nothing)

Validate parameters specific to population margins functions.
Currently identical to common validation but provided for API symmetry.
"""
function validate_population_parameters(type, scale, backend, measure=:effect, vars=nothing)
    validate_common_parameters(type, scale, backend, measure, vars)
end

# `ci_alpha` was the 2.x name of the significance level; it is `alpha` everywhere now (population_margins,
# profile_margins, contrast, second_differences). Passing the old keyword says so.
function _reject_ci_alpha(ci_alpha)
    isnothing(ci_alpha) && return nothing
    throw(ArgumentError("`ci_alpha` was renamed `alpha` in 3.0: use alpha=$(ci_alpha)"))
end

"""
    validate_vcov_parameter(vcov, model) -> Matrix{Float64}

Validate the `vcov` parameter for margins functions and return the covariance matrix Σ it
gives for `model`. `vcov` is a function of the model returning a covariance matrix, a
CovarianceMatrices estimator, or a covariance matrix itself (square, symmetric, one row per
coefficient). The rows and columns of coefficients the fit dropped as collinear are set to 0
(see `_vcov_and_dropped`, which also returns which coefficients those are).

This is the one place `vcov` is evaluated: `_normalize_request` calls it once per call and
keeps the returned Σ (a fresh copy, never aliasing the caller's matrix) in the request, so a
robust or clustered estimator is computed once per call, also for the model-level
`second_differences`, which runs several computations from one request.

# Arguments
- `vcov`: Function that computes covariance matrix from model
- `model`: Statistical model to test the vcov function against

# Examples
```julia
Σ = validate_vcov_parameter(GLM.vcov, model)
Σ = validate_vcov_parameter(CovarianceMatrices.HC1(), model)
```
"""
validate_vcov_parameter(vcov, model)::Matrix{Float64} = first(_vcov_and_dropped(vcov, model))

"""
    _vcov_and_dropped(vcov, model) -> (Σ::Matrix{Float64}, dropped::Vector{Int})

Σ for `vcov` (see `validate_vcov_parameter`) and the coefficients the fit dropped as collinear.

**Rank-deficient fits.** GLM.jl (`dropcollinear=true`), MixedModels and FixedEffectModels drop a
coefficient that is collinear with the others (or with the absorbed fixed effects): they report
it as 0 with NaN rows and columns in `vcov`. Margins treats a dropped coefficient as fixed at 0,
as R does (`lm` reports it as `NA` and `marginaleffects` drops it from the Jacobian): its rows and
columns of Σ are set to 0, whatever `vcov` gives there (NaN from the model's own `vcov` or a
CovarianceMatrices estimator, or any value in a matrix passed as `vcov=`). The zeroed Σ is what
`result.inference.vcov` holds. Which coefficients are dropped:

1. the model's own rank information, `_dropped_coefficients(model)` (GLM.jl's pivoted Cholesky
   rank and pivot, MixedModels' `feterm` rank and pivot, FixedEffectModels' zero coefficient with
   a NaN variance in its stored `vcov`);
2. for a model class without it (`_dropped_coefficients` returns `nothing`): the coefficients
   whose estimate is 0 and whose variance (diagonal of Σ) is not finite.

Any other non-finite entry of Σ is an `ArgumentError` (a broken matrix or estimator is not
silently zeroed), and so is a Σ of the wrong size. CovarianceMatrices estimators on a
rank-deficient GLM.jl fit (CovarianceMatrices 0.30):
- `HC0` gives NaN rows for the dropped coefficients and the refit's values elsewhere;
- `HC1`/`HR1` and HAC scale by n/(n − p) with p counting the dropped coefficients; Σ is rescaled
  to n/(n − rank), the value of the refit without the redundant terms (and of R's `sandwich`);
- `HC2` to `HC5` compute the leverage incorrectly and are refused (`_check_rank_deficient_estimator`);
- the cluster-robust `CR` estimators throw inside CovarianceMatrices (a `DimensionMismatch`),
  reported as an `ArgumentError` that names the dropped coefficients.

A full-rank fit's Σ is the matrix `vcov` gives, unchanged.
"""
function _vcov_and_dropped(vcov, model)::Tuple{Matrix{Float64}, Vector{Int}}
    β = _coef(model)
    k = length(β)
    known = _dropped_coefficients(model)

    raw = if vcov isa AbstractMatrix
        size(vcov) == (k, k) ||
            throw(ArgumentError("vcov matrix is $(size(vcov, 1))×$(size(vcov, 2)) but the model has $k coefficients"))
        vcov
    else
        !isnothing(known) && !isempty(known) && _check_rank_deficient_estimator(vcov, model, known)
        # Evaluate the estimator on the model
        r = try
            _evaluate_vcov(vcov, model)
        catch e
            # The default estimator failing means the model has no covariance method
            vcov === GLM.vcov &&
                throw(ArgumentError("model must support vcov() method (covariance matrix required for standard errors)"))
            msg = "vcov failed when applied to provided model: $e"
            if !isnothing(known) && !isempty(known)
                msg *= ". The model is rank deficient (coefficient(s) dropped as collinear by the fit: " *
                       "$(_coef_list(model, known))), which some CovarianceMatrices estimators (the " *
                       "cluster-robust CR estimators on GLM.jl fits) do not support. Refit without the " *
                       "collinear term(s), or pass the covariance matrix itself as vcov=<matrix>."
            end
            throw(ArgumentError(msg))
        end
        r isa AbstractMatrix ||
            throw(ArgumentError("vcov failed when applied to provided model: " *
                                "vcov must produce a matrix when applied to the model"))
        size(r) == (k, k) || throw(ArgumentError(
            "vcov gave a $(size(r, 1))×$(size(r, 2)) matrix but the model has $k coefficients"))
        r
    end
    Σ = Matrix{Float64}(raw)   # a fresh copy, never the caller's matrix

    dropped = isnothing(known) ? _nonfinite_zero_coefs(Σ, β) : known
    if !isempty(dropped)
        for j in dropped
            Σ[j, :] .= 0.0
            Σ[:, j] .= 0.0
        end
        vcov isa AbstractMatrix || _rank_dof_rescale!(Σ, vcov, model, k - length(dropped))
    end

    if !all(isfinite, Σ)
        bad = [i for i in 1:k if any(!isfinite, view(Σ, i, :))]
        throw(ArgumentError(
            "The covariance matrix (vcov) has non-finite values in the rows of coefficient(s) " *
            "$(_coef_list(model, bad)), which the fit did not drop as collinear" *
            (isempty(dropped) ? "" : " (dropped: $(_coef_list(model, dropped)))") *
            ". Check the vcov matrix or estimator."))
    end
    vcov isa AbstractMatrix && !isapprox(Σ, Σ'; rtol=1e-8, atol=1e-12) &&
        throw(ArgumentError("vcov matrix is not symmetric"))
    return (Σ, dropped)
end

# Coefficients with estimate 0 and a non-finite variance: dropped as collinear, for a model class
# that does not expose its rank (the fallback of `_vcov_and_dropped`)
_nonfinite_zero_coefs(Σ::Matrix{Float64}, β::Vector{Float64}) =
    [i for i in eachindex(β) if iszero(β[i]) && !isfinite(Σ[i, i])]

# Coefficient names for messages (indices if the model gives no names)
function _coef_list(model, idx::Vector{Int})
    names = try
        string.(StatsModels.coefnames(model))
    catch
        nothing
    end
    labels = [(!isnothing(names) && length(names) >= i) ? names[i] : "β[$i]" for i in idx]
    return join(labels, ", ")
end

# CovarianceMatrices' leverage-based estimators (HC2-HC5) on a rank-deficient GLM.jl fit: its
# leverage for a pivoted Cholesky takes the first `rank` columns of X instead of the kept (pivoted)
# ones (CovarianceMatrices 0.30, GLMExt `_leverage`), so the result is silently wrong (5x on a test
# logit). Refused rather than passed on.
_check_rank_deficient_estimator(vcov, model, dropped) = nothing
function _check_rank_deficient_estimator(
        vcov::Union{CovarianceMatrices.HR2, CovarianceMatrices.HR3, CovarianceMatrices.HR4,
                    CovarianceMatrices.HR4m, CovarianceMatrices.HR5},
        model::Union{StatsModels.TableRegressionModel, GLM.LinPredModel}, dropped)
    throw(ArgumentError(
        "vcov=$(nameof(typeof(vcov)))(): the model is rank deficient (coefficient(s) dropped as " *
        "collinear by the fit: $(_coef_list(model, dropped))), and CovarianceMatrices' leverage-based " *
        "estimators (HC2 to HC5) compute the leverage of a rank-deficient GLM.jl fit incorrectly. " *
        "Refit without the collinear term(s), use HC0()/HC1(), or pass the covariance matrix itself " *
        "as vcov=<matrix>."))
end

# CovarianceMatrices' HC1/HR1 and HAC degrees-of-freedom factor n/(n − p) counts every coefficient,
# dropped ones included (CovarianceMatrices 0.30); on a rank-deficient GLM.jl fit, rescale it to
# n/(n − rank), as for the refit without the dropped terms and in R's `sandwich`.
_rank_dof_rescale!(Σ, vcov, model, rank::Int) = Σ
function _rank_dof_rescale!(Σ, vcov::Union{CovarianceMatrices.HR1, CovarianceMatrices.HAC},
                            model::Union{StatsModels.TableRegressionModel, GLM.LinPredModel}, rank::Int)
    n = CovarianceMatrices.numobs(model)
    p = length(_coef(model))
    rmul!(Σ, (n - p) / (n - rank))
    return Σ
end

# `vcov` applied to the model: a function of the model (`GLM.vcov`, `m -> ...`) or a
# CovarianceMatrices estimator (`HC1()`, `CR1(cl)`, ...). CovarianceMatrices is a hard
# dependency, so it is called directly.
_evaluate_vcov(vcov::Function, model) = vcov(model)
_evaluate_vcov(::typeof(GLM.vcov), model) = _default_vcov(model)   # the model class's own
_evaluate_vcov(vcov, model) = CovarianceMatrices.vcov(vcov, model)

"""
    _covariance_matrix(model, vcov) -> Matrix{Float64}

Covariance matrix for `vcov` given as a matrix (copied), a function of the model (e.g.
`GLM.vcov`) or a CovarianceMatrices estimator (`HC1()`, ...), without the entry-point
checks of `validate_vcov_parameter`. Used by `build_engine`, which entry points call with
the matrix they already computed.
"""
_covariance_matrix(model, vcov::AbstractMatrix)::Matrix{Float64} = Matrix{Float64}(vcov)
_covariance_matrix(model, vcov)::Matrix{Float64} = Matrix{Float64}(_evaluate_vcov(vcov, model))

"""
    _filter_vars_for_measure(vars, data_nt, measure, vars_explicit) -> Vector{Symbol}

Enforce that econometric measures other than `:effect` are only requested for continuous
variables, and return the variable list to actually compute.

Elasticities and semi-elasticities are defined through a derivative ∂y/∂x scaled by x̄ and/or
ȳ. A discrete level contrast has no such derivative and no meaningful x̄, so applying an
elasticity factor to it is not defined. Stata rejects factor variables in
`eyex()`/`dyex()`/`eydx()`; R's marginaleffects applies slope transformations to numeric
predictors only. Margins.jl previously computed the raw contrast and labeled the result
`AME (elasticity)`, which silently misreported it.

Behaviour:
- A discrete variable the caller named explicitly is an error — the request cannot be
  honoured as stated.
- A discrete variable that was auto-detected (`vars=nothing`) is dropped with an `@info`,
  so `population_margins(model, data; measure=:elasticity)` still works and returns only
  the rows it can define. If nothing continuous remains, that is an error.

# Arguments
- `vars::Vector{Symbol}`: Resolved variable list from `_process_vars_parameter`
- `data_nt::NamedTuple`: Data columns (used for continuity detection)
- `measure::Symbol`: Requested measure
- `vars_explicit::Bool`: Whether the caller supplied `vars` (vs. auto-detection)

# Returns
- `Vector{Symbol}`: `vars` unchanged for `measure === :effect`, otherwise continuous only
"""
function _filter_vars_for_measure(vars, data_nt::NamedTuple, measure::Symbol, vars_explicit::Bool)
    (measure === :effect || isnothing(vars)) && return vars

    discrete = [v for v in vars if !_is_continuous_variable(getproperty(data_nt, v))]
    isempty(discrete) && return vars

    discrete_list = join(string.(discrete), ", ")
    if vars_explicit
        throw(ArgumentError(
            "measure=:$measure is only defined for continuous variables, but vars includes " *
            "discrete variable(s): $discrete_list. Elasticities scale a derivative ∂y/∂x by " *
            "x̄ and/or ȳ; a discrete level contrast has no such derivative. Request the " *
            "discrete variables in a separate call with measure=:effect."
        ))
    end

    continuous = [v for v in vars if !(v in discrete)]
    if isempty(continuous)
        throw(ArgumentError(
            "measure=:$measure requires at least one continuous variable, but all detected " *
            "variables are discrete: $discrete_list. Use measure=:effect for discrete contrasts."
        ))
    end

    @info "Skipping measure=:$measure for discrete variable(s): $discrete_list " *
          "(elasticity measures are defined for continuous variables only)."
    return continuous
end

# End of validation.jl
