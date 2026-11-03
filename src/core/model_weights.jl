# model_weights.jl
# The `weights` keyword of `population_margins` and the model-level `second_differences`:
# `:model` (default: the fitted model's estimation weights, if any), `:none`, a column name or a
# vector. The model's weights are the default because its `vcov` (from which every standard error
# comes) was computed with them: GLM.jl 1.x treats `wts` as frequency weights, so an unweighted
# point estimate paired with that SE would describe two different samples.

"""
    _model_weights(model, data_nt) -> Union{Nothing, Vector{Float64}, UnknownWeights}
    _model_weights(model) -> Union{Nothing, Vector{Float64}, UnknownWeights}

The estimation weights of a fitted model, one per row of the data it was fit to; `nothing` if
the model is unweighted; `UnknownWeights` if it may be weighted but does not store its weights.
The two-argument form is the interface (`data_nt` is the prepared estimation data, for a model
class that stores a weight column name); by default it calls the one-argument form, which reads
the weights the model stores.

- GLM.jl (`lm`/`glm` with `wts=`, bare or wrapped in a `TableRegressionModel`): the response
  object's `wts` field, which is empty for an unweighted fit (GLM 1.9 defines no
  `StatsAPI.weights` method for these models).
- MixedModels.jl (`LinearMixedModel`, `GeneralizedLinearMixedModel`; in `ext/MixedModelsExt.jl`):
  the `wts=` given at fit time. `StatsAPI.weights` returns ones for an unweighted fit, so the stored
  weights field (`sqrtwts` / `wt`) decides whether there are any.
- FixedEffectModels.jl: `UnknownWeights`. A `FixedEffectModel` (1.13) keeps neither the
  `weights=` column nor the weight vector, so `weights=:model` cannot tell whether `reg` was
  weighted; it averages without weights and warns once per session (pass `weights=:w`
  explicitly for a weighted `reg`, or `weights=:none`).
- Any other model class: `nothing` (add a method to support one).
"""
_model_weights(model, data_nt::NamedTuple) = _model_weights(model)
_model_weights(model) = nothing
_model_weights(model::StatsModels.TableRegressionModel) = _model_weights(model.model)
_model_weights(model::Union{GLM.LinearModel, GLM.GeneralizedLinearModel}) = _weights_from_storage(model.rr.wts)

# A model's stored weights as a Float64 vector, or `nothing` if there are none. GLM 1.x stores a
# plain vector (frequency weights for `vcov`). StatsBase weight types (as GLM 2 is expected to use)
# are honoured by their meaning: frequency and probability weights also define the population to
# average over; analytic (precision) weights do not, so the averaging weights must be chosen.
function _weights_from_storage(w)
    (isnothing(w) || isempty(w) || w isa StatsBase.UnitWeights) && return nothing
    w isa StatsBase.AnalyticWeights && throw(ArgumentError(
        "The model was fit with analytic (precision) weights, which set the estimation precision but not " *
        "the population to average over. Choose the averaging weights explicitly: weights=:none for an " *
        "unweighted average, or weights=<column or vector>."))
    return Vector{Float64}(collect(w))
end

const _WEIGHTS_KEYWORDS = (:model, :none)

"""
    _resolve_weights(weights, model, data[, data_nt]; rows=nothing) -> (weights_vec, source, weight_col)

Turn the `weights` keyword into the averaging weights (`nothing` or a `Vector{Float64}` with one
entry per row of `data_nt`), where they came from (`:model`, `:user` or `:none`, recorded as
`metadata[:weights_source]`) and the weight column to exclude from auto-detected variables
(`nothing` unless `weights` is a column name).

`weights = nothing` means `:none` (the 2.x default, so code that passed it explicitly keeps its
unweighted result). `:model` and `:none` are reserved: a data column with one of these names is
passed as a vector (`weights = data.model`).

`rows` are the model's estimation rows (`_estimation_rows`) when `data` has been subset to
them: `data` and `data_nt` are then the subset, and a weight vector given for every row of the
caller's data (`n_full` entries) is subset the same way.

Under `weights = :model`, a model whose weights cannot be read (`UnknownWeights`, e.g.
FixedEffectModels) gives no weights and a warning, once per session (decision D2 of the 3.0
plan); `weights=:none` or explicit weights do not warn.

This is the one weights helper: `_normalize_request` calls it for every entry point,
including the model-level `second_differences`.
"""
function _resolve_weights(weights, model, data, data_nt=Tables.columntable(data);
                          rows=nothing, n_full::Int=0)
    (isnothing(weights) || weights === :none) && return (nothing, :none, nothing)
    if weights === :model
        w = _model_weights(model, data_nt)
        isnothing(w) && return (nothing, :none, nothing)
        if w isa UnknownWeights
            _warn_unknown_weights(w)
            return (nothing, :none, nothing)
        end
        n = length(first(data_nt))
        length(w) == n || throw(ArgumentError(
            "The model has $(length(w)) estimation weights but the data has $n rows, so the weights cannot be " *
            "matched to rows (the fit probably dropped rows with missing values). Pass the weights for these " *
            "rows explicitly with weights=<column or vector>, or weights=:none for an unweighted average."))
        _check_weight_values(w, "model weights")
        return (w, :model, nothing)
    end
    # A vector for every row of the caller's data, when the request uses the estimation rows only
    if !isnothing(rows) && weights isa AbstractVector && length(weights) == n_full
        weights = weights[rows]
    end
    _validate_weights_parameter(weights, data)
    if weights isa Symbol
        # Weights are only read, so a Float64 column (already converted by data preparation)
        # is used as is rather than copied on every call
        col = getproperty(data_nt, weights)
        return (col isa Vector{Float64} ? col : Float64.(col), :user, weights)
    end
    return (Float64.(weights), :user, nothing)
end

# Decision D2: once per session (per logger), not on every call
@noinline _warn_unknown_weights(w::UnknownWeights) =
    @warn "weights=:model: $(w.reason) The estimates are unweighted. Pass the fit's weights " *
          "explicitly with weights=<column or vector>, or weights=:none to silence this warning." maxlog=1

_n_rows(data) = Tables.rowaccess(data) && !Tables.columnaccess(data) ? length(Tables.rows(data)) :
                length(first(Tables.columns(data)))

# Record the weights used on a result (both keys are in `result_metadata_keys`)
function _set_weights_metadata!(metadata::AbstractDict, weights_vec, source::Symbol)
    metadata[:weighted] = !isnothing(weights_vec)
    metadata[:weights_source] = source
    return metadata
end
