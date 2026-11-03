# model_weights.jl
# The `weights` keyword of `population_margins` and the model-level `second_differences`:
# `:model` (default: the fitted model's estimation weights, if any), `:none`, a column name or a
# vector. The model's weights are the default because its `vcov` (from which every standard error
# comes) was computed with them: GLM.jl 1.x treats `wts` as frequency weights, so an unweighted
# point estimate paired with that SE would describe two different samples.

"""
    _model_weights(model) -> Union{Nothing, Vector{Float64}}

The estimation weights of a fitted model, one per row of the data it was fit to, or `nothing` if
the model is unweighted or its weights cannot be recovered.

- GLM.jl (`lm`/`glm` with `wts=`, bare or wrapped in a `TableRegressionModel`): the response
  object's `wts` field, which is empty for an unweighted fit (GLM 1.9 defines no
  `StatsAPI.weights` method for these models).
- MixedModels.jl (`LinearMixedModel`, `GeneralizedLinearMixedModel`; in `ext/MixedModelsExt.jl`):
  the `wts=` given at fit time. `StatsAPI.weights` returns ones for an unweighted fit, so the stored
  weights field (`sqrtwts` / `wt`) decides whether there are any.
- FixedEffectModels.jl: `nothing`. A `FixedEffectModel` keeps neither the `weights=` column nor
  the weight vector, so pass `weights=:w` explicitly for a weighted `reg`.
- Any other model class: `nothing` (add a method to support one).
"""
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
    _resolve_weights(weights, model, data[, data_nt]) -> (weights_vec, source, weight_col)

Turn the `weights` keyword into the averaging weights (`nothing` or a `Vector{Float64}` with one
entry per row of `data`), where they came from (`:model`, `:user` or `:none`, recorded as
`metadata[:weights_source]`) and the weight column to exclude from auto-detected variables
(`nothing` unless `weights` is a column name).

`weights = nothing` means `:none` (the 2.x default, so code that passed it explicitly keeps its
unweighted result). `:model` and `:none` are reserved: a data column with one of these names is
passed as a vector (`weights = data.model`).
"""
function _resolve_weights(weights, model, data, data_nt=Tables.columntable(data))
    (isnothing(weights) || weights === :none) && return (nothing, :none, nothing)
    if weights === :model
        w = _model_weights(model)
        isnothing(w) && return (nothing, :none, nothing)
        n = _n_rows(data)
        length(w) == n || throw(ArgumentError(
            "The model has $(length(w)) estimation weights but the data has $n rows, so the weights cannot be " *
            "matched to rows (the fit probably dropped rows with missing values). Pass the weights for these " *
            "rows explicitly with weights=<column or vector>, or weights=:none for an unweighted average."))
        _check_weight_values(w, "model weights")
        return (w, :model, nothing)
    end
    _validate_weights_parameter(weights, data)
    return (_process_weights_parameter(weights, data, data_nt), :user,
            weights isa Symbol ? weights : nothing)
end

_n_rows(data) = Tables.rowaccess(data) && !Tables.columnaccess(data) ? length(Tables.rows(data)) :
                length(first(Tables.columns(data)))

# Record the weights used on a result (both keys are in `result_metadata_keys`)
function _set_weights_metadata!(metadata::AbstractDict, weights_vec, source::Symbol)
    metadata[:weighted] = !isnothing(weights_vec)
    metadata[:weights_source] = source
    return metadata
end
