# metadata.jl
# The contract of `result.metadata` (a `Dict{Symbol,Any}` on every `EffectsResult` and
# `PredictionsResult`: provenance only; the inference is `result.inference` and the context
# columns `result.context`) and typed accessors for the entries other code reads. Producers
# (`population_margins`, `profile_margins`) set every key below on every result;
# `test/test_result_metadata_contract.jl` checks keys and types, so a producer cannot drop or
# retype one unnoticed. Code that reads metadata uses these accessors, not `get(md, :k, default)`,
# so a result built by hand or restored from an older file gets the same defaults everywhere.

"""
    result_metadata_keys() -> Vector{Pair{Symbol,Type}}

The keys every `EffectsResult` / `PredictionsResult` carries in `metadata`, with their types:

| key | type | meaning |
|:----|:-----|:--------|
| `:analysis_type` | `Symbol` | `:population` or `:profile` |
| `:type` | `Symbol` | `:effects` or `:predictions` |
| `:vars` | `Vector{Symbol}` | focal variables requested |
| `:n_vars` | `Int` | `length(vars)` |
| `:scale` | `Symbol` | `:response` or `:link` |
| `:measure` | `Symbol` | `:effect`, `:elasticity`, … |
| `:backend` | `Symbol` | `:ad` or `:fd` |
| `:n_obs` | `Int` | rows of data used |
| `:weighted` | `Bool` | whether observation weights were used |
| `:weights_source` | `Symbol` | where the weights came from: `:model` (the fit's estimation weights, the default), `:user` (`weights=` a column or vector) or `:none` |
| `:has_contexts` | `Bool` | whether rows are split by `scenarios=` / `groups=` |
| `:groups_vars` | `Vector{Symbol}` | the `groups=` variables of the call (empty if none; their columns are in `result.context`) |
| `:scenarios_vars` | `Vector{Symbol}` | the `scenarios=` variables of the call (empty if none; their columns are in `result.context`) |
| `:model_type` | `String` | type of the fitted model |
| `:timestamp` | `String` | when the result was computed |

Entries that exist only in some results: `:n_profiles` (profile analyses),
`:at_spec` (profile analyses), `:interval` and `:resid_var` (profile predictions),
`:subgroup_n_values` / `:has_subgroup_n` (grouped results).

The covariance matrix, degrees of freedom and `alpha` behind the standard errors are not
metadata but `result.inference` (an [`Inference`](@ref)); Margins 2.x and early 3.0 stored
them as `metadata[:vcov]`, `metadata[:df]` and `metadata[:alpha]`.
"""
result_metadata_keys() = Pair{Symbol,Type}[
    :analysis_type => Symbol, :type => Symbol, :vars => Vector{Symbol}, :n_vars => Int,
    :scale => Symbol, :measure => Symbol, :backend => Symbol, :n_obs => Int,
    :weighted => Bool,
    :weights_source => Symbol,
    :has_contexts => Bool, :groups_vars => Vector{Symbol}, :scenarios_vars => Vector{Symbol},
    :model_type => String, :timestamp => String,
]

_result_alpha(cr::ContrastResult)::Float64 = Float64(get(cr.metadata, :alpha, 0.05))
_result_analysis_type(r, default::Symbol=:unknown)::Symbol = Symbol(get(r.metadata, :analysis_type, default))
_result_has_contexts(r)::Bool = Bool(get(r.metadata, :has_contexts, false))
_result_is_weighted(r)::Bool = Bool(get(r.metadata, :weighted, false))
_result_weights_source(r)::Symbol = Symbol(get(r.metadata, :weights_source, _result_is_weighted(r) ? :user : :none))
