# metadata.jl
# The contract of `result.metadata` (a `Dict{Symbol,Any}` on every `EffectsResult` and
# `PredictionsResult`) and typed accessors for the entries other code reads. Producers
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
| `:alpha` | `Float64` | the `alpha` the intervals were computed with |
| `:df` | `Float64` | degrees of freedom of the reference distribution (`Inf` = z) |
| `:vcov` | `Matrix{Float64}` | the covariance matrix behind the standard errors |
| `:weighted` | `Bool` | whether observation weights were used |
| `:weights_source` | `Symbol` | where the weights came from: `:model` (the fit's estimation weights, the default), `:user` (`weights=` a column or vector) or `:none` |
| `:has_contexts` | `Bool` | whether rows are split by `scenarios=` / `groups=` |
| `:groups_vars` | `Vector{Symbol}` | the `groups=` columns among `profile_values` (empty if none) |
| `:scenarios_vars` | `Vector{Symbol}` | the `scenarios=` columns among `profile_values` (empty if none) |
| `:model_type` | `String` | type of the fitted model |
| `:timestamp` | `String` | when the result was computed |

Entries that exist only in some results: `:model_vcov` and `:model_df` (`Matrix{Float64}`,
`Float64`: the model's own covariance and residual df, linear regressions only; they let second
differences choose t or z from the covariance they are given), `:n_profiles` (profile analyses),
`:at_spec` (profile analyses), `:interval` (profile predictions), `:subgroup_n_values` /
`:has_subgroup_n` (grouped results).
"""
result_metadata_keys() = Pair{Symbol,Type}[
    :analysis_type => Symbol, :type => Symbol, :vars => Vector{Symbol}, :n_vars => Int,
    :scale => Symbol, :measure => Symbol, :backend => Symbol, :n_obs => Int,
    :alpha => Float64, :df => Float64, :vcov => Matrix{Float64}, :weighted => Bool,
    :weights_source => Symbol,
    :has_contexts => Bool, :groups_vars => Vector{Symbol}, :scenarios_vars => Vector{Symbol},
    :model_type => String, :timestamp => String,
]

_result_alpha(r)::Float64 = Float64(get(r.metadata, :alpha, 0.05))
_result_analysis_type(r, default::Symbol=:unknown)::Symbol = Symbol(get(r.metadata, :analysis_type, default))
_result_groups_vars(r)::Vector{Symbol} = Symbol[Symbol(s) for s in get(r.metadata, :groups_vars, Symbol[])]
_result_scenarios_vars(r)::Vector{Symbol} = Symbol[Symbol(s) for s in get(r.metadata, :scenarios_vars, Symbol[])]
_result_has_contexts(r)::Bool = Bool(get(r.metadata, :has_contexts, false))
_result_is_weighted(r)::Bool = Bool(get(r.metadata, :weighted, false))
_result_weights_source(r)::Symbol = Symbol(get(r.metadata, :weights_source, _result_is_weighted(r) ? :user : :none))
