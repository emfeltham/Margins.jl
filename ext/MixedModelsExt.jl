"""
    MixedModelsExt

Package extension supplying the MixedModels.jl methods of Margins' model-class interface
(`src/core/model_interface.jl`): the link function, the estimation weights and the GLMM offset.
The fixed-effects part of the formula comes from FormulaCompiler's `fixed_effects_form`; random
effects are not part of η by default (`random_effects = :zero`: every random effect 0); with
`random_effects = :conditional` the predicted random effects z'b̂ enter η as an offset (below).
The inference df is the default `Inf` (z).

The link used to be found by looking MixedModels up as `Main.MixedModels`, so a linear mixed
model failed with "Cannot determine link function" unless the user had bound the name
`MixedModels` in `Main`. Dispatching on the model types here removes the dependence on `Main`.
"""
module MixedModelsExt

using Margins
using Margins: RandomEffectsSpec, MarginsError
import MixedModels
using MixedModels: LinearMixedModel, GeneralizedLinearMixedModel, MixedModel, RandomEffectsTerm
import GLM
import StatsModels
import Tables
import FormulaCompiler
import LinearAlgebra
const SparseArrays = MixedModels.SparseArrays

# A linear mixed model is Gaussian with the identity link, like `lm`.
Margins._link(::LinearMixedModel) = GLM.IdentityLink()
Margins._link(m::GeneralizedLinearMixedModel) = GLM.Link(m.resp)

# A GLMM's `offset=` (fit(MixedModel, f, df, dist; offset=…)), stored in its GLM response,
# empty without one. A LinearMixedModel has no offset.
Margins._has_offset(m::GeneralizedLinearMixedModel) = Margins._nonzero_offset(m.resp.offset)
Margins._offset(m::GeneralizedLinearMixedModel, data_nt::NamedTuple) =
    Margins._has_offset(m) ? Margins._checked_offset(m.resp.offset, data_nt) : nothing

# Estimation weights (`fit(MixedModel, f, df; wts=w)`), the default averaging weights of
# `weights=:model`. `StatsAPI.weights` returns ones for an unweighted fit, so the stored field
# decides whether the model is weighted: `sqrtwts` (√w) for a LinearMixedModel, `wt` for a
# GeneralizedLinearMixedModel; both are empty without `wts`.
Margins._model_weights(m::LinearMixedModel) = isempty(m.sqrtwts) ? nothing : Vector{Float64}(abs2.(m.sqrtwts))
Margins._model_weights(m::GeneralizedLinearMixedModel) = Margins._weights_from_storage(m.wt)

# Fixed-effect coefficients dropped as collinear (rank-deficient X): MixedModels pivots the
# fixed-effects term and keeps its rank; `coef` reports the dropped ones as (-)0 with NaN `vcov`
# rows. A GLMM holds its fixed-effects term in the embedded LinearMixedModel.
_feterm_dropped(fe) = sort!(fe.piv[(fe.rank + 1):end])
Margins._dropped_coefficients(m::LinearMixedModel) = _feterm_dropped(m.feterm)
Margins._dropped_coefficients(m::GeneralizedLinearMixedModel) = _feterm_dropped(m.LMM.feterm)

# ---- random_effects = :conditional ----------------------------------------------------------
#
# The predicted random effects (conditional modes, `ranef`) enter η as z'b̂, one term per grouping
# factor. A factor is a `ReMat` of `m.reterms` (named by `fnames`, e.g. `:rater` or
# `Symbol("village & rater")` for a nested factor); its columns of Z come from the random-effects
# terms of the fitted formula with that grouping (`(1 + w | rater)` and `(0 + x | rater)` are
# amalgamated into one `ReMat`, so a factor can have several), evaluated on the rows given with
# their schema (`modelcols`), and matched to the `ReMat`'s coefficient names. This is what
# MixedModels' `predict(m, newdata)` computes, without building a new model.

Margins._supports_random_effects(::MixedModel) = true

_lmm(m::LinearMixedModel) = m
_lmm(m::GeneralizedLinearMixedModel) = m.LMM

# The predicted random effects b̂ (one matrix per `ReMat`, coefficients × levels) of the fitted
# linear predictor. For a GLMM they are λu from the fit's spherical conditional modes `m.u`, which
# `fitted(m)` uses: `MixedModels.ranef` reads the working LMM, whose state after a `fast = false`
# fit is not the final one (MixedModels 4.38.1: b̂ off by up to 0.01 on a logit fit, while a
# `fast = true` fit agrees to rounding).
_bhat(m::LinearMixedModel) = MixedModels.ranef(m)
_bhat(m::GeneralizedLinearMixedModel) = [Matrix(r.λ) * u for (r, u) in zip(m.LMM.reterms, m.u)]

Margins._re_factors(m::MixedModel) = Symbol[Symbol(string(r.trm)) for r in _lmm(m).reterms]

# The random-effects terms of the fitted formula (a `zerocorr` term wraps one)
_re_term(t::RandomEffectsTerm) = t
_re_term(t::MixedModels.ZeroCorr) = _re_term(t.term)
_re_term(t) = nothing
function _formula_re_terms(m::MixedModel)
    rhs = _lmm(m).formula.rhs
    out = RandomEffectsTerm[]
    for t in (rhs isa Tuple ? rhs : (rhs,))
        re = _re_term(t)
        isnothing(re) || push!(out, re)
    end
    return out
end

_factor_terms(m::MixedModel, f::Symbol) = [t for t in _formula_re_terms(m) if Symbol(string(t.rhs)) === f]

Margins._re_grouping_vars(m::MixedModel, factors) =
    unique!(Symbol[v for f in factors for t in _factor_terms(m, f) for v in StatsModels.termvars(t.rhs)])
Margins._re_slope_vars(m::MixedModel, factors) =
    unique!(Symbol[v for f in factors for t in _factor_terms(m, f) for v in StatsModels.termvars(t.lhs)])

_level_key(v) = Margins.CategoricalArrays.unwrap(v)

# The level of grouping term `g` at row `i`: the value of its column, or a tuple of values for an
# interaction (nested) factor, as MixedModels stores `ReMat` levels
_row_level(g::StatsModels.InteractionTerm, tbl, i) = Tuple(_row_level(t, tbl, i) for t in g.terms)
_row_level(g, tbl, i) = _level_key(Tables.getcolumn(tbl, g.sym)[i])

_as_matrix(z::AbstractMatrix) = z
_as_matrix(z::AbstractVector) = reshape(z, :, 1)

function Margins._re_offset(m::MixedModel, tbl::NamedTuple, spec::RandomEffectsSpec)
    lmm = _lmm(m)
    n = length(first(tbl))
    off = zeros(Float64, n)
    b_all = _bhat(m)
    for (k, r) in enumerate(lmm.reterms)
        f = Symbol(string(r.trm))
        f in spec.used || continue
        b = b_all[k]                                   # coefficients × levels
        for t in _factor_terms(m, f)
            missing_vars = [v for v in StatsModels.termvars(t) if !haskey(tbl, v)]
            isempty(missing_vars) || throw(ArgumentError(
                "random_effects = :conditional needs $(join(repr.(missing_vars), ", ")) at every " *
                "row (the random effects of $(repr(f)) depend on $(length(missing_vars) == 1 ? "it" : "them"))"))
            z = _as_matrix(StatsModels.modelcols(t.lhs, tbl))
            cn = StatsModels.coefnames(t.lhs)
            cols = [findfirst(==(c), r.cnames) for c in (cn isa AbstractVector ? cn : [cn])]
            any(isnothing, cols) && error("Internal: random-effects columns $cn not found in $(r.cnames) of $f")
            j = _level_indices(t.rhs, tbl, r.levels, n)
            if spec.new_levels === :error && any(iszero, j)
                i = findfirst(iszero, j)
                throw(ArgumentError(
                    "random_effects = :conditional: level $(repr(_row_level(t.rhs, tbl, i))) of $(repr(f)) " *
                    "was not in the fitted data, so it has no predicted random effect. Pass re_new = :zero " *
                    "to set its random effects to 0."))
            end
            _add_zb!(off, z, b, Vector{Int}(cols), j)
        end
    end
    all(isfinite, off) || throw(ArgumentError("The random-effects contribution contains NaN or Inf values"))
    return off
end

# The level of each row in the fitted factor's levels (0: a level absent from the fit). The keys
# are the rows' unwrapped values (a tuple for an interaction), mapped once per distinct value, so
# the lookup over rows is on a concrete key type.
function _level_indices(g, tbl, levels, n::Int)
    index = Dict{Any,Int}(_level_key(l) => j for (j, l) in enumerate(levels))
    keys_ = _row_keys(g, tbl, n)
    memo = Dict{eltype(keys_),Int}()
    out = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        key = keys_[i]
        out[i] = get!(() -> get(index, key, 0), memo, key)
    end
    return out
end
_row_keys(g::StatsModels.InteractionTerm, tbl, n) = [_row_level(g, tbl, i) for i in 1:n]
_row_keys(g, tbl, n) = _unwrapped(Tables.getcolumn(tbl, g.sym))
_unwrapped(col::Margins.CategoricalArrays.CategoricalArray) = Margins.CategoricalArrays.unwrap.(col)
_unwrapped(col) = col

function _add_zb!(off::AbstractVector{Float64}, z::AbstractMatrix, b::AbstractMatrix, cols::Vector{Int}, j::Vector{Int})
    @inbounds for i in eachindex(off)
        l = j[i]
        l == 0 && continue
        for (c, col) in enumerate(cols)
            off[i] += z[i, c] * b[col, l]
        end
    end
    return off
end

# ---- random_effects = :marginal -------------------------------------------------------------
# u = z'b is normal: the sum over factors of zᵢ_f' b_f, with b_f ~ N(0, Σ_f) independent across
# factors and Σ_f = s² λ_f λ_f' (s the dispersion, `MixedModels.dispersion`: the residual SD of an
# LMM, 1 for a Bernoulli or Poisson GLMM). With random intercepts only τ² = Σ_f σ_f² (`m.σs`); with
# random slopes τᵢ² = Σ_f zᵢ_f' Σ_f zᵢ_f varies by row.
function _reterm(m::MixedModel, f::Symbol)
    lmm = _lmm(m)
    return lmm.reterms[findfirst(r -> Symbol(string(r.trm)) === f, lmm.reterms)]
end

Margins._re_has_slopes(m::MixedModel, factors) =
    any(f -> _reterm(m, f).cnames != ["(Intercept)"], factors)

function Margins._re_intercept_variance(m::MixedModel, factors)
    σs = m.σs
    return sum(Float64(first(values(getproperty(σs, f))))^2 for f in factors; init=0.0)
end

# Z of factor `f` at the rows of `tbl`, its columns in the `ReMat`'s coefficient order (several
# formula terms may share one factor)
function _factor_Z(m::MixedModel, f::Symbol, tbl::NamedTuple)
    r = _reterm(m, f)
    n = length(first(tbl))
    Z = zeros(Float64, n, length(r.cnames))
    for t in _factor_terms(m, f)
        missing_vars = [v for v in StatsModels.termvars(t.lhs) if !haskey(tbl, v)]
        isempty(missing_vars) || throw(ArgumentError(
            "the random effects of $(repr(f)) depend on $(join(repr.(missing_vars), ", ")), which " *
            "must be given at every row"))
        z = _as_matrix(StatsModels.modelcols(t.lhs, tbl))
        cn = StatsModels.coefnames(t.lhs)
        for (c, name) in enumerate(cn isa AbstractVector ? cn : [cn])
            Z[:, findfirst(==(name), r.cnames)] .= view(z, :, c)
        end
    end
    return Z
end

_factor_Σ(m::MixedModel, f::Symbol) =
    (Λ = Matrix(_reterm(m, f).λ); Float64(MixedModels.dispersion(m))^2 .* (Λ * Λ'))

function Margins._re_spread_sq(m::MixedModel, tbl::NamedTuple, factors)
    τ² = zeros(Float64, length(first(tbl)))
    for f in factors
        _add_quadform!(τ², _factor_Z(m, f, tbl), _factor_Σ(m, f))
    end
    return τ²
end

# ---- effects of a variable with a random slope, on the `:conditional` offset path ------------
# ∂Z/∂w by a central difference of Z's model columns (exact, up to rounding, for terms linear in w
# such as `(1 + w | rater)`): ∂(zᵢ'b̂)/∂w = (∂zᵢ/∂w)'b̂ of row i's level. (The wrapper paths evaluate
# z in the compiled formula and need none of this.)
function Margins._re_slope_derivatives(m::MixedModel, tbl::NamedTuple, spec::RandomEffectsSpec,
                                       vars::Vector{Symbol})
    n = length(first(tbl))
    doff = zeros(Float64, n, length(vars))
    b_all = _bhat(m)
    lmm = _lmm(m)
    for (k, v) in enumerate(vars)
        col = Float64.(Tables.getcolumn(tbl, v))
        h = 1e-6 .* max.(1.0, abs.(col))
        tp = merge(tbl, NamedTuple{(v,)}((col .+ h,)))
        tm = merge(tbl, NamedTuple{(v,)}((col .- h,)))
        for f in spec.used
            v in Margins._re_slope_vars(m, [f]) || continue
            dZ = (_factor_Z(m, f, tp) .- _factor_Z(m, f, tm)) ./ (2 .* h)
            kf = findfirst(t -> Symbol(string(t.trm)) === f, lmm.reterms)
            r = lmm.reterms[kf]
            j = _level_indices(r.trm, tbl, r.levels, n)
            _add_zb!(view(doff, :, k), dZ, b_all[kf], collect(1:length(r.cnames)), j)
        end
    end
    return doff
end

# τ²[i] += Z[i, :]' Σ Z[i, :]
function _add_quadform!(τ²::Vector{Float64}, Z::Matrix{Float64}, Σ::Matrix{Float64})
    q = size(Z, 2)
    @inbounds for i in eachindex(τ²)
        acc = 0.0
        for a in 1:q, b in 1:q
            acc += Z[i, a] * Σ[a, b] * Z[i, b]
        end
        τ²[i] += acc
    end
    return τ²
end

# ---- the wrapper model: the random-effects columns in the compiled formula ---------------------
#
# z'b is a fixed-effects design: `(1 + w | rater)` is full dummy coding of the rater plus its
# interaction with w, and a nested `(1 | village & rater)` is full dummy coding of the observed
# cells. A wrapped `:conditional` request (`re_se = :joint`, or `:fixed` within the limit) therefore
# swaps the model for a `REWrapperModel`: the fixed-effects formula plus those columns (over hidden
# grouping columns, one per factor), coefficients (β, b̂) and the joint conditional covariance of
# (β̂ − β, b̂ − b) given θ̂ (or blockdiag(Σ_β, 0)). Every kernel, contrast and second difference then
# works in the joint parameter space unchanged, and z is evaluated by the compiled formula (so
# derivatives, contrasts and scenarios of random-slope variables need nothing special).
#
# A `:marginal` request with random slopes appends instead the random-effects model columns of
# the integrated factors themselves (w, the dummy columns of a categorical slope; no grouping
# dummies) with coefficient 0 and variance 0, so η, gradients and standard errors are unchanged and
# each evaluated row carries its z for τ (`Margins.RowSpreadOffset`).
#
# The joint covariance (Henderson's prediction-error covariance), in MixedModels' spherical
# parametrization b = Λu: Cov(u − û, β̂ − β) = s² A⁻¹ with
#   A = [Λ'Z'WZΛ + I  Λ'Z'WX; X'WZΛ  X'WX],
# W the (working) weights, s the dispersion; its β block is `vcov(m)`.

const _RE_ZERO_LEVEL = "(random effect 0)"
const _RE_JOINT_MAX_Q = 2000

struct REWrapperModel{M<:MixedModel} <: StatsModels.RegressionModel
    model::M
    formula::StatsModels.FormulaTerm
    θ::Vector{Float64}             # (β; b̂ in the compiled column order)
    C::Matrix{Float64}             # joint covariance, or blockdiag(Σ_β, 0) for re_se = :fixed
    hidden::Vector{Symbol}         # hidden grouping column of each factor
    factors::Vector{Symbol}        # the factors conditioned on
    keys::Vector{Dict{Any,String}} # per factor: level key → hidden level
    sources::Vector{Vector{Symbol}}# per factor: the grouping columns it is built from
    p::Int
    spread_pos::Vector{Int}        # `:marginal`: the column of each element of z (0: intercept)
    spread_Σ::Matrix{Float64}      # `:marginal`: blockdiag of Σ_f over the integrated factors
end

_hidden_name(f::Symbol) = Symbol("__re_", replace(string(f), " " => ""))
_key_string(k::Tuple) = join(string.(k), " & ")
_key_string(k) = string(k)

# The hidden grouping column of factor k at the rows of `tbl`: a CategoricalArray over the fitted
# levels plus the zero level, built from level codes (no per-row strings)
function _hidden_column(w::REWrapperModel, k::Int, tbl, n::Int, new_levels::Symbol)
    r = _reterm(w.model, w.factors[k])
    lv = _hidden_levels(r)
    zero_code = length(lv)
    codes = if all(v -> haskey(tbl, v), w.sources[k])
        j = _level_indices(r.trm, tbl, r.levels, n)
        if new_levels === :error && any(iszero, j)
            i = findfirst(iszero, j)
            throw(ArgumentError(
                "random_effects = :conditional: level $(repr(_row_level(r.trm, tbl, i))) of " *
                "$(repr(w.factors[k])) was not in the fitted data, so it has no predicted random effect. " *
                "Pass re_new = :zero to set its random effects to 0."))
        end
        replace!(j, 0 => zero_code)
    else
        fill(zero_code, n)
    end
    col = Margins.CategoricalArrays.CategoricalArray{String}(undef, n; levels=lv)
    col.refs .= codes
    return col
end
_hidden_levels(w::REWrapperModel, k::Int) = _hidden_levels(_reterm(w.model, w.factors[k]))
_hidden_levels(r) = [_key_string.(_level_key.(r.levels))..., _RE_ZERO_LEVEL]

function _joint_covariance(m::MixedModel)
    lmm = _lmm(m)
    Z = reduce(hcat, [SparseArrays.sparse(r) for r in lmm.reterms])
    Λ = reduce(SparseArrays.blockdiag,
               [kron(SparseArrays.sparse(LinearAlgebra.I, length(r.levels), length(r.levels)),
                     SparseArrays.sparse(Matrix(r.λ))) for r in lmm.reterms])
    X = lmm.X
    n = size(X, 1)
    sw = isempty(lmm.sqrtwts) ? ones(n) : Vector{Float64}(lmm.sqrtwts)
    ZL = Z * Λ
    ZLw = LinearAlgebra.Diagonal(sw) * ZL
    Xw = LinearAlgebra.Diagonal(sw) * X
    q = size(Z, 2)
    A = [Matrix(ZLw' * ZLw) + LinearAlgebra.I  Matrix(ZLw' * Xw); Matrix(Xw' * ZLw)  Xw' * Xw]
    s² = Float64(MixedModels.dispersion(m))^2
    Ainv = s² .* inv(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A)))
    Λd = Matrix(Λ)
    Cbb = Λd * Ainv[1:q, 1:q] * Λd'
    Cbβ = Λd * Ainv[1:q, q+1:end]
    Cββ = Ainv[q+1:end, q+1:end]
    # offsets of each reterm's coefficients in b (level-major, coefficients within a level)
    offs = cumsum([0; [length(r.levels) * length(r.cnames) for r in lmm.reterms[1:end-1]]])
    return (Cbb, Cbβ, Cββ, offs)
end

# `re_se = :fixed` uses the wrapper when the columns it adds (the conditioned factors' random-effect
# coefficients plus a zero level each) stay within the limit; above it, the offset path (phases 1–2),
# which has no size limit but rejects contrasts of categorical random-slope variables and
# population scenarios on slope variables
function Margins._re_wrap_fixed(m::MixedModel, factors)
    lmm = _lmm(m)
    qa = sum((length(r.levels) + 1) * length(r.cnames) for r in lmm.reterms
             if Symbol(string(r.trm)) in factors; init=0)
    return qa <= _RE_JOINT_MAX_Q
end

# The columns of a random-effects term's lhs (`1 + w`, `0 + w`, a categorical slope): each term
# with its coefficient names; an InterceptTerm is `nothing` (the constant 1, no column of its own)
function _lhs_parts(t)
    lhs = t.lhs isa StatsModels.MatrixTerm ? t.lhs.terms : (t.lhs isa Tuple ? t.lhs : (t.lhs,))
    out = Tuple{Any,Vector{String}}[]
    for lt in lhs
        # `0 + w` (a slope without its intercept, e.g. amalgamated with a `(1 | g)` term)
        # carries an InterceptTerm{false}, which has no column
        lt isa StatsModels.InterceptTerm{false} && continue
        if lt isa StatsModels.InterceptTerm
            push!(out, (nothing, ["(Intercept)"]))
        else
            c = StatsModels.coefnames(lt)
            push!(out, (lt, c isa AbstractVector ? collect(String, c) : [String(c)]))
        end
    end
    return out
end

function Margins._re_wrapper_model(m::MixedModel, cols::NamedTuple, spec::RandomEffectsSpec, vcov)
    lmm = _lmm(m)
    if spec.se === :joint
        q = sum(length(r.levels) * length(r.cnames) for r in lmm.reterms)
        vcov === GLM.vcov || throw(ArgumentError(
            "random_effects = :conditional with re_se = :joint (the default) combines the model's own " *
            "covariance of β with the prediction error of the random effects, so it cannot take another " *
            "vcov. Pass re_se = :fixed to use this vcov (standard errors then omit the prediction error " *
            "of the random effects)."))
        q <= _RE_JOINT_MAX_Q || throw(ArgumentError(
            "random_effects = :conditional with re_se = :joint: the model has $q random-effect " *
            "coefficients, above the limit of $_RE_JOINT_MAX_Q for the joint covariance. Pass re_se = :fixed " *
            "(standard errors without the prediction error of the random effects), or re_terms."))
        isempty(Margins._dropped_coefficients(m)) || throw(ArgumentError(
            "random_effects = :conditional with re_se = :joint is not supported for a rank-deficient fit " *
            "(coefficients dropped as collinear); refit without the collinear terms, or pass re_se = :fixed."))
    end
    β = Margins._coef(m)
    p = length(β)
    fx = FormulaCompiler.fixed_effects_form(m)
    conds = Margins._conditioned(spec)
    # z's columns are needed for τ only with random slopes in the averaged factors
    ints = Margins._re_has_slopes(m, Margins._integrated(spec)) ? Margins._integrated(spec) : Symbol[]
    aug = StatsModels.AbstractTerm[]
    bvals = Float64[]
    bidx = Int[]          # index into b (0: a column with b = 0 and variance 0)
    hidden = Symbol[]; keyd = Dict{Any,String}[]; sources = Vector{Symbol}[]
    spread_pos = Int[]; Σblocks = Matrix{Float64}[]
    (Cbb, Cbβ, Cββ, offs) = spec.se === :joint ? _joint_covariance(m) : (nothing, nothing, nothing, nothing)
    if !isempty(conds)
        B = _bhat(m)
        for f in conds
            kf = findfirst(t -> Symbol(string(t.trm)) === f, lmm.reterms)
            r = lmm.reterms[kf]
            h = _hidden_name(f)
            push!(hidden, h)
            push!(keyd, Dict{Any,String}(_level_key(l) => _key_string(_level_key(l)) for l in r.levels))
            push!(sources, collect(Symbol, StatsModels.termvars(r.trm)))
            lv = _hidden_levels(r)
            g = StatsModels.CategoricalTerm(h, StatsModels.ContrastsMatrix(StatsModels.FullDummyCoding(), lv))
            nc = length(r.cnames)
            for t in _factor_terms(m, f), (lt, cn) in _lhs_parts(t)
                push!(aug, isnothing(lt) ? g : StatsModels.InteractionTerm((g, lt)))
                for c in cn, j in 1:length(lv)
                    ci = findfirst(==(c), r.cnames)
                    if j <= length(r.levels)
                        push!(bvals, B[kf][ci, j]); push!(bidx, offs === nothing ? 0 : offs[kf] + (j - 1) * nc + ci)
                    else
                        push!(bvals, 0.0); push!(bidx, 0)
                    end
                end
            end
        end
    end
    # the averaged factors (`:marginal`, `:hybrid`): z's own columns, coefficient 0; each factor's z
    # in its `ReMat` coefficient order
    if !isempty(ints)
        for f in ints
            r = _reterm(m, f)
            pos = zeros(Int, length(r.cnames))   # the intercept stays 0 (the constant 1)
            for t in _factor_terms(m, f), (lt, cn) in _lhs_parts(t)
                isnothing(lt) && continue
                push!(aug, lt)
                for c in cn
                    push!(bvals, 0.0); push!(bidx, 0)
                    pos[findfirst(==(c), r.cnames)] = p + length(bvals)
                end
            end
            append!(spread_pos, pos)
            push!(Σblocks, _factor_Σ(m, f))
        end
    end
    rhs = StatsModels.MatrixTerm((fx.rhs.terms..., aug...))
    formula = StatsModels.FormulaTerm(fx.lhs, rhs)
    qa = length(bvals)
    C = zeros(p + qa, p + qa)
    if spec.se === :joint
        C[1:p, 1:p] .= Cββ
        @inbounds for a in 1:qa
            ia = bidx[a]; ia == 0 && continue
            for c in 1:p
                C[p + a, c] = Cbβ[ia, c]; C[c, p + a] = Cbβ[ia, c]
            end
            for b in 1:qa
                ib = bidx[b]; ib == 0 && continue
                C[p + a, p + b] = Cbb[ia, ib]
            end
        end
    else
        Σβ = Margins._vcov_and_dropped(vcov, m)[1]
        C[1:p, 1:p] .= Σβ
    end
    Σs = isempty(Σblocks) ? zeros(0, 0) : Matrix(reduce(SparseArrays.blockdiag, SparseArrays.sparse.(Σblocks)))
    w = REWrapperModel(m, formula, [β; bvals], C, hidden, copy(conds),
                       keyd, sources, p, spread_pos, Σs)
    n = length(first(cols))
    hcols = Tuple(_hidden_column(w, k, cols, n, spec.new_levels) for k in eachindex(hidden))
    return (w, merge(cols, NamedTuple{Tuple(hidden)}(hcols)))
end

Margins._re_row_spread(w::REWrapperModel, shift, n::Int, τmax::Float64) =
    Margins.RowSpreadOffset(shift, n, w.spread_pos, w.spread_Σ, τmax)

# The hidden columns at reference-grid rows: from the grouping columns the grid sets, else the
# zero level (that factor's random effects 0)
function Margins._augment_grid(w::REWrapperModel, grid::Margins.DataFrame, new_levels::Symbol)
    g = copy(grid)
    tbl = Tables.columntable(g)
    for k in eachindex(w.hidden)
        g[!, w.hidden[k]] = _hidden_column(w, k, tbl, Margins.nrow(g), new_levels)
    end
    return g
end
Margins._hidden_columns(w::REWrapperModel) = w.hidden
Margins._joint_used(w::REWrapperModel, grid) =
    Symbol[w.factors[k] for k in eachindex(w.factors) if all(v -> string(v) in Margins.names(grid), w.sources[k])]

# The model-class interface, delegated to the mixed model except for the formula, β and Σ
Margins._link(w::REWrapperModel) = Margins._link(w.model)
Margins._coef(w::REWrapperModel) = w.θ
Margins._default_vcov(w::REWrapperModel) = w.C
GLM.vcov(w::REWrapperModel) = w.C
GLM.coef(w::REWrapperModel) = w.θ
StatsModels.coefnames(w::REWrapperModel) = StatsModels.coefnames(w.formula.rhs)
StatsModels.formula(w::REWrapperModel) = w.formula
FormulaCompiler.fixed_effects_form(w::REWrapperModel) = w.formula
Margins._dropped_coefficients(w::REWrapperModel) = Margins._dropped_coefficients(w.model)
Margins._response_name(w::REWrapperModel) = Margins._response_name(w.model)
Margins._model_weights(w::REWrapperModel) = Margins._model_weights(w.model)
Margins._has_offset(w::REWrapperModel) = Margins._has_offset(w.model)
Margins._offset(w::REWrapperModel, data_nt::NamedTuple) = Margins._offset(w.model, data_nt)
Margins.validate_model_methods(::REWrapperModel) = nothing
Margins._supports_random_effects(::REWrapperModel) = true
Margins._re_factors(w::REWrapperModel) = Margins._re_factors(w.model)
Margins._re_grouping_vars(w::REWrapperModel, factors) = Margins._re_grouping_vars(w.model, factors)
Margins._re_slope_vars(w::REWrapperModel, factors) = Margins._re_slope_vars(w.model, factors)
Margins._re_has_slopes(w::REWrapperModel, factors) = Margins._re_has_slopes(w.model, factors)
Margins._re_spread_sq(w::REWrapperModel, tbl::NamedTuple, factors) = Margins._re_spread_sq(w.model, tbl, factors)
Margins._re_intercept_variance(w::REWrapperModel, factors) = Margins._re_intercept_variance(w.model, factors)
Margins._model_type_string(::Type{REWrapperModel{M}}) where {M} = Margins._model_type_string(M)
# The hidden grouping columns are in the compiled formula but are not variables of the model
Margins._get_model_formula_variables(w::REWrapperModel) =
    setdiff!(Margins._termvars!(Set{Symbol}(), w.formula.rhs), w.hidden)

end # module
