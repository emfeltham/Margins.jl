using DataFrames, StatsModels, StatsBase, MixedModels

"""
    ame_numeric(
      df::DataFrame,
      model::GeneralizedLinearMixedModel;
      x::Symbol,
      z::Symbol,
      z_val::Real,
      δ::Real = 1e-6,
      typical = mean,
      invlink = (η -> 1/(1 + exp(-η))),
      dinvlink = nothing,
      d2invlink = nothing,
      vcov = StatsBase.vcov
    ) -> NamedTuple

Compute the **Average Marginal Effect** of a continuous `x` at a single fixed value `z = z_val`
for an arbitrary formula, by numerically approximating ∂η/∂x via a small finite difference (step `δ`).

# Arguments
- `df::DataFrame`
  The original data frame used to fit `model`.
- `model::GeneralizedLinearMixedModel`
  A fitted MixedModels.jl GLMM (family can be Bernoulli, Poisson, etc.).
- `x::Symbol`
  The name of the continuous focal variable.
- `z::Symbol`
  The name of the continuous moderator (whose value will be “frozen” at `z_val`).
- `z_val::Real`
  The single value of `z` at which to compute the AME of `x`.
- `δ::Real = 1e-6`
  Finite‐difference step to approximate ∂η/∂x. Adjust if `x` is on a different scale.

# Keyword Arguments
- `typical::Function = mean`
  Function to compute a typical value of any *other* numeric covariate. Categorical covariates go to their reference level.
- `invlink::Function`
  Inverse‐link μ = g⁻¹(η), e.g. `η -> 1/(1 + exp(-η))` for logit.
- `dinvlink::Union{Function,Nothing}`
  First derivative dμ/dη. If `nothing`, assumes logistic (uses μ·(1−μ)).
- `d2invlink::Union{Function,Nothing}`
  Second derivative d²μ/dη². If `nothing` and `dinvlink === nothing`, assumes logistic (μ·(1−μ)·(1−2μ)).
- `vcov::Function`
  How to extract the fixed‐effects covariance matrix from `model`.

# Returns
A `NamedTuple` with fields:
- `ame`    :: Float64        = average of ∂μ_i/∂x_i over i=1..n.
- `se`     :: Float64        = delta‐method standard error of that AME.
- `grad`   :: Vector{Float64} = p‐vector ∇₍β₎[AME] (for contrasts).
- `n`      :: Int            = sample size.
- `η_base` :: Vector{Float64} = η_i at (x_i, z=z_val, others=typical).
- `μ_base` :: Vector{Float64} = μ_i = invlink(η_i) at base.
"""
function ame_numeric(
    df::DataFrame,
    model::GeneralizedLinearMixedModel;
    x::Symbol,
    z::Symbol,
    z_val::Real,
    δ::Real = 1e-6,
    typical = mean,
    invlink = (η -> 1/(1 + exp(-η))),
    dinvlink = nothing,
    d2invlink = nothing,
    vcov = StatsBase.vcov
)
    # 1) Build df_fixed with z -> z_val, other covariates (except x) at typical
    n = nrow(df)
    df_fixed = deepcopy(df)
    df_fixed[!, z] .= z_val
    for col in names(df)
        if col === x || col === z
            continue
        end
        coldata = df[!, col]
        if eltype(coldata) <: Number
            df_fixed[!, col] .= typical(coldata)
        elseif eltype(coldata) <: CategoricalValue
            df_fixed[!, col] = categorical(
                fill(levels(coldata)[1], n);
                levels = levels(coldata)
            )
        else
            df_fixed[!, col] .= coldata
        end
    end

    # 2) Base design matrix X0 (n×p), β, Σβ
    form = formula(model)
    X0   = modelcols(form.rhs, columntable(df_fixed))   # n×p
    β    = coef(model)                                   # length p
    Σβ   = vcov(model)                                   # p×p

    η_base = X0 * β                # length n
    μ_base = invlink.(η_base)      # length n

    # 3) dμ/dη at each i
    dpdη = if dinvlink !== nothing
        dinvlink.(η_base)
    else
        μ_base .* (1 .- μ_base)    # logistic fallback
    end

    # 4) Build df_plus, df_minus to perturb x by ±δ
    df_plus  = deepcopy(df_fixed)
    df_minus = deepcopy(df_fixed)
    df_plus[!, x]  .= df_fixed[!, x] .+ δ
    df_minus[!, x] .= df_fixed[!, x] .- δ

    # 5) Design matrices X_plus, X_minus
    X_plus  = modelcols(form.rhs, columntable(df_plus))
    X_minus = modelcols(form.rhs, columntable(df_minus))

    η_plus  = X_plus  * β   # length n
    η_minus = X_minus * β   # length n

    # 6) Finite‐difference ∂η_i/∂x_i ≈ (η_i⁺ - η_i⁻)/(2δ)
    δη_δx = (η_plus .- η_minus) ./ (2δ)   # length n

    # 7) Instantaneous ∂μ_i/∂x_i = (dμ/dη)_i * (∂η_i/∂x_i)
    me_vec = dpdη .* δη_δx                   # length n

    # 8) AME = mean over i
    ame_val = mean(me_vec)

    # 9) Compute second derivative d²μ/dη² at each i
    if dinvlink === nothing
        # logistic: d²μ/dη² = μ(1 − μ)(1 − 2μ)
        d2dη2 = μ_base .* (1 .- μ_base) .* (1 .- 2 .* μ_base)
    else
        @assert d2invlink !== nothing "If dinvlink ≠ nothing, you must supply d2invlink."
        d2dη2 = d2invlink.(η_base)
    end

    # 10) Build gradient matrix grads (p×n)
    p = length(β)
    grads = zeros(p, n)

    for i in 1:n
        Xi0      = vec(X0[i, :])      # ∂η_i/∂β
        Xi_plus  = vec(X_plus[i, :])
        Xi_minus = vec(X_minus[i, :])

        # (i) ∂[∂η_i/∂x_i]/∂β = (Xi_plus - Xi_minus)/(2δ)
        d_eta_dx_dβ = (Xi_plus .- Xi_minus) ./ (2δ)  # length p

        # (ii) dμ/dη at i = dpdη[i],   d²μ/dη² at i = d2dη2[i]
        Ai = dpdη[i]
        Bi = d2dη2[i]
        Di = δη_δx[i]

        # product rule: ∂(Ai * Di)/∂β = Bi * Xi0 * Di  +  Ai * d_eta_dx_dβ
        term1 = Bi .* Xi0 .* Di
        term2 = Ai .* d_eta_dx_dβ

        grads[:, i] = term1 .+ term2
    end

    # 11) ∇AME = (1/n) ∑_{i=1}^n grads[:,i]
    grad_AME = sum(grads, dims=2)[:, 1] ./ n   # length p

    # 12) Var(AME) = ∇AME' * Σβ * ∇AME;  se = sqrt(var)
    var_AME = dot(grad_AME, Σβ * grad_AME)
    se_AME  = sqrt(var_AME)

    return (
        ame    = ame_val,
        se     = se_AME,
        grad   = grad_AME,
        n      = n,
        η_base = η_base,
        μ_base = μ_base
    )
end


"""
    ame_contrast_numeric(
      df::DataFrame,
      model::GeneralizedLinearMixedModel;
      x::Symbol,
      z::Symbol,
      z_vals::Tuple{<:Real,<:Real},
      δ::Real = 1e-6,
      typical = mean,
      invlink = (η -> 1/(1 + exp(-η))),
      dinvlink = nothing,
      d2invlink = nothing,
      vcov = StatsBase.vcov
    ) -> NamedTuple

Compute the **difference of two AMEs**,
    ΔAME = AME(x | z = z_high) − AME(x | z = z_low)
and its standard error by calling `ame_numeric` at each `z_low`/`z_high` and using the delta‐method.

# Arguments
- `df::DataFrame`
- `model::GeneralizedLinearMixedModel`
- `x::Symbol`
- `z::Symbol`
- `z_vals::Tuple{(z_low, z_high)}`
- `δ::Real = 1e-6`
- `typical`, `invlink`, `dinvlink`, `d2invlink`, `vcov` same as in `ame_numeric`.

# Returns
A `NamedTuple` with:
- `ame_low`   :: Float64        = AME at z_low
- `se_low`    :: Float64        = SE of AME at z_low
- `ame_high`  :: Float64        = AME at z_high
- `se_high`   :: Float64        = SE of AME at z_high
- `ame_diff`  :: Float64        = ame_high − ame_low
- `se_diff`   :: Float64        = SE of ame_diff
- `grad_low`  :: Vector{Float64} = ∇₍β₎[AME @ z_low]
- `grad_high` :: Vector{Float64} = ∇₍β₎[AME @ z_high]
- `n`         :: Int            = sample size
- `names`     :: Vector{String} = coefnames(model)
"""
function ame_contrast_numeric(
    df::DataFrame,
    model::GeneralizedLinearMixedModel;
    x::Symbol,
    z::Symbol,
    z_vals::Tuple{<:Real,<:Real},
    δ::Real = 1e-6,
    typical = mean,
    invlink = (η -> 1/(1 + exp(-η))),
    dinvlink = nothing,
    d2invlink = nothing,
    vcov = StatsBase.vcov
)
    z_low, z_high = z_vals

    out_low = ame_numeric(
        df, model;
        x        = x,
        z        = z,
        z_val    = z_low,
        δ        = δ,
        typical  = typical,
        invlink  = invlink,
        dinvlink = dinvlink,
        d2invlink= d2invlink,
        vcov     = vcov
    )

    out_high = ame_numeric(
        df, model;
        x        = x,
        z        = z,
        z_val    = z_high,
        δ        = δ,
        typical  = typical,
        invlink  = invlink,
        dinvlink = dinvlink,
        d2invlink= d2invlink,
        vcov     = vcov
    )

    ame_low   = out_low.ame
    se_low    = out_low.se
    ame_high  = out_high.ame
    se_high   = out_high.se

    ame_diff  = ame_high - ame_low
    grad_diff = out_high.grad .- out_low.grad
    Σβ        = vcov(model)
    var_diff  = dot(grad_diff, Σβ * grad_diff)
    se_diff   = sqrt(var_diff)

    return (
        ame_low    = ame_low,
        se_low     = se_low,
        ame_high   = ame_high,
        se_high    = se_high,
        ame_diff   = ame_diff,
        se_diff    = se_diff,
        grad_low   = out_low.grad,
        grad_high  = out_high.grad,
        n          = out_low.n,
        names      = coefnames(model)
    )
end

function ame_contrast_numeric(out_low::AME, out_high::AME)

    ## add checks to ensure comparability

    ame_low   = out_low.ame
    se_low    = out_low.se
    ame_high  = out_high.ame
    se_high   = out_high.se

    ame_diff  = ame_high - ame_low
    grad_diff = out_high.grad .- out_low.grad
    Σβ        = vcov(model)
    var_diff  = dot(grad_diff, Σβ * grad_diff)
    se_diff   = sqrt(var_diff)

    return (
        ame_low    = ame_low,
        se_low     = se_low,
        ame_high   = ame_high,
        se_high    = se_high,
        ame_diff   = ame_diff,
        se_diff    = se_diff,
        grad_low   = out_low.grad,
        grad_high  = out_high.grad,
        n          = out_low.n,
        names      = coefnames(model)
    )
end
