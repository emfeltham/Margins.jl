# ame.jl

# ─────────────────────────────────────────────────────────────────────────────
# Define an immutable struct to hold the AME results
# ─────────────────────────────────────────────────────────────────────────────

"""
    ame_interaction_continuous(
      df::DataFrame,
      model,
      x::Symbol,
      z::Symbol,
      z_val::Real;
      δ::Real = 1e-6,
      typical = mean,
      vcov = StatsBase.vcov
    ) -> NamedTuple

Compute the **Average Marginal Effect** of a continuous `x` at a single fixed value `z = z_val`
for an arbitrary formula, by numerically approximating ∂η/∂x via a small finite difference (step `δ`).

# Arguments
- `df::DataFrame`
  The original data frame used to fit `model`.
- `model`
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
function ame_interaction_continuous(
    df::DataFrame,
    model,
    x::Symbol,
    z::Symbol,
    z_val::Real;
    δ::Real           = 1e-6,
    vcov              = StatsBase.vcov,
    typical::Function = mean,
)

    # Link functions
    invlink, dinvlink, d2invlink = link_functions(m)

    # 0) build a “typical” snapshot for all covariates except x and z
    df_typ = deepcopy(df)
    for col in names(df) 
      if col != x && col != z && eltype(df[!,col]) <: Number
        df_typ[!, col] .= typical(df[!, col])
      end
    end

    # 1) now fix z → z_val
    df_typ[!, z] .= z_val

    # 2) extract your base X0 from df_typ (all other covariates at typical, z at z_val)
    X0   = fixed_X(model, df_typ)
    β    = coef(model)
    Σβ   = vcov(model)

    η_base = X0 * β                # length n
    μ_base = invlink.(η_base)      # length n

    # 3) dμ/dη at each i
    dpdη = if dinvlink !== nothing
        dinvlink.(η_base)
    else
        μ_base .* (1 .- μ_base)    # logistic fallback (make this general)
    end

    # 4) Build df_plus, df_minus to perturb x by ±δ
    # approximate derivative calculation

    df_plus  = deepcopy(df_typ)
    df_minus = deepcopy(df_typ)
    df_plus[!, x]  .+= δ
    df_minus[!, x] .-= δ

    # 5) Design matrices X_plus, X_minus
    X_plus  = fixed_X(model, df_plus)
    X_minus = fixed_X(model, df_minus)

    η_plus  = X_plus  * β   # length n
    η_minus = X_minus * β   # length n

    # 6) Finite‐difference ∂η_i/∂x_i ≈ (η_i⁺ - η_i⁻)/(2δ)
    δη_δx = (η_plus .- η_minus) ./ (2δ)   # length n

    # 7) Instantaneous ∂μ_i/∂x_i = (dμ/dη)_i * (∂η_i/∂x_i)
    me_vec = dpdη .* δη_δx                   # length n

    # 8) AME = mean over i
    ame_val = mean(me_vec)

    # 9) Compute second derivative d²μ/dη² at each i
    d2dη2 = d2invlink.(η_base)

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

    return AME(
        ame_val,
        se_AME,
        grad_AME,
        n,
        η_base,
        μ_base
    )
end
