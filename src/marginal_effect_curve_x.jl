
"""
    marginal_effect_curve_x(
      df::DataFrame,
      model::GeneralizedLinearMixedModel,
      x::Symbol,
      z::Symbol,
      z_val::Real,
      K::Int = 50;
      δ::Real = 1e-6,
      typical = mean,
      invlink = (η -> 1/(1 + exp(-η))),
      dinvlink = nothing,
      d2invlink = nothing,
      vcov = StatsBase.vcov
    ) -> DataFrame

Compute the pointwise marginal effect of `x` (i.e. ∂μ/∂x) at `z = z_val` over a grid of `K` values of `x`
(from min(x) to max(x)), holding all other covariates at their typical (mean for numeric, reference level for categorical).
Returns a DataFrame with columns:
  • `x_grid`      — the `K` values of `x`
  • `me`          — estimated ∂μ/∂x at each `x_grid[j]`
  • `se_me`       — standard error of ∂μ/∂x at each `x_grid[j]`

# Arguments
- `df::DataFrame`
- `model::GeneralizedLinearMixedModel`
- `x::Symbol`      — name of the continuous focal variable
- `z::Symbol`      — name of the continuous moderator (fixed at `z_val`)
- `z_val::Real`    — value at which `z` is held
- `K::Int = 50`    — number of grid points over range of `x`
- `δ::Real = 1e-6` — finite‐difference step for approximating ∂η/∂x
- `typical::Function = mean`
- `invlink`, `dinvlink`, `d2invlink`, `vcov` as in `ame_numeric`.

# Returns
A DataFrame with `K` rows and columns:
  - `x_grid::Vector{Float64}`
  - `me::Vector{Float64}`
  - `se_me::Vector{Float64}`
"""
function marginal_effect_curve_x(
    df::DataFrame,
    model::GeneralizedLinearMixedModel,
    x::Symbol,
    z::Symbol,
    z_val::Real;
    K::Int = 10,
    δ::Real = 1e-6,
    typical = mean,
    invlink = (η -> 1/(1 + exp(-η))),
    dinvlink = nothing,
    d2invlink = nothing,
    vcov = StatsBase.vcov
) 
    # 1) Prepare coefficient vector and vcov matrix
    β  = coef(model)            # p-vector
    Σβ = vcov(model)            # p×p
    form = formula(model)

    # 2) Build a “template” row that holds:
    #    - z = z_val
    #    - all other covariates (except x) at their typical values
    # We take the first row of df as a skeleton, then overwrite:
    template = df[1:1, :]  # 1×p DataFrame
    
    template[!, z] .= z_val
    for col in names(df)
        if col === x || col === z
            continue
        end
        coldata = df[!, col]
        if eltype(coldata) <: Number
            template[!, col] .= typical(coldata)
        elseif eltype(coldata) <: CategoricalValue
            template[!, col] = categorical(
                fill(levels(coldata)[1], 1);
                levels = levels(coldata)
            )
        else
            template[!, col] .= df[1, col]
        end
    end

    # 3) Build grid of x values
    x_min, x_max = extrema(df[!, x])
    x_grid = collect(range(x_min, x_max; length=K))

    # 4) Prepare storage
    me_vals  = Vector{Float64}(undef, K)
    se_vals  = Vector{Float64}(undef, K)

    # 5) Loop over grid points
    for j in 1:K
        xj = x_grid[j]
        # a) Base row: x = xj, z = z_val, others typical
        row_base = deepcopy(template)
        row_base[!, x] .= xj

        # Build design matrix for this single row (1×p)
        X0 = modelcols(form.rhs, columntable(row_base))[1]  # returns a 1×p matrix

        η0 = X0 * β                 # 1×1 vector (but treated as scalar)
        μ0 = invlink(η0[1])         # scalar

        # b) Perturbed rows for finite difference: xj ± δ
        row_p = deepcopy(row_base);  row_p[!, x] .= xj + δ
        row_m = deepcopy(row_base);  row_m[!, x] .= xj - δ

        Xp = modelcols(form.rhs, columntable(row_p))[1]   # 1×p
        Xm = modelcols(form.rhs, columntable(row_m))[1]   # 1×p

        ηp = Xp * β    # 1×1
        ηm = Xm * β    # 1×1

        # c) Approximate ∂η/∂x at xj: (ηp - ηm) / (2δ)
        dηdx = (ηp[1] - ηm[1]) / (2δ)  # scalar

        # d) dμ/dη at η0: either dinvlink or logistic fallback
        dpdη = dinvlink !== nothing ? dinvlink(η0[1]) : μ0 * (1 - μ0)

        # e) Marginal effect: ∂μ/∂x = (dμ/dη)* (∂η/∂x)
        me_vals[j] = dpdη * dηdx

        # f) Now compute gradient ∇₍β₎[ ∂μ/∂x ] at this xj for delta‐method:
        #   Let:
        #     A  = dpdη (scalar)
        #     B  = d2μ/dη² (scalar)
        #     D  = dη/dx = dηdx (scalar)
        #   And: ∂η/∂β = X0 (1×p row vector),
        #        ∂(dη/dx)/∂β = (Xp - Xm)/(2δ)  (1×p row vector).
        #
        #   Then: ∂[A*D]/∂β = B * (∂η/∂β) * D   +   A * ∂D/∂β
        #
        #   Where B = d²μ/dη² = if logistic: μ*(1−μ)*(1−2μ), else d2invlink(η0)
        B = if dinvlink === nothing
            μ0 * (1 - μ0) * (1 - 2*μ0)
        else
            @assert d2invlink !== nothing "Must supply d2invlink if dinvlink≠nothing"
            d2invlink(η0[1])
        end

        # Row‐vectors as vectors of length p
        v_X0 = vec(X0[1, :])   # p-vector
        v_diff = (vec(Xp[1, :]) .- vec(Xm[1, :])) ./ (2δ)  # p-vector

        # Gradient of ∂μ/∂x at this xj:
        #   grad_j = B * v_X0 * D   +   A * v_diff
        term1 = B .* v_X0 .* dηdx   # p-vector
        term2 = dpdη .* v_diff      # p-vector
        grad_j = term1 .+ term2     # p-vector

        # g) Standard error: sqrt(grad_j' * Σβ * grad_j)
        var_j = grad_j' * (Σβ * grad_j)
        se_vals[j] = sqrt(var_j)
    end

    return DataFrame(
        x_grid = x_grid,
        me     = me_vals,
        se_me  = se_vals
    )
end
