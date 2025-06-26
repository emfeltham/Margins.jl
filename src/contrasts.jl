# contrasts.jl

# -----------------------------------------------------------------------------
# Internal Δ-method helper
# -----------------------------------------------------------------------------
"""
    _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vcov, ν) -> (Δ, seΔ, tstat, pval)

Compute the difference between two AMEs (θ₁−θ₂), its standard error via the Δ-method,
the t-statistic, and two-sided p-value.

# Arguments
- `θ1`, `θ2`: AME estimates being compared.
- `σ1`, `σ2`: Standard errors of θ₁ and θ₂.
- `g1`, `g2`: Gradient vectors of θ₁ and θ₂ w.r.t. model coefficients.
- `vcov`: Covariance matrix of model coefficients (or `nothing` to assume independence).
- `ν`: Degrees of freedom for the t-test.

# Returns
A 4-tuple `(Δ, seΔ, tstat, pval)`.
"""
function _compute_contrast(
    θ1::Real, θ2::Real,
    σ1::Real, σ2::Real,
    g1::AbstractVector, g2::AbstractVector,
    vcov::Union{AbstractMatrix,Nothing}, ν::Real
)
    cov12 = vcov === nothing ? zero(σ1) : dot(g1, vcov * g2)
    Δ     = θ1 - θ2
    seΔ   = sqrt(abs(σ1^2 + σ2^2 - 2cov12))
    tstat = Δ / seΔ
    pval  = 2 * (1 - cdf(TDist(ν), abs(tstat)))
    return Δ, seΔ, tstat, pval
end

# -----------------------------------------------------------------------------
# Contrast between two AMEResults: single or grid
# -----------------------------------------------------------------------------
"""
    contrast(a, b; var, vcov=nothing) -> ContrastResult

Compare the AME of `var` between two `AMEResult` objects.
If each AME is a dictionary (grid), perform the comparison at every key.
Otherwise, compare the single scalar AMEs.

# Keyword Arguments
- `var::Symbol`: Name of the predictor to contrast.
- `vcov`: Optional covariance matrix of model coefficients (defaults to independent).
"""
function contrast(a::T, b::T; var::Symbol, vcov=nothing) where {T}
    @assert var in a.vars "$(var) not in first result"
    @assert var in b.vars "$(var) not in second result"
    ame_a = a.ame[var]; ame_b = b.ame[var]
    df     = min(a.df_residual, b.df_residual)

    if isa(ame_a, Dict) && isa(ame_b, Dict)
        # grid-based contrast (keys must match)
        keys_a = collect(keys(ame_a)); keys_b = collect(keys(ame_b))
        @assert sort(keys_a) == sort(keys_b) "Grids must match for var=$(var)"
        comps, est, se, t, p = Float64[], Float64[], Float64[], Float64[], Float64[]
        for key in keys_a
            θ1 = ame_a[key]; θ2 = ame_b[key]
            σ1 = a.se[var][key]; σ2 = b.se[var][key]
            g1 = a.grad[var][key]; g2 = b.grad[var][key]
            Δ, seΔ, tstat, pval = _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vcov, df)
            push!(est, Δ); push!(se, seΔ); push!(t, tstat); push!(p, pval)
        end
        return ContrastResult([var], keys_a, est, se, t, p, df)
    else
        # scalar contrast
        θ1  = isa(ame_a, Float64) ? ame_a : only(values(ame_a))
        θ2  = isa(ame_b, Float64) ? ame_b : only(values(ame_b))
        σ1  = isa(a.se[var], Float64) ? a.se[var] : only(values(a.se[var]))
        σ2  = isa(b.se[var], Float64) ? b.se[var] : only(values(b.se[var]))
        g1  = isa(a.grad[var], Vector)    ? a.grad[var]   : only(values(a.grad[var]))
        g2  = isa(b.grad[var], Vector)    ? b.grad[var]   : only(values(b.grad[var]))
        Δ, seΔ, tstat, pval = _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vcov, df)
        return ContrastResult([var], [(:a,:b)], [Δ], [seΔ], [tstat], [pval], df)
    end
end

# -----------------------------------------------------------------------------
# Contrast between two vars within one AMEResult: single or grid
# -----------------------------------------------------------------------------
"""
    contrast(a; var1, var2, vcov=nothing) -> ContrastResult

Contrast the AME between two different predictors in a single `AMEResult`.
Handles both scalar and grid-based AMEs when both variables share the same keys.

# Keyword Arguments
- `var1`, `var2`: Predictor names to compare.
- `vcov`: Optional covariance matrix of model coefficients.
"""
function contrast(a::T; var1::Symbol, var2::Symbol, vcov=nothing) where {T}
    @assert var1 in a.vars; @assert var2 in a.vars
    ame1, ame2 = a.ame[var1], a.ame[var2]
    df         = a.df_residual

    if isa(ame1, Dict) && isa(ame2, Dict)
        keys1 = collect(keys(ame1)); keys2 = collect(keys(ame2))
        @assert sort(keys1) == sort(keys2) "Grids must match for var1 and var2"
        est, se, t, p = Float64[], Float64[], Float64[], Float64[]
        for key in keys1
            θ1, θ2 = ame1[key], ame2[key]
            σ1, σ2 = a.se[var1][key], a.se[var2][key]
            g1, g2 = a.grad[var1][key], a.grad[var2][key]
            Δ, seΔ, tstat, pval = _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vcov, df)
            push!(est, Δ); push!(se, seΔ); push!(t, tstat); push!(p, pval)
        end
        return ContrastResult([var1,var2], keys1, est, se, t, p, df)
    else
        θ1  = isa(ame1, Float64) ? ame1 : only(values(ame1))
        θ2  = isa(ame2, Float64) ? ame2 : only(values(ame2))
        σ1  = isa(a.se[var1], Float64) ? a.se[var1] : only(values(a.se[var1]))
        σ2  = isa(a.se[var2], Float64) ? a.se[var2] : only(values(a.se[var2]))
        g1  = isa(a.grad[var1], Vector)    ? a.grad[var1]   : only(values(a.grad[var1]))
        g2  = isa(a.grad[var2], Vector)    ? a.grad[var2]   : only(values(a.grad[var2]))
        Δ, seΔ, tstat, pval = _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vcov, df)
        return ContrastResult([var1,var2], [(var1,var2)], [Δ], [seΔ], [tstat], [pval], df)
    end
end

# -----------------------------------------------------------------------------
# Pairwise contrasts across levels of a var within one AMEResult
# -----------------------------------------------------------------------------
"""
    contrast_levels(a, var; comparisons=:all, vcov=nothing) -> ContrastResult

Compute pairwise contrasts of AMEs across levels (or representative values) of `var`.

# Keyword Arguments
- `comparisons`: `:all` for every i<j, or `:adjacent` for adjacent pairs.
- `vcov`: Optional covariance matrix of model coefficients.
"""
function contrast_levels(a::T, var::Symbol; comparisons::Symbol = :all, vcov=nothing) where {T}
    @assert var in a.vars
    levels = haskey(a.repvals,var) && !isempty(a.repvals[var]) ? sort(a.repvals[var]) : sort([first(k) for k in keys(a.ame[var])])
    pairs  = comparisons == :adjacent ? collect(zip(levels[1:end-1], levels[2:end])) : [(i,j) for i in levels for j in levels if i<j]
    est, se, t, p = Float64[], Float64[], Float64[], Float64[]

    for (ℓ1,ℓ2) in pairs
        key1 = (ℓ1,); key2 = (ℓ2,)
        θ1, θ2 = a.ame[var][key1], a.ame[var][key2]
        σ1, σ2 = a.se[var][key1], a.se[var][key2]
        g1, g2 = a.grad[var][key1], a.grad[var][key2]
        Δ, seΔ, tstat, pval = _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vcov, a.df_residual)
        push!(est, Δ); push!(se, seΔ); push!(t, tstat); push!(p, pval)
    end

    return ContrastResult([var], pairs, est, se, t, p, a.df_residual)
end
