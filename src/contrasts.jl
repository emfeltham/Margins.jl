# contrasts.jl — updated for new MarginsResult API

############################
# Utility helpers          #
############################
_scalar(v) = v isa Number ? v : only(values(v))
_vec(v)    = v isa AbstractVector ? v : only(values(v))

############################
# Internal Δ‑method helper #
############################
function _compute_contrast(θ1::Real, θ2::Real,
                           σ1::Real, σ2::Real,
                           g1::AbstractVector, g2::AbstractVector,
                           vcov::Union{AbstractMatrix,Nothing}, ν::Real)
    cov12 = vcov === nothing ? zero(σ1) : dot(g1, vcov * g2)
    Δ     = θ1 - θ2
    seΔ   = sqrt(abs(σ1^2 + σ2^2 - 2cov12))
    tstat = Δ / seΔ
    pval  = 2 * (1 - cdf(TDist(ν), abs(tstat)))
    return Δ, seΔ, tstat, pval
end

############################
# Contrast between results #
############################
"""
    contrast(a, b; var, vcov=nothing) -> ContrastResult

Contrast `var` between two `MarginsResult`s. Works for both scalar effects and
rep‑value grids.
"""
function contrast(a::MarginsResult, b::MarginsResult; var::Symbol, vcov=nothing)
    var ∈ a.vars || throw(ArgumentError("$(var) not in first result"))
    var ∈ b.vars || throw(ArgumentError("$(var) not in second result"))

    ame_a, ame_b = a.effects[var], b.effects[var]
    df           = min(a.df_residual, b.df_residual)

    if ame_a isa Dict && ame_b isa Dict
        keys_a = sort(collect(keys(ame_a)))
        keys_b = sort(collect(keys(ame_b)))
        keys_a == keys_b || throw(ArgumentError("rep‑value grids differ for $(var)"))

        est = Float64[]; se = Float64[]; t = Float64[]; p = Float64[]
        for key in keys_a
            θ1, θ2 = ame_a[key], ame_b[key]
            σ1, σ2 = a.ses[var][key], b.ses[var][key]
            g1, g2 = a.grad[var][key], b.grad[var][key]
            Δ,seΔ,tstat,pval = _compute_contrast(θ1,θ2,σ1,σ2,g1,g2,vcov,df)
            push!(est,Δ); push!(se,seΔ); push!(t,tstat); push!(p,pval)
        end
        return ContrastResult([var], keys_a, est, se, t, p, df)
    else
        θ1, θ2 = _scalar(ame_a), _scalar(ame_b)
        σ1, σ2 = _scalar(a.ses[var]), _scalar(b.ses[var])
        g1, g2 = _vec(a.grad[var]), _vec(b.grad[var])
        Δ,seΔ,tstat,pval = _compute_contrast(θ1,θ2,σ1,σ2,g1,g2,vcov,df)
        return ContrastResult([var], [(:a,:b)], [Δ], [seΔ], [tstat], [pval], df)
    end
end

############################
# Contrast within a result #
############################
"""
    contrast(res; var1, var2, vcov=nothing) -> ContrastResult

Contrast two predictors inside the same `MarginsResult`.
"""
function contrast(res::MarginsResult; var1::Symbol, var2::Symbol, vcov=nothing)
    var1 ∈ res.vars || throw(ArgumentError("$(var1) not in result"))
    var2 ∈ res.vars || throw(ArgumentError("$(var2) not in result"))

    ame1, ame2 = res.effects[var1], res.effects[var2]
    df         = res.df_residual

    if ame1 isa Dict && ame2 isa Dict
        keys1 = sort(collect(keys(ame1)))
        keys2 = sort(collect(keys(ame2)))
        keys1 == keys2 || throw(ArgumentError("rep‑value grids differ for $(var1) & $(var2)"))

        est = Float64[]; se = Float64[]; t = Float64[]; p = Float64[]
        for key in keys1
            θ1, θ2 = ame1[key], ame2[key]
            σ1, σ2 = res.ses[var1][key], res.ses[var2][key]
            g1, g2 = res.grad[var1][key], res.grad[var2][key]
            Δ,seΔ,tstat,pval = _compute_contrast(θ1,θ2,σ1,σ2,g1,g2,vcov,df)
            push!(est,Δ); push!(se,seΔ); push!(t,tstat); push!(p,pval)
        end
        return ContrastResult([var1,var2], keys1, est, se, t, p, df)
    else
        θ1, θ2 = _scalar(ame1), _scalar(ame2)
        σ1, σ2 = _scalar(res.ses[var1]), _scalar(res.ses[var2])
        g1, g2 = _vec(res.grad[var1]), _vec(res.grad[var2])
        Δ,seΔ,tstat,pval = _compute_contrast(θ1,θ2,σ1,σ2,g1,g2,vcov,df)
        return ContrastResult([var1,var2], [(var1,var2)], [Δ], [seΔ], [tstat], [pval], df)
    end
end

############################
# Pairwise level contrasts #
############################
"""
    contrast_levels(res, var; comparisons=:all, vcov=nothing) -> ContrastResult

Pairwise contrasts of AMEs across levels of `var`.
"""
function contrast_levels(res::MarginsResult, var::Symbol; comparisons::Symbol=:all, vcov=nothing)
    var ∈ res.vars || throw(ArgumentError("$(var) not in result"))

    levels = haskey(res.repvals,var) && !isempty(res.repvals[var]) ?
             sort(res.repvals[var]) : sort([first(k) for k in keys(res.effects[var])])
    pairs = comparisons == :adjacent ? collect(zip(levels[1:end-1], levels[2:end])) :
            [(i,j) for i in levels for j in levels if i<j]

    est = Float64[]; se = Float64[]; t = Float64[]; p = Float64[]
    for (ℓ1,ℓ2) in pairs
        key1, key2 = (ℓ1,), (ℓ2,)
        θ1, θ2 = res.effects[var][key1], res.effects[var][key2]
        σ1, σ2 = res.ses[var][key1],    res.ses[var][key2]
        g1, g2 = res.grad[var][key1],   res.grad[var][key2]
        Δ,seΔ,tstat,pval = _compute_contrast(θ1,θ2,σ1,σ2,g1,g2,vcov,res.df_residual)
        push!(est,Δ); push!(se,seΔ); push!(t,tstat); push!(p,pval)
    end
    return ContrastResult([var], pairs, est, se, t, p, res.df_residual)
end