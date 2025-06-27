###############################################################################
# demo_ame_mixedmodels_tests.jl
#
# 11.  LMM – random intercept only
# 12.  LMM – random slope
# 13.  LMM – transformation + random intercept
# 14.  GLMM (logistic) – random intercept on cbpp
# 15.  GLMM (logistic) – synthetic random intercept
###############################################################################

using Revise
using RDatasets, DataFrames, CategoricalArrays
using Statistics, Random
using MixedModels, Margins

# ---------------------------------------------------------------------------
# Load sleepstudy for LMMs
# ---------------------------------------------------------------------------
sleep = dataset("lme4","sleepstudy") |> DataFrame
sleep.Subject = categorical(sleep.Subject);

# ---------------------------------------------------------------------------
# 11.  LMM – random intercept only
#      Reaction ~ Days + (1|Subject)
# ---------------------------------------------------------------------------
form11 = @formula(Reaction ~ Days + (1|Subject))
m11    = fit(MixedModel, form11, sleep)
ame11  = ame(m11, :Days, sleep)

println("\n=== Scenario 11: LMM random intercept ===")
println("Formula : ", form11)
println("AME dReaction/dDays = $(round(ame11.ame[:Days]; digits=4))  (se = $(round(ame11.se[:Days]; digits=4)))")

let
    fe   = fixef(m11)
    cn   = coefnames(m11)
    i    = findfirst(isequal("Days"), cn)
    vc   = vcov(m11)
    ame_closed = fe[i]
    se_closed  = sqrt(vc[i,i])

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame11.ame[:Days]
    @assert se_closed   ≈ ame11.se[:Days]
end

# ---------------------------------------------------------------------------
# 12.  LMM – random slope on Days
#      Reaction ~ Days + (Days|Subject)
# ---------------------------------------------------------------------------
form12 = @formula(Reaction ~ Days + (Days|Subject))
m12    = fit(MixedModel, form12, sleep)
ame12  = ame(m12, :Days, sleep)

println("\n=== Scenario 12: LMM random slope ===")
println("Formula : ", form12)
println("AME dReaction/dDays = $(round(ame12.ame[:Days]; digits=4))  (se = $(round(ame12.se[:Days]; digits=4)))")

let
    fe   = fixef(m12)
    cn   = coefnames(m12)
    i    = findfirst(isequal("Days"), cn)
    vc   = vcov(m12)
    ame_closed = fe[i]
    se_closed  = sqrt(vc[i,i])

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame12.ame[:Days]
    @assert se_closed   ≈ ame12.se[:Days]
end

# ---------------------------------------------------------------------------
# 13.  LMM – transformation of x + random intercept (fixed)
#      Reaction ~ log1p(Days) + (1|Subject)
# ---------------------------------------------------------------------------
form13 = @formula(Reaction ~ log1p(Days) + (1|Subject))
m13    = fit(MixedModel, form13, sleep)
ame13  = ame(m13, :Days, sleep)

println("\n=== Scenario 13: LMM w/ log1p(Days) + random intercept ===")
println("Formula : ", form13)
println("AME dReaction/dDays = $(round(ame13.ame[:Days]; digits=4))  (se = $(round(ame13.se[:Days]; digits=4)))")

let
    fe        = fixef(m13)
    cn        = coefnames(m13)
    i_log1p   = findfirst(isequal("log1p(Days)"), cn)
    vc        = vcov(m13)

    # chain‐rule: ∂/∂Days log1p(Days) = 1/(Days+1)
    mean_inv1 = mean(1 ./(sleep.Days .+ 1))

    ame_closed = fe[i_log1p] * mean_inv1
    se_closed  = sqrt(vc[i_log1p, i_log1p] * mean_inv1^2)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame13.ame[:Days]
    @assert se_closed   ≈ ame13.se[:Days]
end

# ---------------------------------------------------------------------------
# 14.  GLMM (logistic) – random intercept on cbpp (fixed)
#       Prop = Incidence/Size, wts = Size
# ---------------------------------------------------------------------------
using RDatasets, DataFrames, CategoricalArrays
using MixedModels, Margins, Statistics
import LinearAlgebra: dot

cbpp        = dataset("lme4","cbpp") |> DataFrame
cbpp.Herd   = categorical(cbpp.Herd)
cbpp.Period = categorical(cbpp.Period)
cbpp.Prop   = cbpp.Incidence ./ cbpp.Size

form14 = @formula(Prop ~ Period + (1|Herd))
m14    = fit(GeneralizedLinearMixedModel,
             form14, cbpp,
             Binomial(), LogitLink(),
             wts = cbpp.Size)
ame14  = ame(m14, :Period, cbpp)

println("\n=== Scenario 14: GLMM logistic w/ random intercept ===")
println("Formula : ", form14, "  |  wts = Size")
println("AMEs of shifting Period ↦")
for ((from,to), eff) in ame14.ame[:Period]
    se = ame14.se[:Period][(from,to)]
    println("  $from → $to : AME=$(round(eff; digits=4))  (se=$(round(se; digits=4)))")
end

# true closed-form for 1→2
let
    fe = fixef(m14)
    V  = vcov(m14)
    cn = coefnames(m14)

    i0 = findfirst(isequal("(Intercept)"), cn)
    i2 = findfirst(isequal("Period: 2"),   cn)

    # linear predictors
    η1 = fe[i0]
    η2 = fe[i0] + fe[i2]

    # probabilities
    p1 = 1 / (1 + exp(-η1))
    p2 = 1 / (1 + exp(-η2))

    # closed-form AME = p2 – p1
    ame_closed = p2 - p1

    # gradient g_j = ∂(p2-p1)/∂β_j
    g = zeros(eltype(fe), length(fe))
    # ∂p1/∂β0 =  p1*(1-p1), ∂p2/∂β0 = p2*(1-p2)
    g[i0] = (p2*(1-p2)) - (p1*(1-p1))
    # ∂p2/∂β2 = p2*(1-p2)
    g[i2] =  p2*(1-p2)

    var_closed = g' * V * g
    se_closed  = sqrt(var_closed)

    println("\n[Closed-form check for 1 → 2]")
    println("  AME   = ", round(ame_closed; digits=4))
    println("  s.e.  = ", round(se_closed;   digits=4))

    @assert ame_closed ≈ ame14.ame[:Period][("1","2")] atol=1e-8
    @assert se_closed   ≈ ame14.se[:Period][("1","2")]   atol=1e-8
end

# ---------------------------------------------------------------------------
# 15.  GLMM (logistic) – synthetic data random intercept
#      y ~ x + (1|group)
# ---------------------------------------------------------------------------
Random.seed!(2025)
n, G = 500, 10
group = rand(1:G, n)
u     = randn(G) .* 0.8
x     = randn(n)
η_s   = 0.2 .+ 1.5 .* x .+ u[group]
p_s   = 1 ./(1 .+ exp.(-η_s))
y_s   = rand.(Bernoulli.(p_s))
df2   = DataFrame(x=x, group=categorical(string.(group)), y=y_s)

form15  = @formula(y ~ x + (1|group))
m15     = fit(GeneralizedLinearMixedModel, form15, df2, Bernoulli())
ame15   = ame(m15, :x, df2)

println("\n=== Scenario 15: GLMM logistic synthetic ===")
println("Formula : ", form15)
println("AME dPr(y=1)/dx = $(round(ame15.ame[:x]; digits=4))  (se = $(round(ame15.se[:x]; digits=4)))")

let
    fe   = fixef(m15)
    cn   = coefnames(m15)
    i    = findfirst(isequal("x"), cn)
    vc   = vcov(m15)

    X    = modelmatrix(m15)
    η    = X * fe
    p    = 1 ./(1 .+ exp.(-η))
    w    = p .* (1 .- p)

    ame_closed = mean(fe[i] .* w)

    k    = length(fe)
    g    = zeros(eltype(fe), k)
    for j in 1:k
        dw    = w .* (1 .- 2p)
        term1 = (j == i ? w : zero(w))
        term2 = fe[i] .* (dw .* X[:, j])
        g[j]  = mean(term1 .+ term2)
    end

    var_closed = g' * vc * g
    se_closed  = sqrt(var_closed)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame15.ame[:x]
    @assert se_closed   ≈ ame15.se[:x]
end

###############################################################################
# End of mixed‐model tests
###############################################################################
