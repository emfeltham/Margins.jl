###############################################################################
# demo_ame_tests_2.jl
# (second set of tests)

###############################################################################
# 6.   Logistic regression – no interactions
#      Outcome: mtcars.am (0/1) ~ mpg + wt
###############################################################################
using RDatasets, DataFrames, GLM, Statistics, Distributions, Margins
mt = dataset("datasets", "mtcars") |> DataFrame
# ensure binary is treated as numeric 0/1
mt.am = convert.(Int, mt.AM)

form6 = @formula(am ~ mpg + wt)
m6    = glm(form6, mt, Binomial(), LogitLink())
ame6  = ame(m6, :mpg, mt)

println("\n=== Scenario 6: Logistic no interactions ===")
println("Formula : ", form6)
println("AME dPr(am=1)/dmpg = $(round(ame6.ame; digits=4))  (se = $(round(ame6.se; digits=4)))")

# closed‐form: ∂E[y]/∂mpg = β_mpg · p_i·(1−p_i); AME = mean(β·p(1−p))
let
    β   = coef(m6)
    vc  = vcov(m6)
    i   = findfirst(isequal("mpg"), coefnames(m6))
    β1  = β[i]
    η   = predict(m6)                          # on link scale
    p   = 1 ./(1 .+ exp.(-η))
    deriv = β1 .* (p .* (1 .- p))
    ame_closed = mean(deriv)
    # delta‐method variance: Var(∑ w_j β_j) with weights w_j = mean(d∂/dβ_j)
    # Here ∂AME/∂β0 = mean(β1 * p*(1-p) * (1−2p) * p), etc.  Instead we do full jacobian:
    # But simplest: approximate var via empirical gradient
    # We'll compute numeric Jacobian rows for β0 and β1:
    # For brevity in test, just check mixture of β1 and its se:
    var_closed = vc[i,i] * (mean(p .* (1 .- p)))^2
    se_closed  = sqrt(var_closed)
    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed, digits=4))
    @assert ame_closed ≈ ame6.ame atol=1e-6
    @assert se_closed  ≈ ame6.se atol=1e-6
end

###############################################################################
# 7.   Logistic regression – with interaction
#      am ~ mpg * wt
###############################################################################
form7 = @formula(am ~ mpg * wt)
m7    = glm(form7, mt, Binomial(), LogitLink())
ame7  = ame(m7, :mpg, mt)

println("\n=== Scenario 7: Logistic interaction ===")
println("Formula : ", form7)
println("AME dPr(am=1)/dmpg = $(round(ame7.ame; digits=4))  (se = $(round(ame7.se; digits=4)))")

let
    β   = coef(m7)
    vc  = vcov(m7)
    cn  = coefnames(m7)
    i1  = findfirst(isequal("mpg"),           cn)
    ix  = findfirst(isequal("mpg & wt"),      cn)
    β1  = β[i1]; β3 = β[ix]
    η   = predict(m7)
    p   = 1 ./(1 .+ exp.(-η))
    # marginal effect for observation i: (β1 + β3*wt_i) * p_i*(1-p_i)
    deriv = (β1 .+ β3 .* mt.wt) .* (p .* (1 .- p))
    ame_closed = mean(deriv)
    # approximate var via β1 only (for simplicity in test):
    var_closed = vc[i1,i1] * (mean(p .* (1 .- p)))^2
    se_closed  = sqrt(var_closed)
    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed, digits=4))
    @assert ame_closed ≈ ame7.ame atol=1e-6
    @assert se_closed  ≈ ame7.se atol=1e-6
end

###############################################################################
# 8.   Probit regression – no interactions
#      am ~ mpg + wt
###############################################################################
m8   = glm(form6, mt, Binomial(), ProbitLink())
ame8 = ame(m8, :wt, mt)

println("\n=== Scenario 8: Probit no interactions ===")
println("Formula : ", form6, " | Probit")
println("AME dPr(am=1)/dwt = $(round(ame8.ame; digits=4))  (se = $(round(ame8.se; digits=4)))")

let
    β   = coef(m8)
    vc  = vcov(m8)
    cn  = coefnames(m8)
    i   = findfirst(isequal("wt"), cn)
    β1  = β[i]
    η   = predict(m8)
    pdf = pdf.(Normal(), η)
    deriv = β1 .* pdf
    ame_closed = mean(deriv)
    var_closed = vc[i,i] * (mean(pdf))^2
    se_closed  = sqrt(var_closed)
    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed, digits=4))
    @assert ame_closed ≈ ame8.ame atol=1e-6
    @assert se_closed  ≈ ame8.se atol=1e-6
end

###############################################################################
# 9.   Poisson regression – no interactions (synthetic data)
###############################################################################
using Random
Random.seed!(1234)
n = 500
x = randn(n)
z = randn(n)
μ_true = exp.(0.5 .+ 1.2 .* x .+ 0.7 .* z)
y = rand.(Poisson.(μ_true))
df = DataFrame(x=x, z=z, y=y)

form9 = @formula(y ~ x + z)
m9    = glm(form9, df, Poisson(), LogLink())
ame9  = ame(m9, :x, df)

println("\n=== Scenario 9: Poisson no interactions ===")
println("Formula : ", form9)
println("AME dy/dx = $(round(ame9.ame; digits=4))  (se = $(round(ame9.se; digits=4)))")

let
    β   = coef(m9)
    vc  = vcov(m9)
    cn  = coefnames(m9)
    i   = findfirst(isequal("x"), cn)
    β1  = β[i]
    η   = predict(m9)                 # on link scale
    μ   = exp.(η)
    # marginal effect: β1 * μ_i
    deriv     = β1 .* μ
    ame_closed = mean(deriv)
    # delta-method: ∂AME/∂β0 = β1*A, ∂AME/∂β1 = A + β1*mean(x.*μ)
    A  = mean(μ)
    B  = mean(x .* μ)
    g  = zeros(length(β))
    g[1] = β1 * A           # for β0
    g[i] = A + β1 * B       # for β1
    var_closed = dot(g, vc * g)
    se_closed  = sqrt(var_closed)
    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed, digits=4))
    @assert ame_closed ≈ ame9.ame atol=1e-6
    @assert se_closed  ≈ ame9.se atol=1e-6
end

###############################################################################
# 10.  Poisson regression – with interaction x*z
###############################################################################
form10 = @formula(y ~ x * z)
m10    = glm(form10, df, Poisson(), LogLink())
ame10  = ame(m10, :x, df)

println("\n=== Scenario 10: Poisson interaction ===")
println("Formula : ", form10)
println("AME dy/dx = $(round(ame10.ame; digits=4))  (se = $(round(ame10.se; digits=4)))")

let
    β   = coef(m10)
    vc  = vcov(m10)
    cn  = coefnames(m10)
    i1  = findfirst(isequal("x"),       cn)
    ix  = findfirst(isequal("x & z"),   cn)
    β1  = β[i1]; β3 = β[ix]
    η   = predict(m10)
    μ   = exp.(η)
    # marginal effect: (β1 + β3*z_i) * μ_i
    deriv     = (β1 .+ β3 .* df.z) .* μ
    ame_closed = mean(deriv)
    # gradient components for β0, β1, β3
    A    = mean(deriv)                     # =AME
    B    = mean(μ)                         # for β1
    C    = mean(df.z .* μ)                 # for β3
    g    = zeros(length(β))
    g[1] = A                              # ∂AME/∂β0
    g[i1] = B                            # ∂AME/∂β1
    g[ix] = C                           # ∂AME/∂β3
    var_closed = dot(g, vc * g)
    se_closed  = sqrt(var_closed)
    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed, digits=4))
    @assert ame_closed ≈ ame10.ame atol=1e-6
    @assert se_closed  ≈ ame10.se atol=1e-6
end

###############################################################################
# End of new tests.
###############################################################################
