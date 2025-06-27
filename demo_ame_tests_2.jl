###############################################################################
# demo_ame_tests_2.jl
# (second set of tests)

###############################################################################
# 6.   Logistic regression – no interactions (fixed)
###############################################################################
form6 = @formula(am ~ MPG + WT)
m6    = glm(form6, mt, Binomial(), LogitLink())
ame6  = ame(m6, :MPG, mt)

println("\n=== Scenario 6: Logistic no interactions (fixed) ===")
println("Formula : ", form6)
println("AME dPr(am=1)/dMPG = $(round(ame6.ame[:MPG]; digits=4))  (se = $(round(ame6.se[:MPG]; digits=4)))")

# closed‐form check with full delta‐method
let
    β    = coef(m6)
    V    = vcov(m6)
    cn   = coefnames(m6)

    # locate indices
    i0   = findfirst(isequal("(Intercept)"), cn)
    i1   = findfirst(isequal("MPG"),         cn)

    # design matrix, link, and p_i
    X    = modelmatrix(m6)                               # n×k
    p    = predict(m6)                                   # prob. scale

    # raw per‐obs derivatives ∂p_i/∂x = β1·p(1-p)
    deriv_i    = β[i1] .* p .* (1 .- p)
    ame_closed = mean(deriv_i)

    # build gradient vector g_j = mean( ∂[β1·p(1-p)]/∂β_j )
    n    = length(p)
    w    = p .* (1 .- p)                                 # ∂p/∂η
    dw   = w .* (1 .- 2p)                                # ∂[p(1-p)]/∂η

    g    = zeros(eltype(β), length(β))
    for j in 1:length(β)
        # when j==i1 we get ∂/∂β1 [β1·w] = w  + β1·(dw·X[:,j])
        # otherwise it's just β1·dw·X[:,j]
        g[j] = mean( (j==i1 ? w : zero(w)) .+ β[i1] .* dw .* X[:,j] )
    end

    var_closed = g' * V * g
    se_closed  = sqrt(var_closed)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame6.ame[:MPG] atol=1e-8
    @assert se_closed  ≈ ame6.se[:MPG]  atol=1e-8
end

###############################################################################
# 7.   Logistic regression – with interaction (fixed)
#      am ~ MPG * WT
###############################################################################
form7 = @formula(am ~ MPG * WT)
m7    = glm(form7, mt, Binomial(), LogitLink())
ame7  = ame(m7, :MPG, mt)

println("\n=== Scenario 7: Logistic interaction (fixed) ===")
println("Formula : ", form7)
println("AME dPr(am=1)/dMPG = $(round(ame7.ame[:MPG]; digits=4))  (se = $(round(ame7.se[:MPG]; digits=4)))")

# closed‐form check with full delta‐method
let
    β    = coef(m7)
    V    = vcov(m7)
    cn   = coefnames(m7)
    i1   = findfirst(isequal("MPG"),       cn)
    ix   = findfirst(isequal("MPG & WT"),  cn)

    # design matrix and fitted probabilities
    X    = modelmatrix(m7)                              # n×k
    η    = predict(m7)                                  # prob scale
    w    = p .* (1 .- p)                                # ∂p/∂η
    dw   = w .* (1 .- 2p)                               # ∂(p(1-p))/∂η

    # per-obs marginal effect: A_i * w_i
    A    = β[i1] .+ β[ix] .* mt.WT
    ame_closed = mean(A .* w)

    # build the gradient vector g_j = mean[ ∂(A_i·w_i)/∂β_j ]
    g    = zeros(eltype(β), length(β))
    for j in eachindex(β)
        # ∂A_i/∂β_j = 1 if j==i1, = WT_i if j==ix, else 0
        dA = (j == i1 ? ones(length(w)) : zero(w)) .+
             (j == ix ? mt.WT : zero(w))
        # ∂(A_i·w_i)/∂β_j = dA * w  +  A * (dw .* X[:,j])
        g[j] = mean( dA .* w .+ A .* (dw .* X[:,j]) )
    end

    var_closed = g' * V * g
    se_closed  = sqrt(var_closed)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame7.ame[:MPG] atol=1e-8
    @assert se_closed  ≈ ame7.se[:MPG]  atol=1e-8
end

###############################################################################
# 8.   Probit regression – no interactions (fixed, manual η)
#      am ~ MPG + WT
###############################################################################
m8   = glm(form6, mt, Binomial(), ProbitLink())
ame8 = ame(m8, :WT, mt)

println("\n=== Scenario 8: Probit no interactions (fixed) ===")
println("Formula : ", form6, " | Probit")
println("AME dPr(am=1)/dWT = $(round(ame8.ame[:WT]; digits=4))  (se = $(round(ame8.se[:WT]; digits=4)))")

let
    β   = coef(m8)
    V   = vcov(m8)
    cn  = coefnames(m8)
    i   = findfirst(isequal("WT"), cn)

    # build the linear predictor by hand
    X   = modelmatrix(m8)             # n×k design matrix
    η   = X * β                       # link scale

    # for probit: φ(η) and its derivative
    ϕ   = pdf.(Normal(), η)           # φ(η)
    dϕ  = -η .* ϕ                     # dφ/dη = -η φ(η)

    # per-obs marginal effect ∂Pr/∂WT = β_WT * φ(η)
    me_i   = β[i] .* ϕ
    ame_closed = mean(me_i)

    # gradient g_j = mean[ ∂(β_i·φ(η_i))/∂β_j ]
    g   = zeros(eltype(β), length(β))
    for j in eachindex(β)
        # ∂/∂β_j [β_i·φ(η)] = (j==i ? φ(η) : 0) + β_i * (dφ/dη * X[:,j])
        term1 = (j == i ? ϕ : zero(ϕ))
        term2 = β[i] .* (dϕ .* X[:, j])
        g[j]  = mean(term1 .+ term2)
    end

    var_closed = g' * V * g
    se_closed  = sqrt(var_closed)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame8.ame[:WT] atol=1e-8
    @assert se_closed  ≈ ame8.se[:WT]  atol=1e-8
end

###############################################################################
# 9.   Poisson regression – no interactions (fixed)
###############################################################################
using Random
import LinearAlgebra:dot
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

println("\n=== Scenario 9: Poisson no interactions (fixed) ===")
println("Formula : ", form9)
println("AME dy/dx = $(round(ame9.ame[:x]; digits=4))  (se = $(round(ame9.se[:x]; digits=4)))")

let
    β   = coef(m9)
    V   = vcov(m9)
    cn  = coefnames(m9)
    ix  = findfirst(isequal("x"), cn)

    # build linear predictor and μ_i
    X   = modelmatrix(m9)          # n×k
    η   = X * β                    # link scale
    μ   = exp.(η)                  # response E[y|x]

    # per‐obs marginal effect: β_x * μ_i
    me_i       = β[ix] .* μ
    ame_closed = mean(me_i)

    # full delta‐method gradient
    k   = length(β)
    g   = zeros(eltype(β), k)
    for j in 1:k
        # ∂[β_x·μ_i]/∂β_j = (j==ix ? μ : 0) + β_x · (μ .* X[:,j])
        term1 = (j == ix ? μ : zero(μ))
        term2 = β[ix] .* (μ .* X[:, j])
        g[j]  = mean(term1 .+ term2)
    end

    var_closed = g' * V * g
    se_closed  = sqrt(var_closed)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame9.ame[:x] atol=1e-8
    @assert se_closed  ≈ ame9.se[:x]  atol=1e-8
end

###############################################################################
# 10.  Poisson regression – with interaction (fixed)
#      y ~ x * z
###############################################################################
form10 = @formula(y ~ x * z)
m10    = glm(form10, df, Poisson(), LogLink())
ame10  = ame(m10, :x, df)

println("\n=== Scenario 10: Poisson interaction (fixed) ===")
println("Formula : ", form10)
println("AME dy/dx = $(round(ame10.ame[:x]; digits=4))  (se = $(round(ame10.se[:x]; digits=4)))")

let
    β   = coef(m10)
    V   = vcov(m10)
    cn  = coefnames(m10)
    i1  = findfirst(isequal("x"),       cn)
    ix  = findfirst(isequal("x & z"),   cn)

    # build the linear predictor and μ_i
    X   = modelmatrix(m10)         # n×k design matrix
    η   = X * β                     # link scale
    μ   = exp.(η)                   # E[y|x,z]

    # per‐obs marginal effect: (β_x + β_{x&z}·z_i) * μ_i
    A         = β[i1] .+ β[ix] .* df.z
    me_i      = A .* μ
    ame_closed = mean(me_i)

    # full delta‐method gradient: g_j = mean[∂me_i/∂β_j]
    k   = length(β)
    g   = zeros(eltype(β), k)
    for j in 1:k
        # ∂A_i/∂β_j
        dA = (j == i1 ? ones(length(μ)) : zero(μ)) .+
             (j == ix ? df.z : zero(μ))
        # ∂μ_i/∂β_j = μ_i * X[:,j]
        dμ = μ .* X[:, j]
        # ∂me_i/∂β_j = dA * μ + A * dμ
        g[j] = mean(dA .* μ .+ A .* dμ)
    end

    var_closed = g' * V * g
    se_closed  = sqrt(var_closed)

    println("Closed-form AME   = ", round(ame_closed, digits=4))
    println("Closed-form s.e.  = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame10.ame[:x] atol=1e-8
    @assert se_closed  ≈ ame10.se[:x]  atol=1e-8
end

###############################################################################
# End of new tests.
###############################################################################
