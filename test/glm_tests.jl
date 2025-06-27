# glm_tests.jl

# Load datasets
mt = dataset("datasets", "mtcars") |> DataFrame

# Prepare Poisson simulation data for scenarios 9 & 10
Random.seed!(1234)
n = 500
x = randn(n)
z = randn(n)
μ_true = exp.(0.5 .+ 1.2 .* x .+ 0.7 .* z)
y = rand.(Poisson.(μ_true))
df = DataFrame(x=x, z=z, y=y)

@testset "Scenario 6: Logistic regression – no interactions (fixed)" begin
    form6 = @formula(AM ~ MPG + WT)
    m6 = glm(form6, mt, Binomial(), LogitLink())
    ame6 = ame(m6, :MPG, mt)

    # Closed-form delta-method
    β = coef(m6)
    V = vcov(m6)
    cn = coefnames(m6)
    i1 = findfirst(isequal("MPG"), cn)
    X = modelmatrix(m6)
    p = predict(m6)
    deriv_i = β[i1] .* p .* (1 .- p)
    ame_closed = mean(deriv_i)
    w = p .* (1 .- p)
    dw = w .* (1 .- 2 .* p)
    g = zeros(eltype(β), length(β))
    for j in 1:length(β)
        g[j] = mean((j == i1 ? w : zero(w)) .+ β[i1] .* dw .* X[:, j])
    end
    var_closed = g' * V * g
    se_closed = sqrt(var_closed)

    @test isapprox(ame6.ame[:MPG], ame_closed; atol=1e-8)
    @test isapprox(ame6.se[:MPG], se_closed; atol=1e-8)
end

@testset "Scenario 7: Logistic regression – with interaction (fixed)" begin
    form7 = @formula(AM ~ MPG * WT)
    m7 = glm(form7, mt, Binomial(), LogitLink())
    ame7 = ame(m7, :MPG, mt)

    β = coef(m7)
    V = vcov(m7)
    cn = coefnames(m7)
    i1 = findfirst(isequal("MPG"), cn)
    ix = findfirst(isequal("MPG & WT"), cn)
    X = modelmatrix(m7)
    p = predict(m7)
    w = p .* (1 .- p)
    dw = w .* (1 .- 2 .* p)
    A = β[i1] .+ β[ix] .* mt.WT
    ame_closed = mean(A .* w)
    g = zeros(eltype(β), length(β))
    for j in eachindex(β)
        dA = (j == i1 ? ones(length(w)) : zero(w)) .+ (j == ix ? mt.WT : zero(w))
        g[j] = mean(dA .* w .+ A .* (dw .* X[:, j]))
    end
    var_closed = g' * V * g
    se_closed = sqrt(var_closed)

    @test isapprox(ame7.ame[:MPG], ame_closed; atol=1e-8)
    @test isapprox(ame7.se[:MPG], se_closed; atol=1e-8)
end

@testset "Scenario 8: Probit regression – no interactions (fixed, manual η)" begin
    form6 = @formula(AM ~ MPG + WT)
    m8 = glm(form6, mt, Binomial(), ProbitLink())
    ame8 = ame(m8, :WT, mt)

    β = coef(m8)
    V = vcov(m8)
    cn = coefnames(m8)
    i = findfirst(isequal("WT"), cn)
    X = modelmatrix(m8)
    η = X * β
    ϕ = pdf.(Normal(), η)
    dϕ = -η .* ϕ
    me_i = β[i] .* ϕ
    ame_closed = mean(me_i)
    g = zeros(eltype(β), length(β))
    for j in eachindex(β)
        term1 = (j == i ? ϕ : zero(ϕ))
        term2 = β[i] .* (dϕ .* X[:, j])
        g[j] = mean(term1 .+ term2)
    end
    var_closed = g' * V * g
    se_closed = sqrt(var_closed)

    @test isapprox(ame8.ame[:WT], ame_closed; atol=1e-8)
    @test isapprox(ame8.se[:WT], se_closed; atol=1e-8)
end

@testset "Scenario 9: Poisson regression – no interactions (fixed)" begin
    form9 = @formula(y ~ x + z)
    m9 = glm(form9, df, Poisson(), LogLink())
    ame9 = ame(m9, :x, df)

    β = coef(m9)
    V = vcov(m9)
    cn = coefnames(m9)
    ix = findfirst(isequal("x"), cn)
    X = modelmatrix(m9)
    η = X * β
    μ = exp.(η)
    me_i = β[ix] .* μ
    ame_closed = mean(me_i)
    k = length(β)
    g = zeros(eltype(β), k)
    for j in 1:k
        term1 = (j == ix ? μ : zero(μ))
        term2 = β[ix] .* (μ .* X[:, j])
        g[j] = mean(term1 .+ term2)
    end
    var_closed = g' * V * g
    se_closed = sqrt(var_closed)

    @test isapprox(ame9.ame[:x], ame_closed; atol=1e-8)
    @test isapprox(ame9.se[:x], se_closed; atol=1e-8)
end

@testset "Scenario 10: Poisson regression – with interaction (fixed)" begin
    form10 = @formula(y ~ x * z)
    m10 = glm(form10, df, Poisson(), LogLink())
    ame10 = ame(m10, :x, df)

    β = coef(m10)
    V = vcov(m10)
    cn = coefnames(m10)
    i1 = findfirst(isequal("x"), cn)
    ix = findfirst(isequal("x & z"), cn)
    X = modelmatrix(m10)
    η = X * β
    μ = exp.(η)
    A = β[i1] .+ β[ix] .* df.z
    me_i = A .* μ
    ame_closed = mean(me_i)
    k = length(β)
    g = zeros(eltype(β), k)
    for j in 1:k
        dA = (j == i1 ? ones(length(μ)) : zero(μ)) .+ (j == ix ? df.z : zero(μ))
        dμ = μ .* X[:, j]
        g[j] = mean(dA .* μ .+ A .* dμ)
    end
    var_closed = g' * V * g
    se_closed = sqrt(var_closed)

    @test isapprox(ame10.ame[:x], ame_closed; atol=1e-8)
    @test isapprox(ame10.se[:x], se_closed; atol=1e-8)
end

