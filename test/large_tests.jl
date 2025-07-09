
@testset "large data OLS (with repvals)" begin
    # simulate
    Random.seed!(42)
    n = 2_000_000
    x = randn(n)
    d = rand(n) .> 0.5       # Bool
    z = randn(n)
    # true model: y = β0 + βx x + βd d + βz z + βxd x*d + βxz x*z + βdz d*z + βxdz x*d*z + ε
    β = (β0=1.0, βx=2.0, βd= -1.5, βz=0.5, βxd=0.8, βxz=1.2, βdz=-0.7, βxdz=0.4)
    μ = β.β0 .+ β.βx*x .+ β.βd*(d .== true) .+ β.βz*z .+
        β.βxd*(x .* (d .== true)) .+ β.βxz*(x .* z) .+ β.βdz*((d .== true) .* z) .+
        β.βxdz*(x .* (d .== true) .* z)
    y = μ .+ randn(n)*0.1
    df = DataFrame(y=y, x=x, d=CategoricalArray(d), z=z)

    m = lm(@formula(y ~ x * d * z), df)
    # AME of x at d=false, z= mean(z)
    zv = mean(z)
    # closed-form derivative: ∂μ/∂x = βx + βxd*d + βxz*z + βxdz*d*z
    # at d=false => d=0: deriv = βx + βxz*zv
    ame_closed = β.βx + β.βxz*zv
    # SE via delta method: g = ∂β/∂x vector; compute var(g'β̂)
    cn = coefnames(m); coefs = coef(m); V = vcov(m)
    # build contrast c
    c = zeros(length(coefs))
    # intercept excluded
    iβx  = findfirst(isequal("x"), cn)
    iβxz = findfirst(isequal("x & z"), cn)
    c[iβx]  = 1
    c[iβxz] = zv
    se_closed = sqrt(c' * V * c)

    ame   = margins(m, :x, df; repvals=Dict(:d => categorical([false]), :z => [zv]))

    # fails at high tolerance (above 1e-2)
    # @test isapprox(ame.effects[:x][(false, zv)], ame_closed; atol=1e-6)

    # instead use estimated coefficients
    β̂ = coef(m)
    ame_closed_est = β̂[iβx] + β̂[iβxz] * zv
    @test isapprox(ame.effects[:x][(false, zv)], ame_closed_est; atol=1e-6)

    @test isapprox(ame.ses[:x][(false, zv)], se_closed; atol=1e-6)
end
