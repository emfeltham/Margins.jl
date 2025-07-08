# mm_tests.jl

atol_ = 1e-6

# Load data for LMMs
sleep = dataset("lme4", "sleepstudy") |> DataFrame
sleep.Subject = categorical(sleep.Subject)

# Load data for GLMM logistic (cbpp)
cbpp = dataset("lme4", "cbpp") |> DataFrame
cbpp.Herd   = categorical(cbpp.Herd)
cbpp.Period = categorical(cbpp.Period)
cbpp.Prop   = cbpp.Incidence ./ cbpp.Size

# Prepare synthetic data for GLMM logistic
Random.seed!(2025)
n, G = 500, 10
group = rand(1:G, n)
u     = randn(G) .* 0.8
x     = randn(n)
η_s   = 0.2 .+ 1.5 .* x .+ u[group]
p_s   = 1 ./(1 .+ exp.(-η_s))
y_s   = rand.(Bernoulli.(p_s))
df2   = DataFrame(x=x, group=categorical(string.(group)), y=y_s)

@testset "AME MixedModels Tests" begin
    @testset "Scenario 11: LMM random intercept only" begin
        form11 = @formula(Reaction ~ Days + (1|Subject))
        m11 = fit(MixedModel, form11, sleep)
        ame11 = margins(m11, :Days, sleep)

        fe = fixef(m11)
        cn = coefnames(m11)
        i = findfirst(isequal("Days"), cn)
        vc = vcov(m11)
        ame_closed = fe[i]
        se_closed = sqrt(vc[i, i])

        @test isapprox(ame11.effects[:Days], ame_closed; atol=atol_)
        @test isapprox(ame11.ses[:Days], se_closed; atol=atol_)
    end

    @testset "Scenario 12: LMM random slope on Days" begin
        form12 = @formula(Reaction ~ Days + (Days|Subject))
        m12 = fit(MixedModel, form12, sleep)
        ame12 = margins(m12, :Days, sleep)

        fe = fixef(m12)
        cn = coefnames(m12)
        i = findfirst(isequal("Days"), cn)
        vc = vcov(m12)
        ame_closed = fe[i]
        se_closed = sqrt(vc[i, i])

        @test isapprox(ame12.effects[:Days], ame_closed; atol=atol_)
        @test isapprox(ame12.ses[:Days], se_closed; atol=atol_)
    end

    @testset "Scenario 13: LMM with transformation + random intercept" begin
        form13 = @formula(Reaction ~ log1p(Days) + (1|Subject))
        m13 = fit(MixedModel, form13, sleep)
        ame13 = margins(m13, :Days, sleep)

        fe = fixef(m13)
        cn = coefnames(m13)
        idx = findfirst(isequal("log1p(Days)"), cn)
        vc = vcov(m13)
        mean_inv = mean(1 ./(sleep.Days .+ 1))
        ame_closed = fe[idx] * mean_inv
        se_closed = sqrt(vc[idx, idx] * mean_inv^2)

        @test isapprox(ame13.effects[:Days], ame_closed; atol=atol_)
        @test isapprox(ame13.ses[:Days], se_closed; atol=atol_)
    end

    @testset "Scenario 14: GLMM logistic random intercept on cbpp" begin
        form14 = @formula(Prop ~ Period + (1|Herd))
        m14 = fit(GeneralizedLinearMixedModel, form14, cbpp,
                 Binomial(), LogitLink(), wts = cbpp.Size)
        ame14 = margins(m14, :Period, cbpp)

        fe = fixef(m14)
        V = vcov(m14)
        cn = coefnames(m14)
        i0 = findfirst(isequal("(Intercept)"), cn)
        i2 = findfirst(isequal("Period: 2"),   cn)
        η1 = fe[i0]
        η2 = fe[i0] + fe[i2]
        p1 = 1 / (1 + exp(-η1))
        p2 = 1 / (1 + exp(-η2))
        ame_closed = p2 - p1
        g = zeros(eltype(fe), length(fe))
        g[i0] = p2*(1-p2) - p1*(1-p1)
        g[i2] = p2*(1-p2)
        var_closed = g' * V * g
        se_closed = sqrt(var_closed)

        @test isapprox(ame14.effects[:Period][("1","2")], ame_closed; atol=atol_)
        @test isapprox(ame14.ses[:Period][("1","2")], se_closed; atol=atol_)
    end

    @testset "Scenario 15: GLMM logistic synthetic random intercept" begin
        form15 = @formula(y ~ x + (1|group))
        m15 = fit(GeneralizedLinearMixedModel, form15, df2, Bernoulli())
        ame15 = margins(m15, :x, df2)

        fe = fixef(m15)
        cn = coefnames(m15)
        i = findfirst(isequal("x"), cn)
        vc = vcov(m15)
        X = modelmatrix(m15)
        η = X * fe
        p = 1 ./(1 .+ exp.(-η))
        w = p .* (1 .- p)
        ame_closed = mean(fe[i] .* w)
        dw = w .* (1 .- 2 .* p)
        k = length(fe)
        g = zeros(eltype(fe), k)
        for j in 1:k
            term1 = (j == i ? w : zero(w))
            term2 = fe[i] .* (dw .* X[:, j])
            g[j] = mean(term1 .+ term2)
        end
        var_closed = g' * vc * g
        se_closed = sqrt(var_closed)

        @test isapprox(ame15.effects[:x], ame_closed; atol=atol_)
        @test isapprox(ame15.ses[:x], se_closed; atol=atol_)
    end
end
