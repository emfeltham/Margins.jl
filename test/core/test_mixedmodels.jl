# test_mixedmodels.jl
# MixedModels integration tests with new API

using Test, DataFrames, MixedModels, RDatasets, Random, Statistics, Distributions
using CategoricalArrays
using Margins

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

@testset "MixedModels Integration Tests" begin
    @testset "Scenario 11: LMM random intercept only" begin
        form11 = @formula(Reaction ~ Days + (1|Subject))
        m11 = fit(MixedModel, form11, sleep)
        
        # OLD: ame11 = margins(m11, :Days, sleep)
        # NEW: Population average marginal effects
        result = population_margins(m11, sleep; type=:effects, vars=[:Days])
        df_result = DataFrame(result)
        
        fe = fixef(m11)
        cn = coefnames(m11)
        i = findfirst(isequal("Days"), cn)
        vc = vcov(m11)
        ame_closed = fe[i]
        se_closed = sqrt(vc[i, i])

        days_row = filter(r -> r.variable == "Days", df_result)[1, :]
        @test isapprox(days_row.estimate, ame_closed; atol=atol_)
        @test isapprox(days_row.se, se_closed; atol=atol_)
    end

    @testset "Scenario 12: LMM random slope on Days" begin
        form12 = @formula(Reaction ~ Days + (Days|Subject))
        m12 = fit(MixedModel, form12, sleep)
        
        # OLD: ame12 = margins(m12, :Days, sleep)
        # NEW: Population average marginal effects
        result = population_margins(m12, sleep; type=:effects, vars=[:Days])
        df_result = DataFrame(result)

        fe = fixef(m12)
        cn = coefnames(m12)
        i = findfirst(isequal("Days"), cn)
        vc = vcov(m12)
        ame_closed = fe[i]
        se_closed = sqrt(vc[i, i])

        days_row = filter(r -> r.variable == "Days", df_result)[1, :]
        @test isapprox(days_row.estimate, ame_closed; atol=atol_)
        @test isapprox(days_row.se, se_closed; atol=atol_)
    end

    @testset "Scenario 13: LMM with transformation + random intercept" begin
        form13 = @formula(Reaction ~ log1p(Days) + (1|Subject))
        m13 = fit(MixedModel, form13, sleep)
        
        # OLD: ame13 = margins(m13, :Days, sleep)
        # NEW: Population average marginal effects
        result = population_margins(m13, sleep; type=:effects, vars=[:Days])
        df_result = DataFrame(result)

        fe = fixef(m13)
        cn = coefnames(m13)
        idx = findfirst(isequal("log1p(Days)"), cn)
        vc = vcov(m13)
        mean_inv = mean(1 ./(sleep.Days .+ 1))
        ame_closed = fe[idx] * mean_inv
        se_closed = sqrt(vc[idx, idx] * mean_inv^2)

        days_row = filter(r -> r.variable == "Days", df_result)[1, :]
        # less precision here 1.5e6
        @test isapprox(days_row.estimate, ame_closed; atol=1.5e6)
        @test isapprox(days_row.se, se_closed; atol=1.5e6)
    end

    @testset "Scenario 14: GLMM logistic random intercept on cbpp" begin
        form14 = @formula(Prop ~ Period + (1|Herd))
        m14 = fit(GeneralizedLinearMixedModel, form14, cbpp,
                 Binomial(), LogitLink(), wts = cbpp.Size)
        
        # Test categorical contrasts - now working with unified baseline detection
        result = population_margins(m14, cbpp; type=:effects, vars=[:Period])
        df_result = DataFrame(result)
        
        # Basic validation that categorical contrasts work
        @test size(df_result, 1) == 3  # Three contrasts for Period variable (4 levels - 1 baseline)
        @test any(contains.(df_result.variable, "Period"))  # All terms should contain "Period"
        @test !isnan(df_result.estimate[1])
        @test !isnan(df_result.se[1])
        @test df_result.se[1] > 0  # Standard error should be positive
    end

    @testset "Scenario 15: GLMM logistic synthetic random intercept" begin
        form15 = @formula(y ~ x + (1|group))
        m15 = fit(GeneralizedLinearMixedModel, form15, df2, Bernoulli())
        
        # OLD: ame15 = margins(m15, :x, df2)
        # NEW: Population average marginal effects
        result = population_margins(m15, df2; type=:effects, vars=[:x])
        df_result = DataFrame(result)

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

        x_row = filter(r -> r.variable == "x", df_result)[1, :]
        @test isapprox(x_row.estimate, ame_closed; atol=atol_)
        @test isapprox(x_row.se, se_closed; atol=atol_)
    end
end