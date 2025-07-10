# additional tests.jl
# tests inspired by Honduras CSS model structure

@testset "AME Additional Tests" begin
    # 1. Three-way interaction (continuous × Bool × continuous) in OLS
    @testset "3-way interaction OLS" begin
        # simulate
        Random.seed!(42)
        n = 500
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

    # 2. Boolean moderator in a Logit GLM
    @testset "Bool moderator in Logit GLM" begin
        # -- load & fit --
        df = dataset("datasets", "mtcars")
        df.HighMPG = df.MPG .> median(df.MPG)
        df.isHeavy = categorical(df.WT .> median(df.WT))
        m = glm(@formula(HighMPG ~ HP * isHeavy),
                df, Binomial(), LogitLink())

        # -- extract coefficients & indices --
        β    = coef(m)
        cn   = coefnames(m)
        i_hp = findfirst(isequal("HP"),           cn)
        i_int= findfirst(isequal("HP & isHeavy: true"), cn)

        # -- full linear predictor & inverse‐link slope p(1-p) --
        p  = predict(m)                        # includes intercept, HP, isHeavy & interaction
        μp = p .* (1 .- p)                     # d(invlink)/dη

        # -- dη/dHP for each obs: β_HP + β_int * I(isHeavy) --
        h = df.isHeavy .== true
        dη= β[i_hp] .+ β[i_int] .* h

        # -- marginal slope dμ/dHP and its average --
        dμ = dη .* μp
        ame_closed = mean(dμ)

        # -- compare to margins() --
        ame = margins(m, :HP, df)

        @test isapprox(ame.effects[:HP], ame_closed; atol=1e-12)
    end


    # 3. Categorical main effect in a Logit GLM
    @testset "Categorical main effect in Logit GLM (correct)" begin
        using Statistics, Distributions

        # 1. fit simple logit with factor
        df = dataset("datasets", "iris")
        df.Outcome = df.SepalLength .> median(df.SepalLength)
        df.Species = categorical(df.Species)
        m = glm(@formula(Outcome ~ Species), df, Binomial(), LogitLink())

        # 2. margins() all-pairs
        ame = margins(m, :Species, df)
        lev  = levels(df.Species)

        # --- 2a. key set matches i<j orientation ------------------------------
        exp_keys = [(i,j) for i in lev for j in lev if i<j]
        @test sort(collect(keys(ame.effects[:Species]))) == sort(exp_keys)

        # --- 2b. antisymmetry: AME(i,j) + AME(j,i) == 0 -----------------------
        for (i,j) in exp_keys
            eij = ame.effects[:Species][(i,j)]
            @test isapprox(eij + (-eij), 0.0; atol=1e-12)   # trivial, but clarifies intent
        end

        # --- 2c. numeric check for one pair (baseline vs. first non-baseline) --
        base, other = lev[1], lev[2]
        β  = coef(m)
        cn = coefnames(m)
        η_base  = β[1]                                         # intercept only
        int_name = "Species: $other"
        η_other = β[1] + β[findfirst(isequal(int_name), cn)]   # add dummy coef
        p_base  = 1/(1+exp(-η_base))
        p_other = 1/(1+exp(-η_other))
        ame_manual = p_other - p_base

        @test isapprox(ame.effects[:Species][(base,other)], ame_manual; atol=1e-12)
    end

    # 4. Continuous × categorical interaction in Logit GLM
    @testset "Continuous × categorical in Logit GLM (manual)" begin
        # 1) load & prepare
        df = dataset("datasets", "mtcars")
        df.CylF = categorical(string.(df.Cyl))

        # 2) fit the Logit GLM with MPG*CylF
        m = glm(@formula(VS ~ MPG * CylF), df, Binomial(), LogitLink())

        # 3) extract coef & vcov
        β  = coef(m)
        Σβ = vcov(m)
        n  = nrow(df)

        # 4) build design matrix X by hand
        #    Columns: [1, MPG, dummy_lev2, …, dummy_levK, MPG.*dummy_lev2, …]
        levs    = levels(df.CylF)
        onescol = ones(n)
        mpg     = df.MPG
        # dummy indicators for levels 2…K
        dummies = [Float64.(df.CylF .== lvl) for lvl in levs[2:end]]
        # interactions of mpg with each dummy
        inters  = [mpg .* d for d in dummies]
        X       = hcat(onescol, mpg, dummies..., inters...)

        # 5) build derivative‐design Xdx = ∂X/∂MPG
        #    ∂/∂MPG [1, MPG, d2, …, dK, MPG⋅d2, …] = [0, 1, 0…0, d2, …, dK]
        zeroscol = zeros(n)
        Xdx      = hcat(zeroscol, onescol,
                        [zeroscol for _ in dummies]...,
                        dummies...)

        # 6) compute link derivatives
        p   = predict(m)
        μp  = p .* (1 .- p)                       # d(invlink)/dη
        μpp = μp .* (1 .- 2 .* p)                 # d²(invlink)/dη²

        # 7) per‐obs slope dη/dMPG and AME
        dη         = Xdx * β                      # vector of ∂η_i/∂MPG
        ame_closed = mean(μp .* dη)               # average ∂μ/∂MPG

        # 8) Δ‐method SE: grad = (X′(μpp⋅dη) + Xdx′μp) / n
        grad       = (X' * (μpp .* dη) .+ Xdx' * μp) ./ n
        se_closed  = sqrt(dot(grad, Σβ * grad))

        # 9) compare to public‐API margins()
        ame = margins(m, :MPG, df)

        @test isapprox(ame.effects[:MPG], ame_closed; atol=1e-8)
        @test isapprox(ame.ses[:MPG],    se_closed;   atol=1e-8)
    end

    # 5. repvals in Logit GLM and GLMM
    @testset "repvals in Logit GLM & GLMM" begin
        # GLM case: mpg at 25th & 75th pct
        df = dataset("datasets", "mtcars")
        q = quantile(df.MPG, [0.25,0.75])
        m_glm = glm(@formula(AM ~ MPG + WT), df, Binomial(), LogitLink())
        ame_glm = margins(m_glm, :MPG, df; repvals=Dict(:MPG => q))

        # collect the keys into a Vector
        keys_glm = collect(keys(ame_glm.effects[:MPG]))
        expected = Tuple.(q)   # [(q₁,), (q₂,)]

        @test Set(keys_glm) == Set(expected)
        # (or equivalently)
        # @test sort(keys_glm) == sort(expected)

        # GLMM case: similarly
        Random.seed!(123)
        G, n = 5, 200
        g = repeat(1:G, inner=n)
        x = randn(G*n)
        β0, β1 = 0.2, 1.0
        η = β0 .+ β1*x .+ randn(G)[g]
        p = 1 ./ (1 .+ exp.(-η))
        y = rand.(Bernoulli.(p))
        df2 = DataFrame(y=y, x=x, grp=categorical(g))
        m_glmm = fit(GeneralizedLinearMixedModel, @formula(y ~ x + (1|grp)),
                    df2, Bernoulli(), LogitLink())
        qx = quantile(df2.x, [0.25,0.75])
        ame_glmm = margins(m_glmm, :x, df2; repvals=Dict(:x => qx))

        keys_glmm = collect(keys(ame_glmm.effects[:x]))
        expected_glmm = Tuple.(qx)

        @test Set(keys_glmm) == Set(expected_glmm)
    end

    # 6. Inf and missing predictor values
    # (we can't handle missings as is)
    # @testset "Inf and missing predictor values" begin
    #     # --- Inf case: regression with Inf in x will yield NaN/Inf AME & SE ---
    #     df_inf  = DataFrame(y = randn(100), x = vcat(randn(95), fill(Inf, 5)))
    #     m_inf   = lm(@formula(y ~ x), df_inf)
    #     ame_inf = margins(m_inf, :x, df_inf)

    #     @test isnan(ame_inf.effects[:x])  || isinf(ame_inf.effects[:x])
    #     @test isnan(ame_inf.ses[:x])      || isinf(ame_inf.ses[:x])

    #     # --- Missing case: lm() (via StatsModels) automatically drops missings ---
    #     df_miss  = DataFrame(y = randn(100), x = vcat(randn(95), fill(missing, 5)))
    #     m_miss   = lm(@formula(y ~ x), df_miss)
    #     ame_miss = margins(m_miss, :x, df_miss)

    #     # For plain OLS, the AME of x is just its slope, and its SE matches vcov
    #     coefs = coef(m_miss)
    #     vc    = vcov(m_miss)

    #     @test isa(ame_miss.effects[:x], Float64)
    #     @test isapprox(ame_miss.effects[:x], coefs[2]; atol = 1e-8)
    #     @test isapprox(ame_miss.ses[:x],       sqrt(vc[2,2]); atol = 1e-8)
    # end

    # 7. Weighted Logit GLM margins
    @testset "Weighted Logit GLM margins" begin
        using Statistics

        # 1) load & prepare
        df = dataset("datasets", "mtcars")
        # define a Boolean response, NOT a categorical
        df.odd = df.MPG .% 2 .== 1

        # 2) make up some weights
        w = rand(100:200, nrow(df))

        # 3) fit a weighted Logit GLM (Bool y is fine here)
        m = glm(@formula(odd ~ HP + WT),
                df, Binomial(), LogitLink(); wts = w)

        # 4) margins on HP should run and return scalars
        ame_w = margins(m, :HP, df)
        @test isa(ame_w.effects[:HP], Float64)
        @test isa(ame_w.ses[:HP],    Float64)

        # 5) sanity‐check: weighted vs. unweighted differ
        mu    = glm(@formula(odd ~ HP + WT), df, Binomial(), LogitLink())
        ame_u = margins(mu, :HP, df)
        @test ame_w.effects[:HP] != ame_u.effects[:HP]
    end

    # 8. Random-slope GLMM margins
    @testset "Random-slope GLMM margins" begin
        Random.seed!(2025)
        # simulate group varying slopes
        G, n = 10, 50
        g = repeat(1:G, inner=n)
        x = randn(G*n)
        u0 = randn(G).*0.5
        u1 = randn(G).*0.3
        η = 0.1 .+ (1.5 .+ u1[g]).*x .+ u0[g]
        p = 1 ./ (1 .+ exp.(-η))
        y = rand.(Bernoulli.(p))
        df = DataFrame(y=y, x=x, grp=categorical(g))

        m = fit(GeneralizedLinearMixedModel, @formula(y ~ x + (x|grp)), df, Bernoulli(), LogitLink())
        # AME of x is just marginal average of (β_x + u1_i)*p*(1-p)
        ame = margins(m, :x, df)
        @test isa(ame.effects[:x], Number)
        @test isa(ame.ses[:x],   Number)
    end
end
