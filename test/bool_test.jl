@testset "Scenario 1b: Boolean predictor" begin
    # 1) add a Bool column
    df_bool = copy(iris)
    df_bool.is_large = df_bool.SepalLength .> 5.0  # Bool

    # 2) fit a simple model with that Bool as the sole predictor
    form_bool = @formula(PetalWidth ~ is_large)
    m_bool   = lm(form_bool, df_bool)

    # 3) run margins()
    ame_bool = margins(m_bool, :is_large, df_bool)

    # 4) extract the “true vs false” coefficient and SE
    names  = coefnames(m_bool)
    coefs  = coef(m_bool)
    vc_mat = vcov(m_bool)
    idx    = findfirst(isequal("is_large"), names)

    # 5) margins should produce one pair: (false,true)
    keys_bool = sort(collect(keys(ame_bool.effects[:is_large])))
    @test keys_bool == [(false, true)]

    # 6) the AME is exactly the slope, and its SE matches the vcov diagonal
    pair = keys_bool[1]
    @test isapprox(ame_bool.effects[:is_large][pair], coefs[idx]; atol=1e-8)
    @test isapprox(ame_bool.ses[:is_large][pair], sqrt(vc_mat[idx,idx]); atol=1e-8)
end

@testset "Scenario 1c: Boolean predictor and not(bool) function" begin
    # 1) add a Bool column
    df_bool = copy(iris)
    df_bool.is_large = df_bool.SepalLength .> 5.0  # Bool

    # ------------------------------------------------------------------
    # 1b.1) raw Bool as factor → expect one (false,true) contrast
    # ------------------------------------------------------------------
    form_bool = @formula(PetalWidth ~ is_large)
    m_bool   = lm(form_bool, df_bool)

    ame_bool = margins(m_bool, :is_large, df_bool)

    names  = coefnames(m_bool)
    coefs  = coef(m_bool)
    vc_mat = vcov(m_bool)
    idx    = findfirst(isequal("is_large"), names)

    keys_bool = sort(collect(keys(ame_bool.effects[:is_large])))
    @test keys_bool == [(false, true)]

    pair = keys_bool[1]
    @test isapprox(ame_bool.effects[:is_large][pair], coefs[idx]; atol=1e-8)
    @test isapprox(ame_bool.ses[:is_large][pair], sqrt(vc_mat[idx,idx]); atol=1e-8)

    # ------------------------------------------------------------------
    # 1b.2) not(bool) inside the formula → margins on :is_large
    # ------------------------------------------------------------------
    # your `not(x::Bool)=Float64(!x)` must be in scope here
    form_not = @formula(PetalWidth ~ not(is_large))
    m_not   = lm(form_not, df_bool)

    # find the coefficient and SE for the not term
    names_not = coefnames(m_not)
    coefs_not = coef(m_not)
    vc_not    = vcov(m_not)
    idx_not   = findfirst(isequal("not(is_large)"), names_not)

    # run margins on the original :is_large
    ame_not = margins(m_not, :is_large, df_bool)

    # because not(x)=1−x, ∂Y/∂x = −∂Y/∂not(x), so AME = −β_not
    expected_ame = -coefs_not[idx_not]
    expected_se  = sqrt(vc_not[idx_not, idx_not])

    @test isapprox(ame_not.effects[:is_large][pair], expected_ame; atol=1e-8)
    @test isapprox(ame_not.ses[:is_large][pair],      expected_se;  atol=1e-8)
end

@testset "Scenario 1b2: Boolean predictor with repvals" begin
    # 1) augment iris with a Bool column
    df_bool = copy(iris)
    df_bool.is_large = df_bool.SepalLength .> 5.0  # Bool

    # 2) fit a model that includes that Bool plus a continuous covariate
    form = @formula(PetalWidth ~ is_large + SepalWidth)
    m    = lm(form, df_bool)

    # 3) choose two representative SepalWidth values
    quant_sw = quantile(df_bool.SepalWidth, [0.25, 0.75])
    repvals  = Dict(:SepalWidth => quant_sw)

    # 4) compute margins for the Bool, *with* repvals
    ame_rep = margins(m, :is_large, df_bool; repvals = repvals)

    # 5) the raw slope & SE for is_large
    names = coefnames(m)
    idx   = findfirst(isequal("is_large"), names)
    slope = coef(m)[idx]
    se_h  = sqrt(vcov(m)[idx, idx])

    # 6) we should get one AME per rep‐value, all equal to `slope` (no interaction)
    ky = sort(collect(keys(ame_rep.effects[:is_large])))
    @test ky == [(quant_sw[1],),(quant_sw[2],)]

    for q in quant_sw
        @test isapprox(ame_rep.effects[:is_large][(q,)], slope; atol=1e-8)
        @test isapprox(ame_rep.ses[:is_large][(q,)],      se_h;  atol=1e-8)
    end
end
