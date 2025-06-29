# lm_tests.jl
# tests for linear models

# Load data
iris = dataset("datasets", "iris") |> DataFrame
iris.Species = categorical(iris.Species)

@testset "Scenario 1: No interactions - continuous predictors" begin
    form = @formula(SepalLength ~ SepalWidth + PetalLength + PetalWidth)
    m = lm(form, iris)
    coefs = coef(m)
    ses = sqrt.(diag(vcov(m)))
    names = coefnames(m)
    idx_sw = findfirst(isequal("SepalWidth"), names)
    idx_pl = findfirst(isequal("PetalLength"), names)
    idx_pw = findfirst(isequal("PetalWidth"), names)

    ame_sw = margins(m, :SepalWidth, iris)
    ame_pl = margins(m, :PetalLength, iris)
    ame_pw = margins(m, :PetalWidth, iris)
    ame_all = margins(m, [:SepalWidth, :PetalLength, :PetalWidth], iris)

    @test isapprox(ame_sw.effects[:SepalWidth], coefs[idx_sw]; atol=1e-8)
    @test isapprox(ame_sw.ses[:SepalWidth], ses[idx_sw]; atol=1e-8)
    @test isapprox(ame_pl.effects[:PetalLength], coefs[idx_pl]; atol=1e-8)
    @test isapprox(ame_pl.ses[:PetalLength], ses[idx_pl]; atol=1e-8)
    @test isapprox(ame_pw.effects[:PetalWidth], coefs[idx_pw]; atol=1e-8)
    @test isapprox(ame_pw.ses[:PetalWidth], ses[idx_pw]; atol=1e-8)
    @test all(isapprox.(ame_all.effects[:SepalWidth], coefs[idx_sw]; atol=1e-8))
end

@testset "Scenario 1a: Categorical predictor" begin
    form_cat = @formula(SepalLength ~ SepalWidth + PetalWidth + Species)
    m_cat = lm(form_cat, iris)

    ame_species = margins(m_cat, :Species, iris)
    levels_list = levels(iris.Species)
    expected_pairs = sort([(i, j) for i in levels_list for j in levels_list if i < j])
    @test sort(collect(keys(ame_species.effects[:Species]))) == expected_pairs
end

@testset "Scenario 2: Interaction" begin
    form2 = @formula(SepalLength ~ SepalWidth * PetalLength + PetalWidth)
    m2 = lm(form2, iris)
    names2 = coefnames(m2)
    coefs2 = coef(m2)
    vc2 = vcov(m2)
    i_sw = findfirst(isequal("SepalWidth"), names2)
    i_swpl = findfirst(isequal("SepalWidth & PetalLength"), names2)
    meanPL = mean(iris.PetalLength)

    ame2 = margins(m2, :SepalWidth, iris)
    ame_closed = coefs2[i_sw] + coefs2[i_swpl] * meanPL
    var_closed = vc2[i_sw,i_sw] + meanPL^2 * vc2[i_swpl,i_swpl] + 2*meanPL * vc2[i_sw,i_swpl]
    se_closed = sqrt(var_closed)

    @test isapprox(ame2.effects[:SepalWidth], ame_closed; atol=1e-8)
    @test isapprox(ame2.ses[:SepalWidth], se_closed; atol=1e-8)
end

@testset "Scenario 3: Transformation of predictor" begin
    form3 = @formula(SepalLength ~ log(SepalWidth) + PetalLength + PetalWidth)
    m3 = lm(form3, iris)
    coefs3 = coef(m3)
    vc3 = vcov(m3)
    β_log = coefs3[2]
    mean_inv_sw = mean(1 ./ iris.SepalWidth)

    ame3 = margins(m3, :SepalWidth, iris)
    ame_closed3 = β_log * mean_inv_sw
    se_closed3 = sqrt((mean_inv_sw)^2 * vc3[2,2])

    @test isapprox(ame3.effects[:SepalWidth], ame_closed3; atol=1e-8)
    @test isapprox(ame3.ses[:SepalWidth], se_closed3; atol=1e-8)
end

@testset "Scenario 4: Representative values of moderator" begin
    form4 = @formula(SepalLength ~ SepalWidth * PetalWidth + PetalLength)
    m4 = lm(form4, iris)
    quantiles_pw = quantile(iris.PetalWidth, [0.25, 0.75])
    repvals = Dict(:PetalWidth => quantiles_pw)
    ame4 = margins(m4, :SepalWidth, iris; repvals=repvals)
    names4 = coefnames(m4)
    coefs4 = coef(m4)
    vc4 = vcov(m4)
    i_sw = findfirst(isequal("SepalWidth"), names4)
    i_sw_pw = findfirst(isequal("SepalWidth & PetalWidth"), names4)

    for pw in quantiles_pw
        ame_val = ame4.effects[:SepalWidth][(pw,)]
        se_val = ame4.ses[:SepalWidth][(pw,)]
        closed = coefs4[i_sw] + coefs4[i_sw_pw] * pw
        var_closed = vc4[i_sw,i_sw] + pw^2 * vc4[i_sw_pw,i_sw_pw] + 2*pw * vc4[i_sw,i_sw_pw]
        se_closed = sqrt(var_closed)

        @test isapprox(ame_val, closed; atol=1e-8)
        @test isapprox(se_val, se_closed; atol=1e-8)
    end
end

@testset "Scenario 5: Interaction with categorical moderator" begin
    form5 = @formula(SepalLength ~ SepalWidth * Species)
    m5 = lm(form5, iris)
    levels_list = levels(iris.Species)
    repvals_cat = Dict(:Species => categorical(levels_list; levels=levels_list))
    ame5 = margins(m5, :SepalWidth, iris; repvals=repvals_cat)

    β5 = coef(m5)
    V5 = vcov(m5)
    names5 = coefnames(m5)

    function me_and_se(level)
        c = zeros(eltype(β5), length(β5))
        i_sw = findfirst(isequal("SepalWidth"), names5)
        c[i_sw] = 1
        if level != levels_list[1]
            iname = "SepalWidth & Species: $level"
            i_int = findfirst(isequal(iname), names5)
            c[i_int] = 1
        end
        eff = dot(c, β5)
        var = c' * V5 * c
        return eff, sqrt(var)
    end

    for lvl in levels_list
        eff, se = me_and_se(string(lvl))
        ame_val = ame5.effects[:SepalWidth][(lvl,)]
        se_val = ame5.ses[:SepalWidth][(lvl,)]

        @test isapprox(ame_val, eff; atol=1e-8)
        @test isapprox(se_val, se; atol=1e-8)
    end
end
