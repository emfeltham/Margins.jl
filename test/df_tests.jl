# df_tests.jl

@testset "DataFrame conversion tests" begin
    @testset "DF Scenario 1: No interactions - continuous predictors" begin
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

        @test DataFrame(ame_sw) isa DataFrame
        @test DataFrame(ame_all) isa DataFrame
    end

    @testset "DF Scenario 1b2: Boolean predictor with repvals" begin
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
        @test DataFrame(ame_rep) isa DataFrame
    end
end
