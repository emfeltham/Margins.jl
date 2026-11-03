# test_categorical_mixtures.jl
# Margins' use of categorical mixtures: profile_margins on grids holding mix(...) values.
#
# The mixture API itself (mix, CategoricalMixture, validate_mixture_against_data,
# mixture_to_scenario_value) is FormulaCompiler's and is tested in its
# test/test_mixture_api.jl. A single mixture in a main-effect term (logit predictions, SEs and
# effects; two mixtures in one grid; a mixture after a plain level) is checked exactly in
# regression/test_mixture_cache.jl, and frequency mixtures at the means by
# validation/test_feature_matrix_oracle.jl. Here: mixtures inside an interaction term.

using Test
using Random
using Statistics
using DataFrames, GLM, StatsModels, CategoricalArrays
using Margins

@testset "Mixtures in an interaction term (profile_margins)" begin
    rng = MersenneTwister(06515)
    n = 100
    df = DataFrame(
        education = categorical(rand(rng, ["high_school", "college", "graduate"], n)),
        region = categorical(rand(rng, ["urban", "rural"], n)),
        age = rand(rng, 25:65, n),
        income = randn(rng, n) * 10000 .+ 50000,
        employed = rand(rng, Bool, n),
    )
    df.outcome = 10 .+ 0.001 .* (df.income .- 50000) .+ 0.02 .* (df.age .- 45) .+
                 0.5 .* (df.education .== "college") .* (df.region .== "urban") .+ 2 .* randn(rng, n)
    m = lm(@formula(outcome ~ education * region + age + income + employed), df)
    β, V, rhs = coef(m), vcov(m), formula(m).rhs

    # The design row at a mixture is linear in each factor's indicators, so with two mixtures
    # it is the share-weighted sum of the design rows at the level pairs (an interaction column
    # gets the product of the two shares)
    shares(c) = c isa CategoricalMixture ? collect(zip(c.levels, c.weights)) : [(c, 1.0)]
    level_row(e, r) = vec(modelcols(rhs, DataFrame(
        education = categorical([e]; levels = levels(df.education)),
        region = categorical([r]; levels = levels(df.region)),
        age = [40], income = [60000.0], employed = [true])))
    mixed_row(e, r) = sum(we * wr .* level_row(le, lr) for (le, we) in shares(e), (lr, wr) in shares(r))

    edu = mix("high_school" => 0.3, "college" => 0.5, "graduate" => 0.2)
    reg = mix("urban" => 0.6, "rural" => 0.4)
    for (e, r) in ((edu, reg), (edu, "rural"), ("college", reg))
        G = DataFrame(education = [e], region = [r], age = [40], income = [60000.0], employed = [true])
        x = mixed_row(e, r)
        p = DataFrame(profile_margins(m, df, G; type = :predictions))
        @test nrow(p) == 1
        @test p.estimate[1] ≈ x' * β rtol = 1e-12
        @test p.se[1] ≈ sqrt(x' * V * x) rtol = 1e-10

        # identity link: each effect is its coefficient, at any mixture
        ef = DataFrame(profile_margins(m, df, G; type = :effects, vars = [:age, :income, :employed]))
        cn = coefnames(m)
        @test ef.variable == ["age", "income", "employed"]
        @test ef.estimate ≈ β[[findfirst(==(c), cn) for c in ("age", "income", "employed")]] rtol = 1e-8
    end
end
