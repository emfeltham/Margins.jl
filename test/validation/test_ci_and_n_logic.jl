# test_ci_and_n_logic.jl - CI columns of the :confidence table and the :n / :N counts
#
# The t interval of the default table (half-width, df, alpha = 0.05) is checked in
# core/test_output_schema.jl and core/test_typed_results.jl, the column names and order of
# every table format in core/test_public_api.jl, and the counts with weights in
# core/test_weights.jl. What is left here: the :confidence table at a non-default alpha, the
# per-group counts without weights, the Stata table's N and the profile results' n.

using Random, DataFrames, CategoricalArrays, GLM, Statistics, Distributions

@testset "CI Columns and Sample Size Logic" begin
    Random.seed!(1234)
    n = 100
    df = DataFrame(
        x = randn(n),
        z = randn(n),
        group1 = categorical(rand(["A", "B"], n)),
        treatment = categorical(rand(["control", "treated"], n))
    )
    df.y = randn(n) .+ 0.5 * df.x .+ 0.3 * df.z
    m = lm(@formula(y ~ x + z + treatment), df)

    @testset "Confidence table at alpha = 0.10" begin
        res = population_margins(m, df; type=:effects, vars=[:x], alpha=0.10)
        df_result = DataFrame(res; format=:confidence)
        # the :confidence table has no se column; take it from the standard table
        se = DataFrame(res; format=:standard).se
        tcrit = quantile(TDist(dof_residual(m)), 0.95)
        @test df_result.ci_lower ≈ df_result.estimate .- tcrit .* se
        @test df_result.ci_upper ≈ df_result.estimate .+ tcrit .* se
    end

    @testset ":n per group (unweighted)" begin
        r = DataFrame(population_margins(m, df; type=:effects, vars=[:x], groups=:group1))
        @test r.n == [count(==(g), df.group1) for g in r.over_group1]
        p = DataFrame(population_margins(m, df; type=:predictions, groups=:treatment))
        @test p.n == [count(==(g), df.treatment) for g in p.over_treatment]
    end

    @testset "Stata table: N is the sample size" begin
        df_stata = DataFrame(population_margins(m, df; type=:effects, vars=[:x]); format=:stata)
        @test df_stata.N == [n]
    end

    @testset "Profile results: n is the full sample" begin
        res = profile_margins(m, df, means_grid(df); type=:effects, vars=[:x])
        @test DataFrame(res).n == [n]
        @test DataFrame(profile_margins(m, df, cartesian_grid(x=[-1.0, 1.0]); type=:predictions)).n == [n, n]
    end
end
