# test_weights.jl - the user `weights=` keyword of population_margins
# Uniform weights reproduce the unweighted result exactly (estimates and SEs); in a linear model
# weights leave the AME (a coefficient) unchanged, in a logit they move it; `n` counts only rows
# with a positive weight. Weighted estimates and SEs against an independent oracle are in
# validation/test_feature_matrix_oracle.jl; invalid weights in core/test_model_weights.jl.

@testset "Observation Weights" begin
    @testset "Weighted vs Unweighted Results - Linear Model" begin
        # Linear model: marginal effects are coefficients (weights shouldn't affect estimates much)
        Random.seed!(08540)
        n = 200
        data = DataFrame(
            y = randn(n),
            x1 = randn(n),
            x2 = randn(n)
        )
        
        # Create weights correlated with x1
        data.weights_uniform = ones(n)
        data.weights_nonuniform = 1.0 .+ 0.3 .* data.x1
        data.weights_nonuniform = max.(data.weights_nonuniform, 0.1)  # Ensure positive
        
        model = lm(@formula(y ~ x1 + x2), data)
        
        # Compare results
        result_unweighted = population_margins(model, data; type=:effects, vars=[:x1, :x2])
        result_weighted = population_margins(model, data; type=:effects, vars=[:x1, :x2], 
                                           weights=data.weights_nonuniform)
        result_uniform_weights = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                                  weights=data.weights_uniform)
        
        # Extract DataFrames for comparison
        df_unwt = DataFrame(result_unweighted)
        df_wt = DataFrame(result_weighted)
        df_uniform = DataFrame(result_uniform_weights)
        
        # For linear models, estimates should be identical (coefficients don't change)
        # But standard errors might differ slightly due to weighted gradient computation
        @test all(abs.(df_wt.estimate .- df_unwt.estimate) .< 1e-10)
        
        # Uniform weights should exactly match unweighted
        @test all(abs.(df_uniform.estimate .- df_unwt.estimate) .< 1e-12)
        @test all(abs.(df_uniform.se .- df_unwt.se) .< 1e-12)
        
        # Check that metadata correctly indicates weighting
        @test result_unweighted.metadata[:weighted] == false
        @test result_weighted.metadata[:weighted] == true
        @test result_uniform_weights.metadata[:weighted] == true
    end
    
    @testset "Weighted vs Unweighted Results - GLM Model" begin
        # GLM model: marginal effects depend on data points (weights should affect estimates)
        Random.seed!(06515)
        n = 300
        data = DataFrame(
            x1 = randn(n),
            x2 = randn(n)
        )
        
        # Create binary outcome for logistic regression  
        data.y_binary = rand(n) .< (0.5 .+ 0.1 .* data.x1)
        
        # Create weights that favor certain observations
        data.weights_uniform = ones(n)
        data.weights_skewed = 1.0 .+ 0.5 .* data.x1  # Weights correlated with x1
        data.weights_skewed = max.(data.weights_skewed, 0.1)  # Ensure positive
        
        model = glm(@formula(y_binary ~ x1 + x2), data, Binomial(), LogitLink())
        
        # Compare results
        result_unweighted = population_margins(model, data; type=:effects, vars=[:x1, :x2])
        result_weighted = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                           weights=data.weights_skewed)
        result_uniform_weights = population_margins(model, data; type=:effects, vars=[:x1, :x2],
                                                  weights=data.weights_uniform)
        
        # Extract DataFrames for comparison
        df_unwt = DataFrame(result_unweighted)
        df_wt = DataFrame(result_weighted)
        df_uniform = DataFrame(result_uniform_weights)
        
        # For GLM, weighted results should differ from unweighted when weights are non-uniform
        @test any(abs.(df_wt.estimate .- df_unwt.estimate) .> 1e-6)
        @test any(abs.(df_wt.se .- df_unwt.se) .> 1e-6)
        
        # Uniform weights should exactly match unweighted
        @test all(abs.(df_uniform.estimate .- df_unwt.estimate) .< 1e-12)
        @test all(abs.(df_uniform.se .- df_unwt.se) .< 1e-12)
    end

    @testset "n counts only rows with a positive weight (ISSUES #3)" begin
        rng = MersenneTwister(303)
        n = 2000
        d = DataFrame(x = randn(rng, n), z = randn(rng, n),
                      g = categorical(rand(rng, ["a", "b", "c"], n)),
                      h = categorical(rand(rng, ["u", "v"], n)))
        d.y = d.x .+ 0.5 .* d.z .+ (d.g .== "b") .+ randn(rng, n)
        d.w = [i <= 500 ? 0.0 : 1.0 + mod(i, 3) for i in 1:n]   # 500 of 2000 weights are zero
        npos = count(>(0), d.w)
        m = lm(@formula(y ~ x + z + g + h), d)

        # No contexts: metadata and the `n` column, effects (continuous and categorical) and predictions
        ame = population_margins(m, d; vars=[:x, :g], weights=:w)
        @test ame.metadata[:n_obs] == npos
        @test all(==(npos), DataFrame(ame).n)
        aap = population_margins(m, d; type=:predictions, weights=:w)
        @test aap.metadata[:n_obs] == npos
        @test DataFrame(aap).n == [npos]
        # A vector gives the same counts as the column
        @test DataFrame(population_margins(m, d; vars=[:x], weights=d.w)).n == [npos]

        # Per group: rows of the group with a positive weight; scenarios keep every group's count
        for type in (:effects, :predictions)
            kw = type === :effects ? (; vars=[:x, :g]) : (;)
            r = DataFrame(population_margins(m, d; type, groups=:h, weights=:w, kw...))
            for lvl in ("u", "v")
                rows = string.(d.h) .== lvl
                @test all(==(count(>(0), d.w[rows])), r.n[string.(r.over_h) .== lvl])
            end
            rs = DataFrame(population_margins(m, d; type, scenarios=(z=[0.0, 1.0],), weights=:w, kw...))
            @test all(==(npos), rs.n)
        end
        # Binned groups count positive-weight rows too (a bin's zero-weight rows are not counted)
        rb = DataFrame(population_margins(m, d; vars=[:z], groups=(:x, 4), weights=:w))
        @test sum(rb.n) == npos

        # Unweighted and weights=:none are unchanged: every row
        @test population_margins(m, d; vars=[:x]).metadata[:n_obs] == n
        @test DataFrame(population_margins(m, d; vars=[:x], weights=:none)).n == [n]
        @test sum(DataFrame(population_margins(m, d; vars=[:x], groups=:h)).n) == n

        # Model weights (the default) are counted the same way
        mw = lm(@formula(y ~ x + z + g + h), d; wts=d.w)
        @test population_margins(mw, d; vars=[:x]).metadata[:n_obs] == npos
        @test DataFrame(population_margins(mw, d; type=:predictions)).n == [npos]
    end

end