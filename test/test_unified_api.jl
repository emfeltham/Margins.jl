# A call with continuous, Bool and categorical variables together gives the same rows (estimate,
# SE, gradient) as one call per variable: the variables are computed independently and the
# joint call only stacks them. (The variable-type detection and AD/FD agreement this file used
# to check loosely are covered by core/test_automatic_variable_detection.jl and
# statistical_validation/backend_consistency.jl.)

using Test, Margins, GLM, DataFrames, CategoricalArrays, Random

@testset "Mixed variable types: joint call equals separate calls" begin
    rng = MersenneTwister(4242)
    n = 300
    df = DataFrame(x_continuous = randn(rng, n), x_boolean = rand(rng, Bool, n),
                   x_categorical = categorical(rand(rng, ["A", "B", "C"], n)))
    df.y = df.x_continuous .+ 0.5 .* df.x_boolean .+ (df.x_categorical .== "B") .+ randn(rng, n)
    model = lm(@formula(y ~ x_continuous * x_boolean * x_categorical), df)

    for contrasts in (:baseline, :pairwise), backend in (:ad, :fd)
        joint = DataFrame(population_margins(model, df; vars=[:x_continuous, :x_boolean, :x_categorical],
                                             contrasts, backend))
        parts = vcat([DataFrame(population_margins(model, df; vars=[v], contrasts, backend))
                      for v in (:x_continuous, :x_boolean, :x_categorical)]...)
        @test nrow(joint) == 1 + 1 + (contrasts === :baseline ? 2 : 3)
        @test joint.variable == parts.variable
        @test joint.contrast == parts.contrast
        @test joint.estimate ≈ parts.estimate rtol = 1e-12
        @test joint.se ≈ parts.se rtol = 1e-12
    end

    # The same within groups
    joint = DataFrame(population_margins(model, df; vars=[:x_continuous, :x_categorical], groups=:x_boolean))
    parts = [DataFrame(population_margins(model, df; vars=[v], groups=:x_boolean)) for v in (:x_continuous, :x_categorical)]
    for p in parts
        rows = joint[joint.variable .== p.variable[1], :]
        @test rows.estimate ≈ p.estimate rtol = 1e-12
        @test rows.se ≈ p.se rtol = 1e-12
    end
end
