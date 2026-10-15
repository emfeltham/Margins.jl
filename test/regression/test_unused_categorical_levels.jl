# Regression test for: categorical predictions/effects misaligned with unused levels
#
# When a model is fit on a CategoricalArray whose pool contains an unused (zero-count) level
# and/or a non-alphabetical level ordering, both profile_margins() and population_margins()
# previously misaligned estimates with their stated levels:
#   - Predictions attributed each level's estimate to the wrong level (off-by-one shift when a
#     leading unused level was present) — root cause was in FormulaCompiler, which indexed the
#     contrast matrix by the pool ref code (`levelcode`) instead of the model's level ordering.
#   - Categorical effects generated a spurious contrast for the unused level and over-counted
#     relative to the results-DataFrame sizing, truncating (dropping) a real contrast.
#
# See the original GitHub issue MWE reproduced below.

using Test
using Margins, GLM, DataFrames, CategoricalArrays
using Random

@testset "Unused / reordered categorical levels" begin
    Random.seed!(934)
    n = 3000
    cat_vals = repeat(["A", "B", "C", "D", "E", "F"], inner = n ÷ 6)
    tp = Dict("A"=>0.5, "B"=>0.65, "C"=>0.5, "D"=>0.45, "E"=>0.5, "F"=>0.55)
    y = [rand() < tp[c] ? 1 : 0 for c in cat_vals]

    # Pool with a leading unused level AND non-alphabetical ordering, plus a custom base level.
    cat = categorical(cat_vals)
    levels!(cat, ["Unused", "A", "C", "F", "E", "D", "B"])
    df = DataFrame(cat = cat, y = y)
    model = glm(@formula(y ~ cat), df, Binomial(), LogitLink(),
                contrasts = Dict(:cat => DummyCoding(base = "C")))

    grid = DataFrame(cat = unique(string.(cat_vals)))
    # Ground truth from GLM's own predict()
    truth = Dict(string(r.cat) => r.pred
                 for r in eachrow(hcat(grid, DataFrame(pred = predict(model, grid)))))

    @testset "profile_margins predictions match predict()" begin
        m = DataFrame(profile_margins(model, df, grid; type = :predictions))
        for r in eachrow(m)
            @test isapprox(r.estimate, truth[string(r.cat)]; atol = 1e-8)
        end
        # The bug produced a duplicated estimate (D == B); ensure all six are distinct.
        @test length(unique(round.(m.estimate; digits = 10))) == 6
    end

    @testset "population_margins predictions (AAP) match predict()" begin
        pm = DataFrame(population_margins(model, df; type = :predictions, groups = :cat))
        for r in eachrow(pm)
            @test isapprox(r.estimate, truth[r.cat]; atol = 1e-8)
        end
    end

    @testset "population_margins baseline effects: correct set and values" begin
        eff = DataFrame(population_margins(model, df; type = :effects, vars = [:cat]))
        # Exactly 5 baseline contrasts (6 observed levels minus base C); no unused level.
        @test nrow(eff) == 5
        @test !any(occursin("Unused", c) for c in eff.contrast)
        for r in eachrow(eff)
            mobj = match(r"^(\w+) - (\w+)$", r.contrast)
            @test mobj !== nothing
            lvl, base = mobj.captures[1], mobj.captures[2]
            @test isapprox(r.estimate, truth[lvl] - truth[base]; atol = 1e-6)
        end
        # The real B - C contrast must be present (it was previously truncated away).
        @test any(==("B - C"), eff.contrast)
    end

    @testset "population_margins pairwise effects: no unused level, correct count" begin
        pw = DataFrame(population_margins(model, df; type = :effects,
                                          vars = [:cat], contrasts = :pairwise))
        @test nrow(pw) == 15                       # C(6, 2) over the 6 observed levels
        @test !any(occursin("Unused", c) for c in pw.contrast)
    end
end
