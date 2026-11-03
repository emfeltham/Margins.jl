# sd_categorical_focal.jl
# Second differences of a categorical focal variable: baseline and pairwise contrasts on the local
# derivative, a categorical Bool focal variable whose one contrast is found automatically,
# selecting one contrast with contrast=, and variables that appear only inside an interaction
# (no main effect) as focal variables and modifiers, against hand computations from modelcols.
# The last block also checks the AME of such a variable (population_margins), which the second
# differences build on.

using Test
using Margins
using DataFrames, CategoricalArrays, GLM, StatsModels, Statistics, Random, LinearAlgebra

@testset "second_differences(model, data, x, m; at=…) - Categorical Focal Variables" begin
    n = 200
    # Create data with categorical focal variable and interaction
    Random.seed!(999)
    df_cat = DataFrame(
        religion = categorical(rand(["Catholic", "Protestant", "None"], n)),
        modifier = randn(n),
        other = randn(n)
    )

    # Create interaction: religion effects vary with modifier
    df_cat.y = 0.5 .+
               (df_cat.religion .== "Protestant") .* 0.3 .+
               (df_cat.religion .== "None") .* 0.2 .+
               0.1 .* df_cat.modifier .+
               (df_cat.religion .== "Protestant") .* df_cat.modifier .* 0.5 .+
               (df_cat.religion .== "None") .* df_cat.modifier .* 0.3 .+
               randn(n) .* 0.3

    model_cat = lm(@formula(y ~ religion * modifier + other), df_cat)
    Σ_cat = vcov(model_cat)

    @testset "Baseline and Pairwise Contrasts" begin
        # Default should use baseline contrasts
        sd = second_differences(model_cat, df_cat, :religion, :modifier; vcov=Σ_cat, modifier_type=:continuous, at=:mean)

        # Should have K-1 contrasts for K=3 levels, one row per contrast
        @test nrow(sd) == 2
        @test hasproperty(sd, :contrast)
        @test all(sd.variable .== :religion)

        # Contrasts should not be "dy/dx" (that's for continuous only)
        @test !("dy/dx" in sd.contrast)

        # Statistical properties
        @test all(isfinite.(sd.second_diff))
        @test all(sd.se .> 0)
        @test all(0 .<= sd.p_value .<= 1)

        # Explicit pairwise contrasts
        sd_pairwise = second_differences(model_cat, df_cat, :religion, :modifier; vcov=Σ_cat, modifier_type=:continuous,
                                            at=:mean, contrasts=:pairwise)

        # Should have K(K-1)/2 = 3 contrasts for K=3 levels
        @test nrow(sd_pairwise) == 3
        @test hasproperty(sd_pairwise, :contrast)
        @test all(sd_pairwise.variable .== :religion)

        # All derivatives should be finite
        @test all(isfinite.(sd_pairwise.second_diff))
        @test all(sd_pairwise.se .> 0)

        # contrasts=:baseline is the default
        sd_baseline = second_differences(model_cat, df_cat, :religion, :modifier; vcov=Σ_cat, modifier_type=:continuous,
                                           at=:mean, contrasts=:baseline)
        @test nrow(sd_baseline) == 2  # K-1
        @test sd_baseline.contrast == sd.contrast

        # Some contrasts should overlap (baseline contrasts appear in pairwise)
        @test length(intersect(Set(sd_baseline.contrast), Set(sd_pairwise.contrast))) >= 1
    end

    @testset "Categorical with Vector Profile" begin
        # Add binary variable for stratification
        df_cat.treated = rand([false, true], n)
        model_cat2 = lm(@formula(y ~ religion * modifier + treated + other), df_cat)

        sd = second_differences(
            model_cat2, df_cat, :religion, :modifier; vcov=vcov(model_cat2), modifier_type=:continuous,
            at=:mean,
            profile=(treated=[false, true],))

        # Should have rows for each contrast × treated level
        @test hasproperty(sd, :treated)
        n_contrasts = length(unique(sd.contrast))
        @test nrow(sd) == n_contrasts * 2  # contrasts × 2 treated levels

        # Check stratification worked
        @test Set(sd.treated) == Set([false, true])
    end

    @testset "Mixed: Continuous and Categorical Focal Variables" begin
        # Should handle both types in one call
        sd = second_differences(
            model_cat, df_cat, [:religion, :other], :modifier; vcov=Σ_cat, modifier_type=:continuous,
            at=:mean)

        # Other is continuous (1 row), religion is categorical (multiple rows)
        religion_rows = sd[sd.variable .== :religion, :]
        other_rows = sd[sd.variable .== :other, :]

        @test nrow(religion_rows) >= 2  # Multiple contrasts
        @test nrow(other_rows) == 1     # Single derivative

        # Continuous variable should have "dy/dx" contrast
        @test other_rows.contrast[1] == "dy/dx"

        # Categorical should not
        @test !("dy/dx" in religion_rows.contrast)
    end
end

@testset "Second differences: categorical Bool focal variable, contrast found automatically" begin
    rng = MersenneTwister(12345)
    n = 400
    df = DataFrame(x = randn(rng, n), treated = rand(rng, Bool, n),
                   region = categorical(rand(rng, ["north", "south", "west"], n)))
    df.treated_cat = categorical(df.treated)
    df.y = 0.5 .+ 0.3 .* df.x .+ 0.4 .* df.treated .+ 0.2 .* df.x .* df.treated .+
           0.5 .* df.treated .* (df.region .== "south") .+ randn(rng, n) .* 0.4

    # The treatment effect differs by region through the treated_cat & region interaction
    model = lm(@formula(y ~ x * treated_cat + treated_cat * region), df)
    β = Dict(zip(coefnames(model), coef(model)))
    regions = ["north", "south", "west"]

    ames = population_margins(model, df; vars=[:treated_cat], scenarios=(region=regions,),
                              contrasts=:pairwise)
    contrasts = unique(DataFrame(ames).contrast)
    @test contrasts == ["true - false"]

    # contrast=nothing: the one contrast is used, without naming it
    sd = second_differences(ames, :treated_cat, :region)
    @test nrow(sd) == 3
    @test all(sd.contrast .== "true - false")
    # Linear model: the effect of treated in region r is β_treated + β_{x×treated}·mean(x)
    # + β_{treated×r}, so the difference between regions is the difference of the interaction
    # coefficients (zero for the reference region north)
    inter(r) = r == "north" ? 0.0 : β["treated_cat: true & region: $r"]
    for row in eachrow(sd)
        @test row.second_diff ≈ inter(string(row.modifier_level2)) - inter(string(row.modifier_level1)) atol = 1e-10
    end
    @test all(isfinite, sd.se) && all(>(0), sd.se)
    @test all(0 .<= sd.p_value .<= 1)

    # Naming the contrast gives the same table
    sd_explicit = second_differences(ames, :treated_cat, :region; contrast="true - false")
    @test sd_explicit.second_diff == sd.second_diff
    @test sd_explicit.se == sd.se
end

@testset "second differences: contrast= selects one contrast" begin
    Random.seed!(21)
    n = 600
    df = DataFrame(x = randn(n), w = randn(n), z = rand([false, true], n), t = rand([0, 1], n),
                   g = rand(["a", "b", "c"], n))
    df.y = 0.2 .+ 0.4df.x .+ 0.3df.w .+ 0.5df.z .+ 0.6df.x .* df.z .- 0.4df.x .* df.w .+ randn(n)

    @testset "contrast= selects one contrast (both forms); all_contrasts is gone" begin
        d = copy(df); d.cg = categorical(d.g)
        m = lm(@formula(y ~ x * z + cg * z + w), d)
        all_rows = second_differences(m, d, :cg, :z; warn = false)
        @test sort(unique(all_rows.contrast)) == ["b - a", "c - a"]
        one = second_differences(m, d, :cg, :z; contrast = "b - a", warn = false)
        @test one.contrast == ["b - a"]
        @test one.second_diff == all_rows[all_rows.contrast .== "b - a", :].second_diff
        # p_adjust family is the selected rows only
        adj = second_differences(m, d, :cg, :z; contrast = "b - a", p_adjust = :bonferroni, warn = false)
        @test adj.p_adjusted == adj.p_value
        # a continuous focal variable: "dy/dx" and its alias
        @test second_differences(m, d, :x, :z; contrast = "dy/dx").contrast == ["dy/dx"]
        @test second_differences(m, d, :x, :z; contrast = "derivative").contrast == ["dy/dx"]
        @test second_differences(m, d, :x, :w; contrast = "dy/dx", at = 0.0).contrast == ["dy/dx"]
        @test_throws ArgumentError second_differences(m, d, :cg, :z; contrast = "c - b", warn = false)  # not built under :baseline
        @test_throws ArgumentError second_differences(m, d, :x, :w; contrast = "b - a")
        @test second_differences(m, d, :cg, :z; contrast = "c - b", contrasts = :pairwise, warn = false).contrast == ["c - b"]
        # result-level form
        ames = population_margins(m, d; vars = :cg, scenarios = (z = [false, true],), type = :effects)
        r = second_differences(ames, :cg, :z; warn = false)
        @test nrow(r) == 2
        @test second_differences(ames, :cg, :z; contrast = "c - a", warn = false).contrast == ["c - a"]
        @test_throws ArgumentError second_differences(ames, :cg, :z; contrast = "dy/dx", warn = false)
        err = try second_differences(ames, :cg, :z; all_contrasts = false) catch e; sprint(showerror, e) end
        @test occursin("all_contrasts", err) && occursin("contrast=", err)
    end
end

# Hand computations for a categorical variable: average over rows of the prediction with every row
# set to `lv` minus every row set to `base` (other variables as observed, and `set` applied), with
# the delta-method SE from the modelcols gradients
function _hand_contrast(m, d, var, lv, base; set = (;), link = :identity)
    f = formula(m).rhs
    β = coef(m)
    function X(level)
        dd = copy(d)
        dd[!, var] = categorical(fill(level, nrow(dd)); levels = levels(d[!, var]))
        for (k, v) in pairs(set)
            dd[!, k] = fill(v, nrow(dd))
        end
        modelcols(f, dd)
    end
    Xl, Xb = X(lv), X(base)
    if link === :identity
        return mean(Xl * β .- Xb * β), vec(mean(Xl .- Xb; dims = 1))
    end
    μ(η) = 1 / (1 + exp(-η))
    dμ(η) = μ(η) * (1 - μ(η))
    el, eb = Xl * β, Xb * β
    return mean(μ.(el) .- μ.(eb)), vec(mean(dμ.(el) .* Xl .- dμ.(eb) .* Xb; dims = 1))
end

@testset "a variable that appears only inside an interaction" begin
    Random.seed!(4242)
    n = 600
    d = DataFrame(x = randn(n), w = 1 .+ rand(n), t = rand(0:1, n), b = rand(Bool, n),
                  g = categorical(rand(["a", "b", "c"], n)))
    d.y = d.x .+ 0.5 .* (d.g .== "b") .* d.w .- 0.4 .* (d.g .== "c") .* d.x .* d.w .+ 0.3 .* d.t .+ randn(n)
    ηb = 0.2 .+ d.x .+ 0.8 .* (d.g .== "b") .* d.w .- 0.5 .* (d.g .== "c") .* d.t .* d.w
    d.yb = rand(n) .< 1 ./ (1 .+ exp.(-ηb))

    @testset "AME of g (no main effect) against the hand computation: $(fm)" for (fm, link) in (
            (@formula(y ~ x + g & w), :identity),
            (@formula(y ~ x + g & x & w), :identity),
            (@formula(y ~ x + g & log(w)), :identity),
            (@formula(yb ~ x + g & w + g & t & w), :logit))
        m = link === :identity ? lm(fm, d) : glm(fm, d, Binomial(), LogitLink())
        r = DataFrame(population_margins(m, d; vars = [:g]))
        @test r.contrast == ["b - a", "c - a"]          # full-rank coding: the first level is the reference
        for row in eachrow(r)
            est, grad = _hand_contrast(m, d, :g, split(row.contrast, " - ")[1], "a"; link)
            @test row.estimate ≈ est atol = 1e-12
            @test row.se ≈ sqrt(dot(grad, vcov(m), grad)) atol = 1e-12
        end
        # auto-detected variables include it too
        @test "g" in DataFrame(population_margins(m, d)).variable
        # pairwise contrasts
        @test nrow(DataFrame(population_margins(m, d; vars = [:g], contrasts = :pairwise))) == 3
        # profile_margins (row-specific baseline at a grid row)
        # (a grid row that sets g gives the row-specific contrast of that level against the reference)
        @test DataFrame(profile_margins(m, d, cartesian_grid(g = ["b"]); type = :effects, vars = [:g])).contrast == ["b - a"]
    end

    @testset "full-rank main effect (y ~ 0 + g + x): contrasts are coefficient differences" begin
        m0 = lm(@formula(y ~ 0 + g + x), d)
        r = DataFrame(population_margins(m0, d; vars = [:g]))
        cn = coefnames(m0)
        @test r.estimate[1] ≈ coef(m0)[findfirst(==("g: b"), cn)] - coef(m0)[findfirst(==("g: a"), cn)] atol = 1e-12
    end

    m = glm(@formula(yb ~ x + g & w + g & t & w), d, Binomial(), LogitLink())
    @testset "as a focal variable of second differences" begin
        sd = second_differences(m, d, :g, :t)
        @test sd.contrast == ["b - a", "c - a"]
        for row in eachrow(sd)
            lv = split(row.contrast, " - ")[1]
            e0, g0 = _hand_contrast(m, d, :g, lv, "a"; set = (t = 0,), link = :logit)
            e1, g1 = _hand_contrast(m, d, :g, lv, "a"; set = (t = 1,), link = :logit)
            @test row.second_diff ≈ e1 - e0 atol = 1e-12
            @test row.se ≈ sqrt(dot(g1 .- g0, vcov(m), g1 .- g0)) atol = 1e-12
        end
        @test nrow(second_differences(m, d, :g, :w)) == 2          # continuous modifier, at=:observed
        @test nrow(second_differences(m, d, :g, :t; groups = :b)) == 4
    end

    @testset "as a modifier" begin
        sd = second_differences(m, d, :x, :g; warn = false)
        @test nrow(sd) == 3                                         # pairs of a, b, c
        r = population_margins(m, d; vars = [:x], scenarios = (g = ["a", "b", "c"],))
        cols = collect(Margins.SECOND_DIFFERENCE_COLUMNS)
        @test isequal(sd[!, cols], second_differences(r, :x, :g; warn = false)[!, cols])
        sd = second_differences(m, d, :w, :g; warn = false)
        @test nrow(sd) == 3 && all(isfinite, sd.se)
    end

    @testset "Bool and continuous variables only inside an interaction" begin
        mb = glm(@formula(yb ~ x + b & w + x & w), d, Binomial(), LogitLink())
        rb = DataFrame(population_margins(mb, d; vars = [:b, :w]))
        @test Set(rb.variable) == Set(["b", "w"])
        # Bool: the true - false contrast, by hand
        f = formula(mb).rhs
        Xt = modelcols(f, transform(d, :b => ByRow(_ -> true) => :b))
        Xf = modelcols(f, transform(d, :b => ByRow(_ -> false) => :b))
        μ(η) = 1 / (1 + exp(-η)); dμ(η) = μ(η) * (1 - μ(η))
        β = coef(mb)
        est = mean(μ.(Xt * β) .- μ.(Xf * β))
        grad = vec(mean(dμ.(Xt * β) .* Xt .- dμ.(Xf * β) .* Xf; dims = 1))
        rowb = rb[rb.variable .== "b", :]
        @test rowb.estimate[1] ≈ est atol = 1e-12
        @test rowb.se[1] ≈ sqrt(dot(grad, vcov(mb), grad)) atol = 1e-12
        # second differences: Bool focal and Bool modifier; continuous w only in interactions
        @test nrow(second_differences(mb, d, :b, :x; warn = false)) == 1
        @test nrow(second_differences(mb, d, :x, :b)) == 1
        @test nrow(second_differences(mb, d, :w, :x)) == 1
    end
end
