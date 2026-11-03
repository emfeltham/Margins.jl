# test_derived_columns.jl
# One counterfactual mechanism (SetTo / Shift, `_apply_counterfactuals`) and the derived-column
# detector (Margins 3.0, step 7; ISSUES #2).

using Test
using Margins
using DataFrames, GLM, CategoricalArrays, Statistics, Random, Logging, Tables

const DERIVED_WARN = r"look like functions of a variable this call changes"

# Every warning a call logs (the detector logs one per call)
function _warnings(f)
    logger = Test.TestLogger(min_level=Logging.Warn)
    with_logger(f, logger)
    return [string(r.message) for r in logger.logs if r.level == Logging.Warn]
end
_detector_warnings(f) = filter(m -> occursin(DERIVED_WARN, m), _warnings(f))

@testset "counterfactual mechanism" begin
    nt = (x = [1.0, 2.0, 3.0], b = [true, false, true],
          g = categorical(["a", "b", "a"]), k = [1.5, 2.5, 3.5])

    @testset "SetTo, Shift, plain values" begin
        out = Margins._apply_counterfactuals(nt, Dict(:x => Margins.SetTo(5), :b => "false"))
        @test out.x == [5.0, 5.0, 5.0] && eltype(out.x) == Float64
        @test out.b == [false, false, false] && eltype(out.b) == Bool
        @test out.g === nt.g && out.k === nt.k          # unchanged columns are shared
        @test keys(out) == keys(nt)                     # column order kept
        @test nt.x == [1.0, 2.0, 3.0]                   # the data is not mutated

        sh = Margins._apply_counterfactuals(nt, (x = Margins.Shift(-0.5),))
        @test sh.x == [0.5, 1.5, 2.5] && sh.k === nt.k

        cg = Margins._apply_counterfactuals(nt, Dict(:g => "b"))
        @test all(==("b"), string.(cg.g)) && levels(cg.g) == levels(nt.g)   # pool kept

        @test Margins._apply_counterfactuals(nt, Dict()) === nt
        @test_throws ArgumentError Margins._apply_counterfactuals(nt, Dict(:nope => 1))
        @test_throws ArgumentError Margins._apply_counterfactuals(nt, Dict(:g => Margins.Shift(1.0)))
        @test_throws ArgumentError Margins._apply_counterfactuals(nt, Dict(:b => Margins.Shift(1.0)))
    end

    @testset "labels and grid completion" begin
        @test Margins._context_label(Margins.Shift(0.25)) == 0.25
        @test Margins._context_label(Margins.SetTo("a")) == "a"
        @test Margins._context_label((kind = :bin, label = "Q1")) == "Q1"
        @test Margins._counterfactual_column(Margins.SetTo(2), 3) == [2.0, 2.0, 2.0]
        @test eltype(Margins._counterfactual_column(Margins.SetTo(true), 2)) == Bool
    end

    @testset "a Shift scenario is the at=:observed computation" begin
        Random.seed!(3)
        n = 300
        df = DataFrame(x = randn(n), m = randn(n), z = randn(n))
        df.y = (0.2 .+ df.x .+ 0.5 .* df.x .* df.m .+ 0.3df.z .+ randn(n)) .> 0
        fit_ = glm(@formula(y ~ x * m + z), df, Binomial(), LogitLink())
        δ = 0.01
        r = population_margins(fit_, df; vars = [:x], scenarios = (m = [Margins.Shift(-δ), Margins.Shift(δ)],))
        @test r.context.columns.m == [-δ, δ]
        sd = second_differences(fit_, df, :x, :m; at = :observed, delta = δ, warn = false)
        @test sd.second_diff[1] == (r.estimates[2] - r.estimates[1]) / (2δ)
    end
end

@testset "derived-column detector" begin
    Random.seed!(20261027)
    n = 1000
    df = DataFrame(x = 0.5 .+ 3 .* rand(n), z = randn(n), u = randn(n), t = rand(Bool, n),
                   age = Float64.(rand(20:70, n)), year = Float64.(rand(2001:2010, n)),
                   g = categorical(rand(["a", "b", "c"], n)), c0 = fill(1.0, n))
    df.x2 = df.x .^ 2
    df.logx = log.(df.x)
    df.xs = (df.x .- mean(df.x)) ./ std(df.x)
    df.xz = df.x .* df.z
    df.zt = df.z .* df.t
    df.band = Float64.(df.age .>= 40) .+ Float64.(df.age .>= 60)
    gdp = Dict(y => randn() for y in 2001:2010)
    df.gdp = [gdp[Int(y)] for y in df.year]
    df.y = df.x .+ 0.3df.x2 .+ df.z .+ 0.2df.xz .+ 0.01df.age .+ 0.1df.band .+ 0.2df.gdp .+
           0.05df.year .+ 0.1df.u .+ randn(n)

    @testset "positive cases" begin
        m = lm(@formula(y ~ x + x2 + z), df)
        w = _detector_warnings(() -> population_margins(m, df))
        @test length(w) == 1 && occursin("`x2` is x^2", w[1]) && occursin("changing `x` holds `x2` fixed", w[1])
        # scenarios, profiles and the model-level second differences run it too
        @test length(_detector_warnings(() -> population_margins(m, df; vars = [:z], scenarios = (x = [1.0, 2.0],)))) == 1
        @test length(_detector_warnings(() -> profile_margins(m, df, means_grid(df); vars = [:z]))) == 1
        @test length(_detector_warnings(() -> profile_margins(m, df, cartesian_grid(x = [1.0, 2.0]); vars = [:z]))) == 1
        sdm = lm(@formula(y ~ z * x + x2), df)
        @test length(_detector_warnings(() -> second_differences(sdm, df, :z, :x))) == 1
        @test length(_detector_warnings(() -> second_differences(sdm, df, :z, :x; at = :mean))) == 1
        @test length(_detector_warnings(() -> second_differences(sdm, df, :z, :x; levels = [1.0, 2.0]))) == 1

        w = _detector_warnings(() -> population_margins(lm(@formula(y ~ x + logx), df), df))
        @test length(w) == 1 && occursin("`logx` is log(x)", w[1])

        w = _detector_warnings(() -> population_margins(lm(@formula(y ~ x + xs & z), df), df; vars = [:x]))
        @test length(w) == 1 && occursin("`xs` is a linear rescaling of x", w[1])

        # precomputed interaction, found from either factor
        mi = lm(@formula(y ~ x + z + xz), df)
        w = _detector_warnings(() -> population_margins(mi, df; vars = [:z]))
        @test length(w) == 1 && occursin("`xz` is z * x (a precomputed interaction)", w[1])
        w = _detector_warnings(() -> population_margins(lm(@formula(y ~ z + t + zt), df), df; vars = [:z], scenarios = (t = [false, true],)))
        @test length(w) == 1 && occursin("`zt`", w[1])

        # one warning per call, listing every pair
        mall = lm(@formula(y ~ x + x2 + logx + z), df)
        w = _detector_warnings(() -> population_margins(mall, df))
        @test length(w) == 1 && occursin("`x2`", w[1]) && occursin("`logx`", w[1])
    end

    @testset "negative cases" begin
        # transforms written in the formula
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ x + x^2 + log(x) + x & z), df), df)))
        @test isempty(_detector_warnings(() -> second_differences(lm(@formula(y ~ z * x + x^2), df), df, :z, :x)))
        # independent covariates, categorical predictors, a Bool
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ x + z + u + g + t), df), df)))
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ x + z + g), df), df; vars = [:x], scenarios = (g = ["a", "b"],))))
        # a near-unique variable with unrelated columns
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ x + u + z), df), df)))
        # a constant column (no information) and the response are never reported
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ x + z), df), df)))
        # the call sets the derived column consistently
        m = lm(@formula(y ~ x + x2 + z), df)
        @test isempty(_detector_warnings(() -> population_margins(m, df; vars = [:z], scenarios = (x = 2.0, x2 = 4.0))))
        @test isempty(_detector_warnings(() -> profile_margins(m, df, DataFrame(x = [1.0, 2.0], x2 = [1.0, 4.0]); vars = [:z])))
        @test length(_detector_warnings(() -> population_margins(m, df; vars = [:z], scenarios = (x = 2.0, x2 = 5.0)))) == 1
        # predictions with no counterfactual change nothing
        @test isempty(_detector_warnings(() -> population_margins(m, df; type = :predictions)))
        # no lookup rule: a column constant within each value of a discrete variable is not reported
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ age + band + z), df), df; vars = [:age])))
        small = DataFrame(v = repeat(1.0:10.0, 2), b = Float64.(repeat([0, 1], 10)), y = randn(20))
        small.b[11:20] .= small.b[1:10]
        @test isempty(_detector_warnings(() -> population_margins(lm(@formula(y ~ v + b), small), small; vars = [:v])))
    end

    # Many sources and columns at once (an AME or MEM over every variable): every rule, found in
    # the same order, and the same warning from the population and the profile path
    @testset "many sources, both paths" begin
        mm = lm(@formula(y ~ x + x2 + logx + xs & z + z + xz + u + t + zt + age + band + year + gdp + g), df)
        srcs = [:x, :x2, :logx, :xs, :z, :xz, :u, :t, :zt, :age, :band, :year, :gdp, :g, :c0]
        found = Margins._derived_columns(mm, Margins._prepare_margins_data(Tables.columntable(df), mm), srcs)
        @test [(d.source, d.derived, d.relation) for d in found] == [
            (:x, :logx, "log(x)"), (:x, :x2, "x^2"),
            (:x, :xs, "a linear rescaling of x (a standardized or rescaled copy)"),
            (:x, :xz, "x * z (a precomputed interaction)"), (:z, :xz, "z * x (a precomputed interaction)"),
            (:z, :zt, "z * t (a precomputed interaction)"), (:t, :zt, "t * z (a precomputed interaction)")]
        wp = _detector_warnings(() -> population_margins(mm, df))
        wg = _detector_warnings(() -> profile_margins(mm, df, means_grid(df)))
        @test length(wp) == 1 && wp == wg
        @test count("\n  - `", wp[1]) == 7
        # sources the call changes only through the grid
        wc = _detector_warnings(() -> profile_margins(mm, df, cartesian_grid(x = [1.0, 2.0], t = [false, true]); vars = [:u]))
        wc2 = _detector_warnings(() -> population_margins(mm, df; vars = [:u], scenarios = (x = [1.0, 2.0], t = [false, true])))
        @test length(wc) == 1 && wc == wc2
        @test occursin("`x2` is x^2", wc[1]) && occursin("`zt` is t * z", wc[1]) && !occursin("`xz` is z", wc[1])
    end

    @testset "a group-level contextual covariate is not reported" begin
        mc = lm(@formula(y ~ year + gdp + z), df)
        @test isempty(_detector_warnings(() -> population_margins(mc, df; vars = [:z], scenarios = (year = 2005,))))
        @test isempty(_detector_warnings(() -> population_margins(mc, df; vars = [:year])))
    end

    @testset "warn=false" begin
        m = lm(@formula(y ~ x + x2 + z), df)
        @test isempty(_detector_warnings(() -> population_margins(m, df; warn = false)))
        @test isempty(_detector_warnings(() -> profile_margins(m, df, means_grid(df); warn = false)))
        @test isempty(_detector_warnings(() -> second_differences(lm(@formula(y ~ z * x + x2), df), df, :z, :x; warn = false)))
    end

    @testset "results do not depend on the detector" begin
        m = lm(@formula(y ~ x + x2 + z), df)
        a = with_logger(NullLogger()) do
            population_margins(m, df)
        end
        b = population_margins(m, df; warn = false)
        @test a.estimates == b.estimates && a.standard_errors == b.standard_errors
    end
end
