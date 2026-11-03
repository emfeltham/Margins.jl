# test_request_layer.jl
# The request-normalization layer (Margins 3.0, step 4): every entry point builds one
# MarginsRequest with `_normalize_request`, which runs every validator and evaluates `vcov`
# once. Covers the validation it adds (B8 `alpha`, B12 `contrasts`), Tables.jl input to the
# model-level `second_differences` (ISSUES #4), the `backend` pass-through, and the one
# evaluation of `vcov` per call.

using Test
using Margins
using DataFrames, CategoricalArrays, GLM, StatsModels, Tables
using Distributions: Normal, quantile

@testset "Request layer" begin
    n = 400
    i = 1:n
    fr(t) = t - floor(t)
    d = DataFrame(x = fr.(i .* 0.618) .* 4, w = fr.(i .* 0.414) .* 3, t = fr.(i .* 0.732) .< 0.5,
                  g = categorical(["a", "b", "c"][1 .+ mod.(i, 3)]), k = 1 .+ mod.(i .* 7, 4))
    d.y = 1.0 .+ 0.5 .* d.x .- 0.3 .* d.w .+ 0.4 .* d.t .* d.x .+ 0.2 .* (d.g .== "b") .+
          0.1 .* d.x .* d.w .+ fr.(i .* 0.91)
    d.yb = d.y .> 2.2
    m = lm(@formula(y ~ x * w + t * x + g), d)
    lg = glm(@formula(yb ~ x * w + t * x + g), d, Binomial(), LogitLink())

    @testset "MarginsRequest" begin
        req = Margins._normalize_request(m, d; vars=[:x], scenarios=(w=[1.0, 2.0],), weights=:model)
        @test req isa Margins.MarginsRequest{:population}
        @test isconcretetype(typeof(req))
        @test req.vars == [:x] && req.vars_explicit
        @test req.Σ == vcov(m) && req.Σ !== vcov(m)
        @test req.df == dof_residual(m)        # linear model, model-based vcov: t
        @test req.alpha === 0.05
        @test req.weights === nothing && req.weights_source === :none
        @test req.scenarios == (w=[1.0, 2.0],) && length(req.scenario_specs) == 2
        @test req.changed_vars == [:w]
        @test eltype(req.data_nt.k) === Float64  # prepared data
        # Pair syntax is normalized to a NamedTuple
        @test Margins._normalize_request(m, d; vars=[:x], scenarios=(:w => 1.0,)).scenarios == (w=1.0,)

        preq = Margins._normalize_request(m, d; vars=[:w], reference_grid=cartesian_grid(x=[0.0, 1.0]))
        @test preq isa Margins.MarginsRequest{:profile}
        @test preq.grid_vars == Set([:x])
        @test :g in propertynames(preq.completed_grid)   # completed with typical values
        @test preq.changed_vars == [:x]

        # second differences record the modifier and the profile variables they set
        sreq = Margins._local_derivative_request(m, Tables.columntable(d), [:x], :w, :observed, (t=true,);
                                                 scale=:response, backend=:ad, measure=:effect,
                                                 contrasts=:baseline, vcov=GLM.vcov, weights=:model, alpha=0.05)
        @test sreq.changed_vars == [:t, :w]
        @test isnothing(sreq.scenarios)
    end

    @testset "alpha: any Real in (0, 1) (B8)" begin
        r = population_margins(m, d; vars=[:x], alpha=0.1)
        for a in (1//10, 0.1f0)
            ra = population_margins(m, d; vars=[:x], alpha=a)
            @test ra.inference.alpha === Float64(a)
            @test DataFrame(ra).estimate == DataFrame(r).estimate
        end
        @test DataFrame(population_margins(m, d; vars=[:x], alpha=1//10)).ci_lower == DataFrame(r).ci_lower
        # the interval is estimate ± the critical value at that alpha × SE (z for a logit)
        rl = DataFrame(population_margins(lg, d; vars=[:x], alpha=0.1))
        @test rl.ci_upper ≈ rl.estimate .+ quantile(Normal(), 0.95) .* rl.se rtol=1e-12
        @test profile_margins(m, d, means_grid(d); vars=[:x], alpha=1//10).inference.alpha == 0.1
        for bad in (1.5, 1, 0, 0.0, -0.1, NaN)
            @test_throws ArgumentError population_margins(m, d; vars=[:x], alpha=bad)
            @test_throws ArgumentError profile_margins(m, d, means_grid(d); vars=[:x], alpha=bad)
        end
        @test_throws ArgumentError population_margins(m, d; vars=[:x], scenarios=(w=[1.0],), alpha=1.5)
        @test_throws TypeError population_margins(m, d; vars=[:x], alpha="0.1")
        # the 2.x keyword `ci_alpha` points to the new name
        msg = try population_margins(m, d; vars=[:x], ci_alpha=0.1); "" catch e; sprint(showerror, e) end
        @test occursin("`ci_alpha` was renamed `alpha`", msg)
        @test_throws ArgumentError profile_margins(m, d, means_grid(d); vars=[:x], ci_alpha=0.1)
        # contrast() takes the same alpha
        pr = profile_margins(m, d, cartesian_grid(x=[0.0, 1.0]); type=:predictions)
        @test contrast(pr, 1, 2; alpha=1//10).ci_lower == contrast(pr, 1, 2; alpha=0.1).ci_lower
        @test contrast(pr, 1, 2; alpha=0.1).metadata[:alpha] == 0.1
        @test_throws ArgumentError contrast(pr, 1, 2; alpha=1.5)
        @test_throws ArgumentError contrast(pr, (x=0.0,), (x=1.0,); alpha=0.0)
    end

    @testset "contrasts validated without categorical variables (B12)" begin
        @test_throws ArgumentError population_margins(m, d; vars=[:x], contrasts=:foo)
        @test_throws ArgumentError population_margins(m, d; type=:predictions, contrasts=:foo)
        @test_throws ArgumentError profile_margins(m, d, means_grid(d); vars=[:x], contrasts=:foo)
        @test_throws ArgumentError second_differences(m, d, :x, :t; contrasts=:foo)    # was silently accepted
        @test_throws ArgumentError second_differences(m, d, :x, :w; contrasts=:foo)    # was an ErrorException
        @test_throws ArgumentError second_differences(m, d, :x, :w; vcov=vcov(m), modifier_type=:continuous, contrasts=:foo)
        msg = try population_margins(m, d; vars=[:x], contrasts=:foo); "" catch e; sprint(showerror, e) end
        @test occursin("contrasts must be :baseline or :pairwise, got :foo", msg)
    end

    @testset "second_differences on any Tables.jl source (ISSUES #4)" begin
        nt = Tables.columntable(d)
        rt = Tables.rowtable(d)
        for (mod, kw) in ((:t, (;)), (:g, (;)), (:w, (;)), (:w, (at=[1.0, 2.0],)), (:w, (at=:mean,)),
                          (:w, (levels=[0.5, 2.5],)), (:w, (profile=(t=[false, true],),)))
            for model in (m, lg)
                ref = second_differences(model, d, :x, mod; warn=false, kw...)
                @test isequal(second_differences(model, nt, :x, mod; warn=false, kw...), ref)
                @test isequal(second_differences(model, rt, :x, mod; warn=false, kw...), ref)
            end
        end
        @test_throws ArgumentError second_differences(m, nt, :x, :nope)
        # profile_margins too
        pnt = profile_margins(m, nt, means_grid(d); vars=[:x])
        pdf = profile_margins(m, d, means_grid(d); vars=[:x])
        @test pnt.estimates == pdf.estimates && pnt.standard_errors == pdf.standard_errors
    end

    @testset "second_differences backend pass-through" begin
        ad = second_differences(lg, d, :x, :t)
        fd = second_differences(lg, d, :x, :t; backend=:fd)
        @test fd.second_diff ≈ ad.second_diff rtol=1e-6
        @test fd.se ≈ ad.se rtol=1e-6
        @test second_differences(lg, d, :x, :w; backend=:fd).second_diff ≈
              second_differences(lg, d, :x, :w).second_diff rtol=1e-6
        @test_throws ArgumentError second_differences(lg, d, :x, :t; backend=:nope)
    end

    @testset "vcov evaluated once per call" begin
        calls = Ref(0)
        counting(model) = (calls[] += 1; vcov(model))
        for f in (() -> population_margins(lg, d; vcov=counting),
                  () -> population_margins(lg, d; vars=[:x], scenarios=(w=[1.0, 2.0],), groups=:g, vcov=counting),
                  () -> profile_margins(lg, d, means_grid(d); vcov=counting),
                  () -> second_differences(lg, d, :x, :g; profile=(t=[false, true],), vcov=counting),
                  () -> second_differences(lg, d, :x, :w; at=[1.0, 2.0], profile=(t=[false, true],), vcov=counting),
                  () -> second_differences(lg, d, :x, :w; levels=[1.0, 2.0], vcov=counting),
                  () -> second_differences(lg, d, :x, :w; vcov=counting))
            calls[] = 0
            f()
            @test calls[] == 1
        end
    end

    @testset "validators run for the second-differences scenarios" begin
        # the modifier is set by scenarios: it must be a predictor, and not also a focal variable
        @test_throws ArgumentError second_differences(m, d, :x, :k; modifier_type=:categorical)
        @test_throws ArgumentError second_differences(m, d, :t, :t)
        # profile variables likewise
        @test_throws ArgumentError second_differences(m, d, :x, :t; profile=(k=1,))
        # a modifier the fitted model treats as categorical has no local derivative
        mk = lm(@formula(y ~ x * k), d; contrasts=Dict(:k => DummyCoding()))
        @test_throws ArgumentError second_differences(mk, d, :x, :k; modifier_type=:continuous, at=:observed)
        @test nrow(second_differences(mk, d, :x, :k; modifier_type=:categorical, warn=false)) == 6
    end
end
