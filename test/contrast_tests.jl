# contrast_tests.jl

import LinearAlgebra.I
import Margins._compute_contrast

@testset "AME Contrast Tests" begin
    # simple Δ-method helper tests
    @testset "_compute_contrast" begin
        θ1, θ2 = 2.0, 1.0
        σ1, σ2 = 1.0, 1.5
        g1 = [1.0, 0.0]; g2 = [0.0, 1.0]
        vc = I(2)  # 2×2 identity
        ν = 10
        Δ, seΔ, tstat, pval = _compute_contrast(θ1, θ2, σ1, σ2, g1, g2, vc, ν)
        @test Δ == θ1 - θ2
        @test isapprox(seΔ, sqrt(σ1^2 + σ2^2); atol=1e-8)
        @test isapprox(tstat, Δ/seΔ; atol=1e-8)
        # p-value should be between 0 and 1
        @test 0.0 <= pval <= 1.0

        # test with vcov = nothing => uncorrelated
        Δ2, se2, t2, p2 = _compute_contrast(5.0, 3.0, 2.0, 2.0, g1, g2, nothing, ν)
        @test Δ2 == 2.0
        @test isapprox(se2, sqrt(2.0^2 + 2.0^2); atol=1e-8)
    end

    # contrast between two results: scalar case
    @testset "contrast between two scalar MarginsResults" begin
        # simulate two MarginsResult objects with one var :x
        # effect 1.0 vs 0.5, se both 0.2, gradient arbitrary
        grad = [1.0]
        res1 = MarginsResult{:dydx}([:x], Dict(), Dict(:x=>1.0), Dict(:x=>0.2), Dict(:x=>grad), 50, 48, "Normal", "identity")
        res2 = MarginsResult{:dydx}([:x], Dict(), Dict(:x=>0.5), Dict(:x=>0.2), Dict(:x=>grad), 50, 48, "Normal", "identity")
        cr = contrast(res1, res2; var=:x)
        @test cr.vars == [:x]
        @test cr.comps == [(:a,:b)]
        @test cr.estimate[1] == 0.5
        @test isapprox(cr.se[1], sqrt(0.2^2 + 0.2^2); atol=1e-8)
        DataFrame(cr) isa DataFrame
    end

    # contrast between two results: grid case
    @testset "contrast between two grid MarginsResults" begin
        # create repvals with two combos
        effects1 = Dict((1,)=>1.0, (2,)=>2.0)
        effects2 = Dict((1,)=>0.5, (2,)=>1.5)
        ses1     = Dict((1,)=>0.1, (2,)=>0.2)
        ses2     = Dict((1,)=>0.1, (2,)=>0.2)
        grad1    = Dict((1,)=>[1.0],    (2,)=>[1.0])
        grad2    = Dict((1,)=>[1.0],    (2,)=>[1.0])
        repvals  = Dict(:x => [1,2])

        # wrap each map under the predictor key :x
        res1 = MarginsResult{:dydx}(
            [:x], repvals,
            Dict(:x=>effects1),
            Dict(:x=>ses1),
            Dict(:x=>grad1),
            10, 8, "Normal", "identity"
        )
        res2 = MarginsResult{:dydx}(
            [:x], repvals,
            Dict(:x=>effects2),
            Dict(:x=>ses2),
            Dict(:x=>grad2),
            10, 8, "Normal", "identity"
        )

        cr_grid = contrast(res1, res2; var=:x)
        @test cr_grid.vars == [:x]
        @test sort(cr_grid.comps) == sort([(1,),(2,)])
        @test cr_grid.estimate == [0.5, 0.5]
        DataFrame(cr_grid) isa DataFrame
    end

    @testset "contrast within result (scalar)" begin
        # two vars x and y
        grad = [1.0]
        effects = Dict(:x=>2.0, :y=>1.0)
        ses     = Dict(:x=>0.1, :y=>0.2)
        gradm   = Dict(:x=>grad,   :y=>grad)
        res = MarginsResult{:dydx}([:x,:y], Dict(), effects, ses, gradm, 30, 28, "Normal", "identity")
        cr = contrast(res; var1=:x, var2=:y)
        @test cr.vars == [:x,:y]
        @test cr.comps == [(:x,:y)]
        @test isapprox(cr.estimate[1], 1.0; atol=1e-8)
        DataFrame(cr) isa DataFrame
    end

    @testset "contrast within result (grid)" begin
        repvals = Dict(:x => [1, 2])
        effects = Dict((1,) => 3.0, (2,) => 4.0)
        ses     = Dict((1,) => 0.1, (2,) => 0.2)
        gradm   = Dict((1,) => [1.0], (2,) => [1.0])

        # Wrap under :x so res.effects[:x] == effects
        res = MarginsResult{:dydx}(
            [:x], repvals,
            Dict(:x => effects),
            Dict(:x => ses),
            Dict(:x => gradm),
            20, 18,
            "Normal", "identity"
        )

        # Now contrast the two levels
        cr = contrast(res; var1 = :x, var2 = :x)
        @test cr.vars == [:x, :x]
        @test sort(cr.comps) == sort([(1,), (2,)])
        @test length(cr.estimate) == 2
        DataFrame(cr) isa DataFrame
    end

    @testset "contrast_levels" begin
        # repvals numeric
        repvals = Dict(:x => [10,20,30])
        effects = Dict((10,)=>5.0,  (20,)=>6.0,  (30,)=>7.0)
        ses     = Dict((10,)=>0.1,  (20,)=>0.1,  (30,)=>0.1)
        gradm   = Dict((10,)=>[1.0],(20,)=>[1.0],(30,)=>[1.0])

        # wrap under :x so res.effects[:x] == effects, etc.
        res = MarginsResult{:dydx}(
        [:x], repvals,
        Dict(:x => effects),
        Dict(:x => ses),
        Dict(:x => gradm),
        15, 13,
        "Normal", "identity"
        )

        cr_all = contrast_levels(res, :x; comparisons = :all)
        @test length(cr_all.comps) == 3    # pairs (10,20),(10,30),(20,30)
        @test DataFrame(cr_all) isa DataFrame

        cr_adj = contrast_levels(res, :x; comparisons = :adjacent)
        @test cr_adj.comps == [(10,20),(20,30)]
        @test DataFrame(cr_adj) isa DataFrame
    end
end
