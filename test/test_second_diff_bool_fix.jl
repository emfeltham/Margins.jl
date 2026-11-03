# Second differences with a categorical Bool focal variable (CategoricalArray{Bool}) and the
# default `contrast=nothing`: the focal variable's one contrast ("true - false") is found
# without naming it, and the result equals the hand computation from the coefficients.
#
# (Rewritten in 3.0: the 2.x version used `scenarios=` on `region`, which was not a predictor
# of the model -- an `ArgumentError` since 3.0 -- so its second differences were structural zeros.)

using Test
using Margins
using DataFrames
using GLM
using Random
using CategoricalArrays

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
