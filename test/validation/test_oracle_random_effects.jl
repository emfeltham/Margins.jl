# test_oracle_random_effects.jl
# `random_effects` for mixed models against R (lme4 2.1.0, marginaleffects 1.0.0, numDeriv):
# fixtures in r_compare_marginaleffects/random_effects/ (see its README). R's data are read from
# the fixture, both languages fit the same models with tight tolerances (β within ~1e-6 relative),
# and every quantity is compared at a tolerance set by that fit agreement.
#
#   zero         marginaleffects with re.form = NA                  → random_effects = :zero
#   cond_fixed   marginaleffects with re.form = NULL (β-only SEs)   → :conditional, re_se = :fixed
#   cond_joint   AAP with Henderson's covariance built in R          → :conditional (re_se = :joint)
#   marginal     R's adaptive `integrate` + numDeriv (GLMM)          → :marginal

using Test, DataFrames, CSV, MixedModels, Statistics
using Margins

const _RE_ORACLE_DIR = joinpath(@__DIR__, "..", "r_compare_marginaleffects", "random_effects")

@testset "random_effects vs R (lme4, marginaleffects)" begin
    d = CSV.read(joinpath(_RE_ORACLE_DIR, "re_data.csv"), DataFrame)
    ref = CSV.read(joinpath(_RE_ORACLE_DIR, "re_reference.csv"), DataFrame; missingstring="NA")
    ref.term = coalesce.(ref.term, "")
    lmm = LinearMixedModel(@formula(y ~ x + w + (1 + w | rater) + (1 | village)), d)
    lmm.optsum.ftol_rel = 1e-14; lmm.optsum.ftol_abs = 1e-14; lmm.optsum.xtol_rel = 1e-12
    fit!(lmm; REML=false, progress=false)
    glmm = GeneralizedLinearMixedModel(@formula(yb ~ x + w + (1 + w | rater) + (1 | village)), d, Bernoulli())
    glmm.optsum.ftol_rel = 1e-14; glmm.optsum.ftol_abs = 1e-14; glmm.optsum.xtol_rel = 1e-12
    fit!(glmm; fast=false, progress=false)       # Laplace, as glmer's default nAGQ = 1
    fits = Dict("lmm" => lmm, "glmm" => glmm)
    # agreement of the fits sets the tolerance (LMM ~1e-7, GLMM ~1e-6 relative in β)
    rtol = Dict("lmm" => 2e-6, "glmm" => 2e-5)
    maxrel = Dict{Symbol,Float64}()
    note!(key, a, b) = (k = Symbol(key); maxrel[k] = max(get(maxrel, k, 0.0), abs(a - b) / abs(b)))
    for r in eachrow(ref)
        m = fits[r.model]
        if r.mode == "fit"
            k = findfirst(==(r.term), coefnames(m))
            @test fixef(m)[k] ≈ r.estimate rtol=rtol[r.model]
            @test sqrt(vcov(m)[k, k]) ≈ r.std_error rtol=rtol[r.model]
            continue
        end
        kw = r.mode == "zero" ? (;) :
             r.mode == "cond_fixed" ? (random_effects=:conditional, re_se=:fixed) :
             r.mode == "cond_joint" ? (random_effects=:conditional,) : (random_effects=:marginal,)
        res = r.quantity == "aap" ? population_margins(m, d; type=:predictions, kw...) :
                                    population_margins(m, d; vars=[Symbol(r.term)], kw...)
        t = DataFrame(res)
        @test t.estimate[1] ≈ r.estimate rtol=rtol[r.model]
        @test t.se[1] ≈ r.std_error rtol=rtol[r.model]
        note!("$(r.model) $(r.mode)", t.estimate[1], r.estimate)
        note!("$(r.model) $(r.mode) se", t.se[1], r.std_error)
    end
    @info "random_effects vs R: max relative difference\n" * join(("  $k: $(round(v; sigdigits=2))" for (k, v) in sort(collect(maxrel))), "\n")
end
