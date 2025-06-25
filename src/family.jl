# function family(model::StatsModels.TableRegressionModel)
#   if model.model <: GeneralizedLinearModel
#     return model.model.rr.d
#   elseif model.model <: LinearModel
#     σ = GLM.dispersion(m.model)  # from GLM.jl: residual standard error
#     Normal(0, σ)
#   else
#     throw(ArgumentError("No family defined for $(typeof(model.model))"))
#   end
# end

# function family(model::Union{LinearMixedModel, GeneralizedLinearMixedModel})
#     if hasproperty(model, :resp) && hasproperty(model.resp, :link)
#         model.resp.d
#     else
#         σ = sdest(m.model)  # from GLM.jl: residual standard error
#         Normal(0, σ)
#     end
# end

"""
`Family` holds the response distribution and link for a fitted model.

Fields:
- `dist`: a Distributions.jl object (e.g. `Normal`, `Binomial`, …)
- `link`: a GLM.jl or MixedModels.jl link object, or `nothing` for identity.
"""
struct Family
    dist::Distribution
    link::Union{Link, Nothing}
end

"""
`family(model)` returns the `Family` of a fitted regression model.

- **GLM** (`GeneralizedLinearModel`): returns the canonical distribution and link.
- **LM** (`LinearModel`): returns `Normal(0, σ̂)` and `nothing`.
- **GLMM** (`GeneralizedLinearMixedModel`): returns the canonical distribution and link.
- **LMM** (`LinearMixedModel`): returns `Normal(0, σ̂)` and `nothing`.
"""
function family(model::StatsModels.TableRegressionModel)
    core = model.model
    if isa(core, GeneralizedLinearModel)
        dist = core.rr.d
        link = core.rr.link
        return Family(dist, link)
    elseif isa(core, LinearModel)
        σ = dispersion(core) # residual standard error
        return Family(Normal(0, σ), IdentityLink())
    else
        throw(ArgumentError("Unsupported TableRegressionModel of type $(typeof(core))"))
    end
end

function family(model::Union{LinearMixedModel, GeneralizedLinearMixedModel})
    if hasproperty(model, :resp) && hasproperty(model.resp, :link)
        dist = model.resp.d
        link = model.resp.link
        return Family(dist, link)
    else
        σ = sdest(model) # residual standard deviation
        return Family(Normal(0, σ), IdentityLink())
    end
end

#=
# -- Example tests using Test.jl --
using Test

# GLM example
df = DataFrame(y = rand(Binomial(1, 0.3), 100), x = rand(100))
glm_fit = glm(@formula(y ~ x), df, Binomial())
@test family(glm_fit) == Family(Binomial(), LogitLink())

# LM example
df2 = DataFrame(y = randn(100), x = randn(100))
lm_fit = lm(@formula(y ~ x), df2)
fam2 = family(lm_fit)
@test typeof(fam2.dist) == Normal && isapprox(std(fam2.dist), stderror(lm_fit); atol=1e-8)
@test fam2.link === nothing

# GLMM example (requires MixedModels)
df3 = DataFrame(y = rand(Binomial(1, 0.5), 200), x = rand(200), g = rand(1:5,200))
glmm_fit = fit!(GeneralizedLinearMixedModel(@formula(y ~ x + (1|g)), df3, Bernoulli()))
@test family(glmm_fit) == Family(Bernoulli(), LogitLink())

# LMM example
df4 = DataFrame(y = randn(200), x = randn(200), g = rand(1:5,200))
lmm_fit = fit!(LinearMixedModel(@formula(y ~ x + (1|g)), df4))
fam4 = family(lmm_fit)
@test typeof(fam4.dist) == Normal && isapprox(std(fam4.dist), sqrt(sigma2(lmm_fit)); atol=1e-8)
@test fam4.link === nothing
=#
