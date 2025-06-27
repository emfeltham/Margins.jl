###############################################################################
# demo_ame_tests_1.jl
# (first set of tests)

# Three quick-start scenarios to sanity-check `ame_continuous_analytic`:
#   1.  No interactions, several covariates
#   2.  One interaction term involving x
#   3.  x appears inside a transformation
#
# Dataset:   iris  (built in via RDatasets)
# Predictor: :SepalWidth  (our “x”)
# Outcome:   :SepalLength
###############################################################################

using RDatasets            # loads the classic R datasets
using DataFrames, CategoricalArrays
using Statistics, GLM
using Margins # development

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
iris = dataset("datasets", "iris") |> DataFrame        # 150 × 5
iris.Species = categorical(iris.Species);

# check categorical process
c = categorical(iris.Species);
isa(c, CategoricalArray)
lvls  = levels(c)
pairs = [(j, k) for j in lvls for k in lvls if j < k]
#

# ---------------------------------------------------------------------------
# 1.   No interactions – several covariates
#      SepalLength ~ SepalWidth + PetalLength + PetalWidth
# ---------------------------------------------------------------------------
form1 = @formula(SepalLength ~ SepalWidth + PetalLength + PetalWidth)
m = lm(form1, iris) # linear regression
ame1  = ame(m, :SepalWidth, iris)
ame2  = ame(m, :PetalLength, iris)
ame3  = ame(m, :PetalWidth, iris)
ame123  = ame(m, [:SepalWidth, :PetalLength, :PetalWidth], iris)

println("\n=== Scenario 1: No interactions ===")
println("Formula : ", form1)
println("AME dSepalLength/dSepalWidth = $(round(ame1.ame[:SepalWidth]; digits = 4))  (se = $(round(ame1.se[:SepalWidth]; digits = 4)))")

#=
Calculation and verification

This result is trivial in that they should match the model coefficients and
standard errors exactly.
=#

####### 1a categorical

# ---------------------------------------------------------------------------
# 1.   No interactions – several covariates
#      SepalLength ~ SepalWidth + PetalWidth + Species
# ---------------------------------------------------------------------------
form1 = @formula(SepalLength ~ SepalWidth + PetalWidth + Species)
m = lm(form1, iris) # linear regression
ame1  = ame(m, :Species, iris)

# ---------------------------------------------------------------------------
# 2.   Interaction with x
#      SepalLength ~ SepalWidth * PetalLength + PetalWidth
#      (i.e. main effects + SepalWidth×PetalLength)
# ---------------------------------------------------------------------------
form2 = @formula(SepalLength ~ SepalWidth * PetalLength + PetalWidth)
m2    = lm(form2, iris)
ame2  = ame(m2, :SepalWidth, iris)

println("\n=== Scenario 2: Interaction ===")
println("Formula : ", form2)
println("AME dSepalLength/dSepalWidth = $(round(ame2.ame[:SepalWidth], digits = 4))  (se = $(round(ame2.se[:SepalWidth], digits = 4)))")

# verify the result against manual calculation
let
    # extract coefs & vcov from your linear model m2
    coefs = coef(m2)
    vc    = vcov(m2)                  # variance–covariance matrix of all β’s
    names = coefnames(m2)             # ["(Intercept)", "SepalWidth", ..., "SepalWidth & PetalLength"]

    # locate the indices for β₁ and β₄
    i_sw   = findfirst(isequal("SepalWidth"),        names)
    i_swpl = findfirst(isequal("SepalWidth & PetalLength"), names)

    # compute mean of the interacting regressor
    meanPL = mean(iris.PetalLength)

    # closed-form AME
    ame_closed = coefs[i_sw] + coefs[i_swpl] * meanPL

    # delta‐method variance: Var(β₁ + meanPL·β₄)
    var_ame = vc[i_sw,i_sw] +
            meanPL^2 * vc[i_swpl, i_swpl] +
            2*meanPL * vc[i_sw, i_swpl]

    se_closed = sqrt(var_ame)

    println("Closed-form AME = ", round(ame_closed, digits=4))
    println("Closed-form s.e. = ", round(se_closed, digits=4))

    @assert ame_closed ≈ ame2.ame[:SepalWidth]
    @assert se_closed ≈ ame2.se[:SepalWidth]
end

#=
Calculation

To verify that your `ame_continuous_analytic` is giving the “right” answer in the interaction case, note that with

```julia
yᵢ = β₀ + β₁·SWᵢ + β₂·PLᵢ + β₃·PWᵢ + β₄·(SWᵢ·PLᵢ) + εᵢ
```

we have

$$
\frac{∂yᵢ}{∂SWᵢ}
= β₁ \;+\; β₄·PLᵢ
$$

so the **average** marginal effect is

$$
\widehat{AME}
= \frac1n\sum_{i=1}^n\bigl(β₁ + β₄·PLᵢ\bigr)
= β₁ + β₄·\overline{PL}\,.
$$

=#

# ---------------------------------------------------------------------------
# 3.   Transformation of x
#      SepalLength ~ log(SepalWidth) + PetalLength + PetalWidth
#      We still differentiate w.r.t. the *original* SepalWidth column.
# ---------------------------------------------------------------------------
form3 = @formula(SepalLength ~ log(SepalWidth) + PetalLength + PetalWidth)
m3    = lm(form3, iris)
ame3  = ame(m3, :SepalWidth, iris)

println("\n=== Scenario 3: Transformation ===")
println("Formula : ", form3)
println("AME dSepalLength/dSepalWidth = $(round(ame3.ame[:SepalWidth], digits=4))  (se = $(round(ame3.se[:SepalWidth], digits=4)))")

# verify
let 
    # extract the log(SepalWidth) coefficient and its variance
    coef_vec = coef(m3)
    vc        = vcov(m3)
    # in the coefnames, "(Intercept)" is first, "log(SepalWidth)" second:
    β_log     = coef_vec[2]
    var_log   = vc[2,2]

    # compute the average of 1/SepalWidth
    mean_inv_SW = mean(1 ./ iris.SepalWidth)  # use .// to be safe, or just 1 ./ iris.SepalWidth

    # closed-form AME and its standard error
    ame_closed = β_log * mean_inv_SW
    se_closed  = sqrt((mean_inv_SW)^2 * var_log)

    println("Closed-form AME       = ", round(ame_closed, digits=4))
    println("Closed-form Std. Err. = ", round(se_closed,  digits=4))

    @assert ame_closed ≈ ame3.ame[:SepalWidth]
    @assert se_closed ≈ ame3.se[:SepalWidth]
end

#=
Calculation

For the model

```julia
yᵢ = β₀ + β₁·log(SWᵢ) + β₂·PLᵢ + β₃·PWᵢ + εᵢ
```

the marginal effect wrt the *original* SepalWidth is by the chain rule

$$
\frac{∂yᵢ}{∂SWᵢ}
= β₁ \;\times\;\frac{d}{dSWᵢ}[\log(SWᵢ)]
= \frac{β₁}{SWᵢ}\,,
$$

so the **average** marginal effect is

$$
\widehat{AME}
= \frac1n\sum_{i=1}^n \frac{β₁}{SWᵢ}
= β₁ \times \frac1n\sum_{i=1}^n \frac1{SWᵢ}
= β₁ \cdot \overline{1/SW}\,.
$$

Explanation of the case (_e.g._, why the effect does not match the coefficient on `log(SepalWidth)`).

$$
y_i = β_0 \;+\; β_1\,\log(\text{SW}_i)\;+\;\cdots
$$

the **coefficient** $β_1$ tells you the change in $y$ for a one‐unit change in $\log(\text{SW})$.  Equivalently,

$$
\frac{\partial y_i}{\partial \log(\text{SW}_i)} = β_1.
$$

But what you asked for in your AME is the change in $y$ for a one‐unit change in the **original** SepalWidth, $\text{SW}_i$.  By the chain rule,

$$
\frac{\partial y_i}{\partial \text{SW}_i}
= β_1 \;\times\; \frac{d}{d\,\text{SW}_i}\bigl[\log(\text{SW}_i)\bigr]
= β_1 \;\times\;\frac1{\text{SW}_i}.
$$

Since $\tfrac{1}{\text{SW}_i}$ varies across observations, the marginal effect isn’t constant; the AME averages those individual slopes:

$$
\widehat{AME}
= \frac1n\sum_{i=1}^n \frac{β_1}{\text{SW}_i}
= β_1 \;\times\;\frac1n\sum_{i=1}^n\frac1{\text{SW}_i}.
$$

In your data, $\overline{1/\text{SW}}\approx0.3339$.  Thus

$$
AME \approx
1.92549\times0.3339 \;\approx\; 0.6427,
$$

which is exactly what your analytic routine—and the closed‐form check—returned.

**In short:**

* **Coefficient** $1.92549$ is “$\Delta y$ per one‐unit increase in $\log(\text{SW})$.”
* **AME** $0.6427$ is “$\Delta y$ per one‐unit increase in $\text{SW}$, averaged over the sample,” so it necessarily differs from $β_1$ because of the $\tfrac1{\text{SW}_i}$ factor.
=#

###############################################################################
# End of demo – you should see three point estimates and standard errors that
# vary across scenarios, confirming that the derivative correctly adapts to
# interactions and transformations just like Stata’s `margins`.
###############################################################################

# ---------------------------------------------------------------------------
# 4.   Marginal effects at representative values of a moderator
#      Model: SepalLength ~ SepalWidth * PetalWidth + PetalLength
# ---------------------------------------------------------------------------
form4 = @formula(SepalLength ~ SepalWidth * PetalWidth + PetalLength)
m4    = lm(form4, iris)

# choose 25th and 75th percentile of PetalWidth as representative values
using Statistics: quantile

repvals = Dict(
    :PetalLength => [
      quantile(iris.PetalLength, 0.25),
      quantile(iris.PetalLength, 0.75),
    ]
)

ame4 = ame(m4, :SepalWidth, iris; repvals=repvals)

println("\n=== Scenario 4: Representative values of PetalWidth ===")
println("Formula : ", form4)
println("PetalWidth rep values: ", round.(repvals[:PetalLength], digits=2))
println("AME of SepalWidth at rep values:")
for (pw,), ame_val in ame4.ame[:SepalWidth]
    se_val = ame4.se[:SepalWidth][(pw,)]
    println("  PetalWidth=$(round(pw, digits=2)) → AME=$(round(ame_val[:SepalWidth], digits=4))  (se=$(round(se_val, digits=4)))")

    # closed-form check
    coefs = coef(m4)
    cn    = coefnames(m4)
    i_sw    = findfirst(isequal("SepalWidth"),           cn)
    i_sw_pw = findfirst(isequal("SepalWidth & PetalWidth"), cn)

    closed    = coefs[i_sw] + coefs[i_sw_pw] * pw
    vc        = vcov(m4)
    var_closed = vc[i_sw,i_sw] + pw^2*vc[i_sw_pw,i_sw_pw] + 2*pw*vc[i_sw,i_sw_pw]
    se_closed = sqrt(var_closed)

    println("    closed-form AME = $(round(closed,   digits=4)), s.e. = $(round(se_closed, digits=4))")
    @assert closed    ≈ ame_val
    @assert se_closed ≈ se_val
end

######## 

# ---------------------------------------------------------------------------
# 5.   Marginal effects at representative values of a moderator
#      Model: SepalLength ~ SepalWidth * Species
# ---------------------------------------------------------------------------
form5 = @formula(SepalLength ~ SepalWidth * Species)
m5    = lm(form5, iris)

repvals = Dict(
    :Species => categorical(unique(iris.Species); levels = levels(iris.Species)),
)

ame5 = ame(m5, :SepalWidth, iris; repvals=repvals)

import Printf.@printf
import LinearAlgebra.dot
import Distributions.cdf

# verify
# grab coef vector and vcov matrix
β = coef(m5)
V = vcov(m5)

# get the names of each coefficient
names = coefnames(m5)
# e.g. ["(Intercept)", "SepalWidth", "Species: versicolor",
#       "Species: virginica", "SepalWidth & Species: versicolor",
#       "SepalWidth & Species: virginica"]

# function to build contrast, compute effect & se for one species level
function me_and_se(level::String)
    c = zeros(eltype(β), length(β))
    # main effect
    i_sw = findfirst(==("SepalWidth"), names)
    c[i_sw] = 1
    # add the interaction if not setosa
    if level != "setosa"
        iname = "SepalWidth & Species: $level"
        i_int = findfirst(==(iname), names)
        @assert i_int !== nothing "Couldn’t find coefficient $iname"
        c[i_int] = 1
    end
    eff = dot(c, β)
    var = c' * V * c
    return eff, sqrt(var)
end

# loop through species
println("Species       |  ME     |  Std.Err   |  z     |  p-value   |  95% CI")
println("----------------------------------------------------------------------")
for lvl in levels(iris.Species)
    eff, se = me_and_se(string(lvl))
    z  = eff/se
    p  = 2 * (1 - cdf(Normal(), abs(z)))
    lo = eff - 1.96*se
    hi = eff + 1.96*se
    @printf("%-13s| %7.4f | %9.4f | %6.2f | %9.3f | [%7.4f, %7.4f]\n",
            string(lvl), eff, se, z, p, lo, hi)
end

# these values printed should match the ame5 values
