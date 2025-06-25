###############################################################################
# demo_ame_tests.jl
#
# Three quick-start scenarios to sanity-check `ame_continuous_analytic`:
#   1.  No interactions, several covariates
#   2.  One interaction term involving x
#   3.  x appears inside a transformation
#
# Dataset:   iris  (built in via RDatasets)
# Predictor: :SepalWidth  (our “x”)
# Outcome:   :SepalLength
###############################################################################

using Margins
using RDatasets            # loads the classic R datasets
using DataFrames, GLM

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
iris = dataset("datasets", "iris") |> DataFrame        # 150 × 5

# ---------------------------------------------------------------------------
# 1.   No interactions – several covariates
#      SepalLength ~ SepalWidth + PetalLength + PetalWidth
# ---------------------------------------------------------------------------
form1 = @formula(SepalLength ~ SepalWidth + PetalLength + PetalWidth)
m    = lm(form1, iris) # linear regression
ame1  = ame_continuous_analytic(iris, m1, :SepalWidth)

iris.set = iris.Species .== "setosa"

m1 = glm(@formula(set ~ SepalWidth + PetalLength + PetalWidth), iris, Bernoulli(), LogitLink())


println("\n=== Scenario 1: No interactions ===")
println("Formula : ", form1)
println("AME dSepalLength/dSepalWidth = $(round(ame1.ame, 4))  (se = $(round(ame1.se, 4)))")

# ---------------------------------------------------------------------------
# 2.   Interaction with x
#      SepalLength ~ SepalWidth * PetalLength + PetalWidth
#      (i.e. main effects + SepalWidth×PetalLength)
# ---------------------------------------------------------------------------
form2 = @formula(SepalLength ~ SepalWidth * PetalLength + PetalWidth)
m2    = lm(form2, iris)
ame2  = ame_continuous_analytic(iris, m2, :SepalWidth)

println("\n=== Scenario 2: Interaction ===")
println("Formula : ", form2)
println("AME dSepalLength/dSepalWidth = $(round(ame2.ame, 4))  (se = $(round(ame2.se, 4)))")

# ---------------------------------------------------------------------------
# 3.   Transformation of x
#      SepalLength ~ log(SepalWidth) + PetalLength + PetalWidth
#      We still differentiate w.r.t. the *original* SepalWidth column.
# ---------------------------------------------------------------------------
form3 = @formula(SepalLength ~ log(SepalWidth) + PetalLength + PetalWidth)
m3    = lm(form3, iris)
ame3  = ame_continuous_analytic(iris, m3, :SepalWidth)

println("\n=== Scenario 3: Transformation ===")
println("Formula : ", form3)
println("AME dSepalLength/dSepalWidth = $(round(ame3.ame, 4))  (se = $(round(ame3.se, 4)))")

###############################################################################
# End of demo – you should see three point estimates and standard errors that
# vary across scenarios, confirming that the derivative correctly adapts to
# interactions and transformations just like Stata’s `margins`.
###############################################################################
