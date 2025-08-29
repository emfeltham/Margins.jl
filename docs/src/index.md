# Margins.jl Documentation

This Julia package provides a suite of functions to compute marginal effects and related contrasts for predictors in GLM/GLMM models:
1. **Adjusted predictions at the mean** (APM) and **marginal effects at the mean** (MEM)
2. **Average Marginal Effects** (AMEs) and **marginal effects at representative values** (MERS)

As it stands, marginal effect calculations, and AME calculations in particular, are a huge gap in statistical modeling in Julia that really limits the ways researchers can take advantage of packages like [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl).[^1]

[^1]: Furthermore, other packages that seek to convert models estimated in Julia into R objects (which can then be used with the mature modeling ecosystem) ultimately feed into another two-language problem (though this strategy may be the best current option in many situations).

Note that this package is similar in spirit to [Effects.jl](https://github.com/beacon-biosignals/Effects.jl), and borrows directly from it for the APM calculations. Ultimately, the design of this package refers heavily to Stata's ["margins"](https://www.stata.com/manuals/cmmargins.pdf) commands.

## Key Features

- **Population vs Profile Framework**: Clean conceptual separation between population-average effects and effects at specific representative scenarios
- **High Performance**: Zero-allocation evaluation using FormulaCompiler.jl for production-scale analysis
- **Categorical Mixtures**: Specify population composition scenarios for policy analysis and counterfactual modeling
- **Flexible Reference Grids**: Both Dict-based and table-based approaches for complex scenario specification  
- **Comprehensive Standard Errors**: Delta-method standard errors with robust/clustered options

## Quick Example

```julia
using Margins, GLM, DataFrames

# Fit model
model = lm(@formula(outcome ~ education + age + income), data)

# Population-average effects (AME-style)  
population_results = population_margins(model, data; type=:effects)

# Profile analysis with categorical mixtures
scenarios = Dict(
    :education => mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2),
    :age => [25, 50, 65],
    :income => :mean
)
profile_results = profile_margins(model, data; at=scenarios, type=:predictions)
```

## Resources

Williams, R. (2012). Using the margins command to estimate and interpret adjusted predictions and marginal effects. The Stata Journal, 12(2), 308–331. [https://www.stata-journal.com/article.html?article=st0260](https://www.stata-journal.com/article.html?article=st0260)

Williams, R. (2021, January 25). Using the margins command to estimate and interpret adjusted predictions and marginal effects [PDF]. University of Notre Dame. [https://www3.nd.edu/~rwilliam/stats/Margins01.pdf](https://www3.nd.edu/~rwilliam/stats/Margins01.pdf)