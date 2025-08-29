# Core API Functions

Margins.jl provides a clean two-function API for marginal effects analysis:

## Population Margins

```@docs
population_margins
```

## Profile Margins

```@docs
profile_margins
```

## Categorical Mixtures

```@docs  
mix
CategoricalMixture
```

## Result Types

```@docs
MarginsResult
```

## Profile Specification

The package supports multiple approaches for profile-based analysis:

1. **Dict-based specification**: Use the `at` parameter with various Dict formats
2. **Table-based specification**: Pass a DataFrame directly for maximum control  
3. **Categorical mixtures**: Use `mix()` for population composition scenarios

See [Reference Grid Specification](reference_grids.md) for comprehensive documentation on all approaches.
