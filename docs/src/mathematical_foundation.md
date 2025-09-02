# Mathematical Foundation

*Understanding the conceptual framework underlying marginal analysis*

## The 2×2 Framework for Marginal Analysis

Margins.jl is built around a clean **2×2 conceptual framework** that resolves the terminology confusion plaguing marginal effects across disciplines. Rather than memorizing acronyms like MEM, AME, APE, MER, etc., the framework organizes all approaches around two key choices:

1. **Where to evaluate**: Profile (specific scenarios) vs Population (across sample distribution)
2. **What to compute**: Effects (derivatives) vs Predictions (levels)

### The Complete Framework

|                    | **Effects**            | **Predictions**              |
|--------------------|------------------------|------------------------------|
| **Profile**        | Marginal Effects at Representative Points | Adjusted Predictions at Representative Points |
| **Population**     | Average Marginal Effects | Average Adjusted Predictions |

*The key question: Do you want inference for a typical case or across your entire population?*

## Terminology Mapping

This framework unifies terminology across disciplines:

### Profile Approaches
- **Marginal Effects at the Mean (MEM)** → Profile Effects
- **Adjusted Predictions at the Mean (APM)** → Profile Predictions  

### Population Approaches  
- **Average Marginal Effects (AME)** → Population Effects
- **Average Partial Effects (APE)** → Population Effects
- **Average Predictions** → Population Predictions

## Mathematical Definitions

### Effects vs Predictions

**Effects** measure how the expected outcome changes with covariates:
- **Mathematical form**: ∂E[Y|X]/∂X (for continuous variables)
- **Interpretation**: "How much does Y change per unit increase in X?"
- **Units**: Same as Y per unit of X

**Predictions** measure the expected outcome level:
- **Mathematical form**: E[Y|X] 
- **Interpretation**: "What is the predicted value of Y?"
- **Units**: Same as Y

### Profile vs Population

**Profile approaches** evaluate at specific covariate scenarios:
- **At sample means**: X̄ (most common)
- **At user scenarios**: Specific combinations like Dict(:age => 30, :education => "College")
- **Advantage**: Concrete, interpretable results for specific cases
- **Limitation**: May not represent broader population

**Population approaches** average across observed sample distribution:
- **Across all observations**: Average effects/predictions weighted by sample composition
- **Advantage**: True population parameters for your sample
- **Limitation**: May not describe any individual case well

## The Marginal Effects Terminology Problem

The terminology for marginal effects varies dramatically across disciplines, creating unnecessary confusion:

### Cross-Disciplinary Inconsistencies
- **Economics**: MEM (Marginal Effects at Mean), AME (Average Marginal Effects)
- **Biostatistics**: APE (Average Partial Effects) for the same concept as AME
- **Machine Learning**: "Partial effects" as catch-all, often conflated with SHAP values
- **Software packages**: Mix terminologies to accommodate different user bases

### Core Problems
1. **Identical concepts with different names** across fields
2. **Same terms meaning different things** in different contexts  
3. **Poor distinction** between predictions and effects
4. **Inconsistent treatment** of "at-the-mean" versus "average" approaches
5. **Conflation** of statistical associations with causal effects

This terminology chaos hampers reproducibility, creates learning barriers, and impedes interdisciplinary collaboration.

## Statistical vs Causal Interpretation

The 2×2 framework applies equally to both descriptive and causal analysis. The mathematical operations are identical, but the interpretation differs:

### Descriptive Interpretation (Always Valid)
- **Question**: "How does Y vary with X in the observed data?"
- **Requirements**: Only requires correct model specification
- **Example**: "In our sample, each additional year of education is associated with 8% higher wages"

### Causal Interpretation (Requires Additional Assumptions)  
- **Question**: "What would happen to Y if we intervened to change X?"
- **Requirements**: Exogeneity, correct functional form, no omitted variables, etc.
- **Example**: "Increasing education by one year would cause 8% higher wages"

**Important**: The choice of profile vs population is orthogonal to causal identification. Both MEM and AME can be interpreted causally or descriptively, depending on research design.

## When Profile ≠ Population

The choice between profile and population approaches matters most when the link function is non-identity:

### Linear Models with Identity Link
- **Profile = Population**: MEM = AME for effects (predictions differ only by constants)
- **Practical implication**: Choice mainly affects interpretation, not numerical results

### GLMs with Non-Identity Links
- **Profile ≠ Population**: All quantities can differ substantially
- **With interactions**: Profile vs Population can yield opposite conclusions  
- **Heterogeneous samples**: Larger differences between approaches

### Example: Logistic Regression
Consider education effects on probability of employment in a logistic model:

- **Profile (MEM)**: "For someone with average characteristics, +1 year education → +0.08 probability of employment"
- **Population (AME)**: "On average across the sample, +1 year education → +0.05 probability of employment"

The difference arises because the logistic function is nonlinear, so the derivative at the mean differs from the mean of derivatives.

*For the computational implications of this choice, see [Computational Architecture](computational_architecture.md) and [Performance Guide](performance.md).*

## Elasticities and Semi-Elasticities

Elasticities are transformations of marginal effects that follow the same 2×2 framework:

### Definitions
- **Elasticity**: % change in Y per % change in X = (∂Y/∂X) × (X/Y)
- **Semi-elasticity (X)**: % change in Y per unit change in X = (∂Y/∂X) × (1/Y)  
- **Semi-elasticity (Y)**: Unit change in Y per % change in X = (∂Y/∂X) × X

### Framework Application
- **Profile elasticities**: Calculate (∂Y/∂X) × (X̄/Ȳ) at representative values
- **Population elasticities**: Average (∂Y/∂X) × (Xᵢ/Yᵢ) across sample observations

In GLMs with non-identity links, profile elasticity ≠ average elasticity, following the same logic as marginal effects.

*For detailed elasticity examples and applications to policy analysis, see [Advanced Features](advanced.md).*

## Implementation in Margins.jl

The package implements this framework through two main functions:

### `population_margins()`
Computes population-level analysis (AME/AAP equivalent):
```julia
# Population average marginal effects
population_margins(model, data; type=:effects)

# Population average predictions  
population_margins(model, data; type=:predictions)

# Population average elasticities
population_margins(model, data; type=:effects, measure=:elasticity)
```

### `profile_margins()`
Computes profile analysis (MEM/APM equivalent):
```julia
# Effects at sample means
profile_margins(model, data; at=:means, type=:effects)

# Effects at specific scenarios
profile_margins(model, data; at=Dict(:age => [25, 45, 65]), type=:effects)

# Predictions at representative points
profile_margins(model, data; at=:means, type=:predictions)
```

## Practical Guidelines

### Choose Profile When:
- Understanding specific, concrete scenarios
- Communicating results to non-technical audiences  
- Sample is relatively homogeneous
- Policy targets specific demographic profiles

### Choose Population When:
- Estimating true population parameters
- Heterogeneity across sample is important
- Broad policy applications affecting diverse groups
- External validity to similar populations is goal

### The Trade-off
- **Profile approaches**: More concrete and interpretable, but may not represent population
- **Population approaches**: True average effects, but may not describe any individual well

---

*This mathematical foundation anchors all marginal analysis in Margins.jl. For implementation details, see [API Reference](api.md). For computational architecture and performance implications, see [Computational Architecture](computational_architecture.md) and [Performance Guide](performance.md). For advanced applications including elasticities and robust inference, see [Advanced Features](advanced.md).*