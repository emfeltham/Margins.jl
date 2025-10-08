# Mathematical Foundation

*Theoretical exposition of the analytical framework underlying marginal effects analysis*

## Unified Framework for Marginal Effects Analysis

The theoretical foundation of marginal effects analysis rests upon a systematic decomposition of the inferential problem into two fundamental analytical dimensions. This framework addresses the terminological inconsistencies that have historically impeded methodological clarity across econometric and statistical disciplines through the establishment of a unified conceptual structure.

The analytical framework distinguishes between two orthogonal methodological choices that completely characterize the space of marginal effects approaches. The evaluation context determines whether inference targets population-level parameters or profile-specific estimates, while the analytical target specifies whether the quantity of interest represents marginal effects or adjusted predictions.

### Complete Methodological Taxonomy

|                    | **Effects Analysis**            | **Predictions Analysis**              |
|--------------------|------------------------|------------------------------|
| **Profile Context**        | Marginal Effects at Representative Scenarios | Adjusted Predictions at Representative Scenarios |
| **Population Context**     | Average Marginal Effects | Average Adjusted Predictions |

This taxonomic structure eliminates the ambiguity inherent in discipline-specific terminological conventions while preserving the essential methodological distinctions that govern appropriate analytical application.

## Cross-Disciplinary Terminological Unification

The proposed framework establishes a systematic correspondence between the established terminology employed across econometric, biostatistical, and computational disciplines. This mapping preserves the essential methodological content while eliminating the terminological inconsistencies that have impeded interdisciplinary communication and methodological clarity.

### Profile Context Methodologies
The profile evaluation context encompasses analytical approaches that evaluate marginal quantities at representative or theoretically motivated points in the covariate space. Traditional terminological variants include Marginal Effects at the Mean and Adjusted Predictions at the Mean, which are unified under the profile effects and profile predictions categories respectively.

### Population Context Methodologies  
The population evaluation context encompasses approaches that compute marginal quantities averaged across the empirical distribution of observed covariates. This category unifies Average Marginal Effects, Average Partial Effects, and related population-averaged measures under a coherent methodological framework that emphasizes the distributional basis of the inference.

## Formal Mathematical Specification

### Marginal Effects and Adjusted Predictions

**Conceptual Foundation**: Marginal effects quantify the expected change in an outcome variable resulting from a unit change in an explanatory variable, holding all other variables constant. This fundamental concept addresses the analytical question "How does the dependent variable respond to marginal changes in specific covariates?" The formal mathematical representation ∂E[Y|X]/∂X captures the instantaneous rate of change in the conditional expectation, providing a rigorous framework for quantifying covariate effects.

Marginal effects analysis concerns the derivative of the conditional expectation function with respect to the covariates of interest. For continuous covariates, this quantity is formally defined as ∂E[Y|X]/∂X, representing the instantaneous rate of change in the expected outcome with respect to marginal changes in the explanatory variable. The interpretation centers on the magnitude of response in the dependent variable per unit increase in the independent variable, with measurement units corresponding to the dependent variable scaled by the units of the explanatory variable.

Adjusted predictions represent the conditional expectation E[Y|X] evaluated at specified covariate configurations. This quantity provides the expected value of the outcome variable conditional on particular covariate realizations, maintaining the same units as the dependent variable. The analytical focus shifts from rates of change to levels of the outcome under specific conditioning scenarios.

### Evaluation Context Specifications

**Analytical Perspective**: The choice between profile and population approaches reflects two different ways to approach marginal effects analysis. Profile analysis examines effects for representative individuals or scenarios (analogous to asking "What happens for a typical 35-year-old college graduate?"), while population analysis characterizes average effects across the entire sample distribution (analogous to asking "What happens on average across all individuals in our dataset?"). This distinction determines both the interpretive scope and the computational approach of the marginal effects analysis.

Profile approaches evaluate marginal quantities at predetermined points in the covariate space, most commonly at sample means X̄ or at theoretically motivated scenario specifications. This approach yields concrete, interpretable estimates for specific covariate combinations, facilitating clear communication of results and policy implications for particular demographic or economic profiles. The limitation lies in the potential lack of representativeness relative to the broader population distribution.

Population approaches compute marginal quantities averaged across the empirical distribution of observed covariates, weighting each observation according to its sample frequency. This methodology yields population-averaged parameters that reflect the heterogeneity present in the data generating process, providing estimates that characterize the broader population represented by the sample. The limitation concerns the potential difficulty in interpreting results that may not correspond to any particular individual or realistic scenario within the population.

## Methodological Inconsistencies Across Disciplines

**Why This Matters for Users**:
Different fields use related but slightly different terminology:
- **Economics**: "Average Marginal Effects" (AME) - mean of individual-level derivatives
- **Biostatistics**: "Average Partial Effects" (APE) - often equivalent to AME, with some nuance for discrete variables
- **Machine Learning**: "Partial Dependence" - conceptually similar, typically in visualization contexts

While these concepts overlap substantially, each field may have specific conventions for handling categorical variables or estimation approaches. This package provides a unified framework that works consistently across disciplines.

The absence of standardized terminological conventions across quantitative disciplines has generated substantial methodological confusion that impedes both theoretical development and practical implementation. This inconsistency manifests through the use of discipline-specific acronyms and conceptual frameworks that obscure the underlying mathematical equivalence of analytical approaches.

### Disciplinary Terminological Variations

Economics employs Marginal Effects at the Mean and Average Marginal Effects as primary analytical categories, while biostatistics utilizes Average Partial Effects to denote methodologically identical procedures. Machine learning contexts frequently employ "partial effects" as an umbrella term that may conflate distinct analytical approaches including SHAP values and traditional marginal effects. Statistical software packages compound these inconsistencies through accommodation of diverse terminological preferences across user communities.

### Fundamental Methodological Impediments

The proliferation of terminological variants generates several significant barriers to methodological advancement. Identical analytical concepts receive different nomenclature across disciplines, while the same terminological constructs may denote different methodological approaches in alternative contexts. The distinction between predictions and effects becomes obscured through inconsistent usage patterns, and the conceptual difference between evaluation at representative points versus population averaging receives inadequate attention.

These inconsistencies hamper reproducibility across research contexts, create unnecessary learning barriers for practitioners attempting to apply methods across disciplines, and impede productive interdisciplinary collaboration through the introduction of artificial communication barriers.

## Statistical vs Causal Interpretation

**Statistical vs Causal Interpretation**:
The same marginal effects computation supports both descriptive and causal analysis:
- **Descriptive**: "In our sample, education is associated with higher wages" 
- **Causal**: "Education causes higher wages" (requires identifying assumptions)

The package provides statistically valid estimates; interpretation depends on research design and identifying assumptions.

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
profile_margins(model, data, means_grid(data); type=:effects)

# Effects at specific scenarios
age_grid = cartesian_grid(age=[25, 45, 65])
profile_margins(model, data, age_grid; type=:effects)

# Predictions at representative points
profile_margins(model, data, means_grid(data); type=:predictions)
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
