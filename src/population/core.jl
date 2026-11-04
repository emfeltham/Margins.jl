# population/core.jl - Main population_margins() function

"""
    population_margins(model, data; type=:effects, vars=nothing, scale=:response, backend=:ad, scenarios=nothing, groups=nothing, measure=:effect, contrasts=:baseline, alpha=0.05, vcov=GLM.vcov, weights=:model, df=nothing, random_effects=:zero, re_new=:zero, re_terms=nothing, re_nodes=:auto, warn=true) -> Union{EffectsResult, PredictionsResult}

Compute population-level marginal effects or adjusted predictions.

This function averages effects/predictions across the observed sample distribution,
providing true population parameters for your sample. It implements the "Population" 
approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Average Marginal Effects (AME): population-averaged derivatives/contrasts
  - `:predictions` - Average Adjusted Predictions (AAP): population-averaged fitted values
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or nothing)
  - Only required when `type=:effects`
  - Defaults to all explanatory variables from the model formula (not all data columns)
  - Only variables that appear in both the model specification and data are considered
  - Extra columns in data that aren't in the model are automatically ignored
- `scale::Symbol=:response`: Target scale for computation
  - `:response` - Response scale (default, applies inverse link function)  
  - `:link` - Linear predictor scale (link scale)
- `backend::Symbol=:ad`: Computational backend
  - `:ad` - Automatic differentiation (exact derivatives; recommended)
  - `:fd` - Finite differences
- `measure::Symbol=:effect`: Effect measure (only for `type=:effects`)
  - `:effect` - Marginal effects (default, current behavior)
  - `:elasticity` - Elasticities (percent change in y for percent change in x)
  - `:semielasticity_dyex` - Semielasticity d(y)/d(ln x) (change in y for percent change in x)
  - `:semielasticity_eydx` - Semielasticity d(ln y)/dx (percent change in y for unit change in x)
  Each is the (weighted) mean over observations of the unit-level quantity — gᵢxᵢ/ŷᵢ,
  gᵢxᵢ or gᵢ/ŷᵢ with gᵢ = ∂ŷᵢ/∂x — as in Stata `margins, eyex()/dyex()/eydx()` and R
  `avg_slopes(slope = ...)`. (Before v3.0.0: a ratio of means such as (x̄/ȳ)·AME.)
  - Measures other than `:effect` apply to continuous variables only: naming a
    discrete variable in `vars` is an error, and auto-detected discrete variables
    are skipped with an informational message
- `scenarios=nothing`: Counterfactual scenarios (NamedTuple mapping variables to values)
  - Example: `(x1=0, x2=[1, 2])` creates scenarios for all combinations
  - Each scenario sets the variable for every row and re-evaluates the formula, so terms the
    formula computes from it (`x1^2`, `log(x1)`, `x1 & z`) follow it, while a column computed
    outside the formula (a precomputed `x1_sq`) is held fixed. See *Derived columns* below.
- `groups=nothing`: Grouping specification for stratified analysis
  - Simple: `:education` or `[:region, :gender]` for categorical grouping
  - Continuous: `(:income, 4)` for quartiles, `(:age, [25, 50, 75])` for thresholds
  - Nested: `:outer => :inner` for hierarchical grouping
- `contrasts::Symbol=:baseline`: Contrast type for categorical variables
  - `:baseline` - Compare each level to reference level
  - `:pairwise` - All pairwise comparisons between levels  
- `alpha::Real=0.05`: significance level α, in (0, 1): confidence intervals have level 1-α (the same keyword as `contrast` and `second_differences`; it was `ci_alpha` before 3.0)
  - Sets the level of the `ci_lower` and `ci_upper` columns of the DataFrame output
- `vcov=GLM.vcov`: Covariance matrix function for standard errors
  (a function of the model, a CovarianceMatrices estimator, or a covariance matrix; a matrix must be square,
  finite, symmetric and match the coefficients). It is evaluated once per call (nothing is cached
  across calls), so a refitted model or a changed matrix is always picked up.
  - `GLM.vcov` - Model-based covariance matrix (default)
  - Custom function or CovarianceMatrices estimator (e.g. `HC1()`) for robust/clustered
    standard errors (the t/z choice does not depend on `vcov`; see *Inference* below)
- `weights=:model`: observation weights for the averages, `Σ(w_i * ∂ŷ_i/∂x_i) / Σ(w_i)`
  - `:model` - the fitted model's estimation weights (`glm(...; wts=w)`, `fit(MixedModel, ...; wts=w)`)
    if it has any, otherwise none (default). The weights must have one entry per row of `data`: if the
    fit dropped rows (missing values), this is an error; pass the weights explicitly. A
    FixedEffectModels `reg(...; weights=:w)` does not keep its weights, so `:model` averages without
    weights there and warns (once per session); pass `weights=:w` (or `weights=:none`) explicitly.
  - `:none` - no weights (equal weight for all observations; the 2.x default). `nothing` means the same.
  - `Symbol` - column name containing weights (e.g., `:sampling_weight`); `:model` and `:none` are
    reserved, so pass a column with one of those names as a vector. A weight column that is also a
    predictor is left out of the auto-detected `vars` (and naming it in `vars` is an error).
  - `Vector` - weight vector matching number of observations. A vector is not matched to a column,
    so `weights=data.z` does not exclude `z` from `vars`: with `z` a predictor, its effect is
    reported (averaged with weights `z`).
  - With weights, `n` (and `metadata[:n_obs]`) counts the rows with a positive weight.
  - `metadata[:weights_source]` records which applied: `:model`, `:user` or `:none`
  - Weights change the point estimates only. Standard errors come from `vcov` (by default the model's,
    which in GLM.jl treats `wts` as frequency weights); they are not design-based survey standard
    errors. For those, pass a design-based covariance estimator or matrix as `vcov`.
- `df=nothing`: degrees of freedom of the t reference distribution of `statistic`, `p_value` and
  the confidence interval, overriding the model-class default (see *Inference*). A positive number,
  or `Inf` for z inference (e.g. `df = G - 1` with a clustered `vcov` on an `lm`, as Stata's
  `vce(cluster)` does). Recorded in `result.inference.df`.
- `random_effects=:zero`: mixed models (MixedModels.jl) only. `:zero` (the default) sets every
  random effect to 0, so the estimates describe a typical group (e.g. a rater and a village whose
  random effects are 0). `:conditional` adds each row's predicted random effects (conditional
  modes, `ranef`) to η, so the estimates describe the groups in the data. Under a nonlinear link
  neither is the population-averaged effect; `:marginal` gives it, averaging the inverse link
  over the estimated random-effects distribution (E[g⁻¹(η + z'b)], b ~ N(0, Σ̂_b)) by
  Gauss–Hermite quadrature: z'b is normal, so the average is one-dimensional however many
  grouping factors are crossed or nested. With `:marginal`:
  - with random slopes the SD of z'b varies by row (it depends on the slope variables); the
    effect of a continuous random-slope variable includes the change in that SD;
  - `re_nodes=:auto` chooses the rule (16 nodes, doubled until ŷ and its first two derivatives
    agree with the next rule to 1e-10 at every η in [−20, 20]); an integer fixes it;
  - under an identity link it equals `:zero`;
  - `metadata[:re_sd]` and `[:re_nodes]` record the SD of z'b (the largest over the rows with
    random slopes) and the rule used.
  Standard errors are delta-method on β with the random-effect variances fixed at their
  estimates. With `:conditional`:
  - standard errors are delta-method on β alone: they omit the prediction error of the random
    effects;
  - the effect of a continuous variable with a random slope in a conditioned factor (`x` in
    `(1 + x | g)`) includes each group's predicted slope;
  - `re_terms` (a Symbol or a vector, grouping factors as `fnames(model)` names them) conditions
    on some factors and sets the others to 0;
  - `re_new=:zero` gives a level absent from the fit random effects 0 (`:error` rejects it).
  With `:conditional` or `:marginal`, not supported yet: contrasts of a categorical variable with
  a random slope, and `scenarios` that set a variable with a random slope (a profile grid may
  set it). `result.metadata[:random_effects]` and `[:re_terms]` record what was used.

# Derived columns
Margins changes a variable (a focal variable's derivative or contrast, a `scenarios` value) by
re-evaluating the model's formula on changed data. The formula defines what depends on what:
anything it computes from the variable follows the change, and anything precomputed outside it
is held fixed. Write transforms in the formula (`y ~ x + x^2`, `log(x)`, `x & z`), not as
precomputed columns. A precomputed spline or polynomial basis means "hold the basis fixed".
- `warn=true`: warn (once per call) when another numeric model column looks like an exact
  transform of a variable the call changes (`x2 == x.^2`, `log(x)`, a standardized copy, a
  precomputed product `x .* z`). `warn=false` silences it, for a column that should stay fixed. See the "Derived columns"
  section of the Population Scenarios documentation for the supported formula transforms.

# Offsets
A model's η offset (`glm(...; offset=log.(exposure))`, a GLMM's `offset=`, and for
FixedEffectModels the estimated absorbed fixed effects, which need `reg(...; save=:fe)`) enters
every prediction and every derivative of the link at each row: η = x'β + offsetᵢ. Under
`scenarios` the offset stays at each row's observed value; within `groups` each row keeps its own.
It must have one entry per row of `data`.

# Estimation rows
For a FixedEffectModels fit, only the rows the model was estimated on (its `esample`: rows
without missing values, with positive weight, and not singletons of a fixed effect) are used, so
averages, `groups`, weights and the fixed effects all refer to the estimation sample. `data` must be
the table passed to `reg` (one row per entry of `esample`; it is subset here), and any other
number of rows is an `ArgumentError`. A weight vector may have one entry per row of `data`.

# Rank-deficient fits
When the fit drops a coefficient as collinear (GLM.jl's `dropcollinear=true`, the default;
MixedModels; FixedEffectModels for a regressor collinear with the fixed effects), the model reports
it as 0 with NaN rows and columns in `vcov`. Margins treats a dropped coefficient as fixed at 0, as
R does (`lm` reports it as `NA` and `marginaleffects` leaves it out of the Jacobian):
- its rows and columns of Σ are set to 0 (`result.inference.vcov`; the dropped coefficients are
  `result.inference.dropped`), whatever `vcov` gives there, so every estimate and standard error
  equals the one from the refit without the redundant term;
- a quantity that depends only on dropped coefficients (the effect of a variable whose every
  coefficient was dropped, e.g. `z` in `y ~ x + z + w` with `w = 2z` when GLM.jl drops `z`) is not
  identified: its estimate is 0 and its standard error, statistic, p-value and interval are NaN
  (R: estimate 0, SE `NA`). Which of the collinear terms is dropped is the fitting package's choice
  (GLM.jl's pivoted Cholesky keeps the columns of largest norm; R keeps the first in the formula);
- each call warns once, naming the dropped coefficients;
- a non-finite entry of Σ in any other row (a broken matrix or estimator) is an `ArgumentError`;
- CovarianceMatrices estimators on a rank-deficient GLM.jl fit: `HC0`/`HC1` work (`HC1`'s
  n/(n − p) is rescaled to n/(n − rank), the refit's value); `HC2` to `HC5` are refused (their
  leverage is computed incorrectly upstream); the cluster-robust `CR` estimators fail inside
  CovarianceMatrices, which is reported as an `ArgumentError`. A matrix passed as `vcov` is used
  as given apart from the dropped rows and columns.

# Returns
`EffectsResult` or `PredictionsResult` containing estimates, delta-method standard errors,
the parameter gradient matrix, the context columns (`result.context`: the `groups` and
`scenarios` values of each row), the inference (`result.inference`: the covariance matrix, the
degrees of freedom and `alpha`) and analysis metadata. `DataFrame(result)` has the columns

- effects: `[over_<g>…] [at_<s>…] type variable contrast estimate se statistic df p_value ci_lower ci_upper n`
- predictions: `[over_<g>…] [at_<s>…] type estimate se statistic df p_value ci_lower ci_upper n`

Group variables appear as `over_<var>` and scenario variables as `at_<var>`, each in the order
given in `groups` / `scenarios`.

# Inference
`statistic = estimate / se`. Its reference distribution, used for `p_value` and the
confidence interval, is t with `df` degrees of freedom. `df` is a property of the model class,
whatever `vcov` is passed (see `Margins._inference_df`), unless the `df` keyword gives it:
- `df = dof_residual(model)` for a linear model — GLM.jl `lm`, or a `glm` with a `Normal`
  family and `IdentityLink` — with any `vcov`, including robust ones (as Stata's `margins` after
  `regress` or `regress, vce(robust)`, `GLM.coeftable`, and R's `coeftest(m, vcovHC(m))`). Every
  margin of such a model is a linear combination of β.
- `df = dof_residual(model)` for a FixedEffectModels fit, which FixedEffectModels adjusts for
  clustering (G − 1 with a cluster `vcov`), as its own `coeftable` does.
- `df = Inf` (standard Normal, z) for all other models (non-identity GLMs, mixed models).

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts (each level minus reference level)
- All computations maintain statistical validity with zero tolerance for approximations

# Examples
```julia
# Average marginal effects for all explanatory variables
result = population_margins(model, data)
DataFrame(result)  # Convert to DataFrame

# Specific variables with response-scale effects
result = population_margins(model, data; vars=[:x1, :x2], scale=:response)

# Average elasticities
result = population_margins(model, data; vars=[:x1, :x2], measure=:elasticity)

# Semi-elasticities
result = population_margins(model, data; vars=[:x1], measure=:semielasticity_dyex)

# Average adjusted predictions  
result = population_margins(model, data; type=:predictions)

# Confidence intervals (99% confidence level)
result = population_margins(model, data; vars=[:x1], alpha=0.01)
DataFrame(result)  # Includes ci_lower and ci_upper columns

# Counterfactual analysis: effects when x2 is set to 0 vs 1
result = population_margins(model, data; vars=[:x1], scenarios=(x2=[0, 1],))

# Grouping examples
result = population_margins(model, data; groups=:education)  # By education level
result = population_margins(model, data; groups=(:income, 4))  # By income quartiles
result = population_margins(model, data; groups=:region => :gender)  # Nested grouping

# High-accuracy computation with automatic differentiation  
result = population_margins(model, data; backend=:ad, scale=:link)

# A weighted fit is averaged with its own weights by default; :none gives the unweighted average
wmodel = glm(@formula(y ~ x1 + x2), data, Binomial(); wts=data.w)
population_margins(wmodel, data)                  # weighted by data.w (metadata[:weights_source] == :model)
population_margins(wmodel, data; weights=:none)   # unweighted

# Averaging weights given explicitly (point estimates; see `vcov` for the standard errors)
result = population_margins(model, data; weights=:sampling_weight)

# Frequency weights for aggregated data
result = population_margins(model, data; weights=data.freq_weights)
```

See also: [`profile_margins`](@ref) for effects at specific covariate combinations.
"""
function population_margins(
    model, data;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, scenarios=nothing, groups=nothing, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    alpha::Real=0.05, vcov=GLM.vcov, weights=:model, df=nothing,
    random_effects::Symbol=:zero, re_new::Symbol=:zero, re_terms=nothing, re_nodes=:auto,
    warn::Bool=true, ci_alpha=nothing
)
    _reject_ci_alpha(ci_alpha)
    # Every option is validated, and `vcov` evaluated, once, here
    req = _normalize_request(model, data; type, vars, scale, backend, scenarios, groups,
                             measure, contrasts, alpha, vcov, weights, df, random_effects, re_new,
                             re_terms, re_nodes, warn)
    return _run(req)
end

"""
    _run(req::MarginsRequest{:population}) -> Union{EffectsResult, PredictionsResult}

Compute a population request: the AME/AAP, with its inference (`result.inference`: the
request's Σ, df and alpha).
"""
_run(req::MarginsRequest{:population}) =
    _with_random_effects_metadata!(_population_result(req), req)

# Record the request's random-effects treatment on its result (`:zero` is the default metadata)
function _with_random_effects_metadata!(res, req::MarginsRequest)
    _set_random_effects_metadata!(res.metadata, req.model, req.random_effects)
    return res
end

"""
    _population_result(req::MarginsRequest{:population}) -> Union{EffectsResult, PredictionsResult}

The AME/AAP of a population request.
"""
function _population_result(req::MarginsRequest{:population})
    (; model, data_nt, type, vars, scale, backend, measure, contrasts) = req
    weights_vec = req.weights

    # Build zero-allocation engine with PopulationUsage optimization.
    # Each call builds its own engine (nothing is cached across calls), so concurrent calls
    # never share mutable buffers and every call sees the current model, data and Σ.
    # The scenarios/groups path builds its own evaluators per context, so its engine gets none.
    has_contexts = _has_contexts(req)
    # The η offset at each data row (held at its observed values under scenarios) is the
    # engine's, so every population kernel reads it.
    engine = build_engine(PopulationUsage, model, data_nt, isnothing(vars) ? Symbol[] : vars, req.Σ, backend;
                          evaluators = !has_contexts, offset = req.offset,
                          re_average = _re_average(req.random_effects))
    # Handle scenarios/groups parameters for counterfactual scenarios and grouping
    if has_contexts
        # Whether a request has contexts is a run-time value (it is a type in the keyword
        # call), so the contexts path is called without inference (`invokelatest`): it is
        # then not compiled into every plain AME/AAP call (first-call latency), only on first use.
        res = Base.invokelatest(_population_margins_with_contexts, engine, req)
        _set_weights_metadata!(res.metadata, weights_vec, req.weights_source)
        return res
    end

    if type === :effects
        df, G = _ame_calculate(engine, data_nt, scale, backend, _measure_type(measure), contrasts, weights_vec)  # → AME (both continuous and categorical)
        metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=_n_used(1:length(first(data_nt)), weights_vec), model_type=typeof(model))
    
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :population
    
        # Store weights info in metadata
        _set_weights_metadata!(metadata, weights_vec, req.weights_source)
    
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
        variables = df.variable  # The "x" in dy/dx
        terms = df.contrast
    
        _mark_not_identified!(standard_errors, G, req.dropped)
        return EffectsResult(estimates, standard_errors, variables, terms, nothing, G, _request_inference(req, engine), metadata)
    else # :predictions  
        df, G = _population_predictions(engine, data_nt; scale, weights=weights_vec)  # → AAP
        metadata = _build_metadata(; type, vars=Symbol[], scale, backend, n_obs=_n_used(1:length(first(data_nt)), weights_vec), model_type=typeof(model))
    
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :population
    
        # Store weights info in metadata
        _set_weights_metadata!(metadata, weights_vec, req.weights_source)
    
        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se
    
        _mark_not_identified!(standard_errors, G, req.dropped)
        return PredictionsResult(estimates, standard_errors, nothing, G, _request_inference(req, engine), metadata)
    end
end

"""
    _get_model_formula_variables(model) -> Set{Symbol}

Extract explanatory variables from the model formula's right-hand side (RHS), excluding the dependent variable.

This function examines the model's formula specification and returns only variables that are 
actually used in the model, not all variables present in the data. This ensures that marginal
effects are computed only for variables that influence the model predictions.

# Arguments
- `model`: Fitted statistical model with a `.mf.f.rhs` formula component

# Returns
- `Set{Symbol}`: Set of variable names (symbols) from the model formula RHS

# Examples
```julia
# For model: y ~ x1 + x2 + region
variables = _get_model_formula_variables(model)  # -> Set([:x1, :x2, :region])
```

This filtering is critical for preventing errors when data contains extra columns
(e.g., ID variables, alternative specifications) that aren't part of the fitted model.
"""
function _get_model_formula_variables(model)
    # Extract all terms from the model formula RHS, handling mixed models properly
    # Use FormulaCompiler's fixed_effects_form to get only fixed effects for mixed models
    fixed_effects_formula = fixed_effects_form(model)
    return _termvars!(Set{Symbol}(), fixed_effects_formula.rhs)
end

# `StatsModels.termvars(t)` as a set, added to `s`: the same variables, but a matrix or tuple of
# terms is walked term by term without specializing on its (long) tuple type. `termvars` of the
# whole tuple is a `mapreduce(termvars, union, …)` over it: about 25 μs on the paper's 55-term
# formula, on every call that asks for the predictors (the derived-column detector does); this
# takes about 6 μs.
function _termvars!(s::Set{Symbol}, @nospecialize(t))
    if t isa StatsModels.MatrixTerm
        _termvars!(s, t.terms)
    elseif t isa Tuple
        for x in t
            _termvars!(s, x)
        end
    else
        union!(s, StatsModels.termvars(t))
    end
    return s
end

"""
    _get_all_effect_variables(model, data_nt::NamedTuple, weight_col=nothing) -> Vector{Symbol}

Extract all explanatory variables from the model that can have marginal effects computed,
including both continuous and categorical variables.

This function performs three levels of filtering:
1. **Model filtering**: Only considers variables present in the model formula (not all data columns)
2. **Type filtering**: Only includes variables with computable effects (numeric, Bool, CategoricalArray)
3. **Weight filtering**: Excludes weight columns to prevent conflicts

# Arguments
- `model`: Fitted statistical model with formula specification
- `data_nt::NamedTuple`: Data in NamedTuple format (from Tables.columntable)
- `weight_col=nothing`: Weight column name to exclude (Symbol or nothing)

# Returns
- `Vector{Symbol}`: Variable names that can have marginal effects computed

# Variable Type Classification
- **Continuous variables**: `Int64`, `Float64` (but not `Bool`) → get derivatives
- **Categorical variables**: `Bool`, `CategoricalArray` → get discrete contrasts  
- **Excluded types**: `String`, `Date`, other non-numeric types

# Examples
```julia
# Model: y ~ x1 + x2 + education + treated
# Data also contains: id, date_collected, alternative_outcome
effect_vars = _get_all_effect_variables(model, data_nt, :sampling_weight)
# -> [:x1, :x2, :education, :treated]  (only model variables, no id/date/weight)
```

This filtering ensures statistical validity by computing effects only for variables
that actually influence the model predictions and have appropriate data types.
"""
function _get_all_effect_variables(model, data_nt::NamedTuple, weight_col=nothing,
                                   fvars=_get_model_formula_variables(model))
    # Variables that are actually in the model formula (`fvars`: computed once per call by the
    # request layer, see `_normalize_request`)
    model_vars = fvars
    
    effect_vars = Symbol[]
    for (name, col) in pairs(data_nt)
        # CRITICAL FIX: Only process variables that are in the model formula
        if !(name in model_vars)
            continue
        end
        
        # Skip weight column - it's not a model variable
        if !isnothing(weight_col) && name == weight_col
            continue
        end
        
        # Include continuous variables (numeric types except Bool)
        # Include categorical variables (Bool, CategoricalArray, etc.)
        el_type = eltype(col)
        if (el_type <: Real && !(el_type <: Bool)) ||  # Continuous (Int, Float, but not Bool)
           el_type <: Bool ||                           # Bool (categorical)  
           hasproperty(col, :pool)                      # CategoricalArray
            push!(effect_vars, name)
        end
    end
    return effect_vars
end

"""
    _get_continuous_variables(model, data_nt::NamedTuple, weight_col=nothing) -> Vector{Symbol}

Extract continuous explanatory variables from the model that can have derivatives computed.

This function applies the same model filtering as `_get_all_effect_variables` but restricts
to continuous variables only, excluding categorical types (Bool, CategoricalArray).

# Arguments
- `model`: Fitted statistical model with formula specification  
- `data_nt::NamedTuple`: Data in NamedTuple format (from Tables.columntable)
- `weight_col=nothing`: Weight column name to exclude (Symbol or nothing)

# Returns
- `Vector{Symbol}`: Continuous variable names that can have derivatives computed

# Variable Type Classification
- **Included**: `Int64`, `Float64` (excluding `Bool`) → get derivatives via finite differences or AD
- **Excluded**: `Bool`, `CategoricalArray`, `String`, `Date`, etc. → not continuous

# Examples
```julia
# Model: y ~ x1 + x2 + education + treated  (x1,x2 continuous; education,treated categorical)
continuous_vars = _get_continuous_variables(model, data_nt)
# -> [:x1, :x2]  (only continuous model variables)
```

This is used when code specifically needs continuous variables only
(derivatives without discrete contrasts for categorical variables).
"""
function _get_continuous_variables(model, data_nt::NamedTuple, weight_col=nothing)
    # Use FormulaCompiler's robust implementation instead of manual type checking
    # FormulaCompiler analyzes compiled operations to correctly distinguish
    # continuous vs categorical variables and automatically handles formula filtering
    compiled = _shared_compiled(model, data_nt)
    continuous_vars = continuous_variables(compiled, data_nt)

    # Weight column filtering (defensive check - FormulaCompiler should exclude this automatically
    # since weight columns shouldn't be in model formulas, but we keep for extra safety)
    if !isnothing(weight_col)
        continuous_vars = filter(v -> v != weight_col, continuous_vars)
    end

    return continuous_vars
end


"""
    _validate_vars_scenarios_overlap(vars, scenarios)

    Teaching validation that provides helpful error messages when users 
incorrectly specify the same variable in both vars and scenarios parameters.
"""
_validate_vars_scenarios_overlap(vars, scenarios::NamedTuple) =
    _validate_vars_scenarios_overlap(vars, collect(Symbol, keys(scenarios)))

function _validate_vars_scenarios_overlap(vars, scenario_vars::AbstractVector{Symbol})
    # Convert vars to vector for consistent handling
    vars_vec = vars isa Symbol ? [vars] : (vars === :all_continuous ? Symbol[] : vars)
    
    # Find overlapping variables
    overlapping_vars = intersect(Set(vars_vec), Set(scenario_vars))
    
    if !isempty(overlapping_vars)
        overlapping_list = join(overlapping_vars, ", ")
        
        # Create teaching error message
        error_msg = """
        Invalid parameter combination: Variables $(overlapping_list) appear in both 'vars' and 'scenarios'.
        
        What you're asking:
           vars = [$(join(vars_vec, ", "))]        -> "What's the marginal effect of these variables?"
           scenarios = (...)  -> "Hold these variables constant at specific values"
        
        This is contradictory! You can't compute the effect of changing a variable while holding it constant.
        
        What you probably want:
        
        1. Effect of OTHER variables when $(overlapping_list) varies:
           population_margins(model, data; vars=[:other_var], scenarios=($(overlapping_list) = [value1, value2],))
           
        2. Predicted outcomes when $(overlapping_list) varies:
           population_margins(model, data; type=:predictions, scenarios=($(overlapping_list) = [value1, value2],))
           
        3. Effect of $(overlapping_list) within subgroups:
           population_margins(model, data; vars=[$(overlapping_list)], groups=:grouping_var)
           
        4. Just the effect of $(overlapping_list) (no scenarios):
           population_margins(model, data; vars=[$(overlapping_list)])
        
        Learn more about marginal effects vs predictions in the documentation.
        """
        
        throw(ArgumentError(error_msg))
    end
end

"""
    _validate_vars_groups_overlap(vars, groups)

    Teaching validation that provides helpful error messages when users
request the marginal effect of a variable that is also used to define the subgroups.

Mirrors `_validate_vars_scenarios_overlap`. Stratifying by a variable pins it within each
subgroup, so the effect of that variable inside strata of itself is not the quantity the
call appears to request; the variable is dropped from the computation, which for
`vars=[:x], groups=:x` leaves nothing to return at all.
"""
function _validate_vars_groups_overlap(vars, groups)
    # Convert vars to vector for consistent handling
    vars_vec = vars isa Symbol ? [vars] : (vars === :all_continuous ? Symbol[] : vars)

    # Only a grouping that PINS a variable to a single value conflicts with asking for that
    # variable's effect. A grouping that BINS it leaves it varying within each subgroup.
    (pinned_vars, _) = _classify_group_variables(groups)
    overlapping_vars = intersect(Set(vars_vec), Set(pinned_vars))

    if !isempty(overlapping_vars)
        overlapping_list = join(overlapping_vars, ", ")
        vars_list = join((":" * string(v) for v in vars_vec), ", ")
        v = first(overlapping_vars)

        # Create teaching error message
        error_msg = """
        Invalid parameter combination: Variables $(overlapping_list) appear in both 'vars' and 'groups'.

        What you're asking:
           vars = [$(vars_list)]        -> "What's the marginal effect of these variables?"
           groups = ...   -> "Split the sample into subgroups defined by these variables"

        Grouping by $(overlapping_list) holds it at one level inside each subgroup, so there is no
        variation left to take an effect with respect to.

        Note this applies to grouping that PINS a variable to a level. Grouping that BINS a
        numeric variable into ranges is fine, because the variable still varies inside each bin:

           population_margins(model, data; vars=[:income], groups=(:income, 4))   # allowed

        What you probably want:

        1. Effect of OTHER variables within strata of $(overlapping_list):
           population_margins(model, data; vars=[:other_var], groups=:$(v))

        2. Effect of $(overlapping_list) at specific values of $(overlapping_list):
           profile_margins(model, data, cartesian_grid($(v)=[value1, value2]); type=:effects, vars=[:$(v)])

        3. Just the effect of $(overlapping_list) (no groups):
           population_margins(model, data; vars=[$(vars_list)])

        Learn more about the vars/groups conflict rule in the documentation.
        """

        throw(ArgumentError(error_msg))
    end
end

"""
    _validate_scenarios_groups_conflict(scenario_vars, groups)

Reject a variable that is simultaneously overridden by `scenarios` (whose variables are
`scenario_vars`) and used by `groups`.

Such a request is silently misleading: `_create_context_data_with_weights` selects each
group's rows using the variable's REAL values, then the scenario override replaces those
values for every row, while the result still carries the group label. The reported subgroup
is defined by data that no longer exists in the context it labels. For a categorical group
the scenario and group values also share one `at_<var>` column, so the scenario label is
overwritten and the scenarios cannot be told apart.
"""
function _validate_scenarios_groups_conflict(scenario_vars, groups)
    (pinned_vars, binned_vars) = _classify_group_variables(groups)
    conflicting = intersect(Set{Symbol}(scenario_vars), Set(vcat(pinned_vars, binned_vars)))

    if !isempty(conflicting)
        conflicting_list = join(sort!(collect(conflicting)), ", ")
        v = first(sort!(collect(conflicting)))
        throw(ArgumentError("""
        Invalid parameter combination: Variables $(conflicting_list) are used by 'groups' and also overridden by 'scenarios'.

        Groups are formed from the observed values of $(conflicting_list), but the scenario then
        replaces those values for every row. Each subgroup would be labelled by a value its rows
        no longer carry, and the scenario and group labels would collide.

        Group by one variable and set a scenario on another:
           population_margins(model, data; groups=$(repr(v)), scenarios=(other_var=[0, 1],))
        """))
    end
end

"""
    _validate_scenarios_are_predictors(model, scenario_vars)

Reject a scenario variable (an entry of `scenario_vars`, the keys of `scenarios`) that is not
a predictor in the model's formula.

Overriding such a column (a typo, the response, the weights column, or any unused column)
changes nothing, so every scenario would silently report the baseline estimate under an
`at_<var>` label that looks like a "no effect" result.
"""
function _validate_scenarios_are_predictors(model, scenario_vars, fvars=_get_model_formula_variables(model))
    bad = _nonpredictor_vars(model, scenario_vars, fvars)
    if !isempty(bad)
        throw(ArgumentError(
            "Scenario variable(s) $(join(bad, ", ")) are not predictors in the model, so " *
            "overriding them would not change any estimate. Check for typos; model " *
            "predictors are: $(join(_predictor_names(model), ", "))."
        ))
    end
end

"""
    _predictor_names(model) -> Vector{Symbol}

Sorted names of the model's predictors: formula variables other than the response. A
model class whose formula omits some predictors adds a method (FixedEffectModels adds its
absorbed fixed effects, which a reference grid may set).
"""
_predictor_names(model, fvars=_get_model_formula_variables(model)) = _formula_predictor_names(model, fvars)

# `fvars`: the formula's variables (`_get_model_formula_variables`), passed by the request layer,
# which computes them once per call
function _formula_predictor_names(model, fvars=_get_model_formula_variables(model))
    predictors = fvars
    response = try
        Set{Symbol}(StatsModels.termvars(StatsModels.formula(model).lhs))
    catch
        Set{Symbol}()
    end
    return sort!(collect(setdiff(predictors, response)))
end

"""
    _nonpredictor_vars(model, vars) -> Vector{Symbol}

The entries of `vars` that are not predictors of `model` (typos, unused columns, the response).
"""
function _nonpredictor_vars(model, vars, fvars=_get_model_formula_variables(model))
    predictors = Set(_predictor_names(model, fvars))
    return [Symbol(v) for v in vars if !(Symbol(v) in predictors)]
end

"""
    _validate_group_columns_complete(groups, data_nt)

Reject a grouping variable that is not a column of the data, or that contains `missing`. Group
membership is decided by comparing values, which is undefined for `missing`; failing here gives
a clear message instead of a `FieldError` or `TypeError` from deep inside context construction.
"""
function _validate_group_columns_complete(groups, data_nt)
    for var in _extract_group_variables(groups)
        haskey(data_nt, var) || throw(ArgumentError(
            "Grouping variable '$(var)' is not a column of the data. Available columns: " *
            "$(join(string.(keys(data_nt)), ", "))"))
        if any(ismissing, data_nt[var])
            throw(ArgumentError(
                "Grouping variable '$(var)' contains missing values, so group membership is " *
                "undefined for those rows. Drop them first, e.g. " *
                "dropmissing(data, $(repr(var))), or recode them to an explicit level."
            ))
        end
    end
end

"""
    _validate_no_duplicate_group_vars(groups)

Reject a variable that appears more than once in `groups`. Each group spec writes its own
value under the variable's name, so a repeat either duplicates every row or silently
overwrites the earlier spec.
"""
function _validate_no_duplicate_group_vars(groups)
    (pinned, binned) = _classify_group_variables(groups; dedup=false)
    all_vars = vcat(pinned, binned)
    dups = unique(v for v in all_vars if count(==(v), all_vars) > 1)
    if !isempty(dups)
        throw(ArgumentError(
            "Grouping variable(s) $(join(dups, ", ")) appear more than once in 'groups'. " *
            "List each variable once, e.g. groups=[:g, (:x, 4)]."
        ))
    end
end

"""
    _validate_groups_parameter(groups)

Validate the unified groups parameter for population margins.
"""
function _validate_groups_parameter(groups)
    _validate_no_duplicate_group_vars(groups)
    # Simple categorical grouping: :education or [:region, :gender]
    if groups isa Symbol 
        return
    end
    
    # Vector grouping (may contain mixed categorical and continuous specs)
    if groups isa AbstractVector
        # All symbols - simple categorical cross-tabulation
        if all(x -> x isa Symbol, groups)
            return
        end
        # Mixed vector - validate each element
        for spec in groups
            _validate_groups_parameter(spec)
        end
        return
    end
    
    # Continuous grouping: (:income, 4) or (:age, [25000, 50000])
    if groups isa Tuple && length(groups) == 2
        var, spec = groups
        if var isa Symbol
            # Quantile specification: (:income, 4)
            if spec isa Integer && spec > 0
                return
            end
            # Threshold specification: (:income, [25000, 50000])
            if spec isa AbstractVector && all(x -> x isa Real, spec)
                return
            end
        end
    end
    
    # Syntax for nested grouping: :region => :education
    if groups isa Pair
        outer_spec = groups.first
        inner_spec = groups.second
        # Recursively validate outer and inner specifications
        _validate_groups_parameter(outer_spec)
        # Inner specification can also be a Vector for multiple groupings
        if inner_spec isa AbstractVector
            for spec in inner_spec
                _validate_groups_parameter(spec)
            end
        else
            _validate_groups_parameter(inner_spec)
        end
        return
    end
    
    throw(ArgumentError("groups must be Symbol, Vector{Symbol}, continuous specification (Symbol, Int/Vector), or nested specification (outer => inner)"))
end

"""
    _process_vars_parameter(model, vars, data_nt::NamedTuple, weight_col=nothing) -> Vector{Symbol}

Process and validate the `vars` parameter for effects analysis, applying model-aware filtering.

This function handles all possible `vars` specifications and ensures only valid model variables
are returned for effects computation. It serves as the central validation point for variable
selection in both `population_margins` and `profile_margins`.

# Arguments
- `model`: Fitted statistical model with formula specification
- `vars`: User-specified variables (Symbol, Vector{Symbol}, :all_continuous, or nothing)
- `data_nt::NamedTuple`: Data in NamedTuple format (from Tables.columntable)  
- `weight_col=nothing`: Weight column name to exclude (Symbol or nothing)

# Returns
- `Vector{Symbol}`: Validated variable names for effects analysis

# Input Processing Rules
- `vars=nothing` → Auto-detect all model variables (continuous + categorical)
- `vars=:all_continuous` → Auto-detect continuous model variables only  
- `vars=:symbol` → Validate single variable exists in model and data
- `vars=[:x1, :x2]` → Validate all variables exist in model and data

# Validation Checks
1. **Model membership**: Variables must appear in model formula
2. **Data availability**: Variables must exist in provided data
3. **Weight conflicts**: Variables cannot conflict with weight column
4. **Non-empty results**: At least one valid variable must be found

# Examples
```julia
# Auto-detection (most common)
vars = _process_vars_parameter(model, nothing, data_nt)  # -> all model variables

# Explicit specification  
vars = _process_vars_parameter(model, [:x1, :x2], data_nt)  # -> [:x1, :x2] (if valid)

# Backward compatibility
vars = _process_vars_parameter(model, :all_continuous, data_nt)  # -> continuous only
```

# Errors Thrown
- `MarginsError`: When no valid variables found for effects analysis
- `ArgumentError`: When vars format is invalid or variables conflict with weights
"""
function _process_vars_parameter(model, vars, data_nt::NamedTuple, weight_col=nothing,
                                 fvars=_get_model_formula_variables(model))
    if isnothing(vars)
        # Auto-detect ALL variables (both continuous and categorical) for comprehensive effects analysis
        effect_vars = _get_all_effect_variables(model, data_nt, weight_col, fvars)
        if isempty(effect_vars)
            throw(MarginsError("No explanatory variables found in data for effects analysis. Available variables: $(collect(keys(data_nt)))"))
        end
        return effect_vars
    elseif vars isa Symbol
        vars_vec = [vars]
        _validate_variables(data_nt, vars_vec)
        _validate_vars_weight_conflict(vars_vec, weight_col)
        return vars_vec
    elseif vars isa Vector{Symbol}
        if isempty(vars)
            throw(ArgumentError("vars cannot be an empty vector. Use vars=nothing to auto-detect variables, or specify at least one variable."))
        end
        _validate_variables(data_nt, vars)
        _validate_vars_weight_conflict(vars, weight_col)
        return vars
    else
        throw(ArgumentError("vars must be Symbol, Vector{Symbol}, or nothing"))
    end
end

"""
    _validate_vars_weight_conflict(vars, weight_col)

Validate that no variable in vars conflicts with the weight column.

# Arguments
- `vars`: Vector{Symbol} of variables for effects analysis
- `weight_col`: Weight column name (Symbol or nothing)

# Throws
- `ArgumentError`: If any variable in vars matches the weight column
"""
function _validate_vars_weight_conflict(vars::Vector{Symbol}, weight_col)
    if !isnothing(weight_col) && weight_col in vars
        throw(ArgumentError(
            "Variable :$weight_col cannot be used both as a weight (weights=:$weight_col) " *
            "and as a variable for effects analysis (vars includes :$weight_col). " *
            "A column cannot serve both roles simultaneously. " *
            "Choose different columns for weights and effects, or remove :$weight_col from vars."
        ))
    end
end

"""
    _validate_weights_parameter(weights, data)

Validate the weights parameter for population_margins().

# Arguments
- `weights`: The weights specification (Symbol, Vector, or nothing)
- `data`: Original data (for Tables.jl compatibility check)

# Throws
- `ArgumentError`: If weights specification is invalid
"""
function _validate_weights_parameter(weights, data)
    if weights isa Symbol
        # Check that the column exists in the data
        if !Tables.columnaccess(data) || !haskey(Tables.columntable(data), weights)
            throw(ArgumentError("weights column :$weights not found in data"))
        end
        # A column gets the same value checks as a vector: negative, NaN, Inf, missing or
        # all-zero weights otherwise flow into different normalizers and skip rules on
        # different code paths and give silently different answers.
        _check_weight_values(Tables.getcolumn(Tables.columns(data), weights), "weights column :$weights")
    elseif weights isa AbstractVector
        # Check that weights vector has right length
        n_data = Tables.rowaccess(data) ? length(Tables.rows(data)) : length(first(Tables.columns(data)))
        if length(weights) != n_data
            throw(ArgumentError("weights vector length ($(length(weights))) must match number of observations ($n_data)"))
        end
        _check_weight_values(weights, "weights vector")
    else
        throw(ArgumentError("weights must be :model, :none, a Symbol (column name) or a Vector{<:Real}"))
    end
end

# Weights must be real, finite, non-negative, and not all zero.
function _check_weight_values(w, label)
    if any(ismissing, w)
        throw(ArgumentError("$label contains missing values; drop those rows or give them weight 0"))
    end
    if !(eltype(w) <: Real) && !all(x -> x isa Real, w)
        throw(ArgumentError("$label must be numeric, got element type $(eltype(w))"))
    end
    if !all(isfinite, w)
        throw(ArgumentError("$label contains NaN or Inf"))
    end
    if !all(x -> x >= 0, w)
        throw(ArgumentError("all weights must be non-negative ($label has negative values)"))
    end
    if all(iszero, w)
        throw(ArgumentError("at least some weights must be positive ($label is all zero)"))
    end
    return nothing
end

# Note: _convert_numeric_to_float64() is defined in core/data_conversion.jl
