# profile/core.jl
# Main profile_margins() function with reference grid approach

"""
    profile_margins(model, data, reference_grid; type=:effects, vars=nothing, scale=:response, backend=:ad, measure=:effect, contrasts=:baseline, alpha=0.05, vcov=GLM.vcov, interval=:confidence, df=nothing, offset=nothing, warn=true) -> Union{EffectsResult, PredictionsResult}

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at representative points or user-specified scenarios,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).
It provides marginal effects at the mean (MEM), marginal effects at representative values (MER),
or adjusted predictions at specific profiles (APM/APR).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame specifying covariate combinations for analysis
  - Use reference grid builders: `means_grid(data)`, `balanced_grid(data; vars...)`, `quantile_grid(data; vars...)`
  - Or use pure grid construction: `cartesian_grid(vars...)` (automatically completed with typical values)
  - Or provide custom DataFrame with desired covariate combinations
  - **Note**: Missing model variables are automatically completed with typical values internally

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles (MEM/MER): derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles (APM/APR): fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
  - Only required when `type=:effects`
  - Defaults to all explanatory variables from the model formula (not all data columns)
  - Only variables that appear in both the model specification and data are considered
  - Extra columns in data that aren't in the model are automatically ignored
- `scale::Symbol=:response`: Target scale for computation
  - `:response` - Response scale (default, applies inverse link function)  
  - `:link` - Linear predictor scale (link scale)
- `backend::Symbol=:ad`: Computational backend
  - `:ad` - Automatic differentiation (higher accuracy, small memory cost)
  - `:fd` - Finite differences (zero allocation, production-ready)
- `measure::Symbol=:effect`: Effect measure (only for `type=:effects`)
  - `:effect` - Marginal effects (default, current behavior)
  - `:elasticity` - Elasticities (percent change in y for percent change in x)
  - `:semielasticity_dyex` - Semielasticity d(y)/d(ln x) (change in y for percent change in x)
  - `:semielasticity_eydx` - Semielasticity d(ln y)/dx (percent change in y for unit change in x)
- `contrasts::Symbol=:baseline`: Contrast type for categorical variables
  - `:baseline` - Compare each level to reference level
  - `:pairwise` - All pairwise comparisons between levels
- `alpha::Real=0.05`: Type I error rate α for confidence intervals (confidence level = 1-α), in (0, 1)
  - When specified, `ci_lower` and `ci_upper` columns are added to DataFrame output
- `vcov=GLM.vcov`: Covariance matrix function for standard errors
  - `GLM.vcov` - Model-based covariance matrix (default)
  - Custom function or CovarianceMatrices estimator for robust/clustered standard errors, or a covariance
    matrix itself (see `population_margins`)
- `df=nothing`: degrees of freedom of the t reference distribution, overriding the model-class
  default (`Inf` for z); see `population_margins`, *Inference*.
- `offset=nothing`: the η offset at each reference-grid row, for a model with an offset
  (`glm(...; offset=log.(exposure))`, a GLMM's `offset=`): a number for every row or one value
  per grid row, on the scale of the model's offset (e.g. `offset=log(1000)` for an exposure of
  1000). An `offset` column of the grid does the same. See *Offsets*.
- `warn=true`: warn (once per call) when a numeric model column looks like an exact transform of
  a grid or focal variable but is a separate column of the data (see *Derived columns*);
  `warn=false` silences it.

# Derived columns
The grid sets variables, and the formula is evaluated at each grid row, so terms the formula
computes (`x^2`, `log(x)`, `x & z`) follow the grid values. A column computed outside the formula
(a precomputed `x_sq`, a spline or polynomial basis) is a separate variable: unless the grid sets
it consistently, it is held at its typical value (or, for a basis, the basis is held fixed). Write
transforms in the formula. See `population_margins`, *Derived columns*, and the "Derived columns"
section of the Population Scenarios documentation.

# Offsets
For a model with an offset (η = x'β + offset), predictions and response-scale effects depend on
the offset, and a reference-grid row has none of its own. `profile_margins` then requires it: the
`offset` keyword, an `offset` column in the grid, or (FixedEffectModels, whose offset is the sum
of the absorbed fixed effects) grid columns setting every fixed-effect variable, whose estimated
fixed effects are looked up. Otherwise it is an `ArgumentError`; the offset is never taken as 0,
which for an exposure model would predict the rate per unit of exposure. Link-scale and
identity-link `:effect` (and `:semielasticity_dyex`) results do not depend on the offset and do
not need it.

# Weights
`profile_margins` evaluates the grid it is given; it has no `weights` keyword. The values it fills
in for model variables the grid leaves out (means of continuous variables, shares of `true` for
Bool variables, frequency mixtures for categorical variables, and the frequency mixture that
replaces a data-only builder's typical value of a variable the model treats as categorical) are
weighted by the model's estimation weights, as Stata's `margins` uses the estimation weights for
`at()` values it computes: `complete_reference_grid(grid, model, data)`'s default `weights=:model`.
For an unweighted fit they are unweighted. Results record what the filled-in values used:
`metadata[:weights_source]` is `:model` (and `metadata[:weighted]` `true`) when model weights were
applied to a filled-in value, `:none` otherwise (an unweighted fit, a grid that sets every model
variable, or a model class whose weights cannot be read). To fill in unweighted values for a
weighted fit (Stata's `noweights`, and the 2.x behaviour), complete the grid yourself:
`profile_margins(m, data, complete_reference_grid(grid, m, data; weights=:none))`.

The grid builders cannot see the model, so their values are unweighted unless you pass
`weights=`: with a weighted fit, `means_grid(data)` holds unweighted means. To evaluate at the
weighted means (Stata's `atmeans`) write `means_grid(data; weights=:w)`, or give only the
variables you set and let `profile_margins` fill in the rest (`cartesian_grid(x=[0, 1])`). See
"Typical values" in the Reference Grids documentation and the Weights page.

# Rank-deficient fits
A coefficient the fit dropped as collinear is treated as fixed at 0, and a quantity that depends
only on dropped coefficients has a NaN standard error, with one warning per call; see
`population_margins`, *Rank-deficient fits*.

# Returns
`EffectsResult` or `PredictionsResult` (estimates, delta-method standard errors, gradients, the
grid columns of each row `result.context`, the inference `result.inference` (covariance matrix,
df, `alpha`) and metadata). `DataFrame(result)` columns:
- effects: `<grid vars…> type variable contrast estimate se statistic df p_value ci_lower ci_upper n`
- predictions: `<grid vars…> type estimate se statistic df p_value ci_lower ci_upper n`
Grid variables keep their bare names and the grid's column order. `statistic = estimate / se`, referred to t(`df`):
`df = dof_residual(model)` for a linear model (with any `vcov`) and a FixedEffectModels fit,
`Inf` (z) otherwise, unless `df=` is given (see `population_margins`, *Inference*).

# Statistical Notes
- Standard errors computed via delta method using full model covariance matrix
- Categorical variables use baseline contrasts (each level minus reference level) at each profile
- Profile approach enables interpretation at specific, meaningful covariate combinations
- More efficient than population approach when analyzing specific scenarios

# Examples
```julia
# Effects at sample means (MEM) - most common case
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1, :x2])
DataFrame(result)  # Convert to DataFrame with profile information

# Elasticities at sample means
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1], measure=:elasticity)

# Effects at specific scenarios (MER) using cartesian grid
result = profile_margins(model, data, cartesian_grid(x1=[0, 1], income=[25000, 50000]); 
                        type=:effects, vars=[:education])

# Semi-elasticities at specific profiles
result = profile_margins(model, data, cartesian_grid(x1=[-1, 0, 1]); 
                        type=:effects, vars=[:x2], measure=:semielasticity_dyex)

# Predictions at the mean (APM)
result = profile_margins(model, data, means_grid(data); type=:predictions)

# Balanced factorial designs
result = profile_margins(model, data, balanced_grid(data; education=:all, region=:all); type=:effects)

# Baseline contrasts for categorical variables (default)
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:baseline)

# Pairwise contrasts for categorical variables
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:pairwise)

# Multiple explicit profiles for complex analysis
reference_grid = DataFrame(
    x1=[0.0, 1.0], 
    x2=[1.0, 0.0], 
    region=["North", "South"]
)
result = profile_margins(model, data, reference_grid; type=:effects)

# High-performance with pre-built reference grid
reference_grid = DataFrame(x1=[0, 1, 2], x2=[10, 20, 30])
result = profile_margins(model, data, reference_grid; type=:predictions)

# Pure Cartesian product for systematic exploration
grid = cartesian_grid(age=[25, 35, 45], education=[12, 16])
result = profile_margins(model, data, grid; type=:effects, vars=[:income], backend=:ad)

# Data-driven range construction
grid = cartesian_grid(
    x1 = collect(range(extrema(data.x1)...; length = 5)),
    x2 = collect(range(extrema(data.x2)...; length = 5))
)
result = profile_margins(model, data, grid; type=:effects)

# Manual completion still available if needed
partial_grid = cartesian_grid(x1=[0, 1, 2])
complete_grid = complete_reference_grid(partial_grid, model, data)
result = profile_margins(model, data, complete_grid; type=:predictions)
```

# Automatic Reference Grid Completion
Missing model variables are automatically completed with typical values:
```julia
# Your data: region = 75% Urban, 25% Rural, treated = 60% true, 40% false
# Model: y ~ x1 + x2 + region + treated

# Specify only variables you care about
grid = cartesian_grid(x1=[0, 1, 2], x2=[10, 20])
result = profile_margins(model, data, grid; type=:effects)

# Internally completed with typical values:
# → x1, x2: your specified values
# → region: frequency-weighted (75% urban, 25% rural) 
# → treated: 0.6 (actual treatment rate)
# But output shows only x1, x2 in at_* columns!

# Effects "at means" for comparison
result = profile_margins(model, data, means_grid(data); type=:effects)
# → Shows all variables: at_x1, at_x2, at_region, at_treated
```

See also: [`population_margins`](@ref) for population-averaged effects and predictions.
"""

"""
    _grid_context(req::MarginsRequest{:profile}, rows::AbstractVector{Int}) -> ResultContext

The context of a profile result: the grid columns shown in the output (`req.grid_vars`: the
user's grid columns, without the response; filled-in typical values are hidden), in the order
of the user's grid, with `rows[i]` the completed-grid row result row `i` was evaluated at.
"""
function _grid_context(req::MarginsRequest{:profile}, rows::AbstractVector{Int})
    grid = req.completed_grid
    shown = Symbol[Symbol(c) for c in names(req.reference_grid) if Symbol(c) in req.grid_vars]
    # grid columns that only completion keeps (none today), after the user's, in grid order
    append!(shown, Symbol[Symbol(c) for c in names(grid) if Symbol(c) in req.grid_vars && !(Symbol(c) in shown)])
    columns = NamedTuple{Tuple(shown)}(Tuple(grid[!, c][rows] for c in shown))
    return ResultContext(columns, fill(:grid, length(shown)))
end

"""
    _run(req::MarginsRequest{:profile}) -> Union{EffectsResult, PredictionsResult}

Compute a profile request: the MEM/APM, with its inference (`result.inference`: the request's
Σ, df and alpha).
"""
_run(req::MarginsRequest{:profile}) = _profile_result(req)

"""
    _profile_result(req::MarginsRequest{:profile}) -> Union{EffectsResult, PredictionsResult}

The MEM/APM of a profile request on its completed reference grid.
`req.offset` is the η offset at each grid row (`nothing` without one).
"""
function _profile_result(req::MarginsRequest{:profile})
    (; model, data_nt, type, vars, scale, backend, measure, contrasts, interval) = req
    completed_reference_grid = req.completed_grid
    original_grid_vars = req.grid_vars
    at_spec = req.reference_grid

    # Build zero-allocation engine with ProfileUsage optimization. It carries no evaluators:
    # effects are evaluated on the reference grid, with evaluators built over the grid.
    # Each call builds its own engine (nothing is cached across calls); Σ is the matrix the
    # request computed from `vcov`, once per call.
    engine = build_engine(ProfileUsage, model, data_nt, isnothing(vars) ? Symbol[] : vars, req.Σ, backend)
    if type === :effects
        # Use the actual working function that already exists from the original utilities.jl
        # Pass original_grid_vars so we know which variables are scenario-defining
        df, G, profile_of_row = _mem_by_grid_blocks(engine, completed_reference_grid, req.offset, original_grid_vars, scale, backend, measure, contrasts, req.vars_explicit)
    
        # Convert symbol terms + profile info to descriptive strings for user display
        df = _convert_profile_terms_to_strings(df)
    
        metadata = _build_metadata(; type, vars, scale, backend, measure, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec)
    
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :profile
        metadata[:n_profiles] = nrow(completed_reference_grid)
        # the weights of the values filled in for model variables the grid left out
        metadata[:weighted] = req.weights_source !== :none
        metadata[:weights_source] = req.weights_source
    
        # Extract raw components from DataFrame  
        estimates = df.estimate
        standard_errors = df.se
        variables = string.(df.variable)  # The "x" in dy/dx
        terms = string.(df.contrast)  # Convert Symbol to String
    
        # Grid columns, one entry per result row: the grid row each result row was evaluated at.
        # (A categorical variable can contribute a different number of contrast rows per grid
        # row; a row pinned to the baseline level contributes none.)
        context = _grid_context(req, profile_of_row)
    
        _mark_not_identified!(standard_errors, G, req.dropped)
        return EffectsResult(estimates, standard_errors, variables, terms, context, G, _request_inference(req, engine), metadata)
    else # :predictions  
        # Reference grid can contain CategoricalMixture objects directly - FormulaCompiler handles them
        df, G = _profile_predictions_by_grid_blocks(engine, completed_reference_grid, scale, req.offset)  # → APM/APR
        metadata = _build_metadata(; type, vars=Symbol[], scale, backend, n_obs=length(first(data_nt)), 
                                  model_type=typeof(model), at_spec=at_spec)
    
        # Add analysis_type for format auto-detection
        metadata[:analysis_type] = :profile
        metadata[:n_profiles] = nrow(completed_reference_grid)
        # the weights of the values filled in for model variables the grid left out
        metadata[:weighted] = req.weights_source !== :none
        metadata[:weights_source] = req.weights_source

        # Interval kind: :confidence (default) or :prediction. For prediction
        # intervals, store the residual variance σ̂² that widens the interval
        # bounds (eligibility to Gaussian/identity is enforced here).
        metadata[:interval] = interval
        if interval === :prediction
            metadata[:resid_var] = _prediction_residual_variance(model)
        end

        # Extract raw components from DataFrame
        estimates = df.estimate
        standard_errors = df.se

        # Grid columns, one entry per result row (one prediction per grid row)
        context = _grid_context(req, repeat(1:nrow(completed_reference_grid), inner=length(estimates) ÷ nrow(completed_reference_grid)))

        _mark_not_identified!(standard_errors, G, req.dropped)
        return PredictionsResult(estimates, standard_errors, context, G, _request_inference(req, engine), metadata)
    end
end

"""
    profile_margins(model, data, reference_grid; type=:effects, vars=nothing, scale=:response, backend=:ad, measure=:effect, contrasts=:baseline, alpha=0.05, vcov=GLM.vcov, interval=:confidence, df=nothing, offset=nothing, warn=true) -> Union{EffectsResult, PredictionsResult}

Compute profile marginal effects or adjusted predictions at specific covariate combinations.

This function evaluates effects/predictions at user-specified scenarios using reference grids,
implementing the "Profile" approach from the 2×2 framework (Population vs Profile × Effects vs Predictions).

# Arguments
- `model`: Fitted statistical model supporting `coef()` and `vcov()` methods
- `data`: Data table (DataFrame, NamedTuple, or any Tables.jl-compatible format)
- `reference_grid`: DataFrame specifying covariate combinations for analysis
  - Use reference grid builders: `means_grid(data)`, `balanced_grid(data; vars...)`, `quantile_grid(data; vars...)`
  - Or use pure grid construction: `cartesian_grid(vars...)` (automatically completed with typical values)
  - Or provide custom DataFrame with desired covariate combinations
  - **Note**: Missing model variables are automatically completed with typical values internally

# Keyword Arguments
- `type::Symbol=:effects`: Analysis type
  - `:effects` - Marginal Effects at profiles: derivatives/contrasts at specific points
  - `:predictions` - Adjusted Predictions at profiles: fitted values at specific points
- `vars=nothing`: Variables for effects analysis (Symbol, Vector{Symbol}, or :all_continuous)
  - Defaults to all explanatory variables from the model formula (not all data columns)
  - Only variables that appear in both the model specification and data are considered
  - Extra columns in data that aren't in the model are automatically ignored
- `scale::Symbol=:response`: Target scale (:response or :link)
- `backend::Symbol=:ad`: Computational backend (:ad or :fd)
- `measure::Symbol=:effect`: Effect measure (:effect, :elasticity, :semielasticity_dyex, :semielasticity_eydx)
  - Measures other than `:effect` apply to continuous variables only: naming a
    discrete variable in `vars` is an error, and auto-detected discrete variables
    are skipped with an informational message
- `contrasts::Symbol=:baseline`: Contrast type for categorical variables (:baseline or :pairwise)
- `alpha::Real=0.05`: Type I error rate α for confidence intervals (confidence level = 1-α), in (0, 1)
- `vcov=GLM.vcov`: Covariance matrix function for standard errors
- `interval::Symbol=:confidence`: Interval kind for `type=:predictions`
  - `:confidence` - interval for the mean prediction (default): `ŷ ± z·√(SE²)`
  - `:prediction` - interval for a single new observation, adding residual variance:
    `ŷ ± z·√(SE² + σ̂²)`. Only supported for Gaussian/identity-link (linear) models;
    errors for GLMs (no additive response-scale residual variance) and for
    `type=:effects`. Matches R's `predict(lm, interval="prediction")`. With a weighted
    fit (`lm(...; wts=w)`), the interval is for a new observation of weight 1, with GLM.jl's
    frequency-weight σ̂² = Σwᵢrᵢ²/(Σwᵢ − p) and df Σwᵢ − p (the convention of `vcov(model)`):
    R's `predict.lm` on the data with each row repeated wᵢ times, not R's `lm(..., weights=w)`
    (precision weights, n − p). A weighted fit warns once per session.
- `df=nothing`: degrees of freedom of the t reference distribution, overriding the model-class
  default (`Inf` for z); see `population_margins`, *Inference*.
- `offset=nothing`: the η offset at each reference-grid row, for a model with an offset
  (`glm(...; offset=log.(exposure))`, a GLMM's `offset=`): a number for every row or one value
  per grid row, on the scale of the model's offset (e.g. `offset=log(1000)` for an exposure of
  1000). An `offset` column of the grid does the same. See *Offsets*.
- `warn=true`: warn (once per call) when a numeric model column looks like an exact transform of
  a grid or focal variable but is a separate column of the data (see *Derived columns*);
  `warn=false` silences it.

# Derived columns
The grid sets variables, and the formula is evaluated at each grid row, so terms the formula
computes (`x^2`, `log(x)`, `x & z`) follow the grid values. A column computed outside the formula
(a precomputed `x_sq`, a spline or polynomial basis) is a separate variable: unless the grid sets
it consistently, it is held at its typical value (or, for a basis, the basis is held fixed). Write
transforms in the formula. See `population_margins`, *Derived columns*, and the "Derived columns"
section of the Population Scenarios documentation.

# Offsets
For a model with an offset (η = x'β + offset), predictions and response-scale effects depend on
the offset, and a reference-grid row has none of its own. `profile_margins` then requires it: the
`offset` keyword, an `offset` column in the grid, or (FixedEffectModels, whose offset is the sum
of the absorbed fixed effects) grid columns setting every fixed-effect variable, whose estimated
fixed effects are looked up. Otherwise it is an `ArgumentError`; the offset is never taken as 0,
which for an exposure model would predict the rate per unit of exposure. Link-scale and
identity-link `:effect` (and `:semielasticity_dyex`) results do not depend on the offset and do
not need it.

# Weights
`profile_margins` evaluates the grid it is given; it has no `weights` keyword. The values it fills
in for model variables the grid leaves out (means of continuous variables, shares of `true` for
Bool variables, frequency mixtures for categorical variables, and the frequency mixture that
replaces a data-only builder's typical value of a variable the model treats as categorical) are
weighted by the model's estimation weights, as Stata's `margins` uses the estimation weights for
`at()` values it computes: `complete_reference_grid(grid, model, data)`'s default `weights=:model`.
For an unweighted fit they are unweighted. Results record what the filled-in values used:
`metadata[:weights_source]` is `:model` (and `metadata[:weighted]` `true`) when model weights were
applied to a filled-in value, `:none` otherwise (an unweighted fit, a grid that sets every model
variable, or a model class whose weights cannot be read). To fill in unweighted values for a
weighted fit (Stata's `noweights`, and the 2.x behaviour), complete the grid yourself:
`profile_margins(m, data, complete_reference_grid(grid, m, data; weights=:none))`.

The grid builders cannot see the model, so their values are unweighted unless you pass
`weights=`: with a weighted fit, `means_grid(data)` holds unweighted means. To evaluate at the
weighted means (Stata's `atmeans`) write `means_grid(data; weights=:w)`, or give only the
variables you set and let `profile_margins` fill in the rest (`cartesian_grid(x=[0, 1])`). See
"Typical values" in the Reference Grids documentation and the Weights page.

# Examples
```julia
# Effects at sample means (most common case)
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:x1, :x2])

# Balanced factorial designs
result = profile_margins(model, data, balanced_grid(data; education=:all, region=:all); type=:effects)

# Effects at specific scenarios using cartesian grid
result = profile_margins(model, data, cartesian_grid(x1=[0, 1], income=[25000, 50000]); 
                        type=:effects, vars=[:education])

# Predictions using quantile-based grid
result = profile_margins(model, data, quantile_grid(data; age=[0.25, 0.5, 0.75]); type=:predictions)

# Direct reference grid specification (maximum control)
reference_grid = DataFrame(age=[25, 35, 45], education=[12, 16, 20], region=["North", "South", "North"])
result = profile_margins(model, data, reference_grid; type=:effects, vars=[:income])

# Baseline contrasts for categorical variables (default behavior)
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:baseline)

# Pairwise contrasts for categorical variables
result = profile_margins(model, data, means_grid(data); type=:effects, vars=[:education], contrasts=:pairwise)

# Complex scenarios with categorical mixtures
using Margins: mix
reference_grid = DataFrame(x1=[0, 1, 2], categorical_var=[mix("A" => 0.7, "B" => 0.3), "A", "B"])
result = profile_margins(model, data, reference_grid; type=:predictions)
```
"""
function profile_margins(
    model, data, reference_grid::DataFrame;
    type::Symbol=:effects, vars=nothing, scale::Symbol=:response,
    backend::Symbol=:ad, measure::Symbol=:effect,
    contrasts::Symbol=:baseline,
    alpha::Real=0.05, vcov=GLM.vcov,
    interval::Symbol=:confidence, df=nothing, offset=nothing, warn::Bool=true, ci_alpha=nothing,
)
    _reject_ci_alpha(ci_alpha)
    # Every option and the reference grid are validated, and `vcov` evaluated, once, here.
    # `profile_margins` takes no weights (see "Weights" above).
    req = _normalize_request(model, data; type, vars, scale, backend, reference_grid, measure,
                             contrasts, alpha, vcov, weights=:none, interval, df, offset, warn)
    return _run(req)
end


# =============================================================================
# Reference grids whose categorical columns mix discrete levels and mixtures
# =============================================================================
#
# FormulaCompiler compiles each column either as a discrete categorical or as ONE mixture
# (`is_mixture_column` requires every cell to hold the same mixture), so a grid column that
# holds a level in some rows and a mixture in others — e.g. a `hierarchical_grid` column only
# some combinations specify, whose other cells hold the frequency mixture — cannot be
# evaluated in one pass. Such grids are evaluated in blocks of rows that are homogeneous in
# every such column, and the results are reassembled in grid-row order (the order a single
# pass produces: profile-major).

_is_mixture_cell(v) = v isa CategoricalMixture

# Row blocks (each a sorted Vector{Int}) to evaluate separately, or `nothing` when the grid
# can be evaluated in one pass.
function _heterogeneous_grid_blocks(grid::DataFrame)
    hetero = Symbol[]
    for c in propertynames(grid)
        col = grid[!, c]
        any(_is_mixture_cell, col) || continue
        first_m = first(col)
        homogeneous = all(v -> _is_mixture_cell(v) && v.levels == first_m.levels && v.weights == first_m.weights, col)
        homogeneous || push!(hetero, c)
    end
    isempty(hetero) && return nothing
    signature(i) = Tuple(_is_mixture_cell(grid[i, c]) ? (grid[i, c].levels, grid[i, c].weights) : nothing for c in hetero)
    blocks = Dict{Any, Vector{Int}}()
    order = Any[]
    for i in 1:nrow(grid)
        sig = signature(i)
        haskey(blocks, sig) || (blocks[sig] = Int[]; push!(order, sig))
        push!(blocks[sig], i)
    end
    return [blocks[sig] for sig in order]
end

# The sub-grid of `rows`, with every column narrowed to its cells' common element type
_grid_block(grid::DataFrame, rows::Vector{Int}) =
    DataFrame([c => identity.(grid[rows, c]) for c in propertynames(grid)])

# The offset at a block's rows
_block_offset(::Nothing, rows) = nothing
_block_offset(offset::Vector{Float64}, rows) = offset[rows]

function _profile_predictions_by_grid_blocks(engine::MarginsEngine, grid::DataFrame, scale, offset=nothing)
    blocks = _heterogeneous_grid_blocks(grid)
    isnothing(blocks) && return _profile_predictions(engine, grid, scale, offset)
    parts = [_profile_predictions(engine, _grid_block(grid, rows), scale, _block_offset(offset, rows)) for rows in blocks]
    perm = sortperm(reduce(vcat, blocks))  # one result row per grid row
    df = reduce((a, b) -> vcat(a, b; cols=:union), first.(parts))[perm, :]
    G = reduce(vcat, last.(parts))[perm, :]
    # The `at_*` columns hold each block's narrowed cells; restore the grid's own columns
    for (c, col) in pairs(eachcol(grid))
        df[!, Symbol("at_$(c)")] = col
    end
    return df, G
end

function _mem_by_grid_blocks(engine::MarginsEngine, grid::DataFrame, offset, args...)
    blocks = _heterogeneous_grid_blocks(grid)
    isnothing(blocks) && return _mem_continuous_and_categorical(engine, grid, offset, args...)
    parts = [_mem_continuous_and_categorical(engine, _grid_block(grid, rows), _block_offset(offset, rows), args...) for rows in blocks]
    profile_of_row = reduce(vcat, [rows[p[3]] for (rows, p) in zip(blocks, parts)])
    perm = sortperm(profile_of_row; alg=Base.Sort.DEFAULT_STABLE)  # keeps within-profile order
    df = reduce(vcat, [p[1] for p in parts])[perm, :]
    G = reduce(vcat, [p[2] for p in parts])[perm, :]
    return df, G, profile_of_row[perm]
end

"""
    _profile_predictions(engine, reference_grid, scale, offset=nothing) -> (DataFrame, Matrix{Float64})

Compute adjusted predictions at profiles (APM/APR) with delta-method standard errors.

This function evaluates predictions at each row of the reference grid, providing
adjusted predictions at the mean (APM) or at representative values (APR). `offset` is the η
offset at each grid row (`nothing` without one).
"""
function _profile_predictions(engine::MarginsEngine{L}, reference_grid, scale, offset=nothing) where L
    n_profiles = nrow(reference_grid)
    n_params = length(engine.β)
    
    predictions = Vector{Float64}(undef, n_profiles)
    G = zeros(n_profiles, n_params)  # One row per profile
    
    # Compile with reference grid (like effects rework)
    # Convert reference grid to NamedTuple format for FormulaCompiler
    data_nt_raw = Tables.columntable(reference_grid)

    # Convert all numeric columns to Float64 for type stability
    _check_model_vars_not_missing(data_nt_raw, engine.model)
    data_nt = _convert_numeric_to_float64(data_nt_raw)

    # Single compilation with complete reference grid (using cache for performance)
    refgrid_compiled = _private_compiled(engine.model, data_nt)

    # Prediction and its gradient at each grid row (prediction kernel)
    predict_rows!(predictions, G, engine.row_buf, refgrid_compiled, data_nt, engine.β,
                  predscale(scale, engine.link), 1:n_profiles, offset)
    
    # Safely use g_buf for SE computation if large enough (performance optimization)
    # NOTE: Buffer fallback is intentional and harmless - engines are often created with
    # minimal buffers (size=1) for flexibility, and gracefully allocate when needed.
    # This is working as designed, not an error.
    if length(engine.g_buf) >= n_profiles
        se_vals = view(engine.g_buf, 1:n_profiles)  # Reuse g_buf if large enough
    else
        @info "Buffer allocation fallback: g_buf too small for $n_profiles profiles (size=$(length(engine.g_buf)))"
        se_vals = Vector{Float64}(undef, n_profiles)  # Fall back to allocation
    end
    for i in 1:n_profiles
        se_vals[i] = sqrt((G[i:i, :] * engine.Σ * G[i:i, :]')[1, 1])
    end
    
    # Create results DataFrame with profile information (no variable/contrast for predictions)
    results = DataFrame()
    # Copy: `se_vals` may be a view of the engine's g_buf, which the next call on this
    # engine overwrites
    results.estimate = predictions
    results.se = collect(se_vals)
    
    # Add profile columns to show which profile each prediction corresponds to
    for (col_name, col_data) in pairs(Tables.columns(reference_grid))
        results[!, Symbol("at_$(col_name)")] = col_data
    end
    
    return results, G
end

"""
    _convert_profile_terms_to_strings(df::DataFrame)

Convert symbol terms + profile descriptions to user-friendly descriptive strings.
Internal computation uses symbols, but user display uses descriptive strings.

`df`` is only arg, since we assume it is filtered to model variables
"""
function _convert_profile_terms_to_strings(df::DataFrame)
    # Clean up contrast descriptions for better display
    descriptive_terms = String[]
    
    for i in 1:nrow(df)
        var = df.variable[i]
        contrast = df.contrast[i]
        
        # Clean up boolean contrasts: convert "value - false" patterns to "true - false"
        if contains(contrast, " - false") && contrast != "true - false"
            # This is a boolean variable with a numeric mean - standardize to "true - false"
            cleaned_contrast = "true - false"
        elseif contrast == "dy/dx"
            # Keep dy/dx as-is for continuous variables
            cleaned_contrast = contrast
        else
            # Keep other contrasts as-is (for future categorical support)
            cleaned_contrast = contrast
        end
        
        push!(descriptive_terms, cleaned_contrast)
    end
    
    # Create new DataFrame with clean contrast terms
    result_df = DataFrame(
        variable = df.variable,  # Include the "x" in dy/dx 
        contrast = descriptive_terms,
        estimate = df.estimate,
        se = df.se
    )
    
    return result_df
end

"""
    _mem_continuous_and_categorical(engine, reference_grid, scale, backend, measure, contrasts) -> (DataFrame, Matrix)

Profile Effects (MEM) properly decomposed from the original working implementation.
This is the correct decomposition that should have been completed during BREAKUP.md.
"""
function _mem_continuous_and_categorical(engine::MarginsEngine{L}, reference_grid, offset, original_grid_vars::Set{Symbol}, scale::Symbol, backend::Symbol, measure::Symbol, contrasts::Symbol=:baseline, vars_explicit::Bool=false) where L
    # Use the decomposed modular approach - break down the original 200-line function

    # Step 1: Calculate dimensions (extracted from original lines 1262-1286)
    n_profiles = nrow(reference_grid)
    continuous_vars = continuous_variables(engine.compiled, engine.data_nt)
    requested_vars = engine.vars
    continuous_requested = [v for v in requested_vars if v ∈ continuous_vars]
    categorical_requested_raw = [v for v in requested_vars if v ∉ continuous_vars]

    # Filter out variables that were ORIGINALLY in the reference grid (before completion)
    # Variables in original reference grid define the scenarios (conditioning variables), not analysis targets
    # Variables added by completion with typical values/mixtures ARE analysis targets
    refgrid_vars = original_grid_vars

    # Decide which categorical variables are analysis targets, and how each is contrasted.
    #
    # A categorical is contrasted only when the caller NAMED it in `vars`. When `vars` is
    # auto-detected, categoricals in the grid stay scenario-defining and categoricals absent
    # from the grid stay backdrop mixtures — the historical behaviour, which
    # test/regression/test_bool_mixture_skip.jl pins down.
    #
    # An explicitly requested categorical is contrasted in one of two modes:
    #   :row_specific — the grid pins it to a discrete level, so each grid row contributes
    #                   exactly one contrast, that row's level against the baseline (the
    #                   row-specific baseline contrast described in CLAUDE.md). Rows already
    #                   at the baseline contribute nothing.
    #   :enumerated   — the grid does not pin it (completion filled a mixture), so each grid
    #                   row contributes the full set of contrasts from the original data's
    #                   levels, holding the row's other variables fixed. This is the usual
    #                   "marginal effect of a factor at this profile".
    # `contrasts=:pairwise` always enumerates: both endpoints of a pair are set explicitly,
    # so a pinned level carries no additional information.
    refgrid_col_table = Tables.columntable(reference_grid)
    categorical_requested = Symbol[]
    categorical_modes = Dict{Symbol, Symbol}()
    for var in categorical_requested_raw
        refgrid_col = getproperty(refgrid_col_table, var)
        orig_col = getproperty(engine.data_nt, var)

        # A mixture column (explicit CategoricalMixture, or a Bool filled with a frequency)
        # does not pin the variable to a level. A Bool written as exact 0.0/1.0 does pin it.
        is_mixture = refgrid_col isa AbstractVector{<:CategoricalMixture} ||
                     (eltype(orig_col) <: Bool && !(eltype(refgrid_col) <: Bool) &&
                      !all(v -> v == 0 || v == 1, refgrid_col))
        pinned = !is_mixture && var ∈ refgrid_vars

        if !vars_explicit
            if is_mixture
                @info "Skipping contrasts for variable $var: specified as mixture in reference grid (use discrete levels if contrasts desired)"
            else
                @debug "Skipping contrasts for variable $var: specified in reference grid (scenario-defining variable)"
            end
            continue
        end

        push!(categorical_requested, var)
        categorical_modes[var] = (pinned && contrasts === :baseline) ? :row_specific : :enumerated
    end

    # Contrast pairs per enumerated variable, taken from the ORIGINAL data so that every
    # level the model knows about is represented (the grid may hold only one). This is the
    # same enumeration the population path uses, so contrast labels agree across the 2x2.
    n_orig = length(first(engine.data_nt))
    categorical_pairs = Dict{Symbol, Vector}()
    for var in categorical_requested
        categorical_pairs[var] = generate_contrast_pairs(
            getproperty(engine.data_nt, var), 1:n_orig, contrasts, engine.model, var, engine.data_nt
        )
    end

    # Rows contributed per grid row, per variable
    categorical_row_counts = Dict{Symbol, Vector{Int}}()
    for var in categorical_requested
        if categorical_modes[var] === :row_specific
            # Resolve levels the same way the handler does, so the count matches exactly
            # (a grid may write a Bool as 0.0/1.0, or a level as a bare String)
            baseline = _profile_baseline_level(engine, var)
            categorical_row_counts[var] = [
                _match_pool_level(engine, var, reference_grid[i, var]) == baseline ? 0 : 1
                for i in 1:n_profiles
            ]
        else
            categorical_row_counts[var] = fill(length(categorical_pairs[var]), n_profiles)
        end
    end

    # NOTE: DO NOT filter continuous variables from reference grid
    # Continuous variables in the grid just specify WHERE to evaluate the derivative
    # We still want to compute marginal effects for them at those points
    # Only categorical variables are filtered (to avoid contrasts for scenario-defining categories)

    # Calculate total terms: one row per continuous variable per profile, plus the exact
    # number of contrast rows each categorical variable contributes (see above)
    total_terms = n_profiles * length(continuous_requested)
    for var in categorical_requested
        total_terms += sum(categorical_row_counts[var])
    end

    # Step 2: Pre-allocate using decomposed function (extracted from original lines 1288-1296)
    results, G = _build_profile_results_structure(total_terms, length(engine.β))

    # Step 3: Prepare reference grid compilation (extracted from original lines 1297-1303)
    refgrid_data = Tables.columntable(reference_grid)
    refgrid_compiled = _private_compiled(engine.model, refgrid_data)
    refgrid_de = _build_refgrid_derivative_evaluator(refgrid_compiled, refgrid_data, continuous_requested, engine, backend)

    # Step 3.5: Build the ContrastEvaluators used for categorical contrasts at grid rows.
    #
    # An evaluator needs the contrasted column to carry the model's FULL level pool (the
    # grid may pin one level, or hold a mixture that is not a level at all), while every
    # other column keeps its grid value — the contrast is evaluated at the grid row's
    # covariates. So the contrasted column is replaced by a grid-length column drawn from
    # the original data's pool. Its stored values are irrelevant: `contrast_modelrow!` sets
    # both endpoints of the contrast explicitly.
    #
    # One evaluator per contrasted variable: when two or more categoricals are requested,
    # contrasting one of them must hold the OTHERS at the grid row's values (a level or a
    # mixture). A single evaluator with every requested column replaced by a placeholder
    # evaluated each contrast with the other categoricals at their placeholder (first
    # level / `false`) instead of their grid values, which gave wrong estimates and SEs.
    #
    # Building it grid-length matters: splicing in the original column (length n_obs) breaks
    # as soon as the grid has more rows than the data.
    refgrid_contrasts = Dict{Symbol, Any}()
    for var in categorical_requested
        hybrid_data_dict = Dict{Symbol, Any}()
        for (k, v) in pairs(refgrid_data)
            hybrid_data_dict[k] = k === var ?
                _pooled_column_for_grid(getproperty(engine.data_nt, k), n_profiles) : v
        end
        hybrid_data = NamedTuple(hybrid_data_dict)

        # Recompile against the hybrid schema: replacing a mixture column with a discrete
        # pooled column changes the column's semantic type, so refgrid_compiled (built for
        # the mixture) does not apply.
        hybrid_compiled = _private_compiled(engine.model, hybrid_data)
        refgrid_contrasts[var] = contrastevaluator(hybrid_compiled, hybrid_data, [var])
    end

    # Step 4: Pre-allocate profile computation buffers (PHASE 3 FIX)
    # These buffers are reused for ALL profiles to achieve 0 allocations
    n_params = length(engine.β)

    # Continuous variables: accumulator workspace and per-profile results for all of them
    n_cont = length(continuous_requested)
    cont_ws = isnothing(refgrid_de) ? MEWorkspace() : MEWorkspace(refgrid_de, n_cont)
    cont_vals = Vector{Float64}(undef, n_cont)
    cont_grads = Matrix{Float64}(undef, n_cont, n_params)
    cont_idx = collect(1:n_cont)   # derivative_variables(refgrid_de) == continuous_requested
    cont_measure = _measure_type(measure)
    cont_xcols = (!_uses_x(cont_measure) || n_cont == 0) ? () :
                 _float_columns(refgrid_data, continuous_requested)
    s_scale = predscale(scale, engine.link)

    # Categorical variable buffers
    contrast_buf = Vector{Float64}(undef, length(refgrid_compiled))

    # Step 5: Process each profile (extracted from original lines 1305-1446)
    # Process only variables that should be computed (continuous + non-mixture categorical)
    # profile_of_row records which grid row each result row was evaluated at; categorical
    # contrasts can contribute a variable number of rows per grid row.
    profile_of_row = Vector{Int}(undef, total_terms)
    row_idx = 1
    for profile_idx in 1:n_profiles
        first_row_of_profile = row_idx
        # Continuous variables, all at once at this grid row
        if !isnothing(refgrid_de)
            row_idx = _process_profile_continuous_variables!(results, G, row_idx, engine,
                                                             continuous_requested, refgrid_de,
                                                             cont_ws, cont_vals, cont_grads,
                                                             cont_idx, cont_xcols, reference_grid,
                                                             profile_idx, s_scale, cont_measure, offset)
        end

        # Process requested categorical variables
        for var in categorical_requested
            row_idx = _process_profile_categorical_variable!(results, G, row_idx, engine, var,
                                                           refgrid_contrasts[var], reference_grid, profile_idx,
                                                           scale, categorical_modes[var],
                                                           categorical_pairs[var],
                                                           contrast_buf, offset)
        end

        if row_idx > first_row_of_profile
            profile_of_row[first_row_of_profile:(row_idx - 1)] .= profile_idx
        end
    end

    # Check if we filled all rows correctly
    actual_rows_filled = row_idx - 1
    if actual_rows_filled != total_terms
        n_cont = length(continuous_requested)
        n_cat = length(categorical_requested)
        @warn "Profile results size mismatch" total_terms actual_rows_filled n_profiles n_cont n_cat contrasts continuous_requested categorical_requested
        # Trim to actual size to avoid UndefRefError
        results = results[1:actual_rows_filled, :]
        G = G[1:actual_rows_filled, :]
        profile_of_row = profile_of_row[1:actual_rows_filled]
    end

    return (results, G, profile_of_row)
end

# ==============================================================================
# DECOMPOSED HELPER FUNCTIONS (extracted from original _mem_continuous_and_categorical)
# ==============================================================================


"""
    _build_profile_results_structure(total_terms, n_params) -> (DataFrame, Matrix)

Build results DataFrame and gradient matrix for profile margins.
Pre-allocates all vectors to avoid push! allocations.
Extracted from original lines 1288-1296.
"""
function _build_profile_results_structure(total_terms, n_params)
    # Pre-allocate all vectors with known size (avoids push! allocations)
    results = DataFrame(
        variable = Vector{String}(undef, total_terms),
        contrast = Vector{String}(undef, total_terms),
        estimate = Vector{Float64}(undef, total_terms),
        se = Vector{Float64}(undef, total_terms),
        profile_desc = Vector{Any}(undef, total_terms)  # Use Any to handle NamedTuple
    )
    G = Matrix{Float64}(undef, total_terms, n_params)
    return (results, G)
end

"""
    _build_refgrid_derivative_evaluator(refgrid_compiled, refgrid_data, continuous_requested, engine)

Derivative evaluator over the reference grid for the requested continuous variables, or
`nothing` when there are none.
"""
function _build_refgrid_derivative_evaluator(refgrid_compiled, refgrid_data, continuous_requested, engine, backend::Symbol)
    if !isempty(continuous_requested)
        # Differentiate with respect to the requested variables only (FormulaCompiler's
        # derivativeevaluator dispatches to the backend). The other continuous variables stay
        # at their grid values either way; this only narrows the Jacobian (and, for AD, the
        # dual-number chunk), not its entries.
        return derivativeevaluator(backend, refgrid_compiled, refgrid_data, continuous_requested)
    else
        return nothing
    end
end

"""
    _process_profile_continuous_variables!(results, G, row_idx, engine, continuous_requested,
                                           refgrid_de, ws, vals, grads, var_idx, xcols,
                                           reference_grid, profile_idx, s::PredScale, measure)
        -> next_row_idx

Effects (or measures) of all requested continuous variables at grid row `profile_idx`, with
the population accumulator over the single row `profile_idx:profile_idx` (weight 1): the
unit quantity at the grid row and its delta-method gradient, including the quotient-rule
term for measures that divide by ŷ. Writes one result row per variable.
"""
function _process_profile_continuous_variables!(results, G, row_idx, engine, continuous_requested,
                                                refgrid_de, ws::MEWorkspace,
                                                vals::Vector{Float64}, grads::Matrix{Float64},
                                                var_idx::Vector{Int}, xcols::XColumns,
                                                reference_grid, profile_idx::Int,
                                                s::PredScale, measure::Measure, offset=nothing)
    fill!(vals, 0.0)
    fill!(grads, 0.0)
    accumulate_continuous!(vals, grads, ws, refgrid_de, engine.β, s, measure, var_idx, xcols,
                           profile_idx:profile_idx, nothing, 1.0, offset)
    profile_nt = _profile_display_row(reference_grid, profile_idx)
    for (k, var) in enumerate(continuous_requested)
        engine.gβ_accumulator .= view(grads, k, :)
        results.variable[row_idx] = string(var)
        results.contrast[row_idx] = "dy/dx"
        results.estimate[row_idx] = vals[k]
        results.se[row_idx] = compute_se_only(engine.gβ_accumulator, engine.Σ)
        results.profile_desc[row_idx] = profile_nt
        G[row_idx, :] = engine.gβ_accumulator
        row_idx += 1
    end
    return row_idx
end

"""
    _process_profile_categorical_variable!(results, G, row_idx, engine, var, refgrid_contrast,
                                         refgrid_compiled, refgrid_data, reference_grid, profile_idx,
                                         scale, backend, contrasts) -> next_row_idx

Process a single categorical variable at a single profile point.
Phase 4 Migration: Uses ContrastEvaluator kernel (same as population path).
Extracted from original lines 1377-1444.
"""
function _process_profile_categorical_variable!(results, G, row_idx, engine, var,
                                              refgrid_contrast, reference_grid, profile_idx,
                                              scale::Symbol, mode::Symbol, contrast_pairs,
                                              contrast_buf::Vector{Float64}, offset=nothing)
    # Which contrasts does this grid row contribute?
    #   :row_specific — the row pins a level, so the only contrast is that level against the
    #                   baseline; a row already at the baseline contributes nothing.
    #   :enumerated   — every pair from the original data's levels, evaluated at this row.
    pairs_for_row = if mode === :row_specific
        baseline = _profile_baseline_level(engine, var)
        current = _match_pool_level(engine, var, reference_grid[profile_idx, var])
        current == baseline ? () : ((baseline, current),)
    else
        ((p.level1, p.level2) for p in contrast_pairs)
    end

    profile_nt = _profile_display_row(reference_grid, profile_idx)

    for (from_level, to_level) in pairs_for_row
        # The categorical kernel over the single grid row, on the requested scale; it leaves
        # the parameter gradient in engine.gβ_accumulator.
        (effect, se) = categorical_contrast_ame!(
            contrast_buf, engine.gβ_accumulator,
            refgrid_contrast, var, from_level, to_level,
            engine.β, engine.Σ, predscale(scale, engine.link), profile_idx:profile_idx, nothing,
            offset
        )

        results.variable[row_idx] = string(var)
        results.contrast[row_idx] = "$(to_level) - $(from_level)"
        results.estimate[row_idx] = effect
        results.se[row_idx] = se
        results.profile_desc[row_idx] = profile_nt
        G[row_idx, :] = engine.gβ_accumulator

        row_idx += 1
    end

    return row_idx
end

"""
    _profile_display_row(reference_grid, profile_idx) -> NamedTuple

Grid row rendered for display, with mixtures stringified.
"""
function _profile_display_row(reference_grid, profile_idx)
    d = Dict{Symbol,Any}()
    for k in names(reference_grid)
        val = reference_grid[profile_idx, k]
        d[Symbol(k)] = val isa CategoricalMixture ? string(val) : val
    end
    return NamedTuple(d)
end

"""
    _profile_baseline_level(engine, var)

The model's baseline level for `var`, as a value the ContrastEvaluator accepts (a
`CategoricalValue` from the original pool, or a `Bool`).
"""
function _profile_baseline_level(engine, var::Symbol)
    orig_col = getproperty(engine.data_nt, var)
    baseline_str = string(_baseline_level(engine.model, var, engine.data_nt))
    return _match_pool_level(engine, var, baseline_str)
end

"""
    _match_pool_level(engine, var, value)

Resolve `value` (a level, a string, or a Bool) to the corresponding level of `var` in the
fitted model (a raw level value), which is what `contrast_modelrow!` accepts.
"""
function _match_pool_level(engine, var::Symbol, value)
    orig_col = getproperty(engine.data_nt, var)
    if eltype(orig_col) <: Bool
        # Grid completion may store a Bool as 0.0/1.0, and a user grid may hold "true"
        value isa Bool && return value
        value isa Real && return !iszero(value)
        lowered = lowercase(string(value))
        lowered in ("true", "t", "1", "1.0") && return true
        lowered in ("false", "f", "0", "0.0") && return false
        throw(ArgumentError("Cannot interpret \"$value\" as a level of the boolean variable :$var."))
    end
    # Levels as the fitted model knows them (raw values, model order), which is also the
    # form `generate_contrast_pairs` uses; the ContrastEvaluator resolves them by value.
    target = string(value isa CategoricalMixture ? value : _unwrap_value(value))
    lvls = get(categorical_levels(engine.compiled), var, nothing)
    if isnothing(lvls)
        lvls = CategoricalArrays.unwrap.(levels(orig_col))
    end
    idx = findfirst(l -> string(l) == target, lvls)
    isnothing(idx) && throw(ArgumentError(
        "Level \"$target\" of :$var is not among the levels of the fitted model ($(lvls))."
    ))
    return lvls[idx]
end

"""
    _pooled_column_for_grid(orig_col, n_rows) -> AbstractVector

A length-`n_rows` column carrying `orig_col`'s full level pool, for building a
ContrastEvaluator over a reference grid. The values are placeholders — every contrast sets
both of its endpoints explicitly — but the pool must be complete so that any level of the
model can be requested.
"""
function _pooled_column_for_grid(orig_col, n_rows::Int)
    if eltype(orig_col) <: Bool
        return fill(false, n_rows)
    elseif eltype(orig_col) <: CategoricalArrays.CategoricalValue
        return CategoricalArrays.categorical(
            fill(first(levels(orig_col)), n_rows);
            levels=levels(orig_col), ordered=CategoricalArrays.isordered(orig_col)
        )
    else
        return fill(first(orig_col), n_rows)
    end
end
