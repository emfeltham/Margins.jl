# engine/core.jl - MarginsEngine struct and construction logic

"""
    MarginsUsage

Abstract type for specifying the usage pattern of a MarginsEngine.
Different usage patterns require different buffer sizing strategies for optimal performance.

See also: [`PopulationUsage`](@ref), [`ProfileUsage`](@ref)
"""
abstract type MarginsUsage end

"""
    PopulationUsage <: MarginsUsage

Usage pattern for population margins (AME, AAP).

Optimizes for:
- Minimal memory footprint (g_buf sized only for FormulaCompiler continuous variables)
- Zero allocation fallbacks for FormulaCompiler derivative operations
- Efficient row-wise processing across sample data

Used by `population_margins()` function.
"""
struct PopulationUsage <: MarginsUsage end

"""
    ProfileUsage <: MarginsUsage

Usage pattern for profile margins (MEM, APM, MER, APR).

Optimizes for:
- Larger g_buf to handle multiple profiles without allocation fallbacks
- Zero allocation fallbacks for typical reference grid sizes (up to 100 profiles)
- Efficient SE computation across profile grids

Used by `profile_margins()` function.
"""
struct ProfileUsage <: MarginsUsage end

"""
    MarginsEngine{L<:GLM.Link, U<:MarginsUsage, C<:UnifiedCompiled, E, CE, O}

Zero-allocation engine for marginal effects computation, optimized for specific usage patterns.

Built on FormulaCompiler with usage-specific pre-allocated buffers. Every field read on a hot
path has a concrete type, so code that receives an engine is compiled for its exact
evaluators and makes no dynamic dispatch through them; the one dispatch is at the function
barrier where an entry point first passes the (runtime-built) engine on.

Type Parameters:
- `L<:GLM.Link`: GLM link function type for the underlying model
- `U<:MarginsUsage`: Usage pattern determining buffer sizing strategy
- `C<:UnifiedCompiled`: Compiled formula type
- `E<:Union{Nothing, AbstractDerivativeEvaluator}`: Type of `de`; `Nothing` when the engine has
  no derivative evaluator (no continuous variable requested, profile engines, scenarios/groups)
- `CE<:Union{Nothing, ContrastEvaluator}`: Type of `contrast`; `Nothing` when the engine has no
  contrast evaluator (no categorical variable requested, profile engines, scenarios/groups)
- `O<:Union{Nothing, Vector{Float64}}`: Type of `offset`; `Nothing` without an η offset, so the
  kernels of a model without one are compiled without it

Fields:
- `compiled::C`: Pre-compiled FormulaCompiler formula
- `de::E`: Derivative evaluator over the data rows for the requested continuous variables
- `contrast::CE`: Categorical contrast evaluator over the data rows for the requested categoricals
- `g_buf::Vector{Float64}`: Usage-optimized buffer for marginal effects results
- `gβ_accumulator::Vector{Float64}`: Pre-allocated buffer for AME gradient accumulation
- `row_buf::Vector{Float64}`: Pre-allocated buffer for model row (design matrix row) computations
- `contrast_buf::Vector{Float64}`: Pre-allocated buffer for categorical contrast vectors (Phase 4)
- `continuous_vars::Vector{Symbol}`: Detected continuous variables
- `categorical_vars::Vector{Symbol}`: Detected categorical variables
- `model::Any`: Reference to original model
- `β::Vector{Float64}`: Model coefficients
- `Σ::Matrix{Float64}`: Model covariance matrix
- `link::L`: GLM link function
- `vars::Vector{Symbol}`: Variables for analysis
- `data_nt::NamedTuple`: Reference data for scenarios/refgrids
- `offset::O`: The η offset at each row of `data_nt` (`_offset`), or `nothing`. Population engines
  only; profile engines get `nothing` and their offset at the grid rows is passed to the kernels

`model` and `data_nt` stay abstractly typed: they are read once per call or per variable,
never per row, and parameterizing on them would only multiply compiled specializations.

# Buffer Sizing Strategy
- `PopulationUsage`: Minimal g_buf (sized for FormulaCompiler continuous variables only)
- `ProfileUsage`: Larger g_buf (sized for typical profile counts ~100 to avoid allocation fallbacks)

# Evaluators
Only population engines evaluate `de` and `contrast` on the data rows. Profile engines build
their own evaluators over the reference grid, and the scenarios/groups path builds its own per
context, so their engines carry neither (`build_engine(...; evaluators=false)`).

# Examples
```julia
# Population margins engine (evaluators for the requested variables)
engine = build_engine(PopulationUsage, model, data_nt, [:x1, :x2], Σ, :ad)

# Profile margins engine (no evaluators over the data rows)
engine = build_engine(ProfileUsage, model, data_nt, [:category], Σ, :ad)
```
"""
struct MarginsEngine{L<:GLM.Link, U<:MarginsUsage, C<:UnifiedCompiled,
                     E<:Union{Nothing, AbstractDerivativeEvaluator},
                     CE<:Union{Nothing, ContrastEvaluator},
                     O<:AnyOffset}
    # FormulaCompiler components (pre-compiled)
    compiled::C
    de::E

    # Variable index mapping
    # Pre-computed mapping from variable Symbol to index in derivative_variables(de)
    # Eliminates O(n×m) linear searches in favor of O(1) hash lookups
    var_index_map::Dict{Symbol, Int}

    contrast::CE # Categorical contrast evaluator

    # Usage-optimized buffers (zero runtime allocation)
    g_buf::Vector{Float64}              # Usage-optimized marginal effects results buffer
    gβ_accumulator::Vector{Float64}     # AME gradient accumulation
    row_buf::Vector{Float64}            # Model row (design matrix row) buffer

    # Batch operation buffers (for zero-allocation batch marginal effects)
    batch_ame_values::Vector{Float64}   # Averaged marginal effects accumulator
    batch_gradients::Matrix{Float64}    # Parameter gradients accumulator (n_vars × n_params)
    cont_var_indices_buf::Vector{Int}   # Scratch mapping for continuous variable indices
    xcols_buf::Vector{Vector{Float64}}  # Scratch for the requested variables' data columns (measures)

    # Workspace of the continuous-effects accumulator (Jacobian, model row, A/B gradient
    # accumulators) for `de`; empty when there is no population derivative evaluator
    me_ws::MEWorkspace

    # Categorical effects buffers (Phase 4 zero-allocation optimization)
    contrast_buf::Vector{Float64}           # Contrast vector buffer (length n_params)


    # Cache continuous and categorical variable metadata
    continuous_vars::Vector{Symbol}
    categorical_vars::Vector{Symbol}

    # Model parameters
    model::Any # Reference to original model
    β::Vector{Float64}
    Σ::Matrix{Float64}
    link::L
    vars::Vector{Symbol}
    data_nt::NamedTuple # Reference for scenarios/refgrids
    offset::O           # η offset per row of data_nt, or nothing
    re_average::Union{Nothing, REAveraging}  # random_effects = :marginal: the quadrature over u = z'b
end

"""
    build_engine(usage::Type{U}, model, data_nt, vars, vcov, backend;
                 evaluators::Bool = U <: PopulationUsage, offset = nothing) where {U<:MarginsUsage} -> MarginsEngine

Construct zero-allocation margins engine with usage-specific optimization.

# Arguments
- `usage::Type{<:MarginsUsage}`: Usage pattern (PopulationUsage or ProfileUsage)
- `model`: Fitted statistical model (GLM.jl, MLJ.jl, etc.)
- `data_nt::NamedTuple`: Data in columntable format from Tables.jl
- `vars::Vector{Symbol}`: Variables to analyze for marginal effects
- `vcov`: Covariance matrix Σ (what entry points pass, computed once per call), or an estimator
  evaluated here (a function of the model or a CovarianceMatrices estimator)
- `backend::Symbol`: Derivative backend of `de` (`:ad` or `:fd`)
- `evaluators::Bool`: Build the derivative evaluator `de` (for the requested continuous
  variables) and the contrast evaluator `contrast` (for the requested categoricals) over the
  data rows. Default `true` for `PopulationUsage` only: profile engines evaluate on the
  reference grid with evaluators of their own, and the scenarios/groups path passes `false`
  because it builds its own per context. Without them `de` and `contrast` are `nothing`.
- `offset`: The η offset at each row of `data_nt` (`nothing` or a `Vector{Float64}` of that
  length), read by the population kernels.
- `re_average`: `random_effects = :marginal`: the Gauss–Hermite rule over the random effects
  (`REAveraging`), which makes the response scale an `AveragedResponseScale`; `nothing` otherwise.

# Returns
- `MarginsEngine{L, U, C, E, CE, O}`: Pre-compiled engine whose type records the exact
  evaluator types (`E`, `CE`) and whether there is an offset (`O`)

# Usage-Specific Buffer Sizing Strategy
- **PopulationUsage**: Minimal g_buf sized for FormulaCompiler continuous variables only
  - Optimizes for minimal memory footprint
  - Eliminates allocation fallbacks for derivative computation
  - Perfect for AME/AAP operations across sample data

- **ProfileUsage**: Larger g_buf sized for typical profile counts (~100)  
  - Optimizes for multiple profiles without allocation fallbacks
  - Handles most reference grids without buffer reallocations
  - Perfect for MEM/APM/MER/APR operations across profile grids

# Buffer Management
Pre-allocates three key buffers with usage-specific optimization:
- `g_buf`: Usage-optimized for marginal effects results
- `gβ_accumulator`: For AME gradient accumulation (sized by coefficients)
- `row_buf`: For design matrix row computations (sized by model columns)

When buffer size is insufficient, functions allocate additional memory
to prevent bounds errors while maintaining correctness. An @info message
is logged when buffer allocation fallback occurs for performance visibility.

# Performance Notes
- Formula compilation is cached by FormulaCompiler; the rest of a build is cheap, so every
  Margins call builds its own engine (nothing in it is shared across calls)
- Usage-specific sizing eliminates major allocation bottlenecks
- The engine's type is not inferable from the arguments (it depends on `vars` and
  `backend`), so callers pass the engine through a function barrier (`_ame_calculate`,
  `_mem_by_grid_blocks`, ...) and everything after it is specialized on the exact type

# Examples
```julia
data_nt = Tables.columntable(data)
Σ = GLM.vcov(model)

# Population margins - continuous and categorical variables
engine = build_engine(PopulationUsage, model, data_nt, [:x1, :category], Σ, :ad)

# Profile margins (evaluators are built over the reference grid, not here)
engine = build_engine(ProfileUsage, model, data_nt, [:x1, :category], Σ, :ad)
```
"""
function build_engine(usage::Type{U}, model, data_nt::NamedTuple, vars::Vector{Symbol}, vcov, backend::Symbol;
                      evaluators::Bool = U <: PopulationUsage,
                      offset::AnyOffset = nothing,
                      re_average::Union{Nothing, REAveraging} = nothing) where {U<:MarginsUsage}
    # Input validation (delegated to utilities.jl)
    _validate_variables(data_nt, vars)
    isnothing(offset) || length(offset) == length(first(data_nt)) ||
        throw(DimensionMismatch("offset has $(length(offset)) entries, data has $(length(first(data_nt))) rows"))

    # Compiled formula with a private scratch buffer (cached ops; see `_private_compiled`),
    # so engines never share mutable evaluation state
    compiled = _private_compiled(model, data_nt)
    all_continuous = continuous_variables(compiled, data_nt)

    # Filter continuous and categorical vars from requested vars (zero allocation)
    # IMPORTANT: Only include vars that were REQUESTED, not all model vars
    # Count continuous and categorical separately
    n_continuous = 0
    n_categorical = 0
    for v in vars
        if v ∈ all_continuous
            n_continuous += 1
        else
            n_categorical += 1
        end
    end

    # Pre-allocate with exact sizes
    continuous_vars = Vector{Symbol}(undef, n_continuous)
    categorical_vars = Vector{Symbol}(undef, n_categorical)

    cont_idx = 1
    cat_idx = 1
    for v in vars
        if v ∈ all_continuous
            continuous_vars[cont_idx] = v
            cont_idx += 1
        else
            categorical_vars[cat_idx] = v
            cat_idx += 1
        end
    end

    # Derivative evaluator over the data rows, for the REQUESTED continuous variables only
    # (FormulaCompiler's derivativeevaluator dispatches to the :ad or :fd backend)
    de = (evaluators && !isempty(continuous_vars)) ?
        derivativeevaluator(backend, compiled, data_nt, continuous_vars) : nothing

    # Contrast evaluator for the requested categorical variables (Phase 4)
    contrast = (evaluators && !isempty(categorical_vars)) ?
        contrastevaluator(compiled, data_nt, categorical_vars) : nothing

    # Build variable index map
    # Populates mapping from variable Symbol to index in derivative_variables(de)
    # Eliminates O(n×m) linear searches during marginal effects computation
    var_index_map = if !isnothing(de)
        Dict{Symbol, Int}(var => i for (i, var) in enumerate(derivative_variables(de)))
    else
        # Empty map if no derivative evaluator
        Dict{Symbol, Int}()
    end

    # Pre-allocate buffers with usage-specific optimization
    n_continuous = length(continuous_vars)
    n_coef = length(compiled)
    n_cols = length(compiled)  # Number of columns in design matrix
    
    # Usage-specific g_buf sizing strategy
    if U <: PopulationUsage
        # PopulationUsage: Minimal g_buf sized only for FormulaCompiler continuous variables
        # Optimizes for minimal memory footprint and zero FormulaCompiler allocation fallbacks
        g_buf_size = max(n_continuous, 1)  # At least size 1 to avoid bounds errors
    elseif U <: ProfileUsage
        # ProfileUsage: Larger g_buf sized for typical profile counts (~100)
        # Optimizes for multiple profiles without allocation fallbacks in SE computation
        # NOTE: ProfileUsage needs this size even without derivatives (predictions use g_buf)
        g_buf_size = max(n_continuous, 100)  # Handle up to 100 profiles without fallback
    else
        error("Unknown MarginsUsage type: $U")
    end
    
    g_buf = Vector{Float64}(undef, g_buf_size)
    gβ_accumulator = Vector{Float64}(undef, n_coef)
    row_buf = Vector{Float64}(undef, n_cols)       # Buffer for design matrix rows

    # Initialize batch operation buffers
    # Size for continuous variables that may be requested for batch operations
    batch_ame_values = Vector{Float64}(undef, max(n_continuous, 1))
    batch_gradients = Matrix{Float64}(undef, max(n_continuous, 1), n_coef)

    # Continuous-effects accumulator workspace for `de` (population engines only)
    me_ws = de === nothing ? MEWorkspace() : MEWorkspace(de, n_continuous)

    # Initialize categorical effects buffers (Phase 4 zero-allocation optimization)
    # These buffers enable zero-allocation categorical AME computation
    contrast_buf = Vector{Float64}(undef, n_coef)        # Contrast vector buffer

    
    # Covariance matrix Σ. Entry points pass the matrix they computed once per call
    # (`validate_vcov_parameter`); a function or CovarianceMatrices estimator is evaluated
    # here. Always a fresh copy, never aliasing the caller's array.
    covariance_matrix = _covariance_matrix(model, vcov)

    # Create engine instance, with the evaluator types as explicit parameters
    link = _link(model)
    return MarginsEngine{typeof(link), U, typeof(compiled), typeof(de), typeof(contrast), typeof(offset)}(
        compiled, de, var_index_map, contrast, g_buf, gβ_accumulator, row_buf,
        batch_ame_values, batch_gradients,
        Vector{Int}(undef, max(n_continuous, 1)),  # cont_var_indices_buf
        Vector{Vector{Float64}}(undef, n_continuous),  # xcols_buf
        me_ws,
        contrast_buf,  # Categorical contrast buffer
        copy(continuous_vars), copy(categorical_vars),
        model, _coef(model), covariance_matrix, link, vars, data_nt, offset, re_average
    )
end

"""
    clear_engine_cache!()

Clear FormulaCompiler's compiled-formula cache (`FormulaCompiler.clear_compiled_cache!()`).

Margins 3.0 keeps no cache of its own: every call builds its own engine, converts the data
columns it needs, validates its inputs and evaluates `vcov` afresh, so results always reflect
the current model (e.g. after a MixedModels `refit!`), data and covariance matrix. The only
state shared across calls is FormulaCompiler's cache of compiled formulas, one small entry
per model and column-type signature; this function empties it. Kept, under its 2.x name, so
existing code (benchmarks that clear caches before every sample) keeps working; ordinary use
never needs it.
"""
function clear_engine_cache!()
    FormulaCompiler.clear_compiled_cache!()
    return nothing
end

# Core validation functions
"""
    _validate_variables(data_nt, vars)

Validate that requested variables exist and are analyzable.
"""
function _validate_variables(data_nt::NamedTuple, vars::Vector{Symbol})
    for var in vars
        if !haskey(data_nt, var)
            throw(MarginsError("Variable $var not found in data"))
        end

        col = getproperty(data_nt, var)
        if length(col) == 0
            throw(MarginsError("Variable $var has no observations"))
        end

        # Check for raw String columns (not CategoricalArray)
        # Raw Strings cause hangs in FormulaCompiler's contrast evaluation system
        if eltype(col) <: AbstractString && !(col isa CategoricalArray)
            throw(MarginsError(
                "Variable :$var is a raw String column. " *
                "String variables must be converted to CategoricalArray for marginal effects computation.\n\n" *
                "Fix: Convert to categorical before fitting the model:\n" *
                "  using CategoricalArrays\n" *
                "  data.$var = categorical(data.$var)\n" *
                "  model = lm(@formula(...), data)"
            ))
        end
    end
end
