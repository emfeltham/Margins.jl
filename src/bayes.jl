# homop functions.jl
# functions for working with Stan model output, effects, etc.

function posterior_predictions(X::Matrix{T}, fixedcoeff::Matrix{S}; 
                               invlink = logistic) where {T<:Real, S<:Real}
    # Dimension validation
    n_obs, n_params = size(X)
    n_samples, n_params_coeff = size(fixedcoeff)
    n_params == n_params_coeff || throw(DimensionMismatch("X and fixedcoeff have incompatible parameter dimensions"))

    # Pre-allocate output with type stability
    U = promote_type(T, S, Float32)  # Handle mixed precision
    ytilde = Matrix{U}(undef, n_obs, n_samples)
    
    predictions!(ytilde, X, fixedcoeff; invlink)
    return ytilde
end

function predictions!(ytilde::AbstractMatrix, X::AbstractMatrix, 
                      fixedcoeff::AbstractMatrix; invlink = logistic)
    # Performance optimizations
    @inbounds @simd for i in axes(fixedcoeff, 1)
        β = @view fixedcoeff[i, :]
        μ = @view ytilde[:, i]
        mul!(μ, X, β)
        broadcast!(invlink, μ, μ)  # Proper in-place transformation
    end
    return ytilde
end

# I need to compute the typical values on the data directly

# this may be the general approach to take for GLMs too, even though the data
# is at hand. the only trick is to get the typical values into design-matrix format.
# rsp, fes, res = modelvariables(m);
# tbx = tbl[!, [rsp, fes...]];
# tbx = columntable(tbx);
# where tbl is columntable(tbl)
# this should handle the transformations
# we need to access the data

# Helper to mirror your existing model matrix construction
function model_matrix(form::FormulaTerm, data::DataFrame)
    schema = schema(form, data)
    mf = ModelFrame(form, data, schema)
    return ModelMatrix(mf).m
end

function design_matrix(form, refgrid)
    # get the fixed effects
    # call rhs, then filter
    fix = form.rhs;
    fixed_loc = .![x <: RandomEffectsTerm for x in typeof.(fix)];
    fix = fix[fixed_loc];

    # create the model matrix for the reference grid
    # check that the transformations are right
    # N.B., this depends on the modified modelcols functions in "effects2.jl"
    return modelcols(fix, columntable(refgrid))[1]; # only grab X mat
end

using StatsBase

"""
effects_predictions(dsn, form, typ, θ; invlink = identity, alpha = 0.05, eff_col=nothing)

## Description

Calculate the posterior of the adjusted predictions at the mean.
- Posterior distribution of fixed effects parameters θ = (α, β)
- Significance level `alpha=0.05`

"""
function effects_predictions(
    dsn, form, typ, θ;
    invlink = identity, alpha = 0.05, eff_col=nothing
)
    refgrid = setup_refgrid(dsn, typ);
    X = design_matrix(form, refgrid);
    ytilde = posterior_predictions(X, θ; invlink);

    mean_pred = vec(mean(ytilde; dims=2))
    lwr, upr = hpdi(ytilde; alpha)

    hpdi(ytilde[1, :]; alpha)
    hpdi(ytilde; alpha)

    refgrid = select(refgrid, keys(dsn)...);
    refgrid[!, something(eff_col, form.lhs.sym)] .= mean_pred;
    refgrid[!, :lower] = lwr;
    refgrid[!, :upper] = upr;
    return refgrid
end

function typicals_from_dict(typs)
    symb = collect(keys(typs[:fes]));
    vl = [typs[:fes][v][1] for v in symb];
    return Dict(s => isa(v, AbstractArray) ? [v] : [v] for (s, v) in zip(symb, vl))
end

"""

effects_pairwise(
    contrast_dsn::Dict{Symbol,<:Any}, form::FormulaTerm, typ::Dict, θ::Matrix;
    invlink = identity, alpha=0.05, specifiedvals = nothing
)

## Description

Setup and calculate pairwise contrasts (discrete marginal effects) over specified variables.
- Each `Dict` entry should represent a pairwise contrast on the focal variable.
- More complex contrasts must be hand-specified.

Calculate within-variable contrasts for each variable present in `contrast_dsn`:

contrast_dsn = Dict(
    :wealth_d1_4_h => [0, 1],
    :age_h => [16, 90],
);

Note, assigns (-1, 1), so that above is the contrast f(1) - f(0).

Specify fixed non-typical values (that overwrite values in typ).

specifiedvals = Dict(
    :notkin431 => true,
    :socio4 => true,
)

"""
function effects_pairwise(
    contrast_dsn::Dict{Symbol,<:Any}, form::FormulaTerm, typ::Dict, θ::Matrix;
    invlink = identity, alpha=0.05, specifiedvals = nothing
)
    # 1. Setup contrast reference grid
    contrast_grid = setup_contrast_grid(contrast_dsn, typ)
    
    # override typical values with optionally scecified values
    if !isnothing(specifiedvals)
        for (k, v) in specifiedvals
            contrast_grid[!, k] .= v
        end
    end

    # 2. Calculate APMs
    # This part should be basically the same effects_stan
    X = design_matrix(form, contrast_grid);
    ytilde = posterior_predictions(X, θ; invlink)

    # track only key variables
    vrs = [keys(contrast_dsn)..., keys(specifiedvals)...]
    select!(contrast_grid, :scenario, :contrast_var, vrs);
    # 4. Extract and calculate contrasts
    results = compute_contrasts(ytilde, contrast_grid, keys(contrast_dsn), alpha)
    
    return results
end

"""

## Description

Calculate the pairwise contrasts.
"""
function compute_contrasts(
    ytilde::Matrix, contrast_grid::DataFrame, contrastvars, alpha::Float64
)
    # Initialize results storage
    results = DataFrame(
        variable = Symbol[],
        Δx = String[],
        ΔyΔx = Float64[],
        lower = Float64[],
        upper = Float64[],
        x_lower = Vector{Any}(),
        x_upper = Vector{Any}(),
    )
    
    # Calculate contrasts for each variable
    for v in contrastvars

        # Get indices for reference and comparison scenarios
        c1 = contrast_grid[!, :contrast_var] .== v;
        idx = findall(c1);

        # Compute posterior contrast distribution
        contrast = vec(sum(ytilde[idx, :] .* contrast_grid[idx, :scenario], dims = 1))
        hpi = hpdi(contrast; alpha)

        c2 = contrast_grid[!, :scenario] .== 1
        c3 = contrast_grid[!, :scenario] .== -1
        
        csl = contrast_grid[c1 .& c3 , v][1];
        csu = contrast_grid[c1 .& c2 , v][1];

        if (typeof(csl) <: Real) & (typeof(csu) <: Real)
            csl = round(csl; digits = 3)
            csu = round(csu; digits = 3)
        end

        cs = string(csu) * " – " * string(csl)

        push!(results, (
            variable = v,
            Δx = cs,
            ΔyΔx = mean(contrast),
            lower = hpi[1],
            upper = hpi[2],
            x_lower = contrast_grid[c1 .& c2 , v][1],
            x_upper = contrast_grid[c1 .& c3 , v][1],
        ))
    end
    
    return results
end
