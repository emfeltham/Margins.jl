# ame_representation.jl

"""
_ame_representation(
    df::DataFrame,
    model,
    focal::Symbol,
    repvals::Dict{Symbol, Vector},
    fe_form,
    β::AbstractVector,
    invlink,
    dinvlink,
    d2invlink,
    vcov::Function
) -> (Dict{Tuple,Float64}, Dict{Tuple,Float64}, Dict{Tuple,Vector{Float64}})

Compute the marginal effect of `focal` at combinations of representative values
specified in `repvals` (a dict from variable ⇒ vector of values to hold that var at).
Returns three dicts mapping each combo (as a tuple of repvals in key order) to:
- `AME`
- `Std.Err`
- `Δ-method gradient`
"""
function _ame_representation(
    df::DataFrame,
    model,
    focal::Symbol,
    repvals,
    fe_form,
    β::AbstractVector,
    invlink,
    dinvlink,
    d2invlink,
    vcov::Function
)
    # Validate representative variables
    repvars = collect(keys(repvals))
    for rv in string.(repvars)
        @assert rv in names(df) "Representative var $(rv) not found in DataFrame"
    end

    # Build Cartesian product of repvals
    lists  = [repvals[rv] for rv in repvars]
    combos = collect(Iterators.product(lists...))

    # Prepare storage
    ame_dict  = Dict{Tuple,Float64}()
    se_dict   = Dict{Tuple,Float64}()
    grad_dict = Dict{Tuple,Vector{Float64}}()
    # Σβ = vcov(model)

    # Loop over each setting
    for combo in combos
        # Create a copy and override representative vars
        df_tmp = copy(df)
        for (rv, val) in zip(repvars, combo)
            col = df_tmp[!, rv]
            if col isa CategoricalArray
                p = pool(col)
                df_tmp[!, rv] .= CategoricalValue(val, p)
            else
                df_tmp[!, rv] .= val
            end
        end

        # Delegate to continuous or factor AME
        if eltype(df_tmp[!, focal]) <: Number
            ame_val, se_val, grad_v = _ame_continuous(
                df_tmp, model, focal, fe_form,
                β, dinvlink, d2invlink, vcov
            )
        else
            # single contrast: baseline = first factor level
            lvl0 = levels(df_tmp[!, focal])[1]
            ame_val, se_val, grad_v = _ame_factor_pair(
                df_tmp, model, focal, lvl0, levels(df_tmp[!,focal])[2],
                fe_form, β, invlink, dinvlink, vcov
            )
        end

        ame_dict[combo]  = ame_val
        se_dict[combo]   = se_val
        grad_dict[combo] = grad_v
    end

    return ame_dict, se_dict, grad_dict
end