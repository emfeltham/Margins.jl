# compute_categorical.jl

"""
    _categorical_effects(model, data_nt, engine; target=:mu, contrasts=:pairwise, rows=:all, at=:none)

Compute discrete-change effects for categorical variables in `engine.vars`.
When `at=:none`, computes average discrete change over `rows` (AME-style for categoricals).
When `at` specifies profiles, computes per-profile contrasts (no averaging).
Supports `:pairwise` (all pairs) and `:baseline` (vs first level).
"""
function _categorical_effects(model, data_nt, engine; target::Symbol=:mu, contrasts::Symbol=:pairwise, rows=:all, at=:none)
    (; compiled, vars, β, Σ, link) = engine
    idxs = rows === :all ? 1 : _nrows(data_nt) : rows
    out = DataFrame(term=Symbol[], level_from=String[], level_to=String[], dydx=Float64[], se=Float64[])
    # Buffers
    X_to = Vector{Float64}(undef, length(compiled))
    X_from = Vector{Float64}(undef, length(compiled))
    Δ = Vector{Float64}(undef, length(compiled))
    for var in vars
        col = getproperty(data_nt, var)
        levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
            string.(CategoricalArrays.levels(col))
        elseif eltype(col) <: Bool
            string.([false, true])
        else
            unique!(string.(collect(col)))
        end
        pairs = if contrasts === :baseline
            [(levels[1], lev) for lev in levels if lev != levels[1]]
        else
            [(levels[i], levels[j]) for i in 1:length(levels), j in (i+1):length(levels)]
        end
        for (from, to) in pairs
            if at === :none
                # Average across rows
                acc_val = 0.0
                acc_gβ = zeros(Float64, length(compiled))
                for row in idxs
                    # η: use ΔX; μ: need X_to/X_from and link derivatives
                    FormulaCompiler.contrast_modelrow!(Δ, compiled, data_nt, row; var=var, from=from, to=to)
                    if target === :eta
                        val_row = dot(β, Δ)
                        gβ = Δ
                    else
                        # Build X_to and X_from by evaluating compiled with overrides
                        # to
                        data_to, ov_to = FormulaCompiler.build_row_override_data(data_nt, [var], row)
                        ov_to[1].replacement = from
                        compiled(X_from, data_to, row)
                        ov_to[1].replacement = to
                        compiled(X_to, data_to, row)
                        η_from = dot(β, X_from)
                        η_to = dot(β, X_to)
                        μ_from = GLM.linkinv(link, η_from)
                        μ_to = GLM.linkinv(link, η_to)
                        val_row = μ_to - μ_from
                        gβ = _dmu_deta_local(link, η_to) .* X_to .- _dmu_deta_local(link, η_from) .* X_from
                    end
                    acc_val += val_row
                    acc_gβ .+= gβ
                end
                val = acc_val / length(idxs)
                gβ_mean = acc_gβ ./ length(idxs)
                se = FormulaCompiler.delta_method_se(gβ_mean, Σ)
                push!(out, (term=var, level_from=from, level_to=to, dydx=val, se=se))
            else
                # Profiles: compute per profile (row=1 equivalent)
                profiles = _build_profiles(at, data_nt)
                for prof in profiles
                    processed_prof = _process_profile_for_scenario(prof, data_nt)
                    scen = FormulaCompiler.create_scenario("profile", data_nt, processed_prof)
                    FormulaCompiler.contrast_modelrow!(Δ, compiled, scen.data, 1; var=var, from=from, to=to)
                    if target === :eta
                        val = dot(β, Δ)
                        gβ = Δ
                    else
                        data_to, ov_to = FormulaCompiler.build_row_override_data(scen.data, [var], 1)
                        ov_to[1].replacement = from
                        compiled(X_from, data_to, 1)
                        ov_to[1].replacement = to
                        compiled(X_to, data_to, 1)
                        η_from = dot(β, X_from)
                        η_to = dot(β, X_to)
                        μ_from = GLM.linkinv(link, η_from)
                        μ_to = GLM.linkinv(link, η_to)
                        val = μ_to - μ_from
                        gβ = _dmu_deta_local(link, η_to) .* X_to .- _dmu_deta_local(link, η_from) .* X_from
                    end
                    se = FormulaCompiler.delta_method_se(gβ, Σ)
                    row = (term=var, level_from=from, level_to=to, dydx=val, se=se)
                    push!(out, row)
                    # attach profile columns
                    for (k,v) in prof
                        out[!, Symbol("at_", k)] = get(out, Symbol("at_", k), fill(v, nrow(out)))
                        out[end, Symbol("at_", k)] = v
                    end
                end
            end
        end
    end
    return out
end
