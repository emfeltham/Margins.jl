#!/usr/bin/env julia

using ArgParse
using DataFrames, CSV, CategoricalArrays
using GLM
using Margins # dev
using CovarianceMatrices
using Statistics, StatsBase

const DATA_DIR = joinpath(@__DIR__, "..", "data")
const R_RESULTS_DIR = joinpath(@__DIR__, "..", "results_r")
const JL_RESULTS_DIR = joinpath(@__DIR__, "..", "results_julia")

mkpath(DATA_DIR)
mkpath(R_RESULTS_DIR)
mkpath(JL_RESULTS_DIR)

function cli_args()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--cov"
            help = "Covariance estimator to use (e.g., HC3). If not set, use model-based."
            arg_type = String
            required = false

        "--scale"
            help = "Scale for predictions/effects: response or link (default: response)"
            arg_type = String
            default = "response"
            required = false
    end
    return ArgParse.parse_args(ARGS, s)
end

"""
get_vcov_estimator(covname) -> estimator or nothing
Return a CovarianceMatrices estimator object to be passed as `vcov=` to Margins.jl,
or `nothing` to use model-based covariance.
"""
function get_vcov_estimator(covname::Union{Nothing,String})
    if covname === nothing
        return nothing
    end
    cname = lowercase(covname)
    if cname == "hc3"
        return HC3()
    elseif cname == "hc1"
        return HC1()
    elseif cname == "hc0"
        return HC0()
    else
        @warn "Unrecognized covariance name; using model-based" covname
        return nothing
    end
end

function prepare_julia_datasets()
    datasets = Dict{String,DataFrame}()
    # Load CSVs written by R
    datasets["mtcars"] = CSV.read(joinpath(DATA_DIR, "data_mtcars.csv"), DataFrame)
    datasets["iris"] = CSV.read(joinpath(DATA_DIR, "data_iris.csv"), DataFrame)
    datasets["toothgrowth"] = CSV.read(joinpath(DATA_DIR, "data_toothgrowth.csv"), DataFrame)
    datasets["titanic"] = CSV.read(joinpath(DATA_DIR, "data_titanic.csv"), DataFrame)
    
    # Convert to proper Julia types - match R exactly
    
    # mtcars: keep numeric variables numeric, only true categoricals as categorical
    # cyl, gear, carb stay numeric (they're counts)
    # vs, am become Bool (like R logicals)
    datasets["mtcars"].vs = Bool.(datasets["mtcars"].vs)  # TRUE = Straight, FALSE = V  
    datasets["mtcars"].am = Bool.(datasets["mtcars"].am)  # TRUE = Manual, FALSE = Automatic

    # iris: Species is naturally categorical
    datasets["iris"].Species = categorical(datasets["iris"].Species)
    levels!(datasets["iris"].Species, ["setosa","versicolor","virginica"])  # baseline setosa

    # toothgrowth: only supp is categorical, dose stays numeric
    datasets["toothgrowth"].supp = categorical(datasets["toothgrowth"].supp)
    levels!(datasets["toothgrowth"].supp, ["OJ","VC"])  # baseline OJ
    # dose stays numeric - no artificial factor versions

    # titanic: Survived becomes Bool, others are naturally categorical
    datasets["titanic"].Survived = Bool.(datasets["titanic"].Survived)  # TRUE = survived
    datasets["titanic"].Class    = categorical(datasets["titanic"].Class)
    levels!(datasets["titanic"].Class, ["1st","2nd","3rd","Crew"])  # baseline 1st
    datasets["titanic"].Sex      = categorical(datasets["titanic"].Sex)
    levels!(datasets["titanic"].Sex,   ["Female","Male"])  # baseline Female
    datasets["titanic"].Age      = categorical(datasets["titanic"].Age)
    levels!(datasets["titanic"].Age,   ["Adult","Child"])  # baseline Adult

    return datasets
end

function estimate_julia_models(datasets)
    models = Dict{String,Dict{String,Any}}()

    models["mtcars"] = Dict(
        "m1" => lm(@formula(mpg ~ wt + hp), datasets["mtcars"]),
        "m2" => lm(@formula(mpg ~ cyl + wt), datasets["mtcars"]),
        "m3" => lm(@formula(mpg ~ wt * am + hp), datasets["mtcars"]),
        "m4" => lm(@formula(mpg ~ cyl + vs + am), datasets["mtcars"]),
        "m5" => lm(@formula(mpg ~ wt * hp + cyl * am), datasets["mtcars"]),
    )

    # iris models: GLM.jl lacks multinomial; skip cross-package here

    models["toothgrowth"] = Dict(
        "m1" => lm(@formula(len ~ supp + dose), datasets["toothgrowth"]),
        "m2" => lm(@formula(len ~ supp * dose), datasets["toothgrowth"]),
        "m3" => lm(@formula(len ~ supp + dose + dose^2), datasets["toothgrowth"]),
        "m4" => lm(@formula(len ~ supp + (dose - 1)), datasets["toothgrowth"]),
        "m5" => lm(@formula(len ~ supp * dose^2), datasets["toothgrowth"]),
    )

    models["titanic"] = Dict(
        "m1" => glm(@formula(Survived ~ Sex + Class), datasets["titanic"], Binomial()),
        "m2" => glm(@formula(Survived ~ Class + Sex + Age), datasets["titanic"], Binomial()),
        "m3" => glm(@formula(Survived ~ Class * Sex + Age), datasets["titanic"], Binomial()),
        # m4 simplified to match R: no Class:Age, only Sex:Age + Class main effect
        "m4" => glm(@formula(Survived ~ Class + Sex * Age), datasets["titanic"], Binomial()),
        "m5" => glm(@formula(Survived ~ (Class + Age) * Sex), datasets["titanic"], Binomial()),
    )

    return models
end

function compute_julia_marginal_effects(models, datasets; covname::Union{Nothing,String}=nothing, scale::Symbol=:response)
    results = Dict{String,Dict{String,Dict{String,DataFrame}}}()
    vcov_est = get_vcov_estimator(covname)

    # Small helpers to conditionally pass vcov
    pm_effects(model, data; vcov_opt=nothing) = vcov_opt === nothing ?
        population_margins(model, data; type=:effects) :
        population_margins(model, data; type=:effects, vcov=vcov_opt)
    pm_preds(model, data; vcov_opt=nothing, scale=:response) = vcov_opt === nothing ?
        population_margins(model, data; type=:predictions, scale=scale) :
        population_margins(model, data; type=:predictions, scale=scale, vcov=vcov_opt)
    prof_effects(model, data, grid; vcov_opt=nothing) = vcov_opt === nothing ?
        profile_margins(model, data, grid; type=:effects) :
        profile_margins(model, data, grid; type=:effects, vcov=vcov_opt)
    prof_preds(model, data, grid; vcov_opt=nothing, scale=:response) = vcov_opt === nothing ?
        profile_margins(model, data, grid; type=:predictions, scale=scale) :
        profile_margins(model, data, grid; type=:predictions, scale=scale, vcov=vcov_opt)
    # Helpers to construct parity profiles matching what R can compare
    build_base_profile(dataset) = begin
        gd = Dict{Symbol,Any}()
        for col in names(dataset)
            colsym = Symbol(col)
            coldata = dataset[!, col]
            if coldata isa CategoricalVector
                first_level = levels(coldata)[1]
                gd[colsym] = categorical([first_level]; levels=levels(coldata), ordered=isordered(coldata))[1]
            elseif eltype(coldata) <: Bool
                # Use mode for booleans to mirror R behavior
                gd[colsym] = mode(coldata)
            elseif eltype(coldata) <: Number
                gd[colsym] = mean(coldata)
            else
                # Fallback: take first element
                gd[colsym] = first(coldata)
            end
        end
        DataFrame(gd)[1:1, :]
    end

    # Identify response and variable types
    get_response_sym(model) = try
        Symbol(model.mf.f.lhs.sym)
    catch
        nothing
    end
    categorical_vars(dataset, response) = begin
        [Symbol(c) for c in names(dataset) if Symbol(c) != response && ((dataset[!, c] isa CategoricalVector) || (eltype(dataset[!, c]) <: Bool))]
    end

    # Build MEM result to match R: derivatives at base row for continuous vars, and one
    # contrast per categorical var at the row where that var is set to its non-baseline.
    function compute_mem_parity(model, dataset; vcov_opt=nothing)
        base = build_base_profile(dataset)
        resp = get_response_sym(model)
        cat_vars = categorical_vars(dataset, resp)

        parts = DataFrame[]
        # Continuous derivatives at base row when present
        try
            cont_effects = vcov_opt === nothing ?
                profile_margins(model, dataset, base; type=:effects, vars=:all_continuous) :
                profile_margins(model, dataset, base; type=:effects, vars=:all_continuous, vcov=vcov_opt)
            push!(parts, DataFrame(cont_effects))
        catch e
            # Gracefully skip when model has no continuous predictors
            msg = sprint(showerror, e)
            if !occursin("No continuous explanatory variables", msg)
                rethrow(e)
            end
        end
        for var in cat_vars
            col = String(var)
            coldata = dataset[!, col]
            # Determine non-baseline level
            if coldata isa CategoricalVector
                bl = levels(coldata)[1]
                lv = first(filter(!=(bl), levels(coldata)))
                row = deepcopy(base)
                row[!, col] .= categorical([lv]; levels=levels(coldata), ordered=isordered(coldata))[1]
            else
                # Bool: toggle
                row = deepcopy(base)
                row[!, col] .= !only(base[!, col])
            end
            try
                mem_one = vcov_opt === nothing ?
                    profile_margins(model, dataset, row; type=:effects, vars=[var]) :
                    profile_margins(model, dataset, row; type=:effects, vars=[var], vcov=vcov_opt)
                push!(parts, DataFrame(mem_one))
            catch e
                msg = sprint(showerror, e)
                # Skip variables not present in the model (e.g., Age in titanic m1)
                if occursin("Could not find categorical variable", msg) || occursin("not found in data", msg)
                    continue
                else
                    rethrow(e)
                end
            end
        end
        vcat(parts...)
    end

    for (dataset_name, dataset_models) in models
        results[dataset_name] = Dict{String,Dict{String,DataFrame}}()
        dataset = datasets[dataset_name]
        for (model_name, model) in dataset_models
            # Per-model vcov; enforce error-first policy if robust estimator fails
            local_vcov = vcov_est
            ame_res = nothing
            try
                ame_res = pm_effects(model, dataset; vcov_opt=local_vcov)
            catch e
                if e isa ArgumentError && occursin("vcov failed", sprint(showerror, e))
                    error("Robust vcov (", covname, ") failed for ", dataset_name, ":", model_name, ". " *
                          "Re-run without --cov or with a different estimator (e.g., --cov HC1).")
                else
                    rethrow(e)
                end
            end

            # Build parity grids
            base_grid = build_base_profile(dataset)

            mem = compute_mem_parity(model, dataset; vcov_opt=local_vcov)
            aap = pm_preds(model, dataset; vcov_opt=local_vcov, scale=scale)
            apm = prof_preds(model, dataset, base_grid; vcov_opt=local_vcov, scale=scale)
            # Compatibility alias to avoid any lingering references
            ame = ame_res
            results[dataset_name][model_name] = Dict(
                "ame" => DataFrame(ame_res),
                "mem" => DataFrame(mem),
                "aap" => DataFrame(aap),
                "apm" => DataFrame(apm),
            )
        end
    end
    return results
end

function export_julia_results(results)
    for (dataset_name, dataset_results) in results
        for (model_name, model_results) in dataset_results
            for (rtype, df) in model_results
                file = joinpath(JL_RESULTS_DIR, "julia_results_$(dataset_name)_$(model_name)_$(rtype).csv")
                CSV.write(file, df)
            end
        end
    end
end

# Comparison is now done separately via compare.jl script

function main()
    parsed = cli_args()
    covname = get(parsed, "cov", nothing)
    scale_str = get(parsed, "scale", "response")
    scale = scale_str == "link" ? :link : :response

    datasets = prepare_julia_datasets()
    models = estimate_julia_models(datasets)

    # Example: compute covariance per model family on the fly is expensive.
    # Here we use model-based if not specified, or HC* if requested (per model).
    # For parity, it's acceptable to pass nothing or robust consistently.

    results = compute_julia_marginal_effects(models, datasets; covname=covname, scale=scale)
    export_julia_results(results)

    println("Julia pipeline completed: results in ", JL_RESULTS_DIR)
end

main()
