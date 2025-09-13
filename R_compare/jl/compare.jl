# compare.jl

using DataFrames, CSV, DataFramesMeta

function load_pair(rfile::String, jlfile::String)
    if !(isfile(rfile) && isfile(jlfile))
        return nothing, nothing
    end
    return CSV.read(rfile, DataFrame), CSV.read(jlfile, DataFrame)
end

function compare_estimates(r_df::DataFrame, jl_df::DataFrame; tol_est=1e-6, tol_se=0.002)
    # Handle new Julia format:
    # - Effects (AME/MEM): Julia has "variable" + "contrast"; R has "term" + "contrast"
    # - Predictions (AAP/APM): Julia has "type" column only, no variable/contrast

    normalize_contrast_r(s) = begin
        if s == "dY/dX"
            return "derivative"
        else
            t = replace(String(s), " - " => " vs ")
            # Normalize boolean tokens to lowercase to match Julia output
            t = replace(t, "TRUE" => "true", "FALSE" => "false")
            return t
        end
    end
    normalize_contrast_jl(s) = String(s)

    if hasproperty(r_df, :term)
        # AME/MEM case: compare on (term, contrast)
        left = select(r_df, :term, :contrast, :estimate, Symbol("std.error") => :se)
        left.contrast = map(normalize_contrast_r, left.contrast)

        right = select(jl_df, :variable => :term, :contrast, :estimate, :se)
        right.contrast = map(normalize_contrast_jl, right.contrast)

        # Rename Julia columns for comparison
        rename!(right, :se => :se_j, :estimate => :estimate_j)
        merged = innerjoin(left, right, on=[:term, :contrast])
    else
        # AAP/APM case: single prediction; synthesize common term key using Julia type
        julia_type = jl_df.type[1]  # "AAP" or "APM"
        left = DataFrame(term = [julia_type], estimate = r_df.estimate, se = r_df[!, Symbol("std.error")])
        right = DataFrame(term = [julia_type], estimate = jl_df.estimate, se = jl_df.se)
        rename!(right, :se => :se_j, :estimate => :estimate_j)
        merged = innerjoin(left, right, on=:term)
    end

    if nrow(merged) == 0
        return DataFrame()
    end

    est_ok = abs.(merged.estimate .- merged.estimate_j) .<= tol_est
    se_ok  = abs.(merged.se .- merged.se_j) .<= tol_se

    out = DataFrame(
        term = merged.term,
        est_diff = abs.(merged.estimate .- merged.estimate_j),
        se_diff = abs.(merged.se .- merged.se_j),
        est_ok = est_ok,
        se_ok = se_ok,
        status = fill(all(est_ok) && all(se_ok) ? "ok" : "diff", nrow(merged)),
    )
    if hasproperty(merged, :contrast)
        out.contrast = merged.contrast
    end
    return out
end

function compare_results(
    ; datasets=["mtcars","toothgrowth","titanic"],
    models=["m1","m2","m3","m4","m5"],
    result_types=["ame","mem","aap","apm"],
    rdir::AbstractString=".", jldir::AbstractString=".",
    detailed::Bool=false
)

    rows = DataFrame(dataset=String[], model=String[], type=String[], n_match=Int[], n_est_ok=Int[], n_se_ok=Int[], status=String[])
    all_details = DataFrame()
    
    for d in datasets, m in models, t in result_types
        rfile = joinpath(rdir, "r_results_$(d)_$(m)_$(t).csv")
        jlfile = joinpath(jldir, "julia_results_$(d)_$(m)_$(t).csv")
        r_df, jl_df = load_pair(rfile, jlfile)
        if isnothing(r_df)
            continue  # Skip missing files instead of erroring
        end
        try
            detail = compare_estimates(r_df, jl_df)
            if nrow(detail) == 0
                push!(rows, (d, m, t, 0, 0, 0, "no_match"))
                continue
            end
            n_match = nrow(detail)
            n_est_ok = count(==(true), detail.est_ok)
            n_se_ok = count(==(true), detail.se_ok)
            status = all(detail.est_ok) && all(detail.se_ok) ? "ok" : "diff"
            push!(rows, (d, m, t, n_match, n_est_ok, n_se_ok, status))
            
            if detailed
                # Add context columns to detail
                detail.dataset .= d
                detail.model .= m  
                detail.result_type .= t
                all_details = vcat(all_details, detail, cols=:union)
            end
        catch e
            println("Error comparing $d/$m/$t: $e")
            push!(rows, (d, m, t, 0, 0, 0, "error"))
        end
    end
    
    return detailed ? (rows, all_details) : rows
end

function detailed_comparison_table(
    ; datasets=["mtcars","toothgrowth","titanic"],
    models=["m1","m2","m3","m4","m5"],
    result_types=["ame","mem","aap","apm"],
    rdir::AbstractString=".", jldir::AbstractString="."
)
    
    all_comparisons = DataFrame()
    
    for d in datasets, m in models, t in result_types
        rfile = joinpath(rdir, "r_results_$(d)_$(m)_$(t).csv")
        jlfile = joinpath(jldir, "julia_results_$(d)_$(m)_$(t).csv")
        r_df, jl_df = load_pair(rfile, jlfile)
        if isnothing(r_df)
            continue
        end
        
        try
            # Get the merged comparison with R and Julia estimates side by side
            if hasproperty(r_df, :term)
                # AME/MEM case
                left = select(r_df, :term, :estimate => :r_estimate, Symbol("std.error") => :r_se)
                right = select(jl_df, :variable => :term, :estimate => :jl_estimate, :se => :jl_se)
            else
                # AAP/APM case
                julia_type = jl_df.type[1]
                left = DataFrame(term = [julia_type], r_estimate = r_df.estimate, r_se = r_df[!, Symbol("std.error")])
                right = DataFrame(term = [julia_type], jl_estimate = jl_df.estimate, jl_se = jl_df.se)
            end
            
            merged = innerjoin(left, right, on=:term)
            if nrow(merged) > 0
                merged.dataset .= d
                merged.model .= m
                merged.result_type .= t
                merged.est_diff = abs.(merged.r_estimate .- merged.jl_estimate)
                merged.se_diff = abs.(merged.r_se .- merged.jl_se)
                
                all_comparisons = vcat(all_comparisons, merged, cols=:union)
            end
        catch e
            println("Error processing $d/$m/$t: $e")
        end
    end
    
    return all_comparisons
end
