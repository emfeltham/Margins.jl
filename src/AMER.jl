# AMER.jl

"""
AMEResult holds average marginal effects (AME) output for one or more predictors.

# Fields
- `vars::Vector{Symbol}`: focal predictors
- `ame::Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}`: AME estimates (scalar or per-combo)
- `se::Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}`: standard errors (scalar or per-combo)
- `grad::Dict{Symbol, Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}`: Δ-method gradients (vector or per-combo)
- `n::Int`: sample size
- `family::String`, `link::String`: model family/link info
"""
struct AMEResult
    vars   :: Vector{Symbol}
    repvals:: Dict{Symbol,AbstractVector} # representative-values for moderators
    ame    :: Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}
    se     :: Dict{Symbol, Union{Float64,Dict{Tuple,Float64}}}
    grad   :: Dict{Symbol, Union{Vector{Float64},Dict{Tuple,Vector{Float64}}}}
    n      :: Int
    family :: String
    link   :: String
end

function Base.show(io::IO, ::MIME"text/plain", res::AMEResult)
    # Header
    println(io, "Average Marginal Effects (Family: $(res.family); Link: $(res.link))")
    println(io, "-"^80)

    # Which moderator columns?
    repvars = collect(keys(res.repvals))
    nrep = length(repvars)

    # --- column headings ---
    for rv in repvars
        @printf(io, "%-12s ", string(rv))
    end
    @printf(io, "%-12s %10s %10s %8s %8s %18s\n",
        "Effect", "AME", "Std.Err", "z", "P>|z|", "95% CI")
    println(io, "-"^80)

    # --- body rows ---
    for v in res.vars
        ame_entry = res.ame[v]
        se_entry  = res.se[v]

        if isa(ame_entry, Number)
            # blank out the moderator columns
            for _ in 1:nrep
                @printf(io, "%-12s ", "")
            end
            # compute stats
            z  = ame_entry / se_entry
            p  = 2*(1 - cdf(Normal(), abs(z)))
            lo = ame_entry - 1.96*se_entry
            hi = ame_entry + 1.96*se_entry
            @printf(io, "%-12s %10.4f %10.4f %8.2f %8.3f [%7.4f, %7.4f]\n",
                "Δ"*string(v), ame_entry, se_entry, z, p, lo, hi)

        else
            # one row per combination of moderator values
            for combo in sort(collect(keys(ame_entry)))
                for val in combo
                    @printf(io, "%-12s ", string(val))
                end
                ame_val = ame_entry[combo]
                se_val  = se_entry[combo]
                z  = ame_val / se_val
                p  = 2*(1 - cdf(Normal(), abs(z)))
                lo = ame_val - 1.96*se_val
                hi = ame_val + 1.96*se_val
                @printf(io, "%-12s %10.4f %10.4f %8.2f %8.3f [%7.4f, %7.4f]\n",
                    "Δ"*string(v), ame_val, se_val, z, p, lo, hi)
            end
        end
    end

    # footer
    println(io)
    println(io, "Observations: ", res.n)
end
