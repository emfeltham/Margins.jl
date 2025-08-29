# core/results.jl - Result types and display

struct MarginsResult
    table::DataFrame
    metadata::NamedTuple
    # Optional gradient storage for post-calculation efficiency
    gradients::Union{Nothing, NamedTuple}
end

# Backward compatibility constructor
MarginsResult(table::DataFrame, metadata::NamedTuple) = MarginsResult(table, metadata, nothing)

function Base.show(io::IO, res::MarginsResult)
    println(io, "MarginsResult:")
    md = res.metadata
    keys_to_show = (:type, :vars, :target, :scale, :at, :backend, :vcov, :n, :link, :dof)
    parts = String[]
    for k in keys_to_show
        if hasproperty(md, k)
            push!(parts, string(k, "=", getproperty(md, k)))
        end
    end
    if !isempty(parts)
        println(io, "  ", join(parts, ", "))
    end
    tbl = res.table
    nshow = min(nrow(tbl), 10)
    if nshow > 0
        show(io, first(tbl, nshow))
        if nrow(tbl) > nshow
            println(io, "\n  … (", nrow(tbl) - nshow, " more rows)")
        end
    else
        println(io, "  (empty table)")
    end
end
