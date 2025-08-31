# types.jl - Result types, error types, and display methods

"""
    MarginsResult

Container for marginal effects results with Tables.jl interface.

Fields:
- `df::DataFrame`: Results table with estimates, standard errors, etc.
- `gradients::Matrix{Float64}`: Parameter gradients (G matrix) for delta-method
- `metadata::Dict`: Analysis metadata (model info, options used, etc.)

# Examples
```julia
result = population_margins(model, data; type=:effects, vars=[:x1, :x2])
DataFrame(result)  # Convert to DataFrame
result.gradients   # Access parameter gradients
result.metadata    # Access analysis metadata
```
"""
struct MarginsResult
    df::DataFrame
    gradients::Matrix{Float64}
    metadata::Dict{Symbol, Any}
end

# Tables.jl interface implementation
Tables.istable(::Type{MarginsResult}) = true
Tables.rowaccess(::Type{MarginsResult}) = true
Tables.rows(mr::MarginsResult) = Tables.rows(mr.df)
Tables.schema(mr::MarginsResult) = Tables.schema(mr.df)

# DataFrame conversion methods
Base.convert(::Type{DataFrame}, mr::MarginsResult) = mr.df
DataFrame(mr::MarginsResult) = mr.df

# Display methods
function Base.show(io::IO, mr::MarginsResult)
    n_effects = nrow(mr.df)
    n_vars = get(mr.metadata, :n_vars, "unknown")
    analysis_type = get(mr.metadata, :type, "unknown")
    
    println(io, "MarginsResult: $n_effects $analysis_type effects")
    println(io, "Variables: $n_vars")
    show(io, mr.df)
end

function Base.show(io::IO, ::MIME"text/plain", mr::MarginsResult)
    show(io, mr)
    println(io, "\nMetadata:")
    for (k, v) in mr.metadata
        println(io, "  $k: $v")
    end
end

# Custom error types for clear user feedback
struct MarginsError <: Exception
    msg::String
end

struct StatisticalValidityError <: Exception
    msg::String
end

# Make error messages display cleanly
Base.showerror(io::IO, e::MarginsError) = print(io, "MarginsError: ", e.msg)
Base.showerror(io::IO, e::StatisticalValidityError) = print(io, "StatisticalValidityError: ", e.msg)