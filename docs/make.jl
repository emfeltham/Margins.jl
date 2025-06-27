using Documenter, Margins

makedocs(sitename = "Margins.jl")

deploydocs(
    repo = "github.com/emfeltham/Margins.jl.git",
    target = "build",
)
