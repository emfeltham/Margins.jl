using Documenter
import Pkg
Pkg.develop(Pkg.PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()
using Margins

makedocs(; 
  modules = [Margins],
  authors = "Eric Feltham",
  repo = "https://github.com/emfeltham/Margins.jl/",
  sitename = "Margins.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    repolink = "https://github.com/emfeltham/Margins.jl",
    # canonical = "https://github.com/emfeltham/Margins.jl/stable",
    assets = String[]
  ),
  pages = [
    "Introduction" => "index.md",
    "Mathematical Foundation" => "mathematical_foundation.md",
    "Computational Architecture" => "computational_architecture.md",
    "User Guide" => [
      "Reference Grids" => "reference_grids.md",
      "Profile Analysis" => "profile_margins.md", 
      "Population Scenarios" => "population_scenarios.md",
      "Weights" => "weights.md",
      "Population Grouping" => "grouping.md",
      "Backend Selection" => "backend_selection.md",
      "Performance Guide" => "performance.md",
      "Advanced Features" => "advanced.md"
    ],
    "Migration Guide" => "stata_migration.md",
    "Package Comparison" => "comparison.md",
    "API Reference" => "api.md",
    "Examples" => "examples.md"
  ],
  doctest = true,    # Enable doctests
  checkdocs = :none    # Skip docstring checking for now
)

deploydocs(; 
  repo = "https://github.com/emfeltham/Margins.jl",
  devbranch = "main",
  versions = [
    "stable" => "v^",
    "dev" => "main",
  ],
  push_preview = true
)
