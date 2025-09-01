using Documenter, Margins

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
    "User Guide" => [
      "Reference Grids" => "reference_grids.md",
      "Profile Analysis" => "profile_margins.md", 
      "Performance Guide" => "performance.md",
      "Advanced Features" => "advanced.md"
    ],
    "API Reference" => "api.md",
    "Examples" => "examples.md"
  ],
  doctest = true,    # Enable doctest validation
  checkdocs = :none    # Skip docstring checking for now
)

deploydocs(;
  repo = "https://github.com/emfeltham/Margins.jl",
  devbranch = "main",
  push_preview = true
)