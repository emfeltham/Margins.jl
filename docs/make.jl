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
    assets = String[]),
  pages = [
    "Introduction" => "index.md",
    "API" => "api.md",
    "Developer" => Any[
      "Performance" => "dev/performance.md",
    ],
  ],
  doctest    = false, # do not test docs
  checkdocs  = :none # ignore missing docstrings
)

deploydocs(;
  repo = "https://github.com/emfeltham/Margins.jl",
  devbranch = "main",
  push_preview = true
)
