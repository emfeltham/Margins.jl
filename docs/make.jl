using Documenter, Margins

makedocs(;
  modules = [Margins],
  authors = "Eric Feltham",
  repo = "https://github.com/emfeltham/Margins.jl/blob/{commit}{path}#{line}",
  sitename = "Margins.jl",
  format = Documenter.HTML(
    ; prettyurls = get(ENV, "CI", "false") == "true",
    repolink = "https://github.com/emfeltham/Margins.jl",
    # canonical="https://beacon-biosignals.github.io/Effects.jl/stable",
    assets = String[]),
  pages = ["Home" => "index.md"],
  doctest    = false, # do not test docs
  checkdocs  = :none # ignore missing docstrings
)

deploydocs(;
  repo = "https://github.com/emfeltham/Margins.jl",
  devbranch = "main",
  push_preview = true
)