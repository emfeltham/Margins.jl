using Documenter, Margins

makedocs(
  sitename = "Margins.jl",
  modules  = [Margins],
  formats  = Documenter.HTML(),
)

deploydocs(
  repo   = "https://github.com/emfeltham/Margins.jl.git",
  branch = "main",    # default branch
  folder = "docs",    # the target folder on main
)