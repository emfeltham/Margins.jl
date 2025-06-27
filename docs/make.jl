using Documenter, Margins

makedocs(
  sitename = "Margins.jl",
  modules  = [Margins],
  format  = Documenter.HTML(),
  doctest  = false
)

deploydocs(
  repo   = "https://github.com/emfeltham/Margins.jl.git",
  branch = "main",    # default branch
  folder = "docs",    # the target folder on main
)