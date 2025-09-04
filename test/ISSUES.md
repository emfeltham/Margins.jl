# Margins.jl Issues

**N.B.,**
- Don't emphasize passing rates. We need to pass every test (unless explicitly noted otherwise for that test).
- Statistical correctness tests are non-negotiable
- Don't CHEAT on tests. If a test fails for a good reason, note it, and we will return to the issue.
- rely on @debug for more information (either new statements with relevant info for testing and diagnostics or the existing ones when useful)
  - appears when explicitly enabled (JULIA_DEBUG=Margins or similar)
- No print statements, icons, or distracting @info. Rely on the formal @test/@testset
- All allocations and timings and performance should be measured with BenchmarkTools.jl -- no @allocated.
- Do not include skip logic. Tests should error or fail!