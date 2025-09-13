# Margins.jl Test Plan & Current Status Analysis

## N.B.

- run individual test files with --project="test"
- run --project="." for Pkg.test()
- No print statements, icons, or distracting @info. Rely on the formal @test/@testset structure, and use `@debug` for extended output instead of `println` or `@info`
- All allocations and timings and performance should be measured with BenchmarkTools.jl -- no @allocated.
- Do not include skip logic. Tests should error or fail!

This document serves two purposes:
1. **Comprehensive Test Plan**: Specifies a correctness-focused test suite for Margins.jl
2. **Current Status Analysis**: Analysis of the ~80 existing test files and integration recommendations

## Test reorganization

Look for and list tests that are not integrated into runtests.jl framework

## More testing?

