#!/usr/bin/env julia

"""
run_comprehensive_tests.jl

Script to run the full comprehensive statistical validation suite.
This runs all 9 tiers of validation including:
- Analytical SE verification (Tiers 1A, 1B)
- Bootstrap validation (Tier 7) 
- Robust SE integration (Tier 8)
- Specialized edge cases (Tier 9)

Usage:
    julia test/run_comprehensive_tests.jl
    
Or via environment variable:
    MARGINS_COMPREHENSIVE_TESTS=true julia test/runtests.jl
"""

ENV["MARGINS_COMPREHENSIVE_TESTS"] = "true"

println("🚀 Starting Margins.jl Comprehensive Statistical Validation")
println("=" ^ 60)
println("This will run all 9 tiers of statistical validation (~60-90 seconds)")
println("Including: analytical verification, bootstrap validation, robust SEs, and specialized edge cases")
println("")

include("runtests.jl")