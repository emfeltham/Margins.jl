# backend_reliability_guide.jl - Backend Testing Utilities
#
# This file provides testing utilities for backend reliability validation.
# The main user-facing documentation is now in docs/src/backend_selection.md
#
# Based on comprehensive testing in the Statistical Correctness Testing Plan.

# Test function to verify backend behavior for specific formula
function test_backend_reliability(model, data; test_backends=[:fd, :ad])
    results = Dict()
    
    for backend in test_backends
        try
            result = population_margins(model, data; type=:effects, backend=backend)
            results[backend] = (
                success = true,
                estimates = DataFrame(result).estimate,
                ses = DataFrame(result).se,
                time = @elapsed population_margins(model, data; type=:effects, backend=backend)
            )
        catch e
            results[backend] = (
                success = false,
                error = e,
                time = nothing
            )
        end
    end
    
    return results
end

# Note: test_backend_reliability function is available for testing but not exported
# The main backend selection guidance is now in docs/src/backend_selection.md