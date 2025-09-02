# validation_control.jl - Statistical Validation Control System
#
# This file provides a control system for running different levels of 
# statistical validation based on the testing context (CI, development, release).
#
# Usage:
#   - CI/CD Pipeline: Fast critical validation (~30 seconds)
#   - Development: Targeted validation for specific areas  
#   - Release: Complete comprehensive validation (~5 minutes)

using Test

"""
    run_ci_validation()

Run fast critical statistical validation suitable for CI/CD pipelines.
Focuses on essential mathematical correctness and core functionality.

Expected duration: < 30 seconds
Coverage: Critical mathematical correctness, integer variables, backend consistency
"""
function run_ci_validation()
    @info "🚀 Running CI Statistical Validation (Fast Critical Subset)"
    @info "Expected duration: < 30 seconds"
    
    include("ci_validation.jl")
    
    @info "✅ CI validation complete - ready for pipeline"
    return true
end

"""
    run_development_validation(; focus=:all)

Run targeted statistical validation for development work.
Allows focusing on specific areas during development.

Arguments:
- `focus`: Symbol indicating focus area
  - `:all` - Complete validation (same as release)  
  - `:core` - Core mathematical correctness only
  - `:backend` - Backend consistency only
  - `:integers` - Integer variable support only
  - `:systematic` - Systematic model coverage only

Expected duration: 30 seconds to 5 minutes depending on focus
"""
function run_development_validation(; focus=:all)
    @info "🔧 Running Development Statistical Validation"
    @info "Focus area: $focus"
    
    if focus == :all
        @info "Running complete validation for development..."
        include("statistical_validation.jl")
        include("backend_consistency.jl")
    elseif focus == :core
        @info "Running core mathematical correctness validation..."
        include("ci_validation.jl")  # CI subset covers core correctness
    elseif focus == :backend
        @info "Running backend consistency validation..."
        include("backend_consistency.jl")
    elseif focus == :integers
        @info "Running integer variable support validation..."
        # Run subset focusing on integer variables
        @testset "Integer Variable Development Validation" begin
            Random.seed!(42)
            include("testing_utilities.jl")
            
            df = make_econometric_data(n=500, seed=123)
            
            # Test key integer variable scenarios
            for int_var in [:int_age, :int_education, :int_experience]
                @testset "$(int_var) validation" begin
                    model = lm(Term(:log_wage) ~ Term(int_var), df)
                    framework_result = test_2x2_framework_quadrants(model, df; test_name="$int_var development")
                    @test framework_result.all_successful
                    @test framework_result.all_finite
                end
            end
            
            @info "✓ Integer variable development validation complete"
        end
    elseif focus == :systematic
        @info "Running systematic model coverage validation..."
        # Run subset of systematic model tests
        @testset "Systematic Model Development Validation" begin
            Random.seed!(42)
            include("testing_utilities.jl")
            
            df = make_econometric_data(n=400, seed=456)
            
            # Test key model types
            key_models = [
                (name="Simple LM", model=lm(@formula(log_wage ~ float_wage), df)),
                (name="Mixed LM", model=lm(@formula(log_wage ~ float_wage + gender), df)),
                (name="Interaction LM", model=lm(@formula(log_wage ~ float_wage * gender), df)),
                (name="Simple GLM", model=glm(@formula(union_member ~ float_wage), df, Binomial(), LogitLink())),
            ]
            
            for (name, model) in key_models
                @testset "$(name) development validation" begin
                    framework_result = test_2x2_framework_quadrants(model, df; test_name=name)
                    @test framework_result.all_successful
                    @test framework_result.all_finite
                end
            end
            
            @info "✓ Systematic model development validation complete"
        end
    else
        error("Unknown focus area: $focus. Use :all, :core, :backend, :integers, or :systematic")
    end
    
    @info "✅ Development validation complete"
    return true
end

"""
    run_release_validation()

Run complete comprehensive statistical validation for release testing.
Includes all tests, performance monitoring, and release diagnostics.

Expected duration: ~5 minutes
Coverage: Complete statistical correctness validation with performance monitoring
"""
function run_release_validation()
    @info "📦 Running Release Statistical Validation (Complete Suite)"
    @info "Expected duration: ~5 minutes"
    @info "This is the complete validation required for production release"
    
    include("release_validation.jl")
    
    @info "✅ Release validation complete - ready for production"
    return true
end

"""
    validation_status_report()

Generate a status report of the statistical validation framework.
Useful for understanding what tests are available and their purposes.
"""
function validation_status_report()
    @info "📊 Statistical Validation Framework Status Report"
    @info ""
    @info "Available Validation Levels:"
    @info "  🚀 CI Validation (run_ci_validation())"
    @info "     - Duration: < 30 seconds"
    @info "     - Purpose: Fast critical validation for CI/CD pipelines"
    @info "     - Coverage: Mathematical correctness, integers, backend consistency"
    @info ""
    @info "  🔧 Development Validation (run_development_validation(; focus=...))"
    @info "     - Duration: 30 seconds - 5 minutes"
    @info "     - Purpose: Targeted validation during development"
    @info "     - Focus areas: :all, :core, :backend, :integers, :systematic"
    @info ""
    @info "  📦 Release Validation (run_release_validation())"
    @info "     - Duration: ~5 minutes"  
    @info "     - Purpose: Complete validation for production release"
    @info "     - Coverage: Full statistical correctness + performance monitoring"
    @info ""
    @info "Framework Components:"
    @info "  ✓ ci_validation.jl - Fast critical subset"
    @info "  ✓ statistical_validation.jl - Complete 6-tier validation"
    @info "  ✓ backend_consistency.jl - AD vs FD consistency"
    @info "  ✓ release_validation.jl - Release-ready comprehensive suite"
    @info "  ✓ testing_utilities.jl - Core testing infrastructure"
    @info ""
    @info "Statistical Guarantees:"
    @info "  🎯 Publication-grade precision (1e-12 analytical validation)"
    @info "  🔢 Complete integer variable support for econometric data"
    @info "  ⚡ FormulaCompiler-level systematic coverage (23 scenarios)"
    @info "  🚫 Zero-tolerance policy for invalid statistical results"
    @info "  📊 Cross-platform numerical consistency"
end

# Export main control functions
export run_ci_validation, run_development_validation, run_release_validation, validation_status_report