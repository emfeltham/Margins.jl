# Test EffectsBuffers container for zero-allocation reuse
using Test, Margins, BenchmarkTools
using Margins: EffectsBuffers, ensure_capacity!, reset!, get_results_view

@testset "EffectsBuffers Tests" begin
    @testset "Construction" begin
        # Test basic construction
        buffers = EffectsBuffers(3, 5)
        @test buffers.capacity_vars == 3
        @test buffers.capacity_params == 5
        @test length(buffers.estimates) == 3
        @test length(buffers.standard_errors) == 3
        @test size(buffers.gradients) == (3, 5)
        @test length(buffers.var_indices) == 3
        @test length(buffers.variables) == 3
        @test buffers.row_count[] == 0

        # Test construction from engine variables and coefficients
        engine_vars = [:x1, :x2, :x3, :x4]
        β = rand(6)
        buffers2 = EffectsBuffers(engine_vars, β)
        @test buffers2.capacity_vars == 4
        @test buffers2.capacity_params == 6
        @test size(buffers2.gradients) == (4, 6)
        @test length(buffers2.variables) == 4
    end

    @testset "Capacity Management" begin
        buffers = EffectsBuffers(2, 3)

        # Test capacity sufficient - should not resize
        resized = ensure_capacity!(buffers, 2, 3)
        @test !resized
        @test buffers.capacity_vars == 2
        @test buffers.capacity_params == 3

        # Test capacity sufficient - smaller request
        resized = ensure_capacity!(buffers, 1, 2)
        @test !resized

        # Test variable capacity expansion
        resized = ensure_capacity!(buffers, 4, 3)
        @test resized
        @test buffers.capacity_vars == 4
        @test length(buffers.estimates) == 4
        @test length(buffers.standard_errors) == 4
        @test length(buffers.var_indices) == 4
        @test length(buffers.variables) == 4
        @test size(buffers.gradients, 1) >= 4

        # Test parameter capacity expansion
        resized = ensure_capacity!(buffers, 4, 6)
        @test resized
        @test buffers.capacity_params == 6
        @test size(buffers.gradients, 2) >= 6

        # Test simultaneous expansion
        buffers_small = EffectsBuffers(1, 1)
        resized = ensure_capacity!(buffers_small, 5, 8)
        @test resized
        @test buffers_small.capacity_vars == 5
        @test buffers_small.capacity_params == 8
        @test size(buffers_small.gradients) == (5, 8)
        @test length(buffers_small.variables) == 5
    end

    @testset "Reset Functionality" begin
        buffers = EffectsBuffers(3, 4)

        # Fill with non-zero values
        fill!(buffers.estimates, 5.0)
        fill!(buffers.standard_errors, 2.0)
        fill!(buffers.gradients, 1.0)
        buffers.row_count[] = 100

        # Reset and verify zeroing
        reset!(buffers, 3)
        @test all(buffers.estimates[1:3] .== 0.0)
        @test all(buffers.standard_errors[1:3] .== 0.0)
        @test all(buffers.gradients[1:3, :] .== 0.0)
        @test buffers.row_count[] == 0

        # Test partial reset
        fill!(buffers.estimates, 7.0)
        reset!(buffers, 2)  # Only reset first 2 variables
        @test all(buffers.estimates[1:2] .== 0.0)
        @test buffers.estimates[3] == 7.0  # Third should be unchanged
    end

    @testset "Results Views" begin
        buffers = EffectsBuffers(4, 3)

        # Fill with test data
        buffers.estimates .= [1.0, 2.0, 3.0, 4.0]
        buffers.standard_errors .= [0.1, 0.2, 0.3, 0.4]
        buffers.gradients .= reshape(1.0:12.0, 4, 3)
        buffers.variables .= [:x1, :x2, :x3, :x4]

        # Get views for subset
        est_view, se_view, grad_view = get_results_view(buffers, 3)

        @test est_view == [1.0, 2.0, 3.0]
        @test se_view == [0.1, 0.2, 0.3]
        @test size(grad_view) == (3, 3)
        @test grad_view[1, :] == [1.0, 5.0, 9.0]

        # Verify views are actually views (modification test)
        est_view[1] = 99.0
        @test buffers.estimates[1] == 99.0
        buffers.variables[1] = :z1
        @test buffers.variables[1] == :z1
    end

    @testset "Zero-Allocation Reuse" begin
        buffers = EffectsBuffers(3, 4)

        # First use - may allocate for warmup
        ensure_capacity!(buffers, 3, 4)
        reset!(buffers, 3)

        # Second use - should be zero allocations using BenchmarkTools
        b1 = @benchmark ensure_capacity!($buffers, 3, 4) samples=50 evals=1
        @test minimum(b1.memory) == 0

        b2 = @benchmark reset!($buffers, 3) samples=50 evals=1
        @test minimum(b2.memory) == 0

        b3 = @benchmark get_results_view($buffers, 3) samples=50 evals=1
        @test minimum(b3.memory) == 0

        # Test with smaller request (should still be zero)
        b4 = @benchmark ensure_capacity!($buffers, 2, 3) samples=50 evals=1
        @test minimum(b4.memory) == 0

        b5 = @benchmark reset!($buffers, 2) samples=50 evals=1
        @test minimum(b5.memory) == 0
    end

    @testset "Data Preservation During Resize" begin
        buffers = EffectsBuffers(2, 2)

        # Set initial data
        buffers.estimates .= [10.0, 20.0]
        buffers.standard_errors .= [1.0, 2.0]
        buffers.gradients .= [100.0 200.0; 300.0 400.0]
        buffers.variables .= [:x1, :x2]

        # Expand and verify data preservation
        ensure_capacity!(buffers, 4, 4)

        @test buffers.estimates[1:2] == [10.0, 20.0]
        @test buffers.standard_errors[1:2] == [1.0, 2.0]
        @test buffers.gradients[1:2, 1:2] == [100.0 200.0; 300.0 400.0]

        # Verify expanded capacity exists
        @test length(buffers.estimates) >= 4
        @test size(buffers.gradients, 1) >= 4
        @test size(buffers.gradients, 2) >= 4
        @test length(buffers.variables) >= 4
    end

    @testset "Realistic Usage Pattern" begin
        # Simulate realistic usage: multiple calls with varying sizes
        buffers = EffectsBuffers(2, 3)

        # First computation: 2 vars, 3 params
        ensure_capacity!(buffers, 2, 3)
        reset!(buffers, 2)
        buffers.estimates[1:2] .= [1.1, 2.2]
        est1, _, _ = get_results_view(buffers, 2)
        @test est1 == [1.1, 2.2]

        # Second computation: 4 vars, 5 params (expansion needed)
        ensure_capacity!(buffers, 4, 5)
        reset!(buffers, 4)
        buffers.estimates[1:4] .= [3.3, 4.4, 5.5, 6.6]
        est2, _, _ = get_results_view(buffers, 4)
        @test est2 == [3.3, 4.4, 5.5, 6.6]

        # Third computation: 3 vars, 4 params (no expansion needed)
        b_third = @benchmark begin
            ensure_capacity!($buffers, 3, 4)
            reset!($buffers, 3)
            get_results_view($buffers, 3)
        end samples=50 evals=1
        @test minimum(b_third.memory) == 0
    end
end
