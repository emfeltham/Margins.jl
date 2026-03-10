#!/bin/bash
# run_performance_marginaleffects.sh
# Automated performance benchmark workflow for Margins.jl vs R marginaleffects
set -e

echo "================================================================================"
echo "Margins.jl vs marginaleffects Performance Benchmark"
echo "================================================================================"
echo ""

# Step 1: Generate data (5K dataset)
echo "Step 1: Generating data (5K observations)..."
julia --project=. generate_data.jl
echo "✓ Data generated"
echo ""

# Step 2: Run Julia benchmarks
echo "Step 2: Running Julia benchmarks..."
julia --project=. performance_benchmark.jl
echo "✓ Julia benchmarks complete"
echo ""

# Step 3: Run R marginaleffects benchmarks
echo "Step 3: Running R marginaleffects benchmarks..."
echo "   (This may take several minutes...)"
Rscript r_marginaleffects_benchmarks.R
echo "✓ R benchmarks complete"
echo ""

# Step 4: Compare performance
echo "Step 4: Comparing performance..."
julia --project=. compare_performance_marginaleffects.jl
echo ""

echo "================================================================================"
echo "BENCHMARK COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - julia_benchmarks.csv"
echo "  - r_marginaleffects_benchmarks.rds"
echo "  - performance_comparison_marginaleffects.csv"
echo ""
