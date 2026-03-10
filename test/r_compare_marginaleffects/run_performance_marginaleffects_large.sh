#!/bin/bash
# run_performance_marginaleffects_large.sh
# Automated large-scale performance benchmark workflow (500K observations)
set -e

echo "================================================================================"
echo "Margins.jl vs marginaleffects LARGE-SCALE Performance Benchmark"
echo "================================================================================"
echo ""

echo "WARNING: This benchmark uses a 500K observation dataset."
echo "         Both Julia and R may require significant time and memory."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Step 1: Generate large dataset
echo "Step 1: Generating large dataset (500K observations)..."
julia --project=. generate_data_large.jl
echo "✓ Large dataset generated"
echo ""

# Step 2: Run Julia benchmarks
echo "Step 2: Running Julia benchmarks on large dataset..."
julia --project=. performance_benchmark_large.jl
echo "✓ Julia benchmarks complete"
echo ""

# Step 3: Run R marginaleffects benchmarks
echo "Step 3: Running R marginaleffects benchmarks on large dataset..."
echo "   (This may take considerable time - potentially 30+ minutes...)"
Rscript r_marginaleffects_benchmarks_large.R
echo "✓ R benchmarks complete"
echo ""

# Step 4: Compare performance
echo "Step 4: Comparing performance..."
julia --project=. compare_performance_marginaleffects_large.jl
echo ""

echo "================================================================================"
echo "LARGE-SCALE BENCHMARK COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - julia_benchmarks_large.csv"
echo "  - r_marginaleffects_benchmarks_large.rds"
echo "  - performance_comparison_marginaleffects_large.csv"
echo ""
