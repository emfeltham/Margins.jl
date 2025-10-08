#!/bin/bash
# run_comparison_auto.sh
# Fully automated R comparison study workflow (no interactive prompts)
# WARNING: Assumes factor levels in r_model.R are already correct

set -e  # Exit on error

echo "================================================================================"
echo "R Comparison Study - Automated Workflow (Non-Interactive)"
echo "================================================================================"
echo ""
echo "WARNING: This assumes factor levels in r_model.R are already updated!"
echo ""

# Check if we're in the right directory
if [ ! -f "test/r_compare/generate_data.jl" ]; then
    echo "Error: Must run from Margins.jl root directory"
    echo "Usage: bash test/r_compare/run_comparison_auto.sh"
    exit 1
fi

# Step 1: Generate data
echo "Step 1/4: Generating synthetic data..."
echo "--------------------------------------------------------------------------------"
julia --project=. test/r_compare/generate_data.jl
echo ""

# Step 2: Run Julia model
echo "Step 2/4: Fitting model in Julia..."
echo "--------------------------------------------------------------------------------"
julia --project=. test/r_compare/julia_model.jl
echo ""

# Step 3: Run R model
echo "Step 3/4: Fitting model in R..."
echo "--------------------------------------------------------------------------------"
Rscript test/r_compare/r_model.R
echo ""

# Step 4: Compare results
echo "Step 4/4: Comparing results..."
echo "--------------------------------------------------------------------------------"
julia --project=. test/r_compare/compare_results.jl
echo ""

echo "================================================================================"
echo "✓ Comparison study complete!"
echo "================================================================================"
echo ""
echo "Results are available in test/r_compare/:"
echo "  - julia_coefficients.csv, r_coefficients.csv (model parameters)"
echo "  - julia_*.csv, r_*.csv (marginal effects)"
echo "  - r_benchmarks.rds (performance benchmarks)"
echo ""
