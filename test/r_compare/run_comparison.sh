#!/bin/bash
# run_comparison.sh
# Complete R comparison study workflow
# Runs data generation, Julia analysis, R analysis, and validation

set -e  # Exit on error

echo "================================================================================"
echo "R Comparison Study - Automated Workflow"
echo "================================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "test/r_compare/generate_data.jl" ]; then
    echo "Error: Must run from Margins.jl root directory"
    echo "Usage: bash test/r_compare/run_comparison.sh"
    exit 1
fi

# Step 1: Generate data
echo "Step 1/4: Generating synthetic data..."
echo "--------------------------------------------------------------------------------"
julia --project=. test/r_compare/generate_data.jl
if [ $? -ne 0 ]; then
    echo "Error: Data generation failed"
    exit 1
fi
echo ""

# Step 2: Run Julia model
echo "Step 2/4: Fitting model in Julia..."
echo "--------------------------------------------------------------------------------"
julia --project=. test/r_compare/julia_model.jl
if [ $? -ne 0 ]; then
    echo "Error: Julia model fitting failed"
    exit 1
fi
echo ""

# Step 3: Check if R factor levels need updating
echo "Step 3/4: Preparing R analysis..."
echo "--------------------------------------------------------------------------------"
echo "⚠️  IMPORTANT: Verify that factor levels in r_model.R match the Julia output above"
echo ""
read -p "Have you updated factor levels in r_model.R? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please update factor levels in test/r_compare/r_model.R before continuing."
    echo "Run the following commands manually:"
    echo "  1. Check the categorical levels printed by generate_data.jl above"
    echo "  2. Edit test/r_compare/r_model.R lines ~62 and ~66"
    echo "  3. Rerun: bash test/r_compare/run_comparison.sh"
    exit 1
fi
echo ""

# Step 4: Run R model
echo "Fitting model in R..."
echo "--------------------------------------------------------------------------------"
Rscript test/r_compare/r_model.R
if [ $? -ne 0 ]; then
    echo "Error: R model fitting failed"
    exit 1
fi
echo ""

# Step 5: Compare results
echo "Step 4/4: Comparing results..."
echo "--------------------------------------------------------------------------------"
julia --project=. test/r_compare/compare_results.jl
if [ $? -ne 0 ]; then
    echo "Error: Results comparison failed"
    exit 1
fi
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
