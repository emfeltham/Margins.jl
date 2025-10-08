# Makefile for R Comparison Study
# Run complete validation workflow

.PHONY: all data julia r compare performance data-large performance-large clean clean-large help

# Default target - run everything
all: data julia r compare
	@echo ""
	@echo "================================================================================"
	@echo "✓ Complete comparison study finished!"
	@echo "================================================================================"

# Performance benchmark - speed and memory only
performance: data
	@echo ""
	@echo "================================================================================"
	@echo "Running performance benchmarks..."
	@echo "================================================================================"
	@echo ""
	@echo "Step 1: Julia benchmark..."
	julia --project=. performance_benchmark.jl
	@echo ""
	@echo "Step 2: R benchmark..."
	Rscript r_benchmarks.R
	@echo ""
	@echo "Step 3: Comparing performance..."
	julia --project=. compare_performance.jl
	@echo ""
	@echo "================================================================================"
	@echo "✓ Performance comparison complete!"
	@echo "================================================================================"

# Generate synthetic data (5K observations for correctness validation)
data:
	@echo "================================================================================"
	@echo "Generating synthetic data (5K observations)..."
	@echo "================================================================================"
	julia --project=. generate_data.jl

# Generate large dataset (500K observations for performance testing)
data-large:
	@echo "================================================================================"
	@echo "Generating large dataset (500K observations)..."
	@echo "================================================================================"
	julia --project=. generate_data_large.jl

# Fit model in Julia
julia: data
	@echo ""
	@echo "================================================================================"
	@echo "Fitting model in Julia..."
	@echo "================================================================================"
	julia --project=. julia_model.jl

# Fit model in R
r: julia
	@echo ""
	@echo "================================================================================"
	@echo "Fitting model in R..."
	@echo "================================================================================"
	@echo "WARNING: Ensure factor levels in r_model.R are updated!"
	@echo ""
	Rscript r_model.R

# Compare results
compare: r
	@echo ""
	@echo "================================================================================"
	@echo "Comparing results..."
	@echo "================================================================================"
	julia --project=. compare_results.jl

# Large-scale performance benchmark (Julia vs R, 500K observations)
performance-large: data-large
	@echo ""
	@echo "================================================================================"
	@echo "Running large-scale performance benchmark (500K observations)..."
	@echo "================================================================================"
	@echo ""
	@echo "Step 1: Julia benchmark..."
	julia --project=. performance_benchmark_large.jl
	@echo ""
	@echo "Step 2: R benchmark..."
	Rscript r_benchmarks_large.R
	@echo ""
	@echo "Step 3: Comparing performance..."
	julia --project=. compare_performance_large.jl
	@echo ""
	@echo "================================================================================"
	@echo "✓ Large-scale performance comparison complete!"
	@echo "================================================================================"

# Clean up generated files (5K dataset)
clean:
	@echo "Cleaning up generated files (5K dataset)..."
	rm -f r_comparison_data.csv
	rm -f julia_*.csv
	rm -f r_*.csv
	rm -f r_*.rds
	@echo "✓ Cleaned"

# Clean up large dataset files
clean-large:
	@echo "Cleaning up large dataset files..."
	rm -f r_comparison_data_large.csv
	rm -f julia_benchmarks_large.csv
	rm -f r_benchmarks_large.rds
	rm -f performance_comparison_large.csv
	@echo "✓ Large dataset files cleaned"

# Help message
help:
	@echo "R Comparison Study - Makefile"
	@echo ""
	@echo "Correctness Validation (5K observations):"
	@echo "  make all              - Run complete workflow (data → julia → r → compare)"
	@echo "  make data             - Generate 5K dataset for correctness validation"
	@echo "  make julia            - Generate data and fit Julia model"
	@echo "  make r                - Run data + Julia + R models"
	@echo "  make compare          - Run everything and compare results"
	@echo "  make performance      - Run Julia vs R performance comparison (5K dataset)"
	@echo "  make clean            - Remove correctness validation files"
	@echo ""
	@echo "Large-Scale Performance Testing (500K observations):"
	@echo "  make data-large       - Generate 500K dataset"
	@echo "  make performance-large - Run Julia vs R performance comparison on 500K dataset"
	@echo "  make clean-large      - Remove large dataset files"
	@echo ""
	@echo "  make help             - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  Correctness:             make all"
	@echo "  Julia vs R performance:  make performance"
	@echo "  Large-scale scaling:     make performance-large"
	@echo ""
	@echo "Note: Before running 'make r', ensure factor levels in r_model.R"
	@echo "      match the output from 'make data'"
