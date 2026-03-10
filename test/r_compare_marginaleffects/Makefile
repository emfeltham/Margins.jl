.PHONY: all performance performance-large data data-large clean clean-large help

all: performance

# Generate 5K dataset
data:
	julia --project=. generate_data.jl

# Generate 500K dataset
data-large:
	julia --project=. generate_data_large.jl

# Run performance benchmark (5K dataset)
performance: data
	@echo "Running Julia benchmarks..."
	julia --project=. performance_benchmark.jl
	@echo ""
	@echo "Running R marginaleffects benchmarks..."
	Rscript r_marginaleffects_benchmarks.R
	@echo ""
	@echo "Comparing performance..."
	julia --project=. compare_performance_marginaleffects.jl

# Run large-scale performance benchmark (500K dataset)
performance-large: data-large
	@echo "WARNING: Large dataset benchmark may take 30+ minutes"
	@echo ""
	@echo "Running Julia benchmarks (large)..."
	julia --project=. performance_benchmark_large.jl
	@echo ""
	@echo "Running R marginaleffects benchmarks (large)..."
	Rscript r_marginaleffects_benchmarks_large.R
	@echo ""
	@echo "Comparing performance (large)..."
	julia --project=. compare_performance_marginaleffects_large.jl

# Clean generated files (5K dataset)
clean:
	rm -f r_comparison_data.csv
	rm -f julia_benchmarks.csv
	rm -f r_marginaleffects_benchmarks.rds
	rm -f performance_comparison_marginaleffects.csv

# Clean generated files (500K dataset)
clean-large:
	rm -f r_comparison_data_large.csv
	rm -f julia_benchmarks_large.csv
	rm -f r_marginaleffects_benchmarks_large.rds
	rm -f performance_comparison_marginaleffects_large.csv

# Help message
help:
	@echo "Margins.jl vs marginaleffects Performance Comparison Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all                - Run performance benchmark (5K dataset)"
	@echo "  make performance        - Run performance benchmark (5K dataset)"
	@echo "  make performance-large  - Run large-scale benchmark (500K dataset)"
	@echo "  make data               - Generate 5K dataset only"
	@echo "  make data-large         - Generate 500K dataset only"
	@echo "  make clean              - Remove generated files (5K)"
	@echo "  make clean-large        - Remove generated files (500K)"
	@echo "  make help               - Show this help message"
	@echo ""
	@echo "Note: marginaleffects is the state-of-the-art R package"
	@echo "      (~1000× faster than the older margins package)"
