# Current Implementation Issues in Margins.jl

**⚠️ STATISTICAL CORRECTNESS IS PARAMOUNT ⚠️**

This document tracks known limitations, approximations, and missing functionality in the current Margins.jl implementation. While the package is production-ready for most use cases, these issues represent **statistical validity gaps** that must be addressed to ensure rigorous econometric inference.

**Critical Principle**: Any approximation or fallback that affects standard error computation compromises statistical inference validity. Users relying on confidence intervals, hypothesis tests, or p-values need mathematically correct uncertainty estimates.

## 🔒 **ZERO TOLERANCE POLICY**

**FUNDAMENTAL PRINCIPLE**: Margins.jl maintains the highest standards of statistical rigor. Any approximation or fallback that affects standard error computation **must error out** rather than provide invalid results.

**🎯 ONGOING COMMITMENT**: All future development must adhere to this principle. **Wrong standard errors are worse than no standard errors.**
