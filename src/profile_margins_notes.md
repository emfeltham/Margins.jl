# profile_margins notes.md

in core.jl

"# Note: DataFrame method removed to avoid dispatch ambiguity"
- this was INCORRECTLY removed, based on a mistake that DataFrames ambiguity was causing this (`DataFrame` wasn't properly imported for extension, now it is.)

I had originally envisioned the following. Ultimately, a reference grid (table-like) is built, regardless of the input style, for this method. So, I wanted a `profile_margins` that took a reference grid (as a Table or DataFrame) directly -- where the user could specify. Then, I wanted another convenience method that called this one, that allowed the user to specify `at` with a dict, which called some other function to build a reference grid containing the unique combinations of the variables specified in at.

How far are we from this?

ANSWER:
✅ IMPLEMENTED - We now have exactly the two-method design you envisioned:

1. **Core method**: `profile_margins(model, data, reference_grid::DataFrame; ...)` 
   - Takes reference grid directly for maximum control and efficiency
   - Bypasses grid building for complex scenarios

2. **Convenience method**: `profile_margins(model, data; at=:means, ...)` 
   - Uses `at` parameter to build reference grid internally
   - Calls `_build_reference_grid()` then routes to same internal implementation

Both methods now work correctly and dispatch properly. The DataFrame import confusion was resolved and the architecture matches your original vision perfectly.