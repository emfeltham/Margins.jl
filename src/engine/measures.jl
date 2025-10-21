"""
    apply_measure_transformation(effect_val, x_val, y_val, measure::Symbol)

Apply measure transformations to a marginal effect value.

# Arguments
- `effect_val`: The raw marginal effect (dy/dx)
- `x_val`: Variable value (x)
- `y_val`: Predicted value (y, either μ or η depending on scale)
- `measure`: Measure type (`:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`)

# Returns
- Tuple of `(transformed_value, transformation_factor)` where:
  - `transformed_value`: The transformed effect value
  - `transformation_factor`: The scaling factor used (for gradient scaling)

# Examples
```julia
effect = 0.5
x = 2.0
y = 1.0

apply_measure_transformation(effect, x, y, :effect)                # Returns (0.5, 1.0)
apply_measure_transformation(effect, x, y, :elasticity)           # Returns (1.0, 2.0)
apply_measure_transformation(effect, x, y, :semielasticity_dyex)  # Returns (1.0, 2.0)
apply_measure_transformation(effect, x, y, :semielasticity_eydx)  # Returns (0.5, 1.0)
```
"""
function apply_measure_transformation(effect_val, x_val, y_val, measure::Symbol)
    if measure === :effect
        return (effect_val, 1.0)
    elseif measure === :elasticity
        factor = x_val / y_val
        return (factor * effect_val, factor)
    elseif measure === :semielasticity_dyex
        factor = x_val
        return (factor * effect_val, factor)
    elseif measure === :semielasticity_eydx
        factor = 1.0 / y_val
        return (factor * effect_val, factor)
    else
        error("Unsupported measure: $measure")
    end
end