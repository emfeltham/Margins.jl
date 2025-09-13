"""
    apply_measure_transformation(effect_val, x_val, y_val, measure::Symbol)

Apply measure transformations to a marginal effect value.

# Arguments
- `effect_val`: The raw marginal effect (dy/dx)
- `x_val`: Variable value (x)  
- `y_val`: Predicted value (y, either μ or η depending on scale)
- `measure`: Measure type (`:effect`, `:elasticity`, `:semielasticity_dyex`, `:semielasticity_eydx`)

# Returns
- Transformed effect value according to the specified measure

# Examples
```julia
effect = 0.5
x = 2.0  
y = 1.0

apply_measure_transformation(effect, x, y, :effect)                # Returns 0.5
apply_measure_transformation(effect, x, y, :elasticity)           # Returns (2.0/1.0) * 0.5 = 1.0  
apply_measure_transformation(effect, x, y, :semielasticity_dyex)  # Returns 2.0 * 0.5 = 1.0
apply_measure_transformation(effect, x, y, :semielasticity_eydx)  # Returns (1/1.0) * 0.5 = 0.5
```
"""
function apply_measure_transformation(effect_val, x_val, y_val, measure::Symbol)
    if measure === :effect
        return effect_val
    elseif measure === :elasticity
        return (x_val / y_val) * effect_val
    elseif measure === :semielasticity_dyex
        return x_val * effect_val  
    elseif measure === :semielasticity_eydx
        return (1 / y_val) * effect_val
    else
        error("Unsupported measure: $measure")
    end
end