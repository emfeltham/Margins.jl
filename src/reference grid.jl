# reference grid.jl

"""

setup_refgrid(design, typs)

## Description

Use typicals to define the reference grid.

"""
function setup_refgrid(design, typs)
    rg = expand_grid(design);
    for (k, v) in typs
        if string(k) ∉ names(rg)
            rg[!, k] = [v for i in 1:nrow(rg)]
        end
    end
    return rg
end

function setup_contrast_grid(dsn::Dict, typicals::Dict)
    # Create two-row grid for each contrast variable
    contrast_dfs = []
    for (var, vals) in dsn
        # Validate exactly two contrast values
        length(vals) == 2 || error("Contrast variable $var requires exactly 2 values")
        
        # Create base scenario with typical values
        base = DataFrame([k => v for (k, v) in typicals])
        
        # Create reference and comparison scenarios
        ref = deepcopy(base)
        ref[!, var] .= vals[1]
        
        comp = deepcopy(base)
        comp[!, var] .= vals[2]
        
        # Combine scenarios with contrast labels
        # second entry - first entry
        scenarios = vcat(ref, comp)
        scenarios[!, :contrast_var] .= var
        scenarios[!, :scenario] = [-1, 1]
        
        push!(contrast_dfs, scenarios)
    end
    
    return vcat(contrast_dfs...)
end
