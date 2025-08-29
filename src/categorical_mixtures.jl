# categorical_mixtures.jl

"""
    CategoricalMixture{T}

Represents a mixture of categorical levels with associated weights for profile analysis.
Used to specify population composition scenarios in Dict-based profile margins.

# Fields
- `levels::Vector{T}`: Categorical levels (strings, symbols, or other types)
- `weights::Vector{Float64}`: Associated weights (must sum to 1.0)

# Example
```julia
edu_mix = CategoricalMixture(["high_school", "college"], [0.6, 0.4])
```
"""
struct CategoricalMixture{T}
    levels::Vector{T}
    weights::Vector{Float64}
    
    function CategoricalMixture(levels::Vector{T}, weights::Vector{Float64}) where T
        # Validation
        length(levels) == length(weights) || 
            throw(ArgumentError("levels and weights must have same length"))
        all(weights .≥ 0.0) || 
            throw(ArgumentError("all weights must be non-negative"))
        abs(sum(weights) - 1.0) < 1e-10 || 
            throw(ArgumentError("weights must sum to 1.0, got $(sum(weights))"))
        length(unique(levels)) == length(levels) || 
            throw(ArgumentError("levels must be unique"))
            
        new{T}(levels, weights)
    end
end

"""
    mix(pairs...)

Convenient constructor for CategoricalMixture from level => weight pairs.

# Example
```julia
# These are equivalent
mix("A" => 0.3, "B" => 0.7)
CategoricalMixture(["A", "B"], [0.3, 0.7])

# Multiple ways to specify education composition
mix("high_school" => 0.4, "college" => 0.4, "graduate" => 0.2)
mix(:urban => 0.7, :rural => 0.3)
mix(true => 0.6, false => 0.4)  # Bool mixture
```
"""
function mix(pairs...)
    isempty(pairs) && throw(ArgumentError("mix() requires at least one level => weight pair"))
    
    levels = [k for (k, v) in pairs]
    weights = [v for (k, v) in pairs]
    return CategoricalMixture(levels, weights)
end

# Convenient display
function Base.show(io::IO, m::CategoricalMixture)
    pairs_str = join(["$(repr(l)) => $(w)" for (l, w) in zip(m.levels, m.weights)], ", ")
    print(io, "mix(", pairs_str, ")")
end

# Support for indexing and iteration if needed
Base.length(m::CategoricalMixture) = length(m.levels)
Base.getindex(m::CategoricalMixture, i) = (m.levels[i], m.weights[i])
Base.iterate(m::CategoricalMixture, state=1) = state > length(m) ? nothing : (m[state], state + 1)

"""
    _validate_mixture_against_data(mixture::CategoricalMixture, col, var::Symbol)

Validate that all levels in the mixture exist in the actual data column.
Throws ArgumentError if any mixture levels are not found in the data.
"""
function _validate_mixture_against_data(mixture::CategoricalMixture, col, var::Symbol)
    # Get actual levels from data
    actual_levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
        string.(CategoricalArrays.levels(col))
    elseif eltype(col) <: Bool
        string.([false, true])
    else
        unique(string.(col))
    end
    
    # Check that all mixture levels exist in data
    mixture_levels_str = string.(mixture.levels)
    missing_levels = setdiff(mixture_levels_str, actual_levels)
    if !isempty(missing_levels)
        throw(ArgumentError("Variable :$var mixture contains levels not found in data: $missing_levels. Available levels: $actual_levels"))
    end
    
    return true
end

"""
    _mixture_to_scenario_value(mixture::CategoricalMixture, original_col)

Convert a categorical mixture to a representative value for FormulaCompiler scenario creation.
Uses weighted average encoding to provide a smooth, continuous representation for derivatives.

# Strategy
- **CategoricalArray**: Weighted average of level indices
- **Bool**: Probability of true (equivalent to current fractional Bool support)
- **Other**: Uses first level as representative (can be enhanced)
"""
function _mixture_to_scenario_value(mixture::CategoricalMixture, original_col)
    if Base.find_package("CategoricalArrays") !== nothing && (original_col isa CategoricalArrays.CategoricalArray)
        # Get level mapping
        actual_levels = string.(CategoricalArrays.levels(original_col))
        level_indices = Dict(level => i for (i, level) in enumerate(actual_levels))
        
        # Compute weighted average of indices
        mixture_levels_str = string.(mixture.levels)
        weighted_sum = sum(mixture.weights[i] * level_indices[mixture_levels_str[i]] 
                          for i in 1:length(mixture.levels))
        
        return weighted_sum
    elseif eltype(original_col) <: Bool
        # Handle Bool as special case - return probability of true
        level_weight_dict = Dict(string.(mixture.levels) .=> mixture.weights)
        false_weight = get(level_weight_dict, "false", 0.0)
        true_weight = get(level_weight_dict, "true", 0.0)
        
        # Validate Bool levels
        if !issubset(keys(level_weight_dict), ["false", "true"])
            throw(ArgumentError("Bool variable mixture must use levels 'false' and 'true' or false and true"))
        end
        
        return true_weight  # Probability of true (matches existing fractional Bool support)
    else
        # Generic categorical - for now, use weighted average of sorted unique values
        # This provides a continuous representation for derivatives
        unique_levels = sort(unique(string.(original_col)))
        level_indices = Dict(level => i for (i, level) in enumerate(unique_levels))
        
        mixture_levels_str = string.(mixture.levels)
        weighted_sum = sum(mixture.weights[i] * level_indices[mixture_levels_str[i]] 
                          for i in 1:length(mixture.levels))
        
        return weighted_sum
    end
end

"""
    MixtureWithLevels{T}

Wrapper that includes original categorical levels with the mixture for FormulaCompiler processing.
"""
struct MixtureWithLevels{T}
    mixture::CategoricalMixture{T}
    original_levels::Vector{String}
end

# Provide access to mixture properties for duck typing in FormulaCompiler
Base.getproperty(mwl::MixtureWithLevels, sym::Symbol) = 
    sym === :levels ? getfield(mwl, :mixture).levels :
    sym === :weights ? getfield(mwl, :mixture).weights :
    sym === :original_levels ? getfield(mwl, :original_levels) :
    getfield(mwl, sym)

Base.hasproperty(::MixtureWithLevels, sym::Symbol) = 
    sym in (:levels, :weights, :original_levels, :mixture)

"""
    _process_profile_for_scenario(prof::Dict, data_nt::NamedTuple)

Process a profile dictionary, passing CategoricalMixture objects directly to FormulaCompiler
with original level information for proper weighted contrast computation.
"""
function _process_profile_for_scenario(prof::Dict, data_nt::NamedTuple)
    processed_prof = Dict{Symbol,Any}()
    
    for (k, v) in prof
        if v isa CategoricalMixture
            # Validate mixture against data and get original levels
            col = getproperty(data_nt, k)
            _validate_mixture_against_data(v, col, k)
            
            # Get original categorical levels
            original_levels = if Base.find_package("CategoricalArrays") !== nothing && (col isa CategoricalArrays.CategoricalArray)
                string.(CategoricalArrays.levels(col))
            elseif eltype(col) <: Bool
                string.([false, true])
            else
                unique(string.(col))
            end
            
            # Pass mixture with original levels info
            processed_prof[k] = MixtureWithLevels(v, original_levels)
        else
            processed_prof[k] = v
        end
    end
    
    return processed_prof
end