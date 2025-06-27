# contrast_do_df.jl

"""
    contrast_to_df(cr::ContrastResult; level=0.95)

Convert a `ContrastResult` into a `DataFrame` with one row per contrast.  
Columns are:

  • `:Comparison` — text like `"a–b"`  
  • `:Estimate`   — difference in AMEs (θ₁−θ₂)  
  • `:StdErr`     — standard error of the difference  
  • `:t`          — t‐statistic (estimate ÷ StdErr)  
  • `:p`          — two‐sided p‐value  
  • `:lower,:upper` — CI bounds at the given level  
"""
function contrast_to_df(cr::ContrastResult; level=0.95)
    α = 1 - level
    crit = quantile(TDist(cr.df_residual), 1 - α/2)

    # build the label and CI vectors
    comps   = cr.comps
    labels  = [ join(string.(cmp), "–") for cmp in comps ]
    ests    = cr.estimate
    ses     = cr.se
    ts      = cr.t
    ps      = cr.p
    lowers  = ests .- crit .* ses
    uppers  = ests .+ crit .* ses

    return DataFrame(
      Comparison = labels,
      Estimate   = ests,
      StdErr     = ses,
      t          = ts,
      p          = ps,
      lower      = lowers,
      upper      = uppers,
    )
end

# ——————————————————————————————
# 1. DataFrame constructor extension
import DataFrames: DataFrame
DataFrame(cr::ContrastResult; kwargs...) = contrast_to_df(cr; kwargs...)

# Now you can do:
#    df = DataFrame(my_contrast_result)
#    df2 = DataFrame(my_contrast_result; level=0.90)

# ——————————————————————————————
# 2. Full Tables.jl integration
Tables.istable(::Type{ContrastResult}) = true

function Tables.schema(cr::ContrastResult)
    Tables.schema(contrast_to_df(cr))
end

Tables.rowiterator(cr::ContrastResult) = Tables.rowiterator(contrast_to_df(cr))

# These should work:
#    CSV.write("contrasts.csv", my_contrast_result)
#    Arrow.write("contrasts.arrow", my_contrast_result)
#    Query(@from c in my_contrast_result begin ... end)
