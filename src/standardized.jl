# copy functions to make modelcols_alt work with StandardizedPredictors.jl

# from scaling.jl
modelcols_alt(t::ScaledTerm, d::NamedTuple) = modelcols_alt(t.term, d) ./ t.scale

# centering.jl
modelcols_alt(t::CenteredTerm, d::NamedTuple) = modelcols_alt(t.term, d) .- t.center

# zscoring.jl
function modelcols_alt(t::ZScoredTerm, d::NamedTuple)
    return zscore(modelcols_alt(t.term, d), t.center, t.scale)
end
