# mueta2.jl

import GLM: 
    linkinv, IdentityLink, LogLink, LogitLink, ProbitLink, CloglogLink,
    InverseLink, InverseSquareLink, SqrtLink, PowerLink, Link
using Distributions: Normal, pdf

"""
    mueta2(link::Link, η) -> Real

Compute the second derivative d²μ/dη² of the inverse link function μ(η).

This function provides analytical second derivatives for all standard GLM link functions,
with ForwardDiff fallback for custom link functions.

# Implemented Links
- IdentityLink: d²μ/dη² = 0
- LogLink: d²μ/dη² = exp(η) 
- LogitLink: d²μ/dη² = μ(1-μ)(1-2μ)
- ProbitLink: d²μ/dη² = -η·φ(η) where φ is standard normal PDF
- CloglogLink: d²μ/dη² = exp(η-exp(η))(1-exp(η))
- InverseLink: d²μ/dη² = 2/η³
- InverseSquareLink: d²μ/dη² = 3/(4η^{5/2})
- SqrtLink: d²μ/dη² = 2
- PowerLink: d²μ/dη² = p(p-1)η^{p-2}

# Arguments
- `link::Link`: GLM link object
- `η`: Linear predictor value(s)

# Returns
Second derivative of inverse link function at η
"""
# — Explicit second‐derivatives (Block 2 style) —
mueta2(::IdentityLink, η) = zero(η)
mueta2(::LogLink, η)      = exp(η)
mueta2(L::LogitLink, η)   =
    begin
      μ = linkinv(L, η)
      μ*(1-μ)*(1-2μ)
    end
mueta2(::ProbitLink, η)   = -η*pdf(Normal(), η)
mueta2(::CloglogLink, η)  =
    begin
      t = exp(η)
      exp(η - t)*(1 - t)
    end
mueta2(::InverseLink, η)  = 2/η^3
mueta2(::InverseSquareLink, η, ) = 3/(4*η^(5/2))
mueta2(::SqrtLink, η)     = 2
mueta2(L::PowerLink, η)   = L.p*(L.p - 1).*η.^(L.p - 2)

# — Fallback: AD for any other link —
mueta2(link::Link, η)     =
    ForwardDiff.derivative(η_val -> linkinv(link, η_val), η)
