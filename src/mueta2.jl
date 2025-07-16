# mueta2.jl

import GLM: linkinv, IdentityLink, LogLink, LogitLink, ProbitLink, CloglogLink,
            InverseLink, InverseSquareLink, SqrtLink, PowerLink, Link
using Distributions: Normal, pdf

@doc raw"""
Compute the second derivative ``d^2\mu / d\eta^2`` of the inverse link function ``\mu(\eta)`` for each GLM.jl `Link` type.

# Arguments
- `L::Link`: A GLM.jl link object.
- `η`: Linear predictor value(s).

# Returns
- The analytic second derivative of the inverse link at `η`.
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
