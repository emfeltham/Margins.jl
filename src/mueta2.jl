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

# Identity: μ(η) = η
function mueta2(::IdentityLink, η)
    zero(η)
end

# Log: μ(η) = exp(η)
function mueta2(::LogLink, η)
    exp(η)
end

# Logit: μ(η) = 1/(1+exp(-η))
function mueta2(L::LogitLink, η)
    μ = linkinv(L, η)
    μ * (1 - μ) * (1 - 2μ)
end

# Probit: μ(η) = Φ(η)
function mueta2(::ProbitLink, η)
    -η * pdf(Normal(), η)
end

# CLogLog: μ(η) = 1 - exp(-exp(η))
function mueta2(::CloglogLink, η)
    t = exp(η)
    exp(η - t) * (1 - t)
end

# Inverse: μ(η) = 1/η
function mueta2(::InverseLink, η)
    2 / η^3
end

# Inverse Square: μ(η) = 1/√η
function mueta2(::InverseSquareLink, η)
    3 / (4 * η^(5/2))
end

# Sqrt: μ(η) = η^2
function mueta2(::SqrtLink, η)
    2
end

# Power: μ(η) = η^p
function mueta2(L::PowerLink, η)
    p = L.p
    p * (p - 1) .* η .^ (p - 2)
end

# Fallback: numeric second derivative for custom/unsupported links
function mueta2(L::Link, η)
    δ = sqrt(eps(eltype(η)))
    f(x) = linkinv(L, x)
    (f(η + δ) + f(η - δ) - 2f(η)) / δ^2
end
