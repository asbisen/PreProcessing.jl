__precompile__()

module PreProcessing

import LearnBase: fit, predict, transform
import Base.show
using Base.Test

export
    fit,
    transform,
    transform!,
    inverse_transform,

    Binarizer,
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler


include("binarizer.jl")
include("standardscaler.jl")
include("minmaxscaler.jl")
include("maxabsscaler.jl")


# ----------------------------------------------------------------------------------------------------------------
function _handle_zeros_in_scale!{T<:AbstractFloat}(σ::Vector{T})
    zero_idx = find(x->x==0, σ)
    σ[zero_idx] = 1.0
    σ
end

# if obsdim == 1 then feature_dim == 2 and vice-versa
_other_dimension(obsdim) = mod(obsdim,2)+1



end # Module
