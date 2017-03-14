
# ----------------------------------------------------------------------------------------------------------------
# Description:
#
#
# TODO:
#   - [x] Finish documenting the code
#   - [ ] Check the correctness of code
#   - [x] Add inverse_transform() function to get to the original matrix
# ----------------------------------------------------------------------------------------------------------------
#
# function _handle_zeros_in_scale!{T<:AbstractFloat}(σ::Vector{T})
#   zero_idx = find(x->x==0, σ)
#   σ[zero_idx] = 1.0
#   σ
# end
#
# # if obsdim == 1 then feature_dim == 2 and vice-versa
# _other_dimension(obsdim) = mod(obsdim,2)+1
#
# ----------------------------------------------------------------------------------------------------------------
"""
`fit(MinMaxScaler; range_min::Real=0, range_max::Real=1, obsdim::Integer=1)`

Transforms features by scaling each feature to a given range. This estimator scales and translates each
feature individually such that it is in given range (`range_min`, `range_max`) on the training set, i.e.
between zero and one (default).

# Methods:
  * `fit(::Type{MinMaxScaler}, X::AbstractMatrix)`
  * `transform(cs::MinMaxScaler, X::AbstractMatrix)`
  * `inverse_transform(cs::MinMaxScaler, X::AbstractMatrix)`

# Example:
```
julia> x=rand(-10:10, 6,4)
6×4 Array{Int64,2}:
 -5   -4    4   7
 10    0    8  -4
  9  -10    2   9
 -4   -5    6   4
  6    1   -7  -7
 -4    5  -10  -9

julia> clf = fit(MinMaxScaler, x, range_min=0, range_max=3);

julia> xnew = transform(clf, x)
6×4 Array{Float64,2}:
 0.0  1.2  2.33333  2.66667
 3.0  2.0  3.0      0.833333
 2.8  0.0  2.0      3.0
 0.2  1.0  2.66667  2.16667
 2.2  2.2  0.5      0.333333
 0.2  3.0  0.0      0.0

julia> inverse_transform(clf, xnew)
6×4 Array{Float64,2}:
 -5.0   -4.0    4.0   7.0
 10.0    0.0    8.0  -4.0
  9.0  -10.0    2.0   9.0
 -4.0   -5.0    6.0   4.0
  6.0    1.0   -7.0  -7.0
 -4.0    5.0  -10.0  -9.0
```
"""
immutable MinMaxScaler{T<:Number, U<:Number}
  min_vec::Vector{T}
  max_vec::Vector{T}
  range_min::U
  range_max::U
  n_features::Integer
  obsdim::Integer
end

function MinMaxScaler{T<:Number}(X::AbstractMatrix{T}; range_min::Real=0, range_max::Real=1, obsdim::Integer=1)
  feature_dim = _other_dimension(obsdim)
  n_features = size(X, feature_dim)

  min_vec = Float64.(minimum(X, obsdim) |> vec)
  max_vec = Float64.(maximum(X, obsdim) |> vec)
  MinMaxScaler(min_vec, max_vec, range_min, range_max, n_features, obsdim)
end

# immutable MinMaxScaler
#   min_vec::Vector{AbstractFloat}
#   max_vec::Vector{AbstractFloat}
#   range_min::Real
#   range_max::Real
#   n_features::Integer
#   obsdim::Integer
#
#   function MinMaxScaler{T<:AbstractFloat}(
#     min_vec::Vector{T}, max_vec::Vector{T},
#     range_min::Real, range_max::Real,
#     n_features::Integer, obsdim::Integer
#     )
#
#     @assert length(min_vec) == length(max_vec)
#
#     new(min_vec, max_vec, range_min, range_max, n_features, obsdim)
#   end
# end
#
#
# function MinMaxScaler{T<:Number}(X::AbstractMatrix{T}; range_min::Real=0, range_max::Real=1, obsdim::Integer=1)
#   feature_dim = _other_dimension(obsdim)
#   n_features = size(X, feature_dim)
#
#   min_vec = Float64.(minimum(X, obsdim) |> vec)
#   max_vec = Float64.(maximum(X, obsdim) |> vec)
#   MinMaxScaler(min_vec, max_vec, range_min, range_max, n_features, obsdim)
# end


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
function fit{T<:Number}(::Type{MinMaxScaler}, X::AbstractMatrix{T}; range_min::Real=0, range_max::Real=1, obsdim::Integer=1)
  MinMaxScaler(X; range_min=range_min, range_max=range_max, obsdim=obsdim)
end


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
function transform{T<:Number}(cs::MinMaxScaler, X::AbstractMatrix{T})
  Xnew = copy(convert(Array{Float64}, X))
  transform!(cs, Xnew)
end


function transform!{T<:Float64}(cs::MinMaxScaler, X::Array{T})
  # if obsdim == 1 then feature_dim == 2 and vice-versa
  feature_dim = _other_dimension(cs.obsdim)
  obsdim = cs.obsdim

  scale = (cs.range_max - cs.range_min) + cs.range_min
  for i in 1:size(X, obsdim)
    for j in 1:size(X, feature_dim)
      if obsdim == 1
        X[i,j] = (X[i,j] - cs.min_vec[j]) / (cs.max_vec[j] - cs.min_vec[j])
        X[i,j] = X[i,j] * scale
      elseif obsdim == 2
        X[j,i] = (X[j,i] - cs.min_vec[j]) / (cs.max_vec[j] - cs.min_vec[j])
        X[j,i] = X[j,i] * scale
      end
    end
  end
  X
end

# ----------------------------------------------------------------------------------------------------------------

function inverse_transform{T<:Number}(cs::MinMaxScaler, X::AbstractMatrix{T})
  feature_dim = _other_dimension(cs.obsdim)
  obsdim = cs.obsdim

  scale = (cs.range_max - cs.range_min) + cs.range_min
  if obsdim == 1
    Xnew = ( (X / scale) .* (cs.max_vec - cs.min_vec)' ) .+ cs.min_vec'
  elseif obsdim == 2
    Xnew = ( (X / scale) .* (cs.max_vec - cs.min_vec) ) .+ cs.min_vec
  else
    error("obsdim should be 1 or 2")
  end
  Xnew
end

# ----------------------------------------------------------------------------------------------------------------

function test_MinMaxScaler(n_obs=10, n_features=6)

  @testset "Transform <-> Inverse Transform" begin
    # Floating Point
    X = rand(n_obs,n_features)
    clf = fit(MinMaxScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test round.(X,2) == round.(X_inversed, 2)

    # Integer
    X = rand(-10:10, n_obs,n_features)
    clf = fit(MinMaxScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test X == Int64.(round.(X_inversed))

    # Integer with obsdim=2
    X = rand(-10:10, n_obs,n_features)
    clf = fit(MinMaxScaler, X, obsdim=2)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test X == Int64.(round.(X_inversed))
  end

end # MinMaxScaler()
