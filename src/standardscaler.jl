
# ----------------------------------------------------------------------------------------------------------------
# Description:
#
#
# TODO:
#   - [ ] Finish documenting the code
#   - [ ] Check the correctness of code
#   - [x] Add inverse_transform() function to get to the original matrix
#   - [ ] Investigate the need for a inplace transform!()
# ----------------------------------------------------------------------------------------------------------------
#
# function _handle_zeros_in_scale!{T<:AbstractFloat}(σ::Vector{T})
#   zero_idx = find(x->x==0, σ)
#   σ[zero_idx] = 1.0
#   σ
# end
#
# _other_dimension(obsdim) = mod(obsdim,2)+1
#
# ----------------------------------------------------------------------------------------------------------------
"""
`fit(StandardScaler; corrected=false, obsdim::Integer=1)`

Standardize features by removing the mean and scaling to unit variance. Standardization of a dataset is a
common requirement for many machine learning estimators: they might behave badly if the individual feature
do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance)

`corrected=false` calculated standard deviation with `n` and `n-1` if `corrected=true`

# Methods:
  * `fit(::Type{MaxAbsScaler}, X::AbstractMatrix)`
  * `transform(cs::MaxAbsScaler, X::AbstractMatrix)`
  * `inverse_transform(cs::MaxAbsScaler, X::AbstractMatrix)`

# Example:

```
julia> x=rand(-10:10, 6,4)
6×4 Array{Int64,2}:
  -8    8   7    0
  -2   -7  -2    9
  -7    6  -9  -10
  -1  -10   9    1
 -10    3  10    3
  -1   -5  -3   -8

julia> clf = fit(StandardScaler, x);

julia> xnew = transform(clf, x)
6×4 Array{Float64,2}:
 -0.873621   1.29577    0.707107   0.128885
  0.781661  -0.904594  -0.565685   1.52084
 -0.597741   1.00239   -1.55563   -1.41773
  1.05754   -1.34467    0.989949   0.283547
 -1.42538    0.562315   1.13137    0.59287
  1.05754   -0.611212  -0.707107  -1.10841

julia> inverse_transform(clf, xnew)
6×4 Array{Float64,2}:
  -8.0    8.0   7.0    0.0
  -2.0   -7.0  -2.0    9.0
  -7.0    6.0  -9.0  -10.0
  -1.0  -10.0   9.0    1.0
 -10.0    3.0  10.0    3.0
  -1.0   -5.0  -3.0   -8.0
```
"""
immutable StandardScaler{T<:Number}
  μ::Vector{T}
  σ::Vector{T}
  n_features::Integer
  obsdim::Integer
end

function StandardScaler{T<:Number}(X::AbstractMatrix{T}; corrected=false, obsdim::Integer=1)
  feature_dim = _other_dimension(obsdim)
  n_features = size(X, feature_dim)

  n_features = size(X, feature_dim)
  μ = mean(X, obsdim) |> vec
  σ = std(X, obsdim, corrected = corrected) |> vec
  _handle_zeros_in_scale!(σ)

  StandardScaler(μ, σ, n_features, obsdim)
end


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
function fit{T<:Number}(::Type{StandardScaler}, X::AbstractMatrix{T}; corrected=false, obsdim::Integer=1)
  StandardScaler(X; corrected=corrected, obsdim=obsdim)
end


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
function transform{T<:Number}(cs::StandardScaler, X::AbstractMatrix{T})
  Xnew = copy(convert(Array{Float64}, X))
  transform!(cs, Xnew)
end


function transform!{T<:Float64}(cs::StandardScaler, X::Array{T})
  # if obsdim == 1 then feature_dim == 2 and vice-versa
  feature_dim = _other_dimension(cs.obsdim)
  obsdim = cs.obsdim

  for i in 1:size(X, obsdim)
    for j in 1:size(X, feature_dim)
      if obsdim == 1
        X[i,j] = (X[i,j] - cs.μ[j]) / cs.σ[j]
      elseif obsdim == 2
        X[j,i] = (X[j,i] - cs.μ[j]) / cs.σ[j]
      end
    end
  end
  X
end

# ----------------------------------------------------------------------------------------------------------------

function inverse_transform{T<:Number}(cs::StandardScaler, X::AbstractMatrix{T})
  feature_dim = _other_dimension(cs.obsdim)
  obsdim = cs.obsdim

  if obsdim == 1
    Xnew = (X .* cs.σ' ) .+ cs.μ'
  elseif obsdim == 2
    Xnew = (X .* cs.σ) .+ cs.μ
  else
    error("obsdim should be 1 or 2")
  end
  Xnew
end

# ----------------------------------------------------------------------------------------------------------------

function test_StandardScaler(n_obs=10, n_features=6)

  @testset "Transform <-> Inverse Transform" begin
    # Floating Point
    X = rand(n_obs,n_features)
    clf = fit(StandardScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test round.(X,2) == round.(X_inversed, 2)

    # Integer
    X = rand(-10:10, n_obs,n_features)
    clf = fit(StandardScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test X == Int64.(round.(X_inversed))

    # Integer with obsdim=2
    X = rand(-10:10, n_obs,n_features)
    clf = fit(StandardScaler, X, obsdim=2)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test X == Int64.(round.(X_inversed))
  end

end # test_StandardScaler()
