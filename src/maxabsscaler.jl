
"""
`fit(MaxAbsScaler; obsdim::Integer=1)`

Scale and translate each feature individually such that the maximal absolute value of each feature in the
training set will be 1.0. This estimator does not shift/center the data.

# Methods:
  * `fit(::Type{MaxAbsScaler}, X::AbstractMatrix)`
  * `transform(cs::MaxAbsScaler, X::AbstractMatrix)`
  * `inverse_transform(cs::MaxAbsScaler, X::AbstractMatrix)`

# Example:

```
julia> x=rand(-10:10, 6,4)
6×4 Array{Int64,2}:
  -9  -4  -4  -9
   5  -6  -3   7
 -10  -5   9  10
  -5   5   1   2
  -4   5  10  -4
  -9  -3   0   0

julia> clf = fit(MaxAbsScaler, x)

julia> xnew = transform(clf, x)
6×4 Array{Float64,2}:
 -0.9  -0.666667  -0.4  -0.9
  0.5  -1.0       -0.3   0.7
 -1.0  -0.833333   0.9   1.0
 -0.5   0.833333   0.1   0.2
 -0.4   0.833333   1.0  -0.4
 -0.9  -0.5        0.0   0.0

julia> inverse_transform(clf, xnew)
6×4 Array{Float64,2}:
  -9.0  -4.0  -4.0  -9.0
   5.0  -6.0  -3.0   7.0
 -10.0  -5.0   9.0  10.0
  -5.0   5.0   1.0   2.0
  -4.0   5.0  10.0  -4.0
  -9.0  -3.0   0.0   0.0
```
"""
immutable MaxAbsScaler{T<:Integer, U<:AbstractFloat}
    scale::Vector{U}
    n_features::T
    obsdim::T
end


function MaxAbsScaler{T<:Number}(X::AbstractMatrix{T}; obsdim::Integer=1)
    feature_dim = _other_dimension(obsdim)
    n_features = size(X, feature_dim)

    scale = Float64.(maximum(abs, X, obsdim) |> vec)
    _handle_zeros_in_scale!(scale)
    MaxAbsScaler(scale, n_features, obsdim)
end


function Base.show(io::IO, t::MaxAbsScaler)
    @printf("MaxAbsScaler transformer with %d features\n", length(t.scale))
end

# ----------------------------------------------------------------------------------------------------------------
function fit{T<:Number}(::Type{MaxAbsScaler}, X::AbstractMatrix{T}; obsdim::Integer=1)
    MaxAbsScaler(X; obsdim=obsdim)
end


# ----------------------------------------------------------------------------------------------------------------
function transform{T<:Number}(cs::MaxAbsScaler, X::AbstractMatrix{T})
    Xnew = copy(convert(Array{Float64}, X))
    transform!(cs, Xnew)
end


function transform!{T<:AbstractFloat}(cs::MaxAbsScaler, X::Array{T})
    # if obsdim == 1 then feature_dim == 2 and vice-versa
    feature_dim = _other_dimension(cs.obsdim)
    obsdim = cs.obsdim

    for i in 1:size(X, obsdim)
        for j in 1:size(X, feature_dim)
            if obsdim == 1
                X[i,j] = (X[i,j] / cs.scale[j])
            elseif obsdim == 2
                X[j,i] = (X[j,i] / cs.scale[j])
            end
        end
    end
    X
end


# ----------------------------------------------------------------------------------------------------------------
function inverse_transform{T<:Number}(cs::MaxAbsScaler, X::AbstractMatrix{T})
    feature_dim = _other_dimension(cs.obsdim)
    obsdim = cs.obsdim

    if obsdim == 1
        Xnew = X .* cs.scale'
    elseif obsdim == 2
        Xnew = X .* cs.scale
    else
        error("obsdim should be 1 or 2")
    end
    Xnew
end


# ----------------------------------------------------------------------------------------------------------------
function test_MaxAbsScaler(n_obs=10, n_features=6)

    @testset "Transform <-> Inverse Transform" begin
        # Floating Point
        X = rand(n_obs,n_features)
        clf = fit(MaxAbsScaler, X)
        X_transformed = transform(clf, X)
        X_inversed = inverse_transform(clf, X_transformed)
        @test round.(X,2) == round.(X_inversed, 2)

        # Integer
        X = rand(-10:10, n_obs,n_features)
        clf = fit(MaxAbsScaler, X)
        X_transformed = transform(clf, X)
        X_inversed = inverse_transform(clf, X_transformed)
        @test X == Int64.(round.(X_inversed))

        # Integer with obsdim=2
        X = rand(-10:10, n_obs,n_features)
        clf = fit(MaxAbsScaler, X, obsdim=2)
        X_transformed = transform(clf, X)
        X_inversed = inverse_transform(clf, X_transformed)
        @test X == Int64.(round.(X_inversed))
    end

end # MaxAbsScaler()
