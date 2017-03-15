

"""
`fit(Binarizer; threshold=0, obsdim::Integer=1)`

Binarize data (set feature values to 0 or 1) according to a threshold Values greater
than the threshold map to 1, while values less than or equal to the threshold map to 0.
With the default threshold of 0, only positive values map to 1.

Binarization is a common operation on text count data where the analyst can decide to
only consider the presence or absence of a feature rather than a quantified number of
occurrences for instance. It can also be used as a pre-processing step for estimators that
consider boolean random variables (e.g. modelled using the Bernoulli distribution in a
Bayesian setting).

# Methods:
  * `fit(::Type{MaxAbsScaler}, X::AbstractMatrix)`
  * `transform(cs::MaxAbsScaler, X::AbstractMatrix)`

# Example:
```
julia> x = rand(-10:10, 8,4);

julia> x = rand(-10:10, 8,4)
8×4 Array{Int64,2}:
  8  10   3   4
 -5   7   6  -2
 -4  -9   8  -3
  7   6  -6   1
  0   1   5  -1
  1  -7  -4   0
  4  10  -7   9
  5   0   1  -8

julia> clf = fit(Binarizer, x)

julia> xnew = transform(clf, x)
8×4 Array{Int64,2}:
 1  1  1  1
 0  1  1  0
 0  0  1  0
 1  1  0  1
 0  1  1  0
 1  0  0  0
 1  1  0  1
 1  0  1  0
```
"""
immutable Binarizer{T<:Number}
    threshold::T
    n_features::Integer
    obsdim::Integer
end

function Binarizer{T<:Number}(X::AbstractMatrix{T}; threshold::T=0, obsdim=1)
    feature_dim = _other_dimension(obsdim)
    n_features = size(X, feature_dim)
    Binarizer(threshold, n_features, obsdim)
end

function fit{T<:Number}(::Type{Binarizer}, X::AbstractMatrix{T}; threshold=0, obsdim::Integer=1)
    Binarizer(X; threshold=threshold, obsdim=obsdim)
end

function transform{T<:Number}(cs::Binarizer, X::AbstractMatrix{T})
    Xnew = convert(AbstractMatrix{Int64}, copy(X))
    transform!(cs, Xnew)
end

function transform!{T<:Number}(cs::Binarizer, X::AbstractMatrix{T})
    # if obsdim == 1 then feature_dim == 2 and vice-versa
    feature_dim = _other_dimension(cs.obsdim)
    obsdim = cs.obsdim

    for i in 1:size(X, obsdim)
        for j in 1:size(X, feature_dim)
            if obsdim == 1
                X[i,j] = (X[i,j] <= cs.threshold) ? 0 : 1
            elseif obsdim == 2
                X[j,i] = (X[j,i] <= cs.threshold) ? 0 : 1
            end
        end
    end
    X
end
