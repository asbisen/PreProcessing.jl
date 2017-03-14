
Library modeled after preprocessing module of
[scikit-learn](http://scikit-learn.org/stable/modules/preprocessing.html).
This library intends to implement the following transformers

* [x] StandardScaler
* [x] MinMaxScaler
* [x] MaxAbsScaler
* [x] Binarizer
* [ ] Normalizer
* [ ] OneHotEncoder

TODO:
* [ ] Generalize to handle 1D arrays

```
julia> using PreProcessing

julia> x = rand(-10:10, 8,4)
8×4 Array{Int64,2}:
   3   8  -2  -10
   3  10  -4    3
  -9   4   9   -5
   0   3   9  -10
   6  -8   4   -4
  -2   5  -5    7
 -10   2   9    3
  -6   6  -8    1

julia> clf = fit(StandardScaler, x)
PreProcessing.StandardScaler{Float64}([-1.875, 3.75, 1.5, -1.875], [5.55512, 5.06828, 6.61438, 5.92532], 4, 1)

julia> xnew = transform(clf, x)
8×4 Array{Float64,2}:
  0.877569    0.838548   -0.52915   -1.37123
  0.877569    1.23316    -0.831522   0.822741
 -1.2826      0.0493264   1.13389   -0.527398
  0.337526   -0.147979    1.13389   -1.37123
  1.41761    -2.31834     0.377964  -0.358631
 -0.0225018   0.246632   -0.982708   1.49781
 -1.46261    -0.345285    1.13389    0.822741
 -0.742558    0.443937   -1.43626    0.485206

julia> inverse_transform(clf, xnew)
8×4 Array{Float64,2}:
   3.0   8.0  -2.0  -10.0
   3.0  10.0  -4.0    3.0
  -9.0   4.0   9.0   -5.0
   0.0   3.0   9.0  -10.0
   6.0  -8.0   4.0   -4.0
  -2.0   5.0  -5.0    7.0
 -10.0   2.0   9.0    3.0
  -6.0   6.0  -8.0    1.0
```

```
julia> x = rand(-10:10, 8,4)
8×4 Array{Int64,2}:
   3   8  -2  -10
   3  10  -4    3
  -9   4   9   -5
   0   3   9  -10
   6  -8   4   -4
  -2   5  -5    7
 -10   2   9    3
  -6   6  -8    1

julia> clf = fit(MinMaxScaler, x, range_min=-4, range_max=4)
PreProcessing.MinMaxScaler{Float64,Int64}([-10.0, -8.0, -8.0, -10.0], [6.0, 10.0, 9.0, 7.0], -4, 4, 4, 1)

julia> xnew = transform(clf, x)
8×4 Array{Float64,2}:
 3.25  3.55556  1.41176   0.0    
 3.25  4.0      0.941176  3.05882
 0.25  2.66667  4.0       1.17647
 2.5   2.44444  4.0       0.0    
 4.0   0.0      2.82353   1.41176
 2.0   2.88889  0.705882  4.0    
 0.0   2.22222  4.0       3.05882
 1.0   3.11111  0.0       2.58824

julia> inverse_transform(clf, xnew)
8×4 Array{Float64,2}:
   3.0   8.0  -2.0  -10.0
   3.0  10.0  -4.0    3.0
  -9.0   4.0   9.0   -5.0
   0.0   3.0   9.0  -10.0
   6.0  -8.0   4.0   -4.0
  -2.0   5.0  -5.0    7.0
 -10.0   2.0   9.0    3.0
  -6.0   6.0  -8.0    1.0

```


```
julia> x = rand(-10:10, 8,4)
8×4 Array{Int64,2}:
   3   8  -2  -10
   3  10  -4    3
  -9   4   9   -5
   0   3   9  -10
   6  -8   4   -4
  -2   5  -5    7
 -10   2   9    3
  -6   6  -8    1

julia> clf = fit(Binarizer, x)
PreProcessing.Binarizer{Int64}(0, 4, 1)

julia> xnew = transform(clf, x)
8×4 Array{Int64,2}:
 1  1  0  0
 1  1  0  1
 0  1  1  0
 0  1  1  0
 1  0  1  0
 0  1  0  1
 0  1  1  1
 0  1  0  1

```

```
julia> x = rand(-10:10, 8,4)
8×4 Array{Int64,2}:
   3   8  -2  -10
   3  10  -4    3
  -9   4   9   -5
   0   3   9  -10
   6  -8   4   -4
  -2   5  -5    7
 -10   2   9    3
  -6   6  -8    1
  
julia> clf = fit(MaxAbsScaler, x)
MaxAbsScaler transformer with 4 features

julia> xnew = transform(clf, x)
8×4 Array{Float64,2}:
  0.3   0.8  -0.222222  -1.0
  0.3   1.0  -0.444444   0.3
 -0.9   0.4   1.0       -0.5
  0.0   0.3   1.0       -1.0
  0.6  -0.8   0.444444  -0.4
 -0.2   0.5  -0.555556   0.7
 -1.0   0.2   1.0        0.3
 -0.6   0.6  -0.888889   0.1

julia> inverse_transform(clf, xnew)
8×4 Array{Float64,2}:
   3.0   8.0  -2.0  -10.0
   3.0  10.0  -4.0    3.0
  -9.0   4.0   9.0   -5.0
   0.0   3.0   9.0  -10.0
   6.0  -8.0   4.0   -4.0
  -2.0   5.0  -5.0    7.0
 -10.0   2.0   9.0    3.0
  -6.0   6.0  -8.0    1.0

```
