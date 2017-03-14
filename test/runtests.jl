using PreProcessing
using Base.Test

function runtests(n_obs=1000, n_features=100)
  @printf("Running Tests\n")

  @testset "Transform, Inverse Transform" begin
    X = rand(n_obs,n_features)
    clf = fit(StandardScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test round.(X,2) == round.(X_inversed, 2)
  end

  @testset "Transform, Inverse Transform" begin
    X = rand(n_obs,n_features)
    clf = fit(MinMaxScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test round.(X,2) == round.(X_inversed, 2)
  end

  @testset "Transform, Inverse Transform" begin
    X = rand(n_obs,n_features)
    clf = fit(MinMaxScaler, X)
    X_transformed = transform(clf, X)
    X_inversed = inverse_transform(clf, X_transformed)
    @test round.(X,2) == round.(X_inversed, 2)
  end

end
