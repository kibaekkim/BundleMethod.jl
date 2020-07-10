using Test
using BundleMethod

const BM = BundleMethod
@testset "Abstract Method" begin
    bundle = BM.BundleModel(1, 1, nothing)
    @test bundle.n == 1
    @test bundle.N == 1
end

m = BM.ProximalMethod(1, 1, nothing)