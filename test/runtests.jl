using Test
using BundleMethod
using JuMP
using Ipopt
using Random

const BM = BundleMethod

@testset "Abstract Method" begin
    bundle = BM.BundleModel(1, 1, nothing)
    @test bundle.n == 1
    @test bundle.N == 1
end

@testset "Proximal Method" begin

    include("../examples/simple.jl")
    
    @show BM.getobjectivevalue(pm)
    @show BM.getsolution(pm)

    @test isapprox(objval, BM.getobjectivevalue(pm), rtol=1e-2)
    for j in 1:n
        @test isapprox(xval[j], BM.getsolution(pm)[j], rtol=1e-2)
    end
end
