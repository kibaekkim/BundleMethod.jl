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

    vm = JuMP.Model(Ipopt.Optimizer)
    @variable(vm, -1 <= vm_x[j=1:n] <= 1)
    @objective(vm, Min, sum(b[i] * (vm_x[j] - a[i,j])^2 for i=1:N, j=1:n))
    optimize!(vm)
    objval = JuMP.objective_value(vm)
    xval = Dict{Int,Float64}()
    for j in 1:n
        xval[j] = JuMP.value(vm_x[j])
    end

    @show BM.getobjectivevalue(pm)
    @show BM.getsolution(pm)

    @test isapprox(objval, BM.getobjectivevalue(pm), rtol=1e-2)
    for j in 1:n
        @test isapprox(xval[j], BM.getsolution(pm)[j], rtol=1e-2)
    end
end
