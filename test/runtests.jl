using Test
using BundleMethod
using JuMP
using Ipopt
using Random

const BM = BundleMethod

@testset "Abstract Method" begin

    mutable struct BaseMethod <: BM.AbstractMethod
        model::BM.BundleModel
        function BaseMethod()
            return new(BM.BundleModel(1,1,nothing))
        end
    end

    bm = BaseMethod()
    empty_model = BM.get_model(bm)
    @test empty_model.n == 0
    @test empty_model.N == 0

    BM.get_model(method::BaseMethod) = method.model

    BM.set_optimizer(bm, Ipopt.Optimizer)
    BM.get_solution(bm)
    @test BM.get_objective_value(bm) == Inf

    BM.build_bundle_model!(bm)
    model = BM.get_jump_model(bm)
    @test JuMP.num_variables(model) == 2
    @test BM.termination_test(bm) == true

    BM.add_constraints!(bm)
    BM.collect_model_solution!(bm)
    BM.evaluate_functions!(bm)
    BM.update_bundles!(bm)
    BM.purge_bundles!(bm)
    BM.add_bundles!(bm)
    BM.update_iteration!(bm)
    BM.display_info!(bm)
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

    @show BM.get_objective_value(pm)
    @show BM.get_solution(pm)
    @test isapprox(objval, BM.get_objective_value(pm), rtol=1e-2)
    for j in 1:n
        @test isapprox(xval[j], BM.get_solution(pm)[j], rtol=1e-2)
    end

    pm2 = BM.ProximalMethod(n, N, evaluate_f)
    pm2.M_g = 3
    pm2.maxiter = 3

    # Set optimization solver to the internal JuMP.Model
    model2 = BM.get_jump_model(pm2)
    set_optimizer(model2, Ipopt.Optimizer)
    set_optimizer_attribute(model2, "print_level", 0)

    BM.build_bundle_model!(pm2)
    BM.run!(pm2)
end

@testset "Trust Region Method" begin

    include("../examples/tr_simple.jl")
    vm = JuMP.Model(Ipopt.Optimizer)
    @variable(vm, -1 <= vm_x[j=1:n] <= 1)
    @objective(vm, Min, sum(b[i] * (vm_x[j] - a[i,j])^2 for i=1:N, j=1:n))
    optimize!(vm)
    objval = JuMP.objective_value(vm)
    xval = Dict{Int,Float64}()
    for j in 1:n
        xval[j] = JuMP.value(vm_x[j])
    end

    @show BM.get_objective_value(pm)
    @show BM.get_solution(pm)
    @test isapprox(objval, BM.get_objective_value(pm), rtol=1e-2)
    for j in 1:n
        @test isapprox(xval[j], BM.get_solution(pm)[j], rtol=1e-2)
    end
end