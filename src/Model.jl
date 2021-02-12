# This structure contains necessary information for bundle methods.
mutable struct BundleModel
    n::Int              # dimension in the solution space
    N::Int              # number of separable functions in the objective
    ncuts_per_iter::Int # number of cuts per iteration 
    cut_indices::Dict{Int,Vector{Int}}
    model::JuMP.Model   # Bundle model

    # user-defined function to evaluate f
    # and return the value and its subgradients
    evaluate_f

    time::Vector{Float64}

    user_data

    function BundleModel(n::Int=0, N::Int=0, ncuts_per_iter::Int=0, func=nothing)
        bundle = new()
        bundle.n = n
        bundle.N = N
        bundle.ncuts_per_iter = ncuts_per_iter
        bundle.cut_indices = Dict(i => [j for j = i:ncuts_per_iter:N] for i = 1:ncuts_per_iter)
        bundle.model = JuMP.Model()
        # bundle.splitvars = splitvars
        bundle.evaluate_f = func
        bundle.time = []
        bundle.user_data = nothing
        return bundle
    end
end

# This returns BundleModel object.
get_model(method::AbstractMethod)::BundleModel = BundleModel()

get_model(model::BundleModel)::JuMP.Model = model.model

set_optimizer(model::BundleModel, optimizer) = JuMP.set_optimizer(get_model(model), optimizer)

function set_user_data(model::BundleModel, user_data)
    model.user_data = user_data
end

function solve_model!(model::BundleModel)
    m = get_model(model)
    @assert !isnothing(m.moi_backend.optimizer)
    JuMP.optimize!(m)
    push!(model.time, JuMP.solve_time(m))
end