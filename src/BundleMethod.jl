"""
Julia package for implementing bundle methods
The current version has implemented a proximal bundle method.

minimize
  sum_{j=1}^N f_j(x)
subject to
  x in a feasible set of dimension n

f_j may be evaluated by calling evaluate_f.
"""

module BundleMethod

using JuMP

# This structure contains necessary information for bundle methods.
mutable struct BundleModel
    n::Int            # dimension in the solution space
    N::Int            # number of separable functions in the objective
    model::JuMP.Model # Bundle model

    # user-defined function to evaluate f
    # and return the value and its subgradients
    evaluate_f

    user_data

    function BundleModel(n::Int=0, N::Int=0, func=nothing)
        bundle = new()
        bundle.n = n
        bundle.N = N
        bundle.model = JuMP.Model()
        # bundle.splitvars = splitvars
        bundle.evaluate_f = func
        bundle.user_data = nothing
        return bundle
    end
end

get_model(model::BundleModel)::JuMP.Model = model.model

set_optimizer(model::BundleModel, optimizer) = JuMP.set_optimizer(get_model(model), optimizer)

function set_user_data(model::BundleModel, user_data)
    model.user_data = user_data
end

function solve_model!(model::BundleModel)
    model = get_model(model)
    @assert !isnothing(model.moi_backend.optimizer)
    JuMP.optimize!(model)
end

include("Abstract.jl")
include("Proximal.jl")
include("TrustRegion.jl")

end
