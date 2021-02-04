"""
This defines abstract method type
    and implements functions for abstract mehtod.
"""
abstract type AbstractMethod end

# This returns BundleModel object.
get_model(method::AbstractMethod)::BundleModel = BundleModel()

# This returns the internal JuMP.Model in BundleModel.
get_jump_model(method::AbstractMethod)::JuMP.Model = get_model(get_model(method))

set_optimizer(method::AbstractMethod, optimizer) = set_optimizer(get_model(method), optimizer)

# This returns solution.
function get_solution(method::AbstractMethod) end

# This returns objective value.
get_objective_value(method::AbstractMethod)::Float64 = Inf

# This sets the termination tolerance.
function set_bundle_tolerance!(method::AbstractMethod, tol::Float64) end

# This builds the initial bundle model.
function build_bundle_model!(method::AbstractMethod)
    add_variables!(method)
    add_objective_function!(method)
    add_constraints!(method)
end

# This creates variables to the bundle model.
function add_variables!(method::AbstractMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    @variable(model, x[i=1:bundle.n])
    @variable(model, θ[j=1:bundle.ncuts_per_iter])
end

# This creates an objective function to the bundle model.
function add_objective_function!(method::AbstractMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    θ = model[:θ]
    @objective(model, Min, sum(θ[j] for j = 1:bundle.ncuts_per_iter))
end

# This creates constraints to the bundle model.
function add_constraints!(method::AbstractMethod) end

# This implements the algorithmic steps.
function run!(method::AbstractMethod)
    add_initial_bundles!(method)
    update_iteration!(method)
    solve_bundle_model!(method)
    display_info!(method)
    while !termination_test(method)
        evaluate_functions!(method)
        update_bundles!(method)
        update_iteration!(method)
        solve_bundle_model!(method)
        display_info!(method)
    end
end

# This implements adding initial bundle to the model.
function add_initial_bundles!(method::AbstractMethod)
    evaluate_functions!(method)
    add_bundles!(method)
end

# This implements the bundle model solution step.
function solve_bundle_model!(method::AbstractMethod)
    model = get_model(method)
    solve_model!(model)
    collect_model_solution!(method)
end

# This may collect solutions from the bundle model.
function collect_model_solution!(method::AbstractMethod) end

# Should the method terminate?
termination_test(method::AbstractMethod) = true

# This should call BundleModel.evaluate_f function
# to get any necessary information (e.g., function value and subgradient).
function evaluate_functions!(method::AbstractMethod) end

# This updates the bundle pool by removing and/or adding bundle objects.
function update_bundles!(method::AbstractMethod)
    purge_bundles!(method)
    add_bundles!(method)
end
function purge_bundles!(method::AbstractMethod) end
function add_bundles!(method::AbstractMethod) end

# This updates any information for the current iteration (e.g., iteration count).
function update_iteration!(method::AbstractMethod) end

# This displays iteration information.
function display_info!(method::AbstractMethod) end
