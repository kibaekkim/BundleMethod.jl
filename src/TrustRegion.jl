"""
Implementation of Bundle-Trust-Region method based on
    Kim, Kibaek, Cosmin G. Petra, and Victor M. Zavala. "An asynchronous bundle-trust-region method for 
    dual decomposition of stochastic mixed-integer programming." SIAM Journal on Optimization 29.1 (2019): 318-342.

"""

using LinearAlgebra
using Printf

mutable struct TrustRegionMethod <: AbstractMethod
    model::BundleModel

    y::Array{Float64,1}  # current iterate of dimension n
    fy::Array{Float64,1} # objective values at y for N functions
    θ::Array{Float64,1}  # θ value of trust region master problem
    g::Dict{Int,SparseVector{Float64}}  # subgradients of dimension n for N functions

    iter::Int # iteration counter
    maxiter::Int # iteration limit
 
    # Algorithm-specific parameters
    linerr::Float64         # linearization error
    Δ_ub::Float64           # trust region bound upper limit
    Δ_lb::Float64           # trust region bound lower limit
    ξ::Float64              # serious step criterion
    ϵ::Float64              # convergence criterion

    x_lb::Array{Float64,1}  # original variable lower bound
    x_ub::Array{Float64,1}  # original variable upper bound
    Δ::Float64              # current trust region size
    x0::Array{Float64,1}	# current trust region center (at iteration k)
    fx0::Array{Float64,1}	# current best objective values

    statistics::Dict{Any,Any} # arbitrary collection of statistics

    null_count::Int         # ineffective search count
    start_time::Float64     # start time

    # Constructor
    function TrustRegionMethod(n::Int, N::Int, func, init::Array{Float64,1}=zeros(n))
        trm = new()

        @assert length(init) == n

        trm.y = copy(init)
        trm.fy, trm.g = func(trm.y)
        trm.θ = zeros(N)
                
        trm.iter = 0
        trm.maxiter = 1000
        
        trm.linerr = 0.0
        trm.Δ_ub = 1.0e+4
        trm.Δ_lb = 1.0e-4
        trm.ξ = 1.0e-4
        trm.ϵ = 1.0e-6

        trm.x_lb = zeros(n)
        trm.x_ub = zeros(n)
        trm.Δ = 100.0
        trm.x0 = copy(init)
        trm.fx0 = copy(trm.fy)

        trm.statistics = Dict()

        trm.null_count = 0
        trm.start_time = time()

        trm.model = BundleModel(n, N, func)
        return trm
    end
end

function store_initial_variable_bounds!(method::TrustRegionMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    x = model[:x]
    for i = 1:bundle.n
        method.x_lb[i] = ifelse(has_lower_bound(x[i]), lower_bound(x[i]), -Inf)
        method.x_ub[i] = ifelse(has_upper_bound(x[i]), upper_bound(x[i]), +Inf)
    end
end

function build_bundle_model!(method::TrustRegionMethod)
    add_variables!(method)
    store_initial_variable_bounds!(method)
    add_objective_function!(method)
    add_constraints!(method)
end

# This returns BundleModel object.
get_model(method::TrustRegionMethod)::BundleModel = method.model

# This returns solution.
get_solution(method::TrustRegionMethod) = method.x0

# This returns objective value.
get_objective_value(method::TrustRegionMethod) = sum(method.fx0)

# This sets the termination tolerance.
function set_bundle_tolerance!(method::TrustRegionMethod, tol::Float64)
    method.ϵ = tol
end

function evaluate_functions!(method::TrustRegionMethod)
    method.fy, method.g = method.model.evaluate_f(method.y)
end

# This will specifically add trust region bounds to model
function add_constraints!(method::TrustRegionMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    Δ = method.Δ
    x = model[:x]
    center = method.x0
    for i = 1:bundle.n
        set_lower_bound(x[i], max(center[i] - Δ, method.x_lb[i]))
        set_upper_bound(x[i], min(center[i] + Δ, method.x_ub[i]))
    end
end

function add_bundle_constraint!(
    method::TrustRegionMethod, y::Array{Float64,1}, fy::Float64, g::SparseVector{Float64}, θ::JuMP.VariableRef)
    bundle = get_model(method)
    model = get_model(bundle)
    x = model[:x]
    @constraint(model, fy + sum(g[i] * (x[i] - y[i]) for i = 1:bundle.n) <= θ)
end

function add_bundles!(method::TrustRegionMethod)
    y = method.y
    fy = method.fy
    g = method.g

    bundle = get_model(method)

    # compute linearization error
    method.linerr = sum(method.fx0) - sum(fy)
    for j = 1:bundle.N, i = 1:bundle.n
        method.linerr -= g[j][i] * (method.x0[i] - y[i])
    end

    # add bundles constraints to the model
    θ = bundle.model[:θ]
    for j = 1:bundle.N
        add_bundle_constraint!(method, y, fy[j], g[j], θ[j])
    end
end

function collect_model_solution!(method::TrustRegionMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    if JuMP.termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        x = model[:x]
        θ = model[:θ]
        for i = 1:bundle.n
            method.y[i] = JuMP.value(x[i])
        end
        for j = 1:bundle.N
            method.θ[j] = JuMP.value(θ[j])
        end
    else
        @error "Unexpected model solution status ($(JuMP.termination_status(model)))"
    end
end

function termination_test(method::TrustRegionMethod)::Bool
    model = get_jump_model(method)
    if JuMP.termination_status(model) ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        return true
    end
    if sum(method.fx0) - sum(method.θ) <= method.ϵ * (1 + abs(sum(method.fx0))) && !is_trust_region_binding(method)
        println("TERMINATION: Optimal")
        return true
    end
    if method.iter >= method.maxiter
        println("TERMINATION: Maximum number of iterations reached.")
        return true
    end
    return false
end

# Update bundles and trust region constraints based on 
function update_bundles!(method::TrustRegionMethod)
    predicted_decrease_ratio = (sum(method.fx0) - sum(method.fy)) / (sum(method.fx0) - sum(method.θ))
    if predicted_decrease_ratio >=  method.ξ
        # serious step
        method.x0 = copy(method.y)
        method.fx0 = copy(method.fy)

        if is_trust_region_binding(method) && predicted_decrease_ratio >= 0.5
            update_Δ_serious_step!(method)
        end
        method.null_count = 0
    else
        # null step
        add_bundles!(method::TrustRegionMethod)

        ρ = min(1.0, method.Δ) * max(
            -predicted_decrease_ratio, 
            method.linerr / ( sum(method.fx0) - sum(method.θ) )
        )
        if ρ > 0
            method.null_count += 1
        end
        if ρ >= 3 || (method.null_count >= 3 && abs(ρ - 2) < 1)
            update_Δ_null_step!(method, ρ)
            method.null_count = 0
        end
    end
    # add new trust region bounds
    add_constraints!(method::TrustRegionMethod)    
end

# This displays iteration information.
function display_info!(method::TrustRegionMethod)
    model = get_jump_model(method)
    nrows = 0
    for tp in [MOI.LessThan{Float64}, MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}]
        nrows += num_constraints(model, AffExpr, tp)
    end
    @printf("Iter %4d: ncols %d, nrows %d, Δ %e, fx0 %+e, m %+e, fy %+e, linerr %+e, time %8.1f sec.\n",
        method.iter, num_variables(model), nrows, method.Δ, sum(method.fx0), sum(method.θ), sum(method.fy), method.linerr, time() - method.start_time)
end

function update_iteration!(method::TrustRegionMethod)
    method.iter += 1
end

function is_trust_region_binding(method::TrustRegionMethod)
    is_binding = false
    bundle = get_model(method)
    model = get_model(bundle)
    x = model[:x]
    for i = 1:bundle.n
        xval = JuMP.value(x[i])
        if isapprox(xval, method.x0[i] + method.Δ) || isapprox(xval, method.x0[i] - method.Δ)
            is_binding = true
            break
        end
    end
    return is_binding
end

# The following functions are specific to trust region method
function update_Δ_serious_step!(method::TrustRegionMethod)
    method.Δ = min(method.Δ * 2, method.Δ_ub)
end

function update_Δ_null_step!(method::TrustRegionMethod, ρ = 4.0)
    method.Δ = max(method.Δ / min(4.0, ρ), method.Δ_lb)
end