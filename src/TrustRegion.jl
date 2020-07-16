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
    Δ_ub::Float64           # trust region bound upper limit
    Δ_lb::Float64           # trust region bound lower limit
    ξ::Float64              # serious step criterion
    ϵ::Float64              # convergence criterion

    Δ::Float64              # current trust region size
	x0::Array{Float64,1}	# current trust region center (at iteration k)
	fx0::Array{Float64,1}	# current best objective values

    tr_pool::Vector{JuMP.ConstraintRef} # references to current trust region bounds
    statistics::Dict{Any,Any} # arbitrary collection of statistics

    # Constructor
    function TrustRegionMethod(n::Int, N::Int, func)
        trm = new()

		trm.y = zeros(n)
        trm.fy, trm.g = func(trm.y)
        trm.θ = zeros(N)
                
        trm.iter = 0
        trm.maxiter = 1000
        
        trm.Δ_ub = 20.0
        trm.Δ_lb = 0.0
        trm.ξ = 0.4
        trm.ϵ = 1.0e-6

        trm.Δ = 10.0
        trm.x0 = zeros(n)
		trm.fx0 = copy(trm.fy)

        trm.tr_pool = []
        trm.statistics = Dict()

        trm.model = BundleModel(n, N, func)
        return trm
    end
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
    method.tr_pool = []
    for i=1:bundle.n
        ref = @constraint(model, x[i] <= center[i] + Δ)
        push!(method.tr_pool, ref)
        ref = @constraint(model, x[i] >= center[i] - Δ)
        push!(method.tr_pool, ref)
    end
end

function add_bundle_constraint!(
    method::TrustRegionMethod, y::Array{Float64,1}, fy::Float64, g::SparseVector{Float64}, θ::JuMP.VariableRef)
    bundle = get_model(method)
    model = get_model(bundle)
    x = model[:x]
    @constraint(model, fy + sum(g[i] * (x[i] - y[i]) for i=1:bundle.n) <= θ)
end

function add_bundles!(method::TrustRegionMethod)
    y = method.y
	fy = method.fy
	g = method.g

	# add bundles constraints to the model
    bundle = get_model(method)
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
		for i=1:bundle.n
			method.y[i] = JuMP.value(x[i])
		end
		for j=1:bundle.N
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
	if sum(method.fx0) - sum(method.θ) <= method.ϵ * (1 + abs(sum(method.fx0)))
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
    if sum(method.fx0) - sum(method.fy) >=  method.ξ * (sum(method.fx0) - sum(method.θ))
        # serious step
        method.x0 = copy(method.y)
        method.fx0 = copy(method.fy)
        update_Δ_serious_step!(method)
    else
        # null step
        update_Δ_null_step!(method)
        add_bundles!(method::TrustRegionMethod)
    end
    # remove old trust region bounds
    model = get_jump_model(method)
    for (i, ref) in enumerate(method.tr_pool)
        delete(model, ref)
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
	@printf("Iter %d: ncols %d, nrows %d, Δ %e, fx0 %e, m %e, fy %e\n",
		method.iter, num_variables(model), nrows, method.Δ, sum(method.fx0), sum(method.θ), sum(method.fy))
end

function update_iteration!(method::TrustRegionMethod)
    method.iter += 1
end

# The following functions are specific to trust region method
function update_Δ_serious_step!(method::TrustRegionMethod)
    method.Δ = (method.Δ + method.Δ_ub) / 2
end

function update_Δ_null_step!(method::TrustRegionMethod)
    method.Δ = (method.Δ + method.Δ_lb) / 2
end