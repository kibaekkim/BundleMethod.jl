"""
Implementation of Proximal Bundle Method.

The implementation is based on
  Krzysztof C. Kiwiel, "Proximity control in bundle methods for convex nondifferentiable minimization"
  Mathematical Programming 46(1-3), 1990
"""

using LinearAlgebra
using SparseArrays
using Printf

const disable_purge_cuts = true

mutable struct ProximalMethod <: AbstractMethod
	model::BundleModel

	y::Array{Float64,1}  # current iterate of dimension n
	fy::Array{Float64,1} # objective values at y for N functions
	g::Dict{Int,SparseVector{Float64}} # subgradients of dimension n for N functions

	dual::Dict{JuMP.ConstraintRef,Float64} # dual variable values to bundle constraints

	iter::Int # iteration counter
	maxiter::Int # iteration limit

	# Algorithm-specific parameters
	u::Float64
	u_min::Float64
	M_g::Int
	ϵ_float::Float64	# tolerance for floating point comparison
	ϵ_s::Float64
	ϵ_v::Float64
	m_L::Float64
	m_R::Float64

	x0::Array{Float64,1}	# current best solution (at iteration k)
	x1::Array{Float64,1}	# new best solution (at iteration k+1)
	fx0::Array{Float64,1}	# current best objective values
	fx1::Array{Float64,1}	# new best objective values (at iteration k+1)
	d::Array{Float64,1}
	v::Array{Float64,1}
	sum_of_v::Float64
	i::Int
	α::Array{Float64,1}

	cut_pool::Vector{JuMP.ConstraintRef}
	statistics::Dict{Any,Any} # arbitrary collection of statistics

	function ProximalMethod(n::Int, N::Int, func)
		pm = new()
		pm.model = BundleModel(n, N, func)
		
		pm.y = zeros(n)
		pm.fy = zeros(N)
		pm.g = Dict()

		pm.dual = Dict()
		
		pm.iter = 0
		pm.maxiter = 3000
		
		pm.u = 0.1
		pm.u_min = 1.0e-2
		pm.M_g = 1e+6
		pm.ϵ_float = 1.0e-8
		pm.ϵ_s = 1.0e-6
		pm.ϵ_v = Inf
		pm.m_L = 1.0e-4
		pm.m_R = 0.5
		
		pm.x0 = zeros(n)
		pm.x1 = zeros(n)
		pm.fx0 = zeros(N)
		pm.fx1 = zeros(N)
		pm.d = zeros(n)
		pm.v = zeros(N)
		pm.sum_of_v = 0.0
		pm.i = 0
		pm.α = zeros(N)
		
		pm.cut_pool = []
		pm.statistics = Dict()
		
		return pm
	end
end

# This returns BundleModel object.
get_model(method::ProximalMethod)::BundleModel = method.model

# This returns solution.
get_solution(method::ProximalMethod) = method.x0

# This returns objective value.
get_objective_value(method::ProximalMethod) = sum(method.fx0)

# This sets the termination tolerance.
function set_bundle_tolerance!(method::ProximalMethod, tol::Float64)
	method.ϵ_s = tol
end

# This creates an objective function to the bundle model.
function add_objective_function!(method::ProximalMethod)
	bundle = get_model(method)
	x = bundle.model[:x]
	θ = bundle.model[:θ]
	@objective(bundle.model, Min,
		  sum(θ[j] for j=1:bundle.N)
		+ 0.5 * method.u * sum((x[i] - method.x0[i])^2 for i=1:bundle.n))
end

# This may collect solutions from the bundle model.
function collect_model_solution!(method::ProximalMethod)
	bundle = get_model(method)
	model = get_model(bundle)
	if JuMP.termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
		x = model[:x]
		θ = model[:θ]
		for i=1:bundle.n
			method.y[i] = JuMP.value(x[i])
			method.d[i] = method.y[i] - method.x0[i]
		end
		for j=1:bundle.N
			method.v[j] = JuMP.value(θ[j]) - method.fx0[j]
		end
		method.sum_of_v = sum(method.v)
		for (i, ref) in enumerate(method.cut_pool)
			@assert JuMP.is_valid(get_jump_model(method), ref)
			method.dual[ref] = JuMP.dual(ref)
		end
	else
		@error "Unexpected model solution status ($(JuMP.termination_status(model)))"
	end
end

# Should the method terminate?
function termination_test(method::ProximalMethod)
	model = get_jump_model(method)
	if JuMP.termination_status(model) ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
		return true
	end
	if method.sum_of_v >= -method.ϵ_s * (1 + abs(sum(method.fx0)))
		println("TERMINATION: Optimal: v = ", method.sum_of_v)
		return true
	end
	if method.iter >= method.maxiter
		println("TERMINATION: Maximum number of iterations reached.")
		return true
	end
	return false
end

"""
This calls function `BundleModel.evaluate_f(y)` to get function value `fy` and gradient `g`.
This method assumes user-defined function of the form
	`evaluate_f(y::Vector{Float64})::Tuple{Float64,Array{Float64,2}}`
	returning function evaluation value and gradient as first and second outputs, resp.
"""
function evaluate_functions!(method::ProximalMethod)
	method.fy, method.g = method.model.evaluate_f(method.y)
end

# This updates the bundle pool by removing and/or adding bundle objects.
function update_bundles!(method::ProximalMethod)
	prox_update!(method)
	purge_bundles!(method)
	add_bundles!(method)
end

function purge_bundles!(method::ProximalMethod)
	model = get_jump_model(method)
	ncuts = length(method.cut_pool)
	ncuts_to_purge = ncuts - method.M_g
	refs_removed = Int[]
	if ncuts_to_purge > 0
		for (i, ref) in enumerate(method.cut_pool)
			@assert JuMP.is_valid(model, ref)
			if method.dual[ref] < -1e-8
				JuMP.delete(model, ref)
				push!(refs_removed, i)
				ncuts_to_purge -= 1
			end
			if ncuts_to_purge == 0
				break
			end
		end
	end

	# remove constraint references from cut_pool
	deleteat!(method.cut_pool, refs_removed)

	return length(refs_removed)
end

function add_bundles!(method::ProximalMethod)
	y = method.y
	fy = method.fy
	g = method.g

	method.fx0 = copy(fy)

	# add bundles as constraints to the model
	bundle = get_model(method)
	for j = 1:bundle.N
		x = bundle.model[:x]
		θ = bundle.model[:θ]

		if method.iter == 0
			add_bundle_constraint!(method, y, fy[j], g[j], θ[j])
		else
			gd = g[j]' * method.d
			method.α[j] = method.fx0[j] - (fy[j] - gd)

			if -method.α[j] + gd > method.v[j] + method.ϵ_float
				add_bundle_constraint!(method, y, fy[j], g[j], θ[j])
			end
		end
	end
end

# This updates the iteration information.
function update_iteration!(method::ProximalMethod)
	# update u
	updated = update_weight(method)

	# Update objective function
	if updated
		update_objective!(method)
	end

	method.iter += 1
	method.x0 = copy(method.x1)
	method.fx0 = copy(method.fx1)
end

# This displays iteration information.
function display_info!(method::ProximalMethod)
	model = get_jump_model(method)
	nrows = 0
	for tp in [MOI.LessThan{Float64}, MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}]
		nrows += num_constraints(model, AffExpr, tp)
	end
	@printf("Iter %d: ncols %d, nrows %d, fx0 %e, fx1 %e, fy %e, v %e, u %e, i %d\n",
		method.iter, num_variables(model), nrows, sum(method.fx0), sum(method.fx1), sum(method.fy), method.sum_of_v, method.u, method.i)
end

"""
The following functions are specific for this method only.
"""

function add_bundle_constraint!(
		method::ProximalMethod, y::Array{Float64,1}, fy::Float64, g::SparseVector{Float64}, θ::JuMP.VariableRef)
	bundle = get_model(method)
	model = get_model(bundle)
	x = model[:x]
	ref = @constraint(model, fy + sum(g[i] * (x[i] - y[i]) for i=1:bundle.n) <= θ)
	push!(method.cut_pool, ref)
	# method.statistics["y"][method.iter] = y
	# method.statistics["fy"][method.iter] = fy
	# bundle.history[j,method.iter] = Bundle(constr, deepcopy(y), fy[j], g[j,:])
end

# Proximal point update based on descent test
function prox_update!(method::ProximalMethod)
	if sum(method.fy) <= sum(method.fx0) + method.m_L * method.sum_of_v
		method.x1 = copy(method.y)
		method.fx1 = copy(method.fy)
	else
		method.x1 = copy(method.x0)
		method.fx1 = copy(method.fx0)
	end
end

# Update proximal term weight and return whether it is updated
function update_weight(method::ProximalMethod)::Bool
	fysum = sum(method.fy)
	fx0sum = sum(method.fx0)
	updated = false

	# update weight u
	u = method.u
	if fysum <= fx0sum + method.m_L * method.sum_of_v
		if method.i > 0 && fysum <= fx0sum + method.m_R * method.sum_of_v
			u = 2 * method.u * (1 - (fysum - fx0sum) / method.sum_of_v)
		elseif method.i > 3
			u = method.u / 2
		end
		# @show (u, bundle.u/10, bundle.u_min)
		newu = max(u, method.u/10, method.u_min)
		method.ϵ_v = max(method.ϵ_v, -2*method.sum_of_v)
		method.i = max(method.i+1,1)
		if newu != method.u
			method.i = 1
		end
		updated = true
	else
		p = norm(-method.u .* method.d, 1)
		α_tilde = -p^2 / method.u - method.sum_of_v

		method.ϵ_v = min(method.ϵ_v, p + α_tilde)
		if sum(method.α) > max(method.ϵ_v, -10*method.sum_of_v) && method.i < -3
			u = 2 * method.u * (1 - (fysum - fx0sum) / method.sum_of_v)
		elseif method.i < 5
			u = method.u * 1.2
		end
		newu = min(u, 10*method.u)
		method.i = min(method.i-1,-1)
		if newu != method.u
			method.i = -1
		end
	end
	# newu = 1.0e-6
	if newu != method.u
		method.u = newu
		updated = true
	end

	return updated
end

function update_objective!(method::ProximalMethod)
	bundle = get_model(method)
	model = get_model(bundle)
	x = model[:x]
	θ = model[:θ]
	@objective(model, Min,
		  sum(θ[j] for j=1:bundle.N)
		+ 0.5 * method.u * sum((x[i] - method.x1[i])^2 for i=1:bundle.n))
end
