#=
	Implementation of Proximal Bundle Method.

	The implementation is based on
	Krzysztof C. Kiwiel, "Proximity control in bundle methods for convex nondifferentiable minimization"
	Mathematical Programming 46(1-3), 1990
=#

abstract type ProximalBundleMethod <: AbstractBundleMethod end

# Algorithm-specific structure
mutable struct BundleModelExt
	# Algorithm-specific parameters
	u::Float64
	u_min::Float64
	M_g::Int64
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
	i::Int64
	α::Array{Float64,1}

	function BundleModelExt(n::Int64, N::Int64)
		ext = new()
		ext.u = 0.1
		ext.u_min = 1.0e-2
		ext.M_g = 1e+6
		ext.ϵ_s = 1.0e-6
		ext.ϵ_v = Inf
		ext.m_L = 1.0e-4
		ext.m_R = 0.5
		ext.x0 = zeros(n)
		ext.x1 = zeros(n)
		ext.fx0 = zeros(N)
		ext.fx1 = zeros(N)
		ext.d = zeros(n)
		ext.v = zeros(N)
		ext.sum_of_v = 0.0
		ext.i = 0
		ext.α = zeros(N)
		return ext
	end
end

const ProximalBundleModel = BundleModel{ProximalBundleMethod}

function initialize!(bundle::ProximalBundleModel)
	# Attach the extended structure
	bundle.ext = BundleModelExt(bundle.n, bundle.N)

	# create the initial bundle model
	@variable(bundle.m, x[i=1:bundle.n])
	@variable(bundle.m, θ[j=1:bundle.N])
	@objective(bundle.m, Min,
		  sum(θ[j] for j=1:bundle.N)
		+ 0.5 * bundle.ext.u * sum((x[i] - bundle.ext.x0[i])^2 for i=1:bundle.n))

	# Add bounding constraints if variables are splitable.
	if bundle.splitvars
		size_of_each_var = Int(bundle.n / bundle.N)
		@constraint(bundle.m, [i=1:size_of_each_var],
			sum(x[(j-1)*size_of_each_var+i] for j in 1:bundle.N) == 0)
	end
end

function add_initial_bundles!(bundle::ProximalBundleModel)
	# initial point evaluation
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)
	bundle.ext.fx0 = bundle.fy

	# add bundles
	for j = 1:bundle.N
		add_cut(bundle, bundle.g[j,:], bundle.fy[j], bundle.y, j)
	end
end

function solve_bundle_model(bundle::ProximalBundleModel)
	# variable references
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)

	status = solve(bundle.m)

	if status == :Optimal
		# get solutions
		for i=1:bundle.n
			bundle.y[i] = getvalue(x[i])
			bundle.ext.d[i] = bundle.y[i] - bundle.ext.x0[i]
		end
		for j=1:bundle.N
			bundle.ext.v[j] = getvalue(θ[j]) - bundle.ext.fx0[j]
		end
		bundle.ext.sum_of_v = sum(bundle.ext.v)
	end

	return status
end

function termination_test(bundle::ProximalBundleModel)
	if bundle.ext.sum_of_v >= -bundle.ext.ϵ_s
		println("TERMINATION: Optimal: v = ", bundle.ext.sum_of_v)
		return true
	end
	if bundle.k >= bundle.maxiter
		println("TERMINATION: Maximum number of iterations reached.")
		return true
	end
	return false
end

function evaluate_functions!(bundle::ProximalBundleModel)
	# evaluation function f
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)

	# descent test
	descent_test(bundle)
end

function manage_bundles!(bundle::ProximalBundleModel)
	ncuts_purged = purge_cuts(bundle)

	# variable references
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)

	# add cuts
	for j = 1:bundle.N
		gd= bundle.g[j,:]' * bundle.ext.d
		bundle.ext.α[j] = bundle.ext.fx0[j] - (bundle.fy[j] - gd)
		if -bundle.ext.α[j] + gd > bundle.ext.v[j]
			add_cut(bundle, bundle.g[j,:], bundle.fy[j], bundle.y, j)
		end
	end
end

function update_iteration!(bundle::ProximalBundleModel)
	# update u
	update_weight(bundle)

	bundle.k += 1
	bundle.ext.x0 = bundle.ext.x1
	bundle.ext.fx0 = bundle.ext.fx1
end

getsolution(bundle::ProximalBundleModel)::Array{Float64,1} = bundle.ext.x0
getobjectivevalue(bundle::ProximalBundleModel)::Float64 = sum(bundle.ext.fx0)

function descent_test(bundle::ProximalBundleModel)
	if sum(bundle.fy) <= sum(bundle.ext.fx0) + bundle.ext.m_L * bundle.ext.sum_of_v
		bundle.ext.x1 = bundle.y
		bundle.ext.fx1 = bundle.fy
	else
		bundle.ext.x1 = bundle.ext.x0
		bundle.ext.fx1 = bundle.ext.fx0
	end
end

function add_cut(bundle::ProximalBundleModel, g::Array{Float64,1}, fy::Float64, y::Array{Float64,1}, j::Int64; store_cuts = true)
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)
	constr = @constraint(bundle.m, fy + sum(g[i] * (x[i] - y[i]) for i=1:bundle.n) <= θ[j])
	if store_cuts
		bundle.history[j,bundle.k] = Bundle(constr, deepcopy(y), fy, g)
	end
end

function purge_cuts(bundle::ProximalBundleModel)
	ncuts = length(bundle.history)
	ncuts_to_purge = ncuts - bundle.ext.M_g
	cuts_to_purge = Tuple{Int64,Int64}[]
	if ncuts_to_purge > 0
		for (refkey,hist) in bundle.history
			if getdual(hist.ref) < -1.0e-8
				push!(cuts_to_purge, refkey)
				ncuts_to_purge -= 1
			end
			if ncuts_to_purge <= 0
				break
			end
		end
	end

	if length(cuts_to_purge) > 0
		for refkey in cuts_to_purge
			delete!(bundle.history, refkey)
		end

		solver = bundle.m.solver
		bundle.m = Model(solver=solver)
		@variable(bundle.m, x[i=1:bundle.n])
		@variable(bundle.m, θ[j=1:bundle.N])
		@objective(bundle.m, Min,
			  sum(θ[j] for j=1:bundle.N)
			+ 0.5 * bundle.ext.u * sum((x[i] - bundle.ext.x1[i])^2 for i=1:bundle.n))

		for (k,h) in bundle.history
			add_cut(bundle, h.g, h.fy, h.y, k[1], store_cuts = false)
			delete!(bundle.history, k)
		end
	end

	return length(cuts_to_purge)
end

function update_weight(bundle::ProximalBundleModel)
	fysum = sum(bundle.fy)
	fx0sum = sum(bundle.ext.fx0)
	update_objfunc = false

	# update weight u
	u = bundle.ext.u
	if fysum <= fx0sum + bundle.ext.m_L * bundle.ext.sum_of_v
		if bundle.ext.i > 0 && fysum <= fx0sum + bundle.ext.m_R * bundle.ext.sum_of_v
			u = 2 * bundle.ext.u * (1 - (fysum - fx0sum) / bundle.ext.sum_of_v)
		elseif bundle.ext.i > 3
			u = bundle.ext.u / 2
		end
		# @show (u, bundle.u/10, bundle.u_min)
		newu = max(u, bundle.ext.u/10, bundle.ext.u_min)
		bundle.ext.ϵ_v = max(bundle.ext.ϵ_v, -2*bundle.ext.sum_of_v)
		bundle.ext.i = max(bundle.ext.i+1,1)
		if newu != bundle.ext.u
			bundle.ext.i = 1
		end
		update_objfunc = true
	else
		p = Compat.norm(-bundle.ext.u .* bundle.ext.d, 1)
		α_tilde = -p^2 / bundle.ext.u - bundle.ext.sum_of_v

		bundle.ext.ϵ_v = min(bundle.ext.ϵ_v, p + α_tilde)
		if sum(bundle.ext.α) > max(bundle.ext.ϵ_v, -10*bundle.ext.sum_of_v) && bundle.ext.i < -3
			u = 2 * bundle.ext.u * (1 - (fysum - fx0sum) / bundle.ext.sum_of_v)
		end
		newu = min(u, 10*bundle.ext.u)
		bundle.ext.i = min(bundle.ext.i-1,-1)
		if newu != bundle.ext.u
			bundle.ext.i = -1
		end
	end
	# newu = 1.0e-6
	if newu != bundle.ext.u
		bundle.ext.u = newu
		update_objfunc = true
	end
	if update_objfunc
		x = getindex(bundle.m, :x)
		θ = getindex(bundle.m, :θ)
		@objective(bundle.m, Min,
			  sum(θ[j] for j=1:bundle.N)
			+ 0.5 * bundle.ext.u * sum((x[i] - bundle.ext.x1[i])^2 for i=1:bundle.n))
	end
end

function display_info!(bundle::ProximalBundleModel)
	Compat.Printf.@printf("Iter %d: ncols %d, nrows %d, fx0 %e, fx1 %e, fy %e, v %e, u %e, i %d\n",
		bundle.k, bundle.m.numCols, length(bundle.m.linconstr), sum(bundle.ext.fx0), sum(bundle.ext.fx1), sum(bundle.fy), bundle.ext.sum_of_v, bundle.ext.u, bundle.ext.i)
end
