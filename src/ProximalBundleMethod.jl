#=
Implementation of Proximal Bundle Method
=#

abstract type ProximalBundleMethod <: AbstractBundleMethod end

# Algorithm-specific structure
mutable struct BundleInfoExt
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

	function BundleInfoExt(n::Int64, N::Int64)
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

const ProximalBundleInfo = BundleInfo{ProximalBundleMethod}

function initialize!(bundle::ProximalBundleInfo)
	# Attach the extended structure
	bundle.ext = BundleInfoExt(bundle.n, bundle.N)

	# create the initial bundle model
	@variable(bundle.m, x[i=1:bundle.n])
	@variable(bundle.m, θ[j=1:bundle.N])
	@objective(bundle.m, Min,
		  sum(θ[j] for j=1:bundle.N)
		+ 0.5 * bundle.ext.u * sum((x[i] - bundle.ext.x0[i])^2 for i=1:bundle.n))
end

function add_initial_bundles!(bundle::ProximalBundleInfo)
	# initial point evaluation
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)
	bundle.ext.fx0 = bundle.fy

	# add bundles
	for j = 1:bundle.N
		addCut(bundle, bundle.g[j,:], bundle.fy[j], bundle.y, j)
	end
end

function solve_bundle_model(bundle::ProximalBundleInfo)
	# variable references
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)

	status = solve(bundle.m)
	# @show (status, getobjectivevalue(bundle.m))

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

function termination_test(bundle::ProximalBundleInfo)
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

function evaluate_functions!(bundle::ProximalBundleInfo)
	# evaluation function f
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)

	# descent test
	descent_test(bundle)
end

function manage_bundles!(bundle::ProximalBundleInfo)
	# @show getdual(bundle.bundleRefs)
	ncuts_purged = purgeCuts(bundle)

	# variable references
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)

	# add cuts
	for j = 1:bundle.N
		gd= bundle.g[j,:]' * bundle.ext.d
		bundle.ext.α[j] = bundle.ext.fx0[j] - (bundle.fy[j] - gd)
		if -bundle.ext.α[j] + gd > bundle.ext.v[j]
			addCut(bundle, bundle.g[j,:], bundle.fy[j], bundle.y, j)
		end
	end
end

function update_iteration!(bundle::ProximalBundleInfo)
	# update u
	update_weight(bundle)

	bundle.k += 1
	bundle.ext.x0 = bundle.ext.x1
	bundle.ext.fx0 = bundle.ext.fx1
end

getsolution(bundle::ProximalBundleInfo)::Array{Float64,1} = bundle.ext.x0
getobjectivevalue(bundle::ProximalBundleInfo)::Float64 = sum(bundle.ext.fx0)

function descent_test(bundle::ProximalBundleInfo)
	if sum(bundle.fy) <= sum(bundle.ext.fx0) + bundle.ext.m_L * bundle.ext.sum_of_v
		bundle.ext.x1 = bundle.y
		bundle.ext.fx1 = bundle.fy
	else
		bundle.ext.x1 = bundle.ext.x0
		bundle.ext.fx1 = bundle.ext.fx0
	end
end

function addCut(bundle::ProximalBundleInfo, g::Array{Float64,1}, fy::Float64, y::Array{Float64,1}, j::Int64; store_cuts = true)
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)
	constr = @constraint(bundle.m, fy + sum(g[i] * (x[i] - y[i]) for i=1:bundle.n) <= θ[j])
	push!(bundle.bundleRefs, constr)
	if store_cuts
		push!(bundle.gk, g)
		push!(bundle.yk, deepcopy(y))
		push!(bundle.fyk, fy)
		push!(bundle.jk, j)
	end
end

function purgeCuts(bundle::ProximalBundleInfo)
	ncuts = length(bundle.bundleRefs)
	ncuts_to_purge = ncuts - bundle.ext.M_g
	cuts_to_purge = Int64[]
	icut = 1
	while ncuts_to_purge > 0
		if getdual(bundle.bundleRefs[icut]) < -1.0e-8
			push!(cuts_to_purge, icut)
			ncuts_to_purge -= 1
		end
		icut += 1
	end

	if length(cuts_to_purge) > 0
		deleteat!(bundle.fyk, cuts_to_purge)
		deleteat!(bundle.yk, cuts_to_purge)
		deleteat!(bundle.gk, cuts_to_purge)
		deleteat!(bundle.jk, cuts_to_purge)

		solver = bundle.m.solver
		bundle.m = Model(solver=solver)
		@variable(bundle.m, x[i=1:bundle.n])
		@variable(bundle.m, θ[j=1:bundle.N])
		@objective(bundle.m, Min,
			  sum(θ[j] for j=1:bundle.N)
			+ 0.5 * bundle.u * sum((x[i] - bundle.x1[i])^2 for i=1:bundle.n))

		bundle.bundleRefs = []
		for i in 1:length(bundle.jk)
			addCut(bundle, bundle.gk[i], bundle.fyk[i], bundle.yk[i], bundle.jk[i], store_cuts = false)
		end
	end

	return length(cuts_to_purge)
end

function update_weight(bundle::ProximalBundleInfo)
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

function display_info!(bundle::ProximalBundleInfo)
	Compat.Printf.@printf("Iter %d: ncols %d, nrows %d, fx0 %e, fx1 %e, fy %e, v %e, u %e, i %d\n",
		bundle.k, bundle.m.numCols, length(bundle.m.linconstr), sum(bundle.ext.fx0), sum(bundle.ext.fx1), sum(bundle.fy), bundle.ext.sum_of_v, bundle.ext.u, bundle.ext.i)
end
