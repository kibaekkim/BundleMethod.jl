#=
Julia package for implementing bundle methods
The current version has implemented a proximal bundle method.
=#

module BundleMethod

export ProximalBundleMethod

using Compat
using JuMP

abstract type AbstractBundleMethod end

mutable struct BundleInfo{T<:AbstractBundleMethod}
	n::Int64
	N::Int64
	u::Float64
	u_min::Float64
	M_g::Int64
	ϵ_s::Float64
	ϵ_v::Float64
	m_L::Float64
	m_R::Float64
	maxiter::Int64
	m::JuMP.Model
	k::Int64 # iteration counter
	x0::Array{Float64,1}
	x1::Array{Float64,1}
	y0::Array{Float64,1}
	y1::Array{Float64,1}
	fx0::Array{Float64,1}
	fx1::Array{Float64,1}
	fy0::Array{Float64,1}
	fy1::Array{Float64,1}
	d::Array{Float64,1}
	v::Array{Float64,1}
	sum_of_v::Float64
	g::Array{Float64,2} # subgradients
	i::Int64
	α::Array{Float64,1}

	evaluate_f
	constrRefs
	# to store cuts at iterations
	gk::Array{Array{Float64,1},1}
	yk::Array{Array{Float64,1},1}
	fyk::Array{Float64,1}
	jk::Array{Int64,1}
end

function BundleInfo(T::DataType, n::Int64, N::Int64, func)
	bundle = BundleInfo{T}(
		n,
		N,
		0.1,		# u
		1.0e-2,		# u_min
		1e+6,		# M_g
		1.0e-6,		# ϵ_s
		Inf,		# ϵ_v
		1.0e-4,		# m_L
		0.5,		# m_R
		500,		# maxiter
		Model(),	# m
		1,			# k
		zeros(n),	# x0
		zeros(n),	# x1
		zeros(n),	# y0
		zeros(n),	# y1
		zeros(N),	# fx0
		zeros(N),	# fx1
		zeros(N),	# fy0
		zeros(N),	# fy1
		zeros(n),	# d
		zeros(N),	# v
		0.0,		# sum_of_v
		zeros(0,0),	# g
		0,			# i
		zeros(N),	# α
		func, 		# user-defined function to evaluate f and return the value and its subgradients
		[],			# constraint references
		Array{Float64,1}[],	# gk
		Array{Float64,1}[],	# yk
		Float64[],	# fyk
		Int64[]	# jk
	)

	# initialize bundle model
	initializeBundleModel(bundle)

	return bundle
end

abstract type ProximalBundleMethod <: AbstractBundleMethod end
const ProximalBundleInfo = BundleInfo{ProximalBundleMethod}

function run(bundle::BundleInfo)
	# initial point evaluation
	bundle.fy0, bundle.g = bundle.evaluate_f(bundle.y0)
	bundle.fx0 = bundle.fy0

	for j = 1:bundle.N
		addCut(bundle, bundle.g[j,:], bundle.fy0[j], bundle.y0, j)
	end

	while true
		status = solveBundleModel(bundle)
		if status != :Optimal
			println("TERMINATION: Invalid status from bundle model.")
			break
		end

		# termination test
		if termination_test(bundle)
			break
		end

		# evaluation function f
		bundle.fy1, bundle.g = bundle.evaluate_f(bundle.y1)

		# descent test
		descent_test(bundle)

		# manage cuts in the bundle model
		manageCuts(bundle)

		# update u
		updateWeight(bundle)

		# display information and update iteration
		displayInfo(bundle)
		nextIter(bundle)
	end
end

getsolution(bundle::BundleInfo) = bundle.x0
getobjectivevalue(bundle::BundleInfo) = sum(bundle.fx0)

function descent_test(bundle::BundleInfo)
	if sum(bundle.fy1) <= sum(bundle.fx0) + bundle.m_L * bundle.sum_of_v
		bundle.x1 = bundle.y1
		bundle.fx1 = bundle.fy1
	else
		bundle.x1 = bundle.x0
		bundle.fx1 = bundle.fx0
	end
end

function termination_test(bundle::BundleInfo)
	if bundle.sum_of_v >= -bundle.ϵ_s
		println("TERMINATION: Optimal: v = ", bundle.sum_of_v)
		return true
	end
	if bundle.k >= bundle.maxiter
		println("TERMINATION: Maximum number of iterations reached.")
		return true
	end
	return false
end

function initializeBundleModel(bundle::BundleInfo)
	# create the initial bundle model
	@variable(bundle.m, x[i=1:bundle.n])
	@variable(bundle.m, θ[j=1:bundle.N])
	@objective(bundle.m, Min,
		  sum(θ[j] for j=1:bundle.N)
		+ 0.5 * bundle.u * sum((x[i] - bundle.x0[i])^2 for i=1:bundle.n))
end

function solveBundleModel(bundle::BundleInfo)
	# variable references
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)

	status = solve(bundle.m)
	# @show (status, getobjectivevalue(bundle.m))

	if status == :Optimal
		# get solutions
		for i=1:bundle.n
			bundle.y1[i] = getvalue(x[i])
			bundle.d[i] = bundle.y1[i] - bundle.x0[i]
		end
		for j=1:bundle.N
			bundle.v[j] = getvalue(θ[j]) - bundle.fx0[j]
		end
		bundle.sum_of_v = sum(bundle.v)
	end

	return status
end

function addCut(bundle::BundleInfo, g::Array{Float64,1}, fy::Float64, y::Array{Float64,1}, j::Int64; store_cuts = true)
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)
	constr = @constraint(bundle.m, fy + sum(g[i] * (x[i] - y[i]) for i=1:bundle.n) <= θ[j])
	push!(bundle.constrRefs, constr)
	if store_cuts
		push!(bundle.gk, g)
		push!(bundle.yk, deepcopy(y))
		push!(bundle.fyk, fy)
		push!(bundle.jk, j)
	end
end

function manageCuts(bundle::BundleInfo)
	# @show getdual(bundle.constrRefs)
	ncuts_purged = purgeCuts(bundle)

	# variable references
	x = getindex(bundle.m, :x)
	θ = getindex(bundle.m, :θ)

	# add cuts
	for j = 1:bundle.N
		gd= bundle.g[j,:]' * bundle.d
		bundle.α[j] = bundle.fx0[j] - (bundle.fy1[j] - gd)
		if -bundle.α[j] + gd > bundle.v[j]
			addCut(bundle, bundle.g[j,:], bundle.fy1[j], bundle.y1, j)
		end
	end
end

function purgeCuts(bundle::BundleInfo)
	ncuts = length(bundle.constrRefs)
	ncuts_to_purge = ncuts - bundle.M_g
	cuts_to_purge = Int64[]
	icut = 1
	while ncuts_to_purge > 0
		if getdual(bundle.constrRefs[icut]) < -1.0e-8
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

		bundle.constrRefs = []
		for i in 1:length(bundle.jk)
			addCut(bundle, bundle.gk[i], bundle.fyk[i], bundle.yk[i], bundle.jk[i], store_cuts = false)
		end
	end

	return length(cuts_to_purge)
end

function updateWeight(bundle::BundleInfo)
	fy1sum = sum(bundle.fy1)
	fx0sum = sum(bundle.fx0)
	update_objfunc = false

	# update weight u
	u = bundle.u
	if fy1sum <= fx0sum + bundle.m_L * bundle.sum_of_v
		if bundle.i > 0 && fy1sum <= fx0sum + bundle.m_R * bundle.sum_of_v
			u = 2 * bundle.u * (1 - (fy1sum - fx0sum) / bundle.sum_of_v)
		elseif bundle.i > 3
			u = bundle.u / 2
		end
		# @show (u, bundle.u/10, bundle.u_min)
		newu = max(u, bundle.u/10, bundle.u_min)
		bundle.ϵ_v = max(bundle.ϵ_v, -2*bundle.sum_of_v)
		bundle.i = max(bundle.i+1,1)
		if newu != bundle.u
			bundle.i = 1
		end
		update_objfunc = true
	else
		p = Compat.norm(-bundle.u .* bundle.d, 1)
		α_tilde = -p^2 / bundle.u - bundle.sum_of_v

		bundle.ϵ_v = min(bundle.ϵ_v, p + α_tilde)
		if sum(bundle.α) > max(bundle.ϵ_v, -10*bundle.sum_of_v) && bundle.i < -3
			u = 2 * bundle.u * (1 - (fy1sum - fx0sum) / bundle.sum_of_v)
		end
		newu = min(u, 10*bundle.u)
		bundle.i = min(bundle.i-1,-1)
		if newu != bundle.u
			bundle.i = -1
		end
	end
	# newu = 1.0e-6
	if newu != bundle.u
		bundle.u = newu
		update_objfunc = true
	end
	if update_objfunc
		x = getindex(bundle.m, :x)
		θ = getindex(bundle.m, :θ)
		@objective(bundle.m, Min,
			  sum(θ[j] for j=1:bundle.N)
			+ 0.5 * bundle.u * sum((x[i] - bundle.x1[i])^2 for i=1:bundle.n))
	end
end

function displayInfo(bundle::BundleInfo)
	Compat.Printf.@printf("Iter %d: ncols %d, nrows %d, fx0 %e, fx1 %e, fy0 %e, fy1 %e, v %e, u %e, i %d\n",
		bundle.k, bundle.m.numCols, length(bundle.m.linconstr), sum(bundle.fx0), sum(bundle.fx1), sum(bundle.fy0), sum(bundle.fy1), bundle.sum_of_v, bundle.u, bundle.i)
end

function nextIter(bundle::BundleInfo)
	bundle.k += 1
	bundle.x0 = bundle.x1
	bundle.y0 = bundle.y1
	bundle.fx0 = bundle.fx1
	bundle.fy0 = bundle.fy1
end

end
