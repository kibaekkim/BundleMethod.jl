#=
	Implementation of Proximal Dual Bundle Method.

	This model is the dual of that in ProximalMethod.jl.
	A main advantage of this model may be the fact that PIPS can solve in parallel,
	particularly when bundle.splitvars = true. This case has been studied in

	Lubin et al. "On parallelizing dual decomposition in stochastic integer programming",
	Operations Research Letters 41, 2013

	Variable w can be seen as the first-stage variable in PIPS.
	There is no first-stage constraint, but there is one second-stage constraint
	for each scenario.
	The objective function has both first- and second-stage variables.
=#
type ProximalModelExt
	# Algorithm-specific parameters
	u::Float64
	u_min::Float64
	M_g::Int64
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
	i::Int64
	α::Array{Float64,1}

	function ProximalModelExt(n::Int64, N::Int64)
		ext = new()
		ext.u = 0.1
		ext.u_min = 1.0e-2
		ext.M_g = 1e+6
		ext.ϵ_float = 1.0e-8
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

type ProximalDualModel
	n::Int64		# dimension in the solution space
	N::Int64		# number of separable functions in the objective
	m::JuMP.Model	# Bundle model
	k::Int64 		# iteration counter
	maxiter::Int64	# maximum number of iterations

	y::Array{Float64,1}		# current iterate
	fy::Array{Float64,1}	# objective values at the current iterate
	g::Array{Float64,2}		# subgradients

	splitvars::Bool	# Are variables decomposed for each function?

	# user-defined function to evaluate f
	# and return the value and its subgradients
	evaluate_f

	# History of bundles
	history::Dict{Tuple{Int64,Int64},Bundle}
	
	solver

	# Placeholder for extended structures
	ext

	function ProximalDualModel(n::Int64, N::Int64, func, splitvars = false) 
		bundle = new()
		bundle.n = n
		bundle.N = N
		# bundle.m = JuMP.Model()
		bundle.m = StructuredModel(num_scenarios=N)
		bundle.k = 0
		bundle.maxiter = 500
		bundle.y = zeros(n)
		bundle.fy = zeros(N)
		bundle.g = zeros(0,0)
		bundle.splitvars = splitvars
		bundle.evaluate_f = func
		bundle.history = Dict{Tuple{Int64,Int64},Bundle}()

		# initialize bundle model
                for j = 1:bundle.N
			cmodel = StructuredModel(parent=bundle.m,id=j)
		end
		initialize!(bundle)

		return bundle
	end
end

function run(bundle::ProximalDualModel)

	add_initial_bundles!(bundle)
	bundle.k += 1

	while true
                build_model!(bundle)
		status = solve_bundle_model(bundle)
		if status != :Optimal
			println("TERMINATION: Invalid status from bundle model.")
			break
		end
		display_info!(bundle)
		if termination_test(bundle)
			break
		end
		evaluate_functions!(bundle)
		manage_bundles!(bundle)
		update_iteration!(bundle)
	end
	# TODO This is not where it should be
	MPI.Finalize()
end

function initialize!(bundle::ProximalDualModel)
	# Attach the extended structure
	bundle.ext = ProximalModelExt(bundle.n, bundle.N)

	# create the empty model (due to no column)
	if bundle.splitvars
		numw = Int(bundle.n / bundle.N)
		@variable(bundle.m, w[i=1:numw])
	end
	# The objective will be set later in add_initial_bundles!.
	@NLobjective(bundle.m, Min, 0)
	for j = 1:bundle.N
		if j ∈ getLocalChildrenIds(bundle.m)
			cmodel = getchildren(bundle.m)[j]
			@constraint(cmodel, cons, 0 == 1)
		end
	end
	bundle.m.ext[:scaling_factor] = 1.0
end

function add_initial_bundles!(bundle::ProximalDualModel)
	# initial point evaluation
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)
	bundle.ext.fx0 = copy(bundle.fy)

	# add bundles
	for j = 1:bundle.N
		if j ∈ getLocalChildrenIds(bundle.m)
			add_var(bundle, vec(bundle.g[j,:]), bundle.fy[j], bundle.y, j)
		end
	end

	# update objective function
	bundle.ext.x1 = copy(bundle.ext.x0)
	update_objective!(bundle)
end

function build_model!(bundle::ProximalDualModel)
	# print(bundle.m)
	bundle.m = StructuredModel(num_scenarios=bundle.N)
	if bundle.splitvars
		numw = Int(bundle.n / bundle.N)
		@variable(bundle.m, w[i=1:numw])
	end
	@NLobjective(bundle.m, Min, 0)
	for j = 1:bundle.N
		if j ∈ getLocalChildrenIds(bundle.m)
			cmodel = StructuredModel(parent=bundle.m,id=j)
			@constraint(cmodel, cons, 0 == 1)
			for i = 0:bundle.k-1
				if haskey(bundle.history, (j,i))
					cons = getconstraint(cmodel, :cons)
					var = @variable(cmodel,
						z >= 0,
						objective = 0.0, # This will be updated later.
						inconstraints = [cons],
						coefficients = [1.0],
						basename = "z[$j,$i]")
					
					bundle.history[j,i].ref = var
				end
			end
		end
	end
	update_objective!(bundle)
	JuMP.setsolver(bundle.m, bundle.solver)
	# print(bundle.m)
end

function solve_bundle_model(bundle::ProximalDualModel)
	# Initialize variables
	if bundle.splitvars
		w = getvariable(bundle.m, :w)
		numw = length(w)
		for i=1:numw
			setvalue(w[i], 0.0)
		end
	end
	
	numz=zeros(bundle.N)
	recv=zeros(bundle.N)
	for j in 1:bundle.N
		numz[j] = 0
		for k in 0:bundle.k
			if haskey(bundle.history, (j,k))
				numz[j] += 1
			end
		end
	end
	MPI.Allreduce!(numz, recv, bundle.N, MPI.SUM, MPI.COMM_WORLD)
	for j in 1:bundle.N
		for k in 0:bundle.k
			if haskey(bundle.history, (j,k))
				setvalue(bundle.history[j,k].ref, 1.0/recv[j])
			end
		end
	end

	# print(bundle.m)
	# solve the bundle model
	status = solve(bundle.m;solver="Ipopt", with_prof=false)
	# status = solve(bundle.m)
	# @show JuMP.getobjectivevalue(bundle.m)
	if status == :Solve_Succeeded
		status = :Optimal
	end
	if status == :Optimal

		# get solutions
		if bundle.splitvars
			nprocs = MPI.Comm_size(MPI.COMM_WORLD)
			id = MPI.Comm_rank(MPI.COMM_WORLD)
			npart = Int(bundle.n/nprocs)
			recvy = zeros(Int(bundle.n/nprocs))
			k = 1
			for i in 1:numw
				for j in 1:bundle.N
					jj = (j - 1) * numw + i
					if j ∈ getLocalChildrenIds(bundle.m)
						bundle.y[jj] = bundle.ext.x0[jj]
						tmp = 0.0
						for k in 0:bundle.k
							if haskey(bundle.history,(j,k))
								tmp -= getvalue(bundle.history[j,k].ref) * bundle.history[j,k].g[jj]
							end
						end
						tmp += getvalue(w[i])
						tmp /= bundle.ext.u
						bundle.y[jj] += tmp
						
						# bundle.y[jj] = bundle.ext.x0[jj] + (
						# getvalue(w[i])
						# - sum(getvalue(bundle.history[j,k].ref) * bundle.history[j,k].g[jj]
						# for k in 0:bundle.k if haskey(bundle.history,(j,k)))
						# 	) / bundle.ext.u
						# bundle.ext.d[jj] = bundle.y[jj] - bundle.ext.x0[jj]
						# recvy[jj-id*npart] = bundle.y[jj]
						# k += 1
					end
				end
			end
			# TODO: This case is necessary because Julia MPI messes the buffer up
			# with one process
			if nprocs > 1
				bundle.y = MPI.Allgather(recvy, MPI.COMM_WORLD)
			end
		else
			for i=1:bundle.n
				# bundle.y[i] = (
				# 	bundle.ext.x0[i]
				# 	- sum(getvalue(hist.ref) * hist.g[i] for hist in values(bundle.history)) / bundle.ext.u
				# )
				bundle.ext.d[i] = bundle.y[i] - bundle.ext.x0[i]
			end
		end

		# Numerical error may occur (particularly with Ipopt).
		numerical_error = sum(bundle.y) / bundle.n
		bundle.y .-= numerical_error
		bundle.ext.d .-= numerical_error
		@assert(isapprox(sum(bundle.y), 0.0, atol = 1.0e-10))

		for j=1:bundle.N
			if j ∈ getLocalChildrenIds(bundle.m)
				# variable/constraint references
				cmodel = getchildren(bundle.m)[j]
				cons = getconstraint(cmodel, :cons)
				
				# We can recover θ from the dual variable value.
				# Don't forget the scaling.
				θ = bundle.m.ext[:scaling_factor] * -getdual(cons)
				bundle.ext.v[j] = θ - bundle.ext.fx0[j]
			end
		end
		bundle.ext.sum_of_v = sum(bundle.ext.v)
		bundle.ext.sum_of_v = MPI.Allreduce(bundle.ext.sum_of_v, MPI.SUM, MPI.COMM_WORLD)
	end

	return status
end

function manage_bundles!(bundle::ProximalDualModel)
	# add variable
	for j = 1:bundle.N
		if j ∈ getLocalChildrenIds(bundle.m)
			gd = (vec(bundle.g[j,:])' * bundle.ext.d)[1]
			bundle.ext.α[j] = bundle.ext.fx0[j] - (bundle.fy[j] - gd)
			if -bundle.ext.α[j] + gd > bundle.ext.v[j] + bundle.ext.ϵ_float
				add_var(bundle, vec(bundle.g[j,:]), bundle.fy[j], bundle.y, j)
			end
		end
	end
end

function update_iteration!(bundle::ProximalDualModel)
	# update u
	updated = update_weight(bundle)

	# Always update the objective function
	update_objective!(bundle)

	bundle.k += 1
	bundle.ext.x0 = copy(bundle.ext.x1)
	bundle.ext.fx0 = copy(bundle.ext.fx1)
end

function update_objective!(bundle::ProximalDualModel)
	# Need to scale the problem, as the gradients g can be large.
	bundle.m.ext[:scaling_factor] = 0.0
	for h in values(bundle.history)
		bundle.m.ext[:scaling_factor] += sqrt((sum(h.g.^2)))
	end
	bundle.m.ext[:scaling_factor] = MPI.Allreduce(bundle.m.ext[:scaling_factor], MPI.SUM, MPI.COMM_WORLD)

	if bundle.splitvars
		# variable references
		w = getvariable(bundle.m, :w)
		numw = length(w)

		# update objective function
		@NLobjective(bundle.m, Min,
		    bundle.N * 0.5 / bundle.ext.u / bundle.m.ext[:scaling_factor] * sum{w[i]^2, i in 1:numw}) 
		for j in getLocalChildrenIds(bundle.m)
			cmodel = getchildren(bundle.m)[j]
			@NLobjective(cmodel, Min,
				-1. / bundle.m.ext[:scaling_factor] * sum{hist.ref
					* (hist.fy + sum{hist.g[i] * (bundle.ext.x1[i] - hist.y[i]), i=1:bundle.n}),
					(key,hist) in bundle.history; key[1] == j
				}   + 
				0.5 / bundle.ext.u / bundle.m.ext[:scaling_factor]
					* -2.0 * (sum{w[i] * (sum{bundle.history[j,k].ref * bundle.history[j,k].g[(j-1)*numw + i], k in 0:bundle.k; haskey(bundle.history,(j,k))}), i in 1:numw})
					+
				0.5 / bundle.ext.u / bundle.m.ext[:scaling_factor]
					* sum{sum{bundle.history[j,k].ref * bundle.history[j,k].g[(j-1)*numw + i], k in 0:bundle.k; haskey(bundle.history,(j,k))}^2, i in 1:numw}
			)
		end
	else
		# update objective function
		@NLobjective(bundle.m, Min,
			-1. / bundle.m.ext[:scaling_factor] * sum{hist.ref
				* (hist.fy + sum{hist.g[i] * (bundle.ext.x1[i] - hist.y[i]), i=1:bundle.n}),
				(key,hist) in bundle.history
			}
			+ 0.5 / bundle.ext.u / bundle.m.ext[:scaling_factor]
				* sum{
					sum{bundle.history[j,k].ref * bundle.history[j,k].g[i], k in 0:bundle.k; haskey(bundle.history,(j,k))}^2,
					i in 1:bundle.n, j in 1:bundle.N
				}
		)
	end
end

function add_var(bundle::ProximalDualModel, g::Array{Float64,1}, fy::Float64, y::Array{Float64,1}, j::Int64; store_vars = true)
	# Add new variable
	cmodel = getchildren(bundle.m)[j]
	cons = getconstraint(cmodel, :cons)
		
	var = @variable(cmodel,
		z >= 0,
		objective = 0.0, # This will be updated later.
		inconstraints = [cons],
		coefficients = [1.0],
		basename = "z[$j,$(bundle.k)]")

	# Keep bundle history
	bundle.history[j,bundle.k] = Bundle(var, deepcopy(y), fy, g)
end

function termination_test(bundle::ProximalDualModel)
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

function evaluate_functions!(bundle::ProximalDualModel)
	# evaluation function f
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)

	# descent test
	descent_test(bundle)
end

function display_info!(bundle::ProximalDualModel)
	@printf("Iter %d: ncols %d, nrows %d, fx0 %e, fx1 %e, fy %e, v %e, u %e, i %d\n",
		bundle.k, bundle.m.numCols, length(bundle.m.linconstr), sum(bundle.ext.fx0), sum(bundle.ext.fx1), sum(bundle.fy), bundle.ext.sum_of_v, bundle.ext.u, bundle.ext.i)
end

function getsolution(bundle::ProximalDualModel)
	return bundle.ext.x0
end

function getobjectivevalue(bundle::ProximalDualModel)
	return sum(bundle.ext.fx0)
end

function descent_test(bundle::ProximalDualModel)
	if sum(bundle.fy) <= sum(bundle.ext.fx0) + bundle.ext.m_L * bundle.ext.sum_of_v
		bundle.ext.x1 = copy(bundle.y)
		bundle.ext.fx1 = copy(bundle.fy)
	else
		bundle.ext.x1 = copy(bundle.ext.x0)
		bundle.ext.fx1 = copy(bundle.ext.fx0)
	end
end

function update_weight(bundle::ProximalDualModel)
	fysum = sum(bundle.fy)
	fx0sum = sum(bundle.ext.fx0)
	updated = false

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
		updated = true
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
		updated = true
	end

	return updated
end
