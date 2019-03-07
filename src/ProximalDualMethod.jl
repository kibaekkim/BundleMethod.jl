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

abstract type ProximalDualMethod <: ProximalMethod end

const ProximalDualModel = Model{ProximalDualMethod}

function initialize!(bundle::ProximalDualModel)
	# Attach the extended structure
	bundle.ext = ProximalModelExt(bundle.n, bundle.N)

	# create the empty model (due to no column)
	if bundle.splitvars
		numw = Int(bundle.n / bundle.N)
		@variable(bundle.m, w[i=1:numw])
	end
	# The objective will be set later in add_initial_bundles!.
	@objective(bundle.m, Max, 0)
	@constraint(bundle.m, cons[j=1:bundle.N], 0 == 1)
	bundle.m.ext[:scaling_factor] = 1.0
end

function add_initial_bundles!(bundle::ProximalDualModel)
	# initial point evaluation
	bundle.fy, bundle.g = bundle.evaluate_f(bundle.y)
	bundle.ext.fx0 = bundle.fy

	# add bundles
	for j = 1:bundle.N
		add_var(bundle, bundle.g[j,:], bundle.fy[j], bundle.y, j)
	end

	# update objective function
	bundle.ext.x1 = bundle.ext.x0
	update_objective!(bundle)
end

function solve_bundle_model(bundle::ProximalDualModel)
	# print(bundle.m)
	# solve the bundle model
	status = solve(bundle.m)
	# @show JuMP.getobjectivevalue(bundle.m)

	if status == :Optimal
		# variable/constraint references
		if bundle.splitvars
			w = getindex(bundle.m, :w)
			numw = length(w)
		end
		cons = getindex(bundle.m, :cons)

		# get solutions
		if bundle.splitvars
			for i in 1:numw
				for j in 1:bundle.N
					jj = (j - 1) * numw + i
					bundle.y[jj] = bundle.ext.x0[jj] + (
						getvalue(w[i])
						- sum(getvalue(bundle.history[j,k].ref) * bundle.history[j,k].g[jj]
							for k in 0:bundle.k if haskey(bundle.history,(j,k)))
						) / bundle.ext.u
					bundle.ext.d[jj] = bundle.y[jj] - bundle.ext.x0[jj]
				end
			end
		else
			for i=1:bundle.n
				bundle.y[i] = (
					bundle.ext.x0[i]
					- sum(getvalue(hist.ref) * hist.g[i] for hist in values(bundle.history)) / bundle.ext.u
				)
				bundle.ext.d[i] = bundle.y[i] - bundle.ext.x0[i]
			end
		end
		for j=1:bundle.N
			# We can recover θ from the dual variable value.
			# Don't forget the scaling.
			θ = bundle.m.ext[:scaling_factor] * getdual(cons[j])
			bundle.ext.v[j] = θ - bundle.ext.fx0[j]
		end
		bundle.ext.sum_of_v = sum(bundle.ext.v)
	end

	return status
end

function manage_bundles!(bundle::ProximalDualModel)
	# add variable
	for j = 1:bundle.N
		gd = bundle.g[j,:]' * bundle.ext.d
		bundle.ext.α[j] = bundle.ext.fx0[j] - (bundle.fy[j] - gd)
		if -bundle.ext.α[j] + gd > bundle.ext.v[j] + bundle.ext.ϵ_float
			add_var(bundle, bundle.g[j,:], bundle.fy[j], bundle.y, j)
		end
	end
end

function update_iteration!(bundle::ProximalDualModel)
	# update u
	updated = update_weight(bundle)

	# Always update the objective function
	update_objective!(bundle)

	bundle.k += 1
	bundle.ext.x0 = bundle.ext.x1
	bundle.ext.fx0 = bundle.ext.fx1
end

function update_objective!(bundle::ProximalDualModel)
	# Need to scale the problem, as the gradients g can be large.
	bundle.m.ext[:scaling_factor] = 0.0
	for h in values(bundle.history)
		bundle.m.ext[:scaling_factor] += sqrt.(sum(h.g.^2))
	end

	if bundle.splitvars
		# variable references
		w = getindex(bundle.m, :w)
		numw = length(w)

		# update objective function
		@objective(bundle.m, Max,
			1. / bundle.m.ext[:scaling_factor] * sum(hist.ref
				* (hist.fy + sum(hist.g[i] * (bundle.ext.x1[i] - hist.y[i]) for i=1:bundle.n))
				for (key,hist) in bundle.history
			)
			- 0.5 / bundle.ext.u / bundle.m.ext[:scaling_factor]
				* sum(
					(w[i] - sum(bundle.history[j,k].ref * bundle.history[j,k].g[(j-1)*numw + i] for k in 0:bundle.k if haskey(bundle.history,(j,k))))^2
					for i in 1:numw for j in 1:bundle.N
				)
		)
	else
		# update objective function
		@objective(bundle.m, Max,
			1. / bundle.m.ext[:scaling_factor] * sum(hist.ref
				* (hist.fy + sum(hist.g[i] * (bundle.ext.x1[i] - hist.y[i]) for i=1:bundle.n))
				for (key,hist) in bundle.history
			)
			- 0.5 / bundle.ext.u / bundle.m.ext[:scaling_factor]
				* sum(
					sum(bundle.history[j,k].ref * bundle.history[j,k].g[i] for k in 0:bundle.k if haskey(bundle.history,(j,k)))^2
					for i in 1:bundle.n for j in 1:bundle.N
				)
		)
	end
end

function add_var(bundle::ProximalDualModel, g::Array{Float64,1}, fy::Float64, y::Array{Float64,1}, j::Int64; store_vars = true)
	# Add new variable
	cons = getindex(bundle.m, :cons)
	var = @variable(bundle.m,
		z >= 0,
		objective = 0.0, # This will be updated later.
		inconstraints = [cons[j]],
		coefficients = [1.0],
		basename = "z[$j,$(bundle.k)]")

	# Keep bundle history
	bundle.history[j,bundle.k] = Bundle(var, deepcopy(y), fy, g)
end
