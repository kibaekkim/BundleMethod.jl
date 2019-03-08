using Compat
using JuMP, CPLEX
using BundleMethod

# User-defined function
function evaluate_f(y)
	k,n = size(a)
	fvals = zeros(k)
	subgrads = zeros(k,n)
	for i=1:k
		for j=1:n
			fvals[i] += b[i] * (y[j] - a[i,j])^2
			subgrads[i,j] += 2 * b[i] * (y[j] - a[i,j])
		end
	end
	return -fvals, -subgrads
end

function test()
	#=
	min_x \sum_{i=1}^10 b_i \sum_{j=1}^5 (x_j - a_{ij})^2
	=#

	Compat.Random.seed!(1)
	global n = 3
	global k = 2
	global a = rand(k, n)
	global b = -rand(k)
	@show a
	@show b

	vm = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))
	@variable(vm, -1 <= vm_x[j=1:n] <= 1)
	@objective(vm, Max, sum(b[i] * (vm_x[j] - a[i,j])^2 for i=1:k for j=1:n))
	solve(vm)
	@show getobjectivevalue(vm)
	@show getvalue(vm_x)

	# initialize bundle method
	bundle = BundleMethod.Model{BundleMethod.ProximalMethod}(n, k, evaluate_f)

	# set bounds
	x = getindex(bundle.m, :x)
	for i=1:bundle.n
		setlowerbound(x[i], -1.0)
		setupperbound(x[i], +1.0)
	end

	# set the underlying solver
	setsolver(bundle.m, CplexSolver(CPX_PARAM_SCRIND=0))
	print(bundle.m)

	# solve!
	BundleMethod.run(bundle)

	# print solution
	@show BundleMethod.getobjectivevalue(bundle)
	@show BundleMethod.getsolution(bundle)
end

test()
