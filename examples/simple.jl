using BundleMethod
using JuMP
using Ipopt
using Random

#=
minimize \sum_{i=1}^k \sum_{j=1}^n b_i * (x_j - a_ij)^2
subject to -1 <= x_j <= 1
=#

Random.seed!(1)
n = 3
N = 2
a = rand(N, n)
b = rand(N)

vm = JuMP.Model(Ipopt.Optimizer)
@variable(vm, -1 <= vm_x[j=1:n] <= 1)
@objective(vm, Min, sum(b[i] * (vm_x[j] - a[i,j])^2 for i=1:N, j=1:n))
optimize!(vm)
objval = JuMP.objective_value(vm)
xval = Dict{Int,Float64}()
for j in 1:n
	xval[j] = JuMP.value(vm_x[j])
end

# User-defined function
function evaluate_f(y)
	N, n = size(a)
	fvals = zeros(N)
	subgrads = zeros(N, n)
	for i = 1:N, j = 1:n
		fvals[i] += b[i] * (y[j] - a[i,j])^2
		subgrads[i,j] += 2 * b[i] * (y[j] - a[i,j])
	end
	return fvals, subgrads
end

pm = BM.ProximalMethod(n, N, evaluate_f)

model = BM.get_model(pm.model)
set_optimizer(model, Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0)

# This creates variables to the bundle model.
function BM.add_variables!(method::BM.ProximalMethod)
	bundle = BM.get_model(method)
	model = BM.get_model(bundle)
	@variable(model, -1 <= x[i=1:bundle.n] <= 1)
	@variable(model, θ[j=1:bundle.N])
end

BM.build_bundle_model!(pm)
BM.run!(pm)
