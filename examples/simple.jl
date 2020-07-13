using BundleMethod
using JuMP
using Ipopt
using Random

#=
This example considers:
	minimize \sum_{i=1}^N \sum_{j=1}^n b_i * (x_j - a_ij)^2
	subject to -1 <= x_j <= 1
where objective function is separable such that
	f_i(x) := \sum_{j=1}^n b_i * (x_j - a_ij)^2
=#

# Randomly generate problem data
Random.seed!(1)
n = 3
N = 2
a = rand(N, n)
b = rand(N)

#=
User-defined function:

This function takes a new trial point `y` of dimension `n`
  and returns the array of separable objective function values
  and gradient of the functions.
=#
function evaluate_f(y)
	N, n = size(a)
	fvals = zeros(N)
	grads = Dict{Int,Vector{Float64}}()
	for i = 1:N
		grad = zeros(n)
		for j = 1:n
			fvals[i] += b[i] * (y[j] - a[i,j])^2
			grad[j] += 2 * b[i] * (y[j] - a[i,j])
		end
		grads[i] = grad
	end
	return fvals, grads
end

# This initializes the proximal bundle method with required arguments.
pm = BM.ProximalMethod(n, N, evaluate_f)

# Set optimization solver to the internal JuMP.Model
model = BM.get_jump_model(pm)
set_optimizer(model, Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0)

# We overwrite the function to have column bounds.
function BM.add_variables!(method::BM.ProximalMethod)
	bundle = BM.get_model(method)
	model = BM.get_model(bundle)
	@variable(model, -1 <= x[i=1:bundle.n] <= 1)
	@variable(model, θ[j=1:bundle.N])
end

# This builds the bundle model.
BM.build_bundle_model!(pm)

# This runs the bundle method.
BM.run!(pm)
