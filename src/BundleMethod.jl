#=
Julia package for implementing bundle methods
The current version has implemented a proximal bundle method.
=#

module BundleMethod

using Compat
using JuMP

abstract type AbstractBundleMethod end

#=
	This structure contains necessary information for bundle methods.
=#
mutable struct BundleInfo{T<:AbstractBundleMethod}
	n::Int64		# dimension in the solution space
	N::Int64		# number of separable functions in the objective
	m::JuMP.Model	# Bundle model
	k::Int64 		# iteration counter
	maxiter::Int64	# maximum number of iterations

	y::Array{Float64,1}		# current iterate
	fy::Array{Float64,1}	# objective values at the current iterate
	g::Array{Float64,2}		# subgradients

	# user-defined function to evaluate f
	# and return the value and its subgradients
	evaluate_f

	# History of bundles
	bundleRefs	# bundle references (can be either constraint or variable)
	yk::Array{Array{Float64,1},1}	# history of iterates
	fyk::Array{Float64,1}			# history of function values at iterates
	gk::Array{Array{Float64,1},1}	# history of subgradients
	jk::Array{Int64,1}				# history of function indices

	# Placeholder for extended structures
	ext

	function BundleInfo(T::DataType, n::Int64, N::Int64, func)
		bundle = new{T}()
		bundle.n = n
		bundle.N = N
		bundle.m = Model()
		bundle.k = 1
		bundle.maxiter = 500
		bundle.y = zeros(n)
		bundle.fy = zeros(N)
		bundle.g = zeros(0,0)
		bundle.evaluate_f = func
		bundle.bundleRefs = []
		bundle.yk = Array{Float64,1}[]
		bundle.fyk = Float64[]
		bundle.gk = Array{Float64,1}[]
		bundle.jk = Int64[]

		# initialize bundle model
		initialize!(bundle)

		return bundle
	end
end

function run(bundle::BundleInfo{<:AbstractBundleMethod})

	add_initial_bundles!(bundle)

	while true
		status = solve_bundle_model(bundle)
		if status != :Optimal
			println("TERMINATION: Invalid status from bundle model.")
			break
		end
		if termination_test(bundle)
			break
		end
		evaluate_functions!(bundle)
		manage_bundles!(bundle)
		update_iteration!(bundle)
		display_info!(bundle)
	end
end

#=
	Abstract functions are defined below.
	For each method, the functions may be implemented.
=#

const AbstractBundleInfo = BundleInfo{AbstractBundleMethod}

function initialize!(bundle::AbstractBundleInfo)
end

function add_initial_bundles!(bundle::AbstractBundleInfo)
end

function solve_bundle_model(bundle::AbstractBundleInfo)
	return :Optimal
end

function termination_test(bundle::AbstractBundleInfo)
	return true
end

function evaluate_functions!(bundle::AbstractBundleInfo)
end

function manage_bundles!(bundle::AbstractBundleInfo)
end

function update_iteration!(bundle::AbstractBundleInfo)
end

function display_info!(bundle::AbstractBundleInfo)
end

getsolution(bundle::AbstractBundleInfo)::Array{Float64,1} = Array{Float64,1}(undef,bundle.n)
getobjectivevalue(bundle::AbstractBundleInfo)::Float64 = NaN

#=
 	Add the implementations of bundle methods
=#

include("ProximalBundleMethod.jl")

end
