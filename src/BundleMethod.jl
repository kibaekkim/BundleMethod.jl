#=
Julia package for implementing bundle methods
The current version has implemented a proximal bundle method.
=#

module BundleMethod

using Compat
using JuMP

abstract type AbstractBundleMethod end

struct Bundle
	ref	# constraint/variable reference
	y	# evaluation point
	fy	# evaluation value
	g	# subgradient
end

#=
	This structure contains necessary information for bundle methods.
=#
mutable struct BundleModel{T<:AbstractBundleMethod}
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

	# Placeholder for extended structures
	ext

	function BundleModel(T::DataType, n::Int64, N::Int64, func, splitvars = false)
		bundle = new{T}()
		bundle.n = n
		bundle.N = N
		bundle.m = Model()
		bundle.k = 0
		bundle.maxiter = 500
		bundle.y = zeros(n)
		bundle.fy = zeros(N)
		bundle.g = zeros(0,0)
		bundle.splitvars = splitvars
		bundle.evaluate_f = func
		bundle.history = Dict{Tuple{Int64,Int64},Bundle}()

		# initialize bundle model
		initialize!(bundle)

		return bundle
	end
end

function run(bundle::BundleModel{<:AbstractBundleMethod})

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

const AbstractBundleModel = BundleModel{AbstractBundleMethod}

function initialize!(bundle::AbstractBundleModel)
end

function add_initial_bundles!(bundle::AbstractBundleModel)
end

function solve_bundle_model(bundle::AbstractBundleModel)
	return :Optimal
end

function termination_test(bundle::AbstractBundleModel)
	return true
end

function evaluate_functions!(bundle::AbstractBundleModel)
end

function manage_bundles!(bundle::AbstractBundleModel)
end

function update_iteration!(bundle::AbstractBundleModel)
end

function display_info!(bundle::AbstractBundleModel)
end

getsolution(bundle::AbstractBundleModel)::Array{Float64,1} = Array{Float64,1}(undef,bundle.n)
getobjectivevalue(bundle::AbstractBundleModel)::Float64 = NaN

#=
 	Add the implementations of bundle methods
=#

include("ProximalBundleMethod.jl")

end
