#=
Julia package for implementing bundle methods
The current version has implemented a proximal bundle method.
=#

module BundleMethod

export ProximalDualModel

using Compat
using JuMP

type Bundle
	ref	# constraint/variable reference
	y	# evaluation point
	fy	# evaluation value
	g	# subgradient
end

include("ProximalDualMethod.jl")

end
