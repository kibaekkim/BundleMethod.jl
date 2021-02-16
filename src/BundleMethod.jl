"""
Julia package for implementing bundle methods
The current version has implemented a proximal bundle method.

minimize
  sum_{j=1}^N f_j(x)
subject to
  x in a feasible set of dimension n

f_j may be evaluated by calling evaluate_f.
"""

module BundleMethod

using JuMP

include("Parameters.jl")
include("Abstract.jl")
include("Model.jl")
include("Proximal.jl")
include("TrustRegion.jl")

end
