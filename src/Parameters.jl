Base.@kwdef mutable struct Parameters
    print_output::Bool = true #print output
    maxiter::Int = 3000     # maximum number of iterations
    ncuts_per_iter::Int = 1 # number of cuts added per iteration
    obj_limit::Float64 = -Inf # termination condition
    time_limit::Float64 = 3600.0

    ϵ_s::Float64 = 1.e-5 # termination tolerance
    m_L::Float64 = 1.e-4 # serious step condition parameter (0, 0.5)

    # Proximal method
    u::Float64 = 1.e-2       # initial proximal penalty value
    u_min::Float64 = 1.e-6   # minimum proximal penalty value
    ϵ_float::Float64 = 1.e-8 # tolerance for floating point comparison
    m_R::Float64 = 0.5       # proximal term update parameter (0.5, 1)
    max_age::Int = 10        # maximum number of iterations before removing inactive cuts

    # Trust region method
    Δ_ub::Float64 = 1.e+3 # trust region bound upper limit
    Δ_lb::Float64 = 1.e-4 # trust region bound lower limit

end

function get_parameter(params::Parameters, pname::String)
    return getfield(params, Symbol(pname))
end

function set_parameter(params::Parameters, pname::String, val)
    setfield!(params, Symbol(pname), val)
    return nothing
end