"""
Implementation of Proximal Bundle Method.

The implementation is based on
  Krzysztof C. Kiwiel, "Proximity control in bundle methods for convex nondifferentiable minimization"
  Mathematical Programming 46(1-3), 1990
"""

using LinearAlgebra
using SparseArrays
using Printf

const disable_purge_cuts = true

mutable struct ProximalMethod <: AbstractMethod
    model::BundleModel

    y::Array{Float64,1}  # current iterate of dimension n
    fy::Array{Float64,1} # objective values at y for N functions
    g::Dict{Int,SparseVector{Float64}} # subgradients of dimension n for N functions

    cuts::Dict{JuMP.ConstraintRef,Dict{String,Any}}

    iter::Int # iteration counter
    maxiter::Int # iteration limit

    # Algorithm-specific parameters
    u::Float64
    u_min::Float64
    M_g::Int
    ϵ_float::Float64	# tolerance for floating point comparison
    ϵ_s::Float64
    ϵ_v::Float64
    m_L::Float64 # serious step condition parameter (0, 0.5)
    m_R::Float64 # proximal term update parameter (0.5, 1)
    max_age::Float64

    x0::Array{Float64,1}	# current best solution (at iteration k)
    fx0::Array{Float64,1}	# current best objective values
    d::Array{Float64,1}
    v::Array{Float64,1}
    sum_of_v::Float64
    i::Int
    α::Array{Float64,1}

    statistics::Dict{Any,Any} # arbitrary collection of statistics
    eval_time::Float64        # function evaluation time
    start_time::Float64       # start time

    function ProximalMethod(n::Int, N::Int, func, init::Array{Float64,1}=zeros(n))
        pm = new()
        pm.model = BundleModel(n, N, func)

        @assert length(init) == n
        
        pm.y = copy(init)
        pm.fy = zeros(N)
        pm.g = Dict()

        pm.cuts = Dict()
        
        pm.iter = 0
        pm.maxiter = 3000
        
        pm.u = 1.0
        pm.u_min = 1.0e-6
        pm.M_g = 1e+6
        pm.ϵ_float = 1.0e-8
        pm.ϵ_s = 1.0e-5
        pm.ϵ_v = Inf
        pm.m_L = 1.0e-4
        pm.m_R = 0.5
        pm.max_age = 10.0
        
        pm.x0 = copy(init)
        pm.fx0 = zeros(N)
        pm.d = zeros(n)
        pm.v = zeros(N)
        pm.sum_of_v = 0.0
        pm.i = 0
        pm.α = zeros(N)
        
        pm.statistics = Dict(
            "total_eval_time" => 0.0)
        pm.start_time = time()
        
        return pm
    end
end

# This returns BundleModel object.
get_model(method::ProximalMethod)::BundleModel = method.model

# This returns solution.
get_solution(method::ProximalMethod) = method.x0

# This returns objective value.
get_objective_value(method::ProximalMethod) = sum(method.fx0)

# This sets the termination tolerance.
function set_bundle_tolerance!(method::ProximalMethod, tol::Float64)
    method.ϵ_s = tol
end

# This creates an objective function to the bundle model.
function add_objective_function!(method::ProximalMethod)
    bundle = get_model(method)
    d = bundle.model[:x]
    v = bundle.model[:θ]
    @objective(bundle.model, Min,
          sum(v[j] for j = 1:bundle.N)
        + 0.5 * method.u * sum(d[i]^2 for i = 1:bundle.n))
end

# This may collect solutions from the bundle model.
function collect_model_solution!(method::ProximalMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    if JuMP.termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        d = model[:x]
        v = model[:θ]
        for i = 1:bundle.n
            method.d[i] = JuMP.value(d[i])
            method.y[i] = method.x0[i] + method.d[i]
        end
        for j = 1:bundle.N
            method.v[j] = JuMP.value(v[j])
        end
        method.sum_of_v = sum(method.v)
        for (ref, cut) in method.cuts
            @assert JuMP.is_valid(get_jump_model(method), ref)
            cut["dual"] = JuMP.dual(ref)
            if cut["dual"] > -1e-6 && cut["α"] > 0.0
                cut["age"] += 1
            end
        end
    else
        @error "Unexpected model solution status ($(JuMP.termination_status(model)))"
    end
end

# Should the method terminate?
function termination_test(method::ProximalMethod)
    model = get_jump_model(method)
    if JuMP.termination_status(model) ∉ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        return true
    end
    if method.sum_of_v >= -method.ϵ_s * (1 + abs(sum(method.fx0)))
        println("TERMINATION: Optimal: v = ", method.sum_of_v)
        return true
    end
    if method.iter >= method.maxiter
        println("TERMINATION: Maximum number of iterations reached.")
        return true
    end
    return false
end

"""
This calls function `BundleModel.evaluate_f(y)` to get function value `fy` and gradient `g`.
This method assumes user-defined function of the form
    `evaluate_f(y::Vector{Float64})::Tuple{Float64,Array{Float64,2}}`
    returning function evaluation value and gradient as first and second outputs, resp.
"""
function evaluate_functions!(method::ProximalMethod)
    stime = time()
    method.fy, method.g = method.model.evaluate_f(method.y)
    method.statistics["total_eval_time"] += time() - stime

    if method.iter == 0
        method.u = 0.0
        bundle = get_model(method)
        for j = 1:bundle.N
            method.u += norm(method.g[j], 1)
        end
        method.u /= bundle.N

        method.x0 = copy(method.y)
        method.fx0 = copy(method.fy)
    end
end

# This updates the bundle pool by removing and/or adding bundle objects.
function update_bundles!(method::ProximalMethod)
    purge_bundles!(method)

    sumfy = sum(method.fy)
    sumfx0 = sum(method.fx0)
    if sumfy - sumfx0 <= method.m_L * method.sum_of_v
        # update bundles first
        for (ref, cut) in method.cuts
            j::Int = cut["j"]
            g::SparseVector{Float64} = cut["g"]
            offset = method.fx0[j] - method.fy[j] + g' * method.d
            JuMP.add_to_function_constant(ref, offset)
            # @show ref
        end

        bundle = get_model(method)
        model = get_model(bundle)
        d = model[:x]
        for i = 1:bundle.n
            if JuMP.has_upper_bound(d[i])
                JuMP.set_upper_bound(d[i], JuMP.upper_bound(d[i]) + method.x0[i] - method.y[i])
            end
            if JuMP.has_lower_bound(d[i])
                JuMP.set_lower_bound(d[i], JuMP.lower_bound(d[i]) + method.x0[i] - method.y[i])
            end
        end

        # serious step
        method.x0 = copy(method.y)
        method.fx0 = copy(method.fy)
        fill!(method.α, 0.0)
    else
        for j in eachindex(method.α)
            method.α[j] = method.fx0[j] - method.fy[j] + method.g[j]' * method.d
        end
    end

    add_bundles!(method)

    u = copy(method.u)
    if sumfy - sumfx0 <= method.m_L * method.sum_of_v
        if method.i > 0 && sumfy - sumfx0 <= method.m_R * method.sum_of_v
            u = 2 * method.u * (1 - (sumfy - sumfx0) / method.sum_of_v)
        elseif method.i > 3
            u = method.u / 2
        end
        # @show u, method.u/10, method.u_min
        newu = max(u, method.u / 10, method.u_min)
        method.ϵ_v = max(method.ϵ_v, -2 * method.sum_of_v)
        method.i = max(method.i + 1, 1)
        if newu != method.u
            method.u = newu
            method.i = 1
        end
        update_objective!(method)
    else
        p = norm(-method.u .* method.d, 1)
        α_tilde = -p^2 / method.u - method.sum_of_v

        method.ϵ_v = min(method.ϵ_v, p + α_tilde)
        if sum(method.α) > max(method.ϵ_v, -10 * method.sum_of_v) && method.i < -3
            u = 2 * method.u * (1 - (sumfy - sumfx0) / method.sum_of_v)
        end
        # @show u, 10 * method.u
        newu = min(u, 10 * method.u)
        method.i = min(method.i - 1, -1)
        if newu != method.u
            method.u = newu
            method.i = -1
            update_objective!(method)
        end
    end
    # update_objective!(method)
end

function purge_bundles!(method::ProximalMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    ncuts_removed = 0
    ncols = JuMP.num_variables(model)
    ncuts = length(method.cuts)
    for (ref, cut) in method.cuts
        if ncuts - ncuts_removed <= ncols
            break
        end
        if cut["age"] >= method.max_age
            JuMP.delete(model, ref)
            delete!(method.cuts, ref)
            ncuts_removed += 1
        end
    end
    if ncuts_removed > 0
        @printf("Removed %d inactive cuts.\n", ncuts_removed)
    end
end

function add_bundles!(method::ProximalMethod)
    # add bundles as constraints to the model
    bundle = get_model(method)
    model = get_model(bundle)
    for j = 1:bundle.N
        if method.iter == 0 || -method.α[j] + method.g[j]' * method.d > method.v[j] + method.ϵ_float
            d = model[:x]
            v = model[:θ]
            g = method.g[j]
            ref = @constraint(model, sum(g[i] * d[i] for i = 1:bundle.n) - v[j] <= method.α[j])
            method.cuts[ref] = Dict(
                "age" => 0.0,
                "dual" => 0.0,
                "j" => j,
                "g" => g,
                "α" => method.α[j]
            )
            # @show ref
        end
    end
end

# This updates the iteration information.
function update_iteration!(method::ProximalMethod)
    method.iter += 1
end

# This displays iteration information.
function display_info!(method::ProximalMethod)
    model = get_jump_model(method)
    nrows = 0
    for tp in [MOI.LessThan{Float64}, MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}]
        nrows += num_constraints(model, AffExpr, tp)
    end
    fx0 = sum(method.fx0)
    @printf("Iter %4d: ncols %5d, nrows %5d, fx0 %+e, fy %+e, m %+e, v %e, u %e, i %+d, master time %6.1fs, eval time %6.1fs, time %6.1fs\n",
        method.iter, num_variables(model), nrows, 
        fx0, 
        sum(method.fy), 
        method.sum_of_v + fx0, 
        method.sum_of_v, 
        method.u, method.i, 
        sum(method.model.time), method.statistics["total_eval_time"], time() - method.start_time)
end

"""
The following functions are specific for this method only.
"""

function update_objective!(method::ProximalMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    d = model[:x]
    v = model[:θ]
    @objective(model, Min,
          sum(v[j] for j = 1:bundle.N)
        + 0.5 * method.u * sum(d[i]^2 for i = 1:bundle.n))
end
