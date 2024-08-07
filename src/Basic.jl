"""
Implementation of Basic Bundle Method.
Enables objective with non-smooth convex and quadratic convex.
∑_j θ_j + 1/2 * x^T P x +q^T x
"""

using LinearAlgebra
using SparseArrays
using Printf

mutable struct BasicMethod <: AbstractMethod
    model::BundleModel
    P::SparseMatrixCSC{Float64}
    q::SparseVector{Float64}


    y::Array{Float64,1}  # current iterate of dimension n
    fy::Array{Float64,1} # objective values at y for N functions
    θ::Array{Float64,1}
    g::Dict{Int,SparseVector{Float64}} # subgradients of dimension n for N evaluate_functions


    x0::Array{Float64,1}	# current best solution (at iteration k)
    fx0::Array{Float64,1}	# current best objective values
    linerr::Float64         # linearization error

    cuts::Dict{JuMP.ConstraintRef,Dict{String,Any}}

    iter::Int # iteration counter

    statistics::Dict{Any,Any} # arbitrary collection of statistics
    start_time::Float64       # start time

    params::Parameters
    status::Int
    #1: Optimal
    #2: Iteration limit
    #3: Dual objective limit
    #4: Time limit

    function BasicMethod(n::Int, N::Int, func;
        init::Array{Float64,1} = zeros(n),
        params::Parameters = Parameters())

        bm = new()
        bm.model = BundleModel(n, N, params.ncuts_per_iter, func)

        @assert length(init) == n

        bm.P = spzeros(n, n)
        bm.q = spzeros(n)

        bm.y = copy(init)
        bm.fy = zeros(N)
        bm.θ = zeros(N)
        bm.g = Dict()

        bm.x0 = copy(init)
        bm.x0 = copy(init)
        bm.fx0 = copy(bm.fy)
        bm.linerr = 0.0

        bm.cuts = Dict()

        bm.iter = 0

        bm.statistics = Dict(
            "total_eval_time" => 0.0)
        bm.start_time = time()

        bm.params = params
        bm.status = 0

        return bm
    end
end

# This returns solution.
get_solution(method::BasicMethod) = method.x0

# This returns objective value.
get_objective_value(method::BasicMethod) = sum(method.fx0)

# This sets the termination tolerance.
set_bundle_tolerance!(method::BasicMethod, tol::Float64) = set_parameter(method.params, "ϵ_s", tol)

# This returns BundleModel object.
get_model(method::BasicMethod)::BundleModel = method.model

# This creates an objective function to the bundle model.
function add_objective_function!(method::BasicMethod)
    bundle = get_model(method)
    x = bundle.model[:x]
    θ = bundle.model[:θ]
    @objective(bundle.model, Min,
          sum(θ[j] for j = 1:bundle.ncuts_per_iter)
        + 0.5 * x' * method.P * x + method.q' * x)
end

"""
This calls function `BundleModel.evaluate_f(y)` to get function value `fy` and gradient `g`.
This method assumes user-defined function of the form
    `evaluate_f(y::Vector{Float64})::Tuple{Float64,Array{Float64,2}}`
    returning function evaluation value and gradient as first and second outputs, resp.
"""
function evaluate_functions!(method::BasicMethod)
    stime = time()
    method.fy, method.g = method.model.evaluate_f(method.y)
    method.statistics["total_eval_time"] += time() - stime
end

function purge_bundles!(method::BasicMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    ncuts_removed = 0
    ncols = JuMP.num_variables(model)
    ncuts = length(method.cuts)
    for (ref, cut) in method.cuts
        if ncuts - ncuts_removed <= ncols
            break
        end
        if cut["age"] >= method.params.max_age
            JuMP.delete(model, ref)
            delete!(method.cuts, ref)
            ncuts_removed += 1
        end
    end
    if ncuts_removed > 0
        if (method.params.print_output)
            @printf("Removed %d inactive cuts.\n", ncuts_removed)
        end
    end
end

function add_bundles!(method::BasicMethod)
    y = method.y
    fy = method.fy
    g = method.g

    bundle = get_model(method)

    # compute linearization error
    method.linerr = sum(method.fx0) - sum(fy)
    for j = 1:bundle.N, i = 1:bundle.n
        method.linerr -= g[j][i] * (method.x0[i] - y[i])
    end

    # add bundles constraints to the model
    θ = bundle.model[:θ]
    for i = 1:bundle.ncuts_per_iter
        agg_fy = sum(fy[j] for j in bundle.cut_indices[i])
        agg_g = sum(g[j] for j in bundle.cut_indices[i])

        cut_violation = agg_fy - method.θ[i]
        if method.iter == 0 || cut_violation > method.params.ϵ_float
            rhs = agg_g' * y - agg_fy
            newcut = true
            if cut_violation < method.params.ϵ_float * 1e3
                for (key, val) in method.cuts
                    if norm(agg_g - val["g"]) < method.params.ϵ_float && norm(rhs - val["rhs"]) < method.params.ϵ_float
                        newcut = false
                        break
                    end
                end 
            end
            if (newcut)
                ref = add_bundle_constraint!(method, y, agg_fy, agg_g, θ[i])
                method.cuts[ref] = Dict(
                    "age" => 0,
                    "dual" => 0.0,
                    "cut_index" => i,
                    "g" => agg_g,
                    "rhs" => agg_g' * y - agg_fy
                )
            end
        end
    end
end

function add_bundle_constraint!(
    method::BasicMethod, y::Array{Float64,1}, fy::Float64, g::SparseVector{Float64}, θ::JuMP.VariableRef)
    bundle = get_model(method)
    model = get_model(bundle)
    x = model[:x]
    ref = @constraint(model, fy + sum(g[i] * (x[i] - y[i]) for i = 1:bundle.n) <= θ)
    return ref
end

# This updates the iteration information.
function update_iteration!(method::BasicMethod)
    method.iter += 1
end

function collect_model_solution!(method::BasicMethod)
    bundle = get_model(method)
    model = get_model(bundle)
    if JuMP.termination_status(model) in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        x = model[:x]
        θ = model[:θ]
        for i = 1:bundle.n
            method.y[i] = JuMP.value(x[i])
        end
        for j = 1:bundle.ncuts_per_iter
            method.θ[j] = JuMP.value(θ[j])
        end
        method.x0 = copy(method.y)
        method.fx0 = copy(method.fy)
        for (ref, cut) in method.cuts
            @assert JuMP.is_valid(get_jump_model(method), ref)
            cut["dual"] = JuMP.dual(ref)
            if cut["dual"] > -1e-6 #&& cut["α"] > 0.0
                cut["age"] += 1
            end
        end
    else
        @error "Unexpected model solution status ($(JuMP.termination_status(model)))"
        println(model)
        println("IIS")
        JuMP.compute_conflict!(model)
        for (type1, type2) in JuMP.list_of_constraint_types(model)
            for constr in JuMP.all_constraints(model, type1, type2)
                if MOI.get(model, MOI.ConstraintConflictStatus(), constr) == MOI.IN_CONFLICT
                    println(constr)
                end
            end
        end
    end
end

# This displays iteration information.
function display_info!(method::BasicMethod)
    model = get_jump_model(method)
    nrows = 0
    for tp in [MOI.LessThan{Float64}, MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}]
        nrows += num_constraints(model, AffExpr, tp)
    end

    if (method.params.print_output)
        @printf("Iter %4d: ncols %5d, nrows %5d, fx0 %+e, m %+e, fy %+e, linerr %+e, master time %6.1fs, eval time %6.1fs, time %6.1fs\n",
            method.iter, num_variables(model), nrows, sum(method.fx0), sum(method.θ), sum(method.fy), method.linerr, 
            sum(method.model.time), method.statistics["total_eval_time"], time() - method.start_time)
    end
end

function termination_test(method::BasicMethod)::Bool
    model = get_jump_model(method)
    if JuMP.termination_status(model) ∉ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.LOCALLY_SOLVED]
        return true
    end
    if sum(method.fx0) - sum(method.θ) <= method.params.ϵ_s * (1 + abs(sum(method.fx0)))
        if (method.params.print_output)
            println("TERMINATION: Optimal")
        end
        method.status = 1
        return true
    end
    if method.iter >= method.params.maxiter
        if (method.params.print_output)
            println("TERMINATION: Maximum number of iterations reached.")
        end
        method.status = 2
        return true
    end
    if sum(method.fx0) <= method.params.obj_limit
        if (method.params.print_output)
            println("TERMINATION: Dual objective limit reached.")
        end
        method.status = 3
        return true
    end
    if time() - method.start_time > method.params.time_limit
        if (method.params.print_output)
            println("TERMINATION: Time limit reached.")
        end
        method.status = 4
        return true
    end
    return false
end

function update_objective!(method::BasicMethod, P::SparseMatrixCSC{Float64}, q::SparseVector{Float64})
    method.P = P
    method.q = q

    add_objective_function!(method)
end
