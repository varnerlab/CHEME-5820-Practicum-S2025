
"""
    ⊗(a::Array{Float64,1},b::Array{Float64,1}) -> Array{Float64,2}

Compute the outer product of two vectors `a` and `b` and returns a matrix.

### Arguments
- `a::Array{Float64,1}`: a vector of length `m`.
- `b::Array{Float64,1}`: a vector of length `n`.

### Returns
- `Y::Array{Float64,2}`: a matrix of size `m x n` such that `Y[i,j] = a[i]*b[j]`.
"""
function ⊗(a::Array{T,1}, b::Array{T,1})::Array{T,2} where T <: Number

    # initialize -
    m = length(a)
    n = length(b)
    Y = zeros(m,n)

    # main loop 
    for i ∈ 1:m
        for j ∈ 1:n
            Y[i,j] = a[i]*b[j]
        end
    end

    # return 
    return Y
end


"""
    recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
        maxiterations::Int64 = 1000, ϵ::Float64 = 1e-10) where T <: Number

This method takes a moderm Hopfield network model and a random state 
and state `s`, the frames and the probabilities of the states at each iteration.

### Arguments
- `model::MyModernHopfieldNetworkModel`: a Hopfield network model.
- `sₒ::Array{T,1}`: a random state.
- `maxiterations::Int64`: the maximum number of iterations.
- `ϵ::Float64`: the convergence threshold.

### Returns
- `s::Array{T,1}`: the final state.
- `frames::Dict{Int64, Array{Float32,1}}`: a dictionary of frames (states at each iteration).
- `probability::Dict{Int64, Array{Float64,1}}`: a dictionary of probabilities (probabilities of the states at each iteration).
"""
function recover(model::MyModernHopfieldNetworkModel, sₒ::Array{T,1}; 
    maxiterations::Int64 = 1000, ϵ::Float64 = 1e-10) where T <: Number

    # initialize -
    X = model.X; # data matrix from the model. This holds the memories on the columns
    β = model.β; # beta parameter (inverse temperature)
    frames = Dict{Int64, Array{Float32,1}}(); # save the iterations -
    probability = Dict{Int64, Array{Float64,1}}(); # save the probabilities
    
    # setup the initial state -
    frames[0] = copy(sₒ); # copy the initial random state
    probability[0] = NNlib.softmax(β*transpose(X)*sₒ); # initial probability
    should_stop_iteration = false; # flag to stop the iteration
    iteration_counter = 1; # iteration counter

    # loop -
    s = copy(sₒ); # initial state
    Δ = Inf; # initial delta
    while (should_stop_iteration == false)
        
        p = NNlib.softmax(β*transpose(X)*s); # compute the probabilities
        s = X*p; # update the state
        
        frames[iteration_counter] = copy(s); # save a copy of the state in the frames dictionary
        probability[iteration_counter] = p; # save the probabilities in the probability dictionary

        # first: compute the difference between the current and previous probabilities
        if (iteration_counter > 1)
            Δ = norm(probability[iteration_counter] - probability[iteration_counter-1]).^2;
        end

        # next: check for convergence. If we are out of iterations or the difference is small, we stop
        if (iteration_counter >= maxiterations || Δ ≤ ϵ)
            should_stop_iteration = true;
        else
            iteration_counter += 1; # increment the iteration counter, we are not done yet. Keep going.
        end
    end

    # return -
    return s,frames,probability
end