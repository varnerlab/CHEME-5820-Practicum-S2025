abstract type AbstractlHopfieldNetworkModel end


"""
    MyModernHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

A mutable struct representing a classical Hopfield network model.

### Fields
- `X::Array{<:Number, 2}`: data matrix.
- `β::Number`: beta parameter (inverse temperature).
"""
mutable struct MyModernHopfieldNetworkModel <: AbstractlHopfieldNetworkModel

    # data -
    X::Array{<:Number, 2} # data matrix
    β::Number; # beta parameter

    # empty constructor -
    MyModernHopfieldNetworkModel() = new();
end