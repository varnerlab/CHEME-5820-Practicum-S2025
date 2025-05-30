# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");

# flag to check if the include file was called -
const _DID_INCLUDE_FILE_GET_CALLED = true;
const _WORD_EMBEDDING_FILE_URL = "https://drive.google.com/file/d/1tP9W4R1Ap7vp2AAmgoVEJ5lMfQAk5Fym/view?usp=share_link";

# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using JLD2
using FileIO
using CSV
using DataFrames
using TSne
using Clustering
using Plots
using Colors
using NNlib
using LinearAlgebra
using Statistics
using Random
using PrettyTables
using Downloads
using Test

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Files.jl"));