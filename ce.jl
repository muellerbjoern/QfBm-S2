"""
Implementation of the Circulant Embedding method for simulation of fractional Brownian motion

Based on:
Perrin E, Harba R, Jennane R, Iribarren I. 2002 Fast and exact synthesis for 1-D fractional Brownian motion and fractional Gaussian noises. IEEE Signal Processing Letters 9, 382-384.(doi:10.1109/LSP.2002.805311)
Dieker T. 2004 Simulation of Fractional Brownian Motion. Master's thesis Vrije Universiteit Amsterdam, Amsterdam; http://www.columbia.edu/~ad3217/fbm.html
"""

module CE_simulation

export CE, generate_sample_path, save_sample_paths_npy, save_generator, load_generator, SavedPath

using LinearSolve
using LinearAlgebra
using Random, Distributions
using Plots
using FFTW
# Use Intel's MKL; remove this line if not on Intel CPU
FFTW.set_provider!("mkl")

using JLD
using NPZ

# To replace np.newaxis
newaxis = [CartesianIndex()]

mutable struct CE
    """
    Struct to store the precomputed information
    in order to allow for faster simulation later
    """
    H::Float64
    max_level::Int
    T::Float64
    t0::Float64
    sqrt_circ_eigvals::Vector{ComplexF64}
    P::FFTW.cFFTWPlan{ComplexF64, -1, true, 1, Tuple{Int64}}
end

function fGn_covmat_firstrow(grid, H)
    """
        fGn_covmat_firstrow(grid, H)
    
    Computation of the first row of the covariance matrix of fractional Gaussian noise (increments of fBm) on a given grid as needed 
    by the precomputation
    grid: n×2 matrix s.t. grid[:, 1] contains the left endpoints and grid[:, 2] the right endpoints of the intervals
    Formula for β an fBm with Hurst parameter H:
    ``E[(β(t) - β(s))(β(u) - β(v))] = ϕ(t, u) - ϕ(t, v) - ϕ(s, u) + ϕ(s, v) = 1/2(|t-u|^(2H) - |t-v|^(2H)  - |s-u|^(2H) + |s-uv|^(2H))``
    """
    tv = (abs.(grid[:, 2] .- grid[1, 2])).^(2*H)
    tu = (abs.(grid[:, 2] .- grid[1, 1])).^(2*H)
    sv = (abs.(grid[:, 1] .- grid[1, 2])).^(2*H)
    su = (abs.(grid[:, 1] .- grid[1, 1])).^(2*H)
    return (tu+sv-tv-su)/2
end

function CE(max_level::Int, H::Float64)
    """
        CE(max_level, H)

    Constructor for CE that performs all the precomputations necessary 
    to efficiently simulate fBm sample paths with Hurst parameter `H` 
    from time ``t_0 = 0`` to ``T=1``on a grid with ``N = 2^(max_level)`` 
    grid points using CE
    """

    T::Float64 = 1.0
    t0::Float64 = 0.0

    # We simulate 1 sample more, in order to get best performance
    # FFT should always be run on vectors of length 2^n for some n
    # Additional sample is simply discarded
    N = 2^max_level+1
    # Rescale T accordingly to be a little bit longer
    T = N*(T/(N-1))

    # Construct a grid
    # Left points of intervals
    grid_l::Vector{Float64} = collect(LinRange(t0, T, N+1)[1:end-1])
    # Right points of intervals (width of interval ``h=(T-t0)/N``)
    grid_r::Vector{Float64} = grid_l .+ ((T-t0)/N)
    # Pack together
    grid = hcat(grid_l, grid_r)

    # First row of the covariance of fGn
    firstrow = fGn_covmat_firstrow(grid, H)
    # First row of the larger matrix `C` 
    circ_mat_row::Vector{ComplexF64} = vcat(firstrow, firstrow[end-1:-1:2])
    # Eigenvalues computed via FFT
    circ_eigvals = fft!(circ_mat_row)

    # Save the square roots of the eigenvalues, used for computation later
    circ_eigvals = sqrt.(circ_eigvals)

    # Precompute what can be precomputed for the FFT during the algorithm
    P = plan_fft!(circ_eigvals)

    # Create generator using default constructor, save everything
    generator = CE(H, max_level, T, t0, circ_eigvals, P)

    return generator
end

# Struct for saving the state of generate_sample_path
# Enables computation of 2 paths at a time, but returning
# them one by one
mutable struct SavedPath
    X::Vector{Float64}
    use::Bool
end


function generate_sample_path(generator::CE, saved_path::SavedPath = SavedPath(zeros(1), false), seed::Union{Nothing, Integer}=nothing)
    """
        generate_sample_path)(generator, saved_path, seed)

    Randomly generates a sample path of fBm from the parameters in `generator`, using 
    the precomputed information.
    Provide a struct SavedPath(zeros(1), false) in order to save the state in it.
    If a seed is provided, saved_path will be discarded and the seed set before creating a new path.
    """

    # If no seed is provided
    if isnothing(seed)
        # If we have a path already saved, use only if seed is not newly set!
        if saved_path.use
            saved_path.use = false
            return saved_path.X, cumsum(saved_path.X)
        end
    # If a seed is provided, use it
    else
        Random.seed!(seed)
    end

    N = 2^generator.max_level+1

    # Draw complex normals s.t. real and imaginary part
    # are independently ``N(0, 1/2)`` distributed
    # Vector W/sqrt(2) in the article (x/sqrt(2) in Perrin)
    normals = randn(ComplexF64, 2N-2) 

    # Multiply the random numbers by square roots of the 
    # eigenvalues and perform the rescaling here that is
    # omitted in FFT algorithm
    # We want the vector ``(sqrt(2)⋅normals[j]⋅√(circ_eigvals[j])/√(2N-2))``
    sqN = sqrt(N-1)
    normals .*= generator.sqrt_circ_eigvals ./ sqN

    # Apply the FFT on the computed vector
    # Written here as a matrix multiplication since it is, in fact, a matrix multiplication
    # and the 'FFT plan' that we precomputed follows the same syntax and semantics
    mul!(normals, generator.P, normals)

    # Real part is one path of fGn
    # Return only N-1 samples, last one is discarded
    # Due to simulating one more than necessary for performance reasons
    X = real.(normals[1:N-1])

    # Imaginary part is another independent path of fGn
    saved_path.X = imag.(normals[1:N-1])
    saved_path.use = true
    
    # Cumulative sum of fGn is the path of fBm
    path = cumsum(X)

    return X, path
end

function save_sample_paths_npy(generator::CE, M::Int, output_directory::String, filename::Union{String, Nothing} = nothing)
    """
        save_sample_paths(generator, M, filename)

    Generate and save `M` increments of sample paths of fBm from the 
    parameters in `generator``, using the precomputed information in `generator`
    """
    N = 2^generator.max_level
    # Collect all `M` sample paths in single array
    all_path_increments = Array{Float64}(undef, N, M)

    # Simulate all `M` sample paths
    for i=1:M
        increments, _ = generate_sample_path(generator)
        all_path_increments[:, i] = increments[:]
    end
    # Write array into file
    if isnothing(filename)
        filename = "CE_H$(generator.H)_N$(generator.max_level)_M$(M).npy"
    end
    npzwrite("$(output_directory)/$(filename)", all_path_increments)
end

function save_generator(generator, output_directory, filename::Union{String, Nothing} = nothing)
    # Write the generator to disk so no computation needs to be repeated
    if isnothing(filename)
        filename = "CE_H$(generator.H)_N$(generator.max_level).jld"
    end
    jldopen("$(output_directory)/$(filename)", "w") do file
        write(file, "generator", generator)
    end 
end

function load_generator(input_directory, filename)
    generator = load("$(input_directory)/$(filename)")["generator"]
    return generator
end

end