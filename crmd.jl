"""
Implementation of the Conditionalized Random Midpoint Displacement method
for simulation of fractional Brownian motion

Based on:
Norros I, Mannersalo P, Wang J. 1999 Simulation of fractional Brownian motion with conditionalized random midpoint displacement. Advances in Performance Analysis 2, 77-101.
Dieker T. 2004 Simulation of Fractional Brownian Motion. Master's thesis Vrije Universiteit Amsterdam, Amsterdam; http://www.columbia.edu/~ad3217/fbm.html
"""

module CRMD_simulation

export CRMD, generate_sample_path, save_sample_paths_npy, get_generator, get_sample_paths_same_randomness,
 save_sample_paths_same_randomness, generate_sample_paths_same_randomness

using LinearSolve
using LinearAlgebra
using Random, Distributions
using Plots

using JLD
using NPZ
using ResumableFunctions

using BenchmarkTools

using ProgressLogging

# To replace np.newaxis
newaxis = [CartesianIndex()]

mutable struct CRMD
    """
    Struct to store the precomputed information
    in order to allow for faster simulation later
    """
    H::Float64
    # Maximal dyadic level: path length N = 2^(max_level)
    max_level::Int
    # end time
    T::Float64
    # start time
    t0::Float64
    # μ, width of conditioning to the left
    μ::Int
    # ν, width of conditioning to the right
    ν::Int
    # cf. Norros for details on eik, vik
    # Vectors used for computation of conditional expectations
    # For boundary depends on how far to the left/right we can
    # actually condition
    eiks::Matrix{Vector{Float64}}
    # Scalars used for computation of conditional variances
    viks::Matrix{Float64}
    # Precomputed relative indices 
    rel_indices::Matrix{Vector{Int}}
end

function fGn_covmat(grid, H)
    """
        covmat_increments(grid, H)
    
    Computation of the covariance matrix of fractional Gaussian noise
    (increments of fBm) on a given grid as needed by the precomputation
    grid: n×2 matrix s.t. grid[:, 1] contains the left endpoints and grid[:, 2] the right endpoints of the intervals
    Formula for β an fBm with Hurst parameter H:
    ``E[(β(t) - β(s))(β(u) - β(v))] = ϕ(t, u) - ϕ(t, v) - ϕ(s, u) + ϕ(s, v) = 1/2(|t-u|^(2H) - |t-v|^(2H)  - |s-u|^(2H) + |s-uv|^(2H))``
    """
    tv = (abs.(grid[:, newaxis, 2] .- grid[newaxis, :, 2])).^(2H)
    tu = (abs.(grid[:, newaxis, 2] .- grid[newaxis, :, 1])).^(2H)
    sv = (abs.(grid[:, newaxis, 1] .- grid[newaxis, :, 2])).^(2H)
    su = (abs.(grid[:, newaxis, 1] .- grid[newaxis, :, 1])).^(2H)
    return (tu+sv-tv-su)/2
end

function CRMD(max_level::Int, H::Float64, μ::Int, ν::Int)
    """
        CRMD(max_level, H, m, n)

    Constructor for CRMD that performs all the precomputations necessary 
    to efficiently simulate fBm sample paths with Hurst parameter `H` 
    from time ``t_0 = 0`` to ``T=1``on a grid with ``N = 2^(max_level)`` 
    grid points using CRMD with approximation parameters `μ` and `ν`
    """
    # We need up to `μ + 1 \times ν` different vectors for the computation
    # of conditional expectation, due to the boundary:
    # E.g., if there exist only ``k < μ`` intervals to the left, we can only
    # condition on `k` left increments
    eiks = Matrix{Vector{Float64}}(undef, μ+1, ν)
    # Similar for the relative indices of these increments
    rel_indices = Matrix{Vector{Int}}(undef, μ+1, ν)
    # Similar for the conditional variances - depend on how many increments we
    # condition on 
    viks = Matrix{Float64}(undef, μ+1, ν)

    # Number of left increments to condition on, at most `μ`
    for left in 0:μ
        # Number of right increments to condition on, at most `ν`
        for right in 1:ν
            # The relative indices of the increments to condition on
            # i.e., index 0 is the place of increment currently to be simulated
            # Assumes that intervals in current step have length 1
            # To be rescaled during simulation
            curr_rel_indices = Array{Int}(undef, left+right)
            # To the left
            curr_rel_indices[1:left] = -1:-1:-left
            # To the right, includes 0 for increment at current position from previous step
            curr_rel_indices[left+1:left+right] = 0:2:2*right-2

            # Same as above, but including another 0 for the newly to be simulated increment
            curr_all_rel_indices = Array{Int}(undef, left+right+1)
            curr_all_rel_indices[2:left+1] = -1:-1:-left
            # Current one must be in very first position, else it makes no sense at all
            curr_all_rel_indices[1] = 0
            curr_all_rel_indices[left+2:left+right+1] = 0:2:2*right-2

            # Create a grid from relative indices
            grid = hcat(curr_all_rel_indices, curr_all_rel_indices)
            grid[1:left+1, 2] .+= 1
            grid[left+2:left+right+1, 2] .+= 2
            # Create covariance matrix from grid
            # FULL Gamma matrix (cf. Norros)!
            Γ = fGn_covmat(grid, H)
            # Compute the vector eik (cf. Norros)
            # Nota bene: Requires to be scaled later!
            eiks[left+1, right] = solve(LinearProblem(Γ[2:end, 2:end], Γ[1, 2:end]))
            # Compute vik (cf. Norros)
            # Nota bene: Requires to be scaled later!
            viks[left+1, right] = det(Γ) / det(Γ[2:end, 2:end])
            # Save relative indices
            rel_indices[left+1, right] = curr_rel_indices
        end
    end
    # Create generator from default constructor
    # Start time 0, end time 1
    generator = CRMD(H, max_level, 1.0, 0.0, μ, ν, eiks, viks, rel_indices)
    # Return it
    return generator
end


function generate_sample_path(generator::CRMD)
    """
        generate_sample_path)(generator)

    Randomly generates a sample path of fBm from the parameters in `generator``, using 
    the precomputed information.
    """
    return generate_sample_path(generator, randn(2^generator.max_level))
end

function generate_sample_path(generator::CRMD, normals::Vector{Float64})
    """
        generate_sample_path)(generator, normals)

    Generates a sample path of fBm from the parameters in `generator`, using 
    the precomputed information and the standard normal numbers given in `normals``
    """
    # Length of path
    N = 2^generator.max_level
    # Storage for the sample path increments, needs to be initialized to 0 for later
    X::Vector{Float64} = zeros(N)
    # Very first increment ``β(T) - β(t0)``
    X[1] = sqrt(generator.T-generator.t0)^generator.H * normals[1]
    # levels of dyadic refinement
    for i in 1:generator.max_level
        # Length of current dyadic level
        curr_N = 2^i
        # Needed to rescale eik and vik according to the current level
        scale_var = 1 / (2^(2*i*generator.H))
        # Step size (size of intervals) to the left - stepsize of current level
        left_step = 2^(generator.max_level-i)
        # Simulate all increments at current level
        # Go steps of 2 straigt away s.t. these 2 increments sum up to the
        # corresponding increment at previous level that is currently being refined
        # Like in Norros, include 1 and 2^i-1
        for k in 1:2:2^(i)
            # How many increments to the left to condition on
            # At most `min(k-1, μ)` where `k-1` is the number of available increments
            left = min(k-1, generator.μ)
            # How many increments to the right to condition on
            # At most `min(curr_N - (k-1))÷2, ν)` where `curr_N - (k-1))÷2` 
            # is the number of available increments
            right = min((curr_N - (k-1))÷2, generator.ν)
            # Get corresponding precomputed data
            eik::Vector{Float64} = generator.eiks[left+1, right]
            rel_indices::Vector{Int} = generator.rel_indices[left+1, right]
            vik = generator.viks[left+1, right] * scale_var

            # Obtain the increment we are now refining
            current_increment = X[(k-1)*left_step+1]

            # Conditional expectation, computed as a dot product
            # of known increments with eik
            expect::Float64 = 0
            @inbounds for idx in eachindex(rel_indices)
                # Affine transformation of relative indices to match 
                # current position and step size
                index = left_step *rel_indices[idx] + (k-1)*left_step + 1
                expect += eik[idx] * X[index]
            end

            # Simulate the new sub-increment using computed conditional expectation
            # and variance
            new_increment = expect + normals[k*left_step+1]*sqrt(vik)
            # Save it into the sample path 
            X[(k-1)*left_step+1] = new_increment
            # Next sub-increment is determined by previous increment that is currently
            # being refined
            X[k*left_step+1] = current_increment - new_increment
        end
    end

    # fBm sample path is a cumulative sum of the increments
    path = cumsum(X)

    # Return increments and path
    return X, path
end

function save_sample_paths_npy(generator::CRMD, M::Int, storage_directory, filename)
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
        increments, path = generate_sample_path(generator)
        all_path_increments[:, i] = increments[:]
    end
    # Write array into file
    npzwrite("$(storage_directory)/sample_paths/CRMD/Julia/$(filename)", all_path_increments)
end

function get_generator(max_level, H, μ, ν, storage_directory, filename::Union{String, Nothing} = nothing)
    """
        get_generator(max_level, H, μ, ν)

    Obtain a generator with the given parameters. If it exists on disk, load from file.
    Else, create the generator and write it to diks.
    Used to avoid recomputing large amounts of data.
    """

    if isnothing(filename)
        filename = "generator_H$(H)_m$(μ)_n$(ν).jld"
    end
    generator = nothing
    try
        # Try loading the generator from the given file
        generator = load("$(storage_directory)/julia_generators/$(filename)")["generator"]
    catch e
        # If the file cannot be loaded, an ArgumentError will be thrown
        # So, we catch it here and create the generator ourselves
        if e isa ArgumentError
            # Create generator
            generator = CRMD(max_level, H, μ, ν)
            # Write it to file
            jldopen("$(storage_directory)/julia_generators/$(filename)", "w") do file
                write(file, "generator", generator)
            end 
        # If a different error happened, we cannot handle it
        else
            rethrow(e)
        end
    end
    # Set the maximum level of the generator, the precomputation does not depend on `max_level`
    generator.max_level = max_level
    generator
end

function generate_sample_paths_same_randomness(max_level, H, μ_max::Int, M, storage_directory)
    """
        generate_sample_paths_same_randomness(max_level, H, μ_max, M)
    
    Generates `M` sets of sample paths containing `μ_max+1`` sample paths per set such that each set is 
    based on the same random numbers but different values of `μ`, setting ``ν = ⌈μ/2⌉``.
    The `i`-th sample path per set is run with ``μ=i``, except the very last one which is run with ``μ = N``
    """

    N = 2^max_level

    # Pre-generate the random numbers to be used
    normals::Matrix{Float64} = randn(M, N)
    # Variable to store all paths
    saved_paths = zeros(Float64, M, μ_max+1, N)
    # Iterate over μ from 1 to μ_max and lastly μ=N
    for (i, μ) in enumerate(Iterators.flatten((1:μ_max, N)))
        # ν = ⌈μ/2⌉, but never 0
        ν = max(1, ceil(Int64, μ/2))

        # Generator object contains precomputed information
        curr_gen = get_generator(max_level, H, μ, ν, storage_directory, "generator_H$(H)_m$(μ)_n$(ν).jld")

        # Actually generate the sample paths here
        for j = 1:M
            saved_paths[j, i, :] .= generate_sample_path(curr_gen, normals[j,:])[2]
        end
    end
    return saved_paths
end

function save_sample_paths_same_randomness(max_level, H, μ_max::Int, M, storage_directory, filename)
    """
        save_sample_paths_same_randomness(max_level, H, μ_max, seed, M, index)
    
    Generates and saves to disk `M` sets of sample paths containing `μ_max+1`` sample paths per set such that 
    each set is based on the same random numbers but different values of `μ`, setting ``ν = ⌈μ/2⌉``.
    The `i`-th sample path per set is run with ``μ=i``, except the very last one which is run with ``μ = N``
    `index` can be provided for the case that multiple such files are created.
    """
    # Generate M sets of paths
    all_paths = generate_sample_paths_same_randomness(max_level, H, μ_max, M, storage_directory)
    # Write them to file
    npzwrite("$(storage_directory)/sample_paths/CRMD/Julia/$(filename)", all_paths)
    return all_paths
end

@resumable function get_sample_paths_same_randomness(max_level, H, μ_max::Int, M::Int, MM::Int, storage_directory; seed=1)
    """
        save_sample_paths_same_randomness(max_level, H, μ_max, seed, M, index)

    Obtains `MM` instances of `M` sets of sample paths containing `μ_max+1`` sample paths per set such that 
    each set is based on the same random numbers but different values of `μ`, setting ``ν = ⌈μ/2⌉``.
    The `i`-th sample path per set is run with ``μ=i``, except the very last one which is run with ``μ = N``
    Attempts reading `MM` files containing `M` sets of sample paths each and returns a single set of sample paths
    at a time. Splitting into `MM` files is done for memory reasons.
    The function is a generator, e.g., to be used in a for loop to iterate over all ``MM × M`` sets of sample paths.
    """

    # Set a seed to obtain reproducible results
    Random.seed!(seed)

    for i in 1:MM
        filename = "CRMD_H$(H)_N$(max_level)_mmax$(μ_max)_seed$(seed)_M$(M)_$(i).npy"
        # If a file with the desired paths exists, just load them
        # Else generate and save them
        try
            a = npzread("$(storage_directory)/sample_paths/CRMD/Julia/$(filename)")
        catch e
            if e isa SystemError
                a = save_sample_paths_same_randomness(max_level, H, μ_max, M, storage_directory, filename)
            else
                rethrow(e)
            end
        end

        # Return each single set of paths separately
        for j in 1:M
            @yield a[j, :, :]
        end
    end
end

end