include("ce.jl")
include("crmd.jl")

import ..CE_simulation
import ..CRMD_simulation

using ProgressLogging
using BenchmarkTools
using NPZ
using HDF5

function fBm_error_L2sup_nm(max_level, H, M, MM, μ_max, output_directory, storage_directory, seed=1)
    """
        fBm_error_L2sup_nm(max_level, H, M, MM, μ_max)
    
    Computes the errors incurred by CRMD with parameter ``μ = 1, …, μ_max`` using ``M × MM` sample paths.
    `M` is the number of samples generated at once, while `MM` describes how often this is done.
    Thus, larger `M` means more memory usage, while larger `MM` means more computation time.
    Suggested parameter is `M = 1000`.
    Sample paths are simulated on a grid with ``2^max_level`` grid points and with Hurst parameter `H`.

    The error is measured as the supremum over ``t=1, …, N`` of the ``L^2(Ω)`` error.
    The reference solution is computed with ``μ = N``.
    """
    # This function performs the strong error analysis of CRMD

    N = 2^max_level

    # We accumulate the L2(Omega) error across all time steps
    errors_l2 = zeros(Float64, μ_max+1, N)
    
    # Process omega by omega for memory reasons
    for (i, paths) in enumerate(CRMD_simulation.get_sample_paths_same_randomness(max_level, H,μ_max, M, MM, storage_directory, seed=seed))
        # Deviations from exact path
        diffs = zeros(Float64, size(paths))
        # For each path generated with parameter μ
        for μ in axes(paths, 1)
            # Compute deviations from exact path
            # Samples are generated such that exact path
            # is at last index: paths[end, :]
            diffs[μ, :] .= paths[μ, :] .- paths[end, :]
        end
        # Accumulate onto error
        errors_l2[:, :] .+= diffs.*diffs
    end
    # Divide by number of samples to obtain mean square error
    errors_l2 /= (M*MM)
    # Root mean square error, L2(Omega)
    errors_l2 .= sqrt.(errors_l2)
    # Maximum over all time
    errors_l2sup = maximum(errors_l2, dims=2)
    errors_l2sup = dropdims(errors_l2sup, dims=2)
    # Write to file
    h5open("$(output_directory)/errors_l2sup_comp_julia_N$(max_level)_H$(H).h5", "w") do file
        write(file, "description", "Errors for CRMD with varying mu, errors[i] = errors(mu[i])")
        write(file, "H", H)
        write(file, "N", max_level)
        write(file, "number_samples", M*MM)
        write(file, "seed", seed)
	write(file, "mu", [collect(1:μ_max); N])
        write(file, "errors", errors_l2sup)
    end
    # Write to numpy as well
    npzwrite("$output_directory/errors_l2sup_comp_julia_N$(max_level)_H$(H).npy", errors_l2sup)
end


function evaluate_performance_CRMD(output_directory)
    """
        evaluate_performance_CRMD()
    
    Perform benchmark on sample path generation with different `N`, `μ`, `ν` 
    """

    # Values of `N` to be tested
    Ns = 10:24
    # Values of `μ`, `ν` to be tested
    μs = [0, 1, 2, 5, 10, 20]
    # ``ν = ⌈μ/2⌉``
    νs = [1, 1, 1, 3, 5, 10]

    # Storage for results
    benchmark_times = Array{Float64}(undef, length(Ns), length(μs))

    # For all parameters
    for (i, μ) in enumerate(μs)

        ν = νs[i]

        @progress for (j, N) in  enumerate(Ns)
            # Initialize generator with current parameters
            generator = CRMD_simulation.CRMD(N, 0.8, μ, ν)
            # Benchmark
            bm = @benchmark CRMD_simulation.generate_sample_path($generator)
            # Save mean of execution times
            benchmark_times[j, i] = mean(bm.times) 
        end
    end
    # Save benchmark results to disk for plotting
    h5open("$(output_directory)/runtimes_CRMD_julia.h5", "w") do file
        write(file, "description", "Computation times for CRMD with varying N and mu, runtimes[i,j] = benchmark(N[i], mu[j])")
        write(file, "H", 0.8)
	    write(file, "N", collect(Ns))
        write(file, "mu", μs)
        write(file, "nu", νs)
        write(file, "runtimes", benchmark_times)
    end
    # Write to numpy as well
    npzwrite("$(output_directory)/runtimes_CRMD_julia.npy", benchmark_times)
end


function evaluate_performance_CE(output_directory)
    """
        evaluate_performance_CE()

    Perform benchmark on sample path generation with different `N`
    """
    # Values of N to be tested
    Ns = 10:24
    # Storage for results
    benchmark_times = Vector{Float64}(undef, length(Ns))
    @progress for (j, N) in  enumerate(Ns)
        # Create a generator, perform necessary precomputation
        # not part of the benchmark
        generator = CE_simulation.CE(N, 0.8)
        # Used for storing the path that was computed but will be returned next time
        sp = CE_simulation.SavedPath(zeros(1), false)
        # Evaluate the benchmark
        # Always evaluate in pairs, since computation is performed for each odd call
        # Each even call only returns a pre-computed sample path
        bm = @benchmark CE_simulation.generate_sample_path($generator, $sp) evals=2 seconds=20
        # Save mean of all runs
        benchmark_times[j] = mean(bm.times)
        
    end
    # Write the results to file
    h5open("$(output_directory)/runtimes_CE_julia.h5", "w") do file
        write(file, "description", "Computation times for CE with varying N, runtimes[i] = benchmark(N[i])")
        write(file, "H", 0.8)
	    write(file, "N", collect(Ns))
        write(file, "runtimes", benchmark_times)
    end
    # Write to numpy as well
    npzwrite("$(output_directory)/runtimes_CE_julia.npy", benchmark_times)
end
