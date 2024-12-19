include("ce.jl")

using FastSphericalHarmonics
using CairoMakie
import Random

# To replace np.newaxis
newaxis = [CartesianIndex()]

function plot_sphere_q_fbm_makie(Θ, Φ, q_fbm; t=1, filename::Union{Nothing, String}=nothing)
    """
    Plots the given array `q_fbm` on the grid `(Θ, Φ)`
    """

    # Surface of the sphere in Cartesian coordinates
    x = [a*b for a in sin.(Θ), b in cos.(Φ)]
    y = [a*b for a in sin.(Θ), b in sin.(Φ)]
    z = [a for a in cos.(Θ), b in Φ]

    # Take Q-fBm at desired point in time
    q_fbm_t = q_fbm[t, :, :]

    # Create a figure
    fig = Figure(backgroundcolor=:transparent, figure_padding=0)
    ax = Axis3(fig[1,1]; aspect=(1,1,1))
    # Remove all the grids and everything except the actual sphere
    hidedecorations!(ax)
    hidespines!(ax)

    # Attempted reduction of margins in output file
    tightlimits!(ax)
    resize_to_layout!(fig)

    # Plot the sphere, with colors determined by the values of Q-fBm
    surface!(ax, x, y, z, color=q_fbm_t, colorrange=(-1,1), colormap=:jet1, shading=NoShading)

    # If no filename given, show the figure
    if isnothing(filename)
        display(fig)
    # If filename provided, save the figure there
    else
        save(filename, fig)
    end
end

function simulate_QfBm(max_level, H, decay, degree; seed=nothing, T=1.0)
    """
    Simulates a Q-fractional Brownian motion with `2^max_level`` grid points,
    Hurst parameter `H`, seed `seed` and final time `T`
    `decay` specifies how quickly the angular power spectrum decays. A_l is proportional to `(l+1)^{-decay/2}``
    `degree` is the parameter `κ`, the truncation parameter of the series expansion
    """

    # We set a global seed here, so we can then follow the RNG 
    # in the simulation of all the independent paths
    if !isnothing(seed)
        Random.seed!(seed)
    end
    N = 2^max_level

    # Angular power spectrum
    A_l = [(l+1)^(-decay) for l in 0:(degree-1)]
    A_l .= sqrt.(A_l)
    A_l = repeat(A_l, inner=[2])

    # Generator for fBm sample paths
    generator = CE_simulation.CE(max_level, H)

    # Array that will be used as input for the Spherical Harmonics Transform
    # Will contain the sample paths in an adapted order and rescaled
    # See https://mikaelslevinsky.github.io/FastTransforms/transforms.html
    # for details on ordering of coefficients
    sample_paths = zeros(N, degree, 2*degree-1)
    for (i, ii) in zip(axes(sample_paths, 2), axes(A_l, 1))
        for j in axes(sample_paths, 3)
            if j <= 2*degree-1 - (2*(i-1))
                # Create a random path and save the fBm path into the array, discarding the increments (fGn)
                sample_paths[:, i, j] .= CE_simulation.generate_sample_path(generator)[2].*(T)^H
            end
        end
        # Rescale the sample_paths according to the angular power spectrum
        for t in 1:N
            sample_paths[t, i, 1:(2*degree-1 - (2*(i-1)))] .*= A_l[2*i:end]
        end
    end

    # Array that will contain the Q-fBm on a grid on the sphere
    q_fbm = Array{Float64, 3}(undef, N,  degree+1, 2*degree)

    for i in axes(q_fbm, 1)
        # Actual (inverse) Spherical Harmonics Transform
        q_fbm[i, 1:(end-1), 1:(end-1)] = sph_evaluate(sample_paths[i, :, :])
        # Plugging together solution at beginning and end
        # Last and first element must be equal, they will be at the same grid points
        # This is required to get closed sphere
        q_fbm[i, :, end] = q_fbm[i, :, 1]
        q_fbm[i, end, :] = q_fbm[i, 1, :]
    end

    Θ, Φ = sph_points(degree)

    # Plugging together grid at beginning and end
    # Last and first elements must be equal
    # This is required to get closed sphere
    Φ = [Φ; Φ[1]]
    Θ = [Θ; Θ[1]]

    return (q_fbm, (Φ, Θ))
end


function plot_QfBm(max_level, H, decay, degree; seed=1)
    """
    Simulates and plots the final time of a Q-fractional Brownian motion with `2^max_level`` grid points,
    Hurst parameter `H`, seed `seed` and final time 1.
    `decay` specifies how quickly the angular power spectrum decays. A_l is proportional to `(l+1)^{-decay/2}``
    `degree` is the parameter `κ`, the truncation parameter of the series expansion
    """
    simulate_QfBm(max_level, H, decay, degree, seed=seed)
    # Plot the Q-fBm on the sphere, at t=1 (default, change to plot later values)
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm)
end

function plot_QfBm_paper(;seed=nothing, directory="./")
    """
    Function that was used to produce the specific plots for the paper
    """

    # Set up parameters
    max_level = 2
    decay = 5.0
    degree = 2^7

    # Generate paths for 3 different values of `H`
    # Up to time `T=4.0`, so we can easily extract t=1,2,3 from a dyadic grid (`T=4.0` is at index 4)
    q_fbm_01, _ = simulate_QfBm(max_level, 0.1, decay, degree, seed=seed, T=4.0)
    q_fbm_05, _ = simulate_QfBm(max_level, 0.5, decay, degree, seed=seed, T=4.0)
    q_fbm_09, (Φ, Θ) = simulate_QfBm(max_level, 0.9, decay, degree, seed=seed, T=4.0)

    # Indices we want to plot, depends on which dyadic level we simulate to
    ts = [1, 2, 3] 

    # Normalize
    max_01 = maximum(abs.(q_fbm_01))
    max_05 = maximum(abs.(q_fbm_05))
    max_09 = maximum(abs.(q_fbm_09))
    absmax = max(max_01, max_05, max_09)


    q_fbm_01 ./= absmax
    q_fbm_05 ./= absmax
    q_fbm_09 ./= absmax


    # Plot the Q-fBm on the sphere
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_01, t=ts[1], filename="$(directory)/QfBm_sphere_H0.1_t1.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_01, t=ts[2], filename="$(directory)/QfBm_sphere_H0.1_t2.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_01, t=ts[3], filename="$(directory)/QfBm_sphere_H0.1_t3.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_05, t=ts[1], filename="$(directory)/QfBm_sphere_H0.5_t1.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_05, t=ts[2], filename="$(directory)/QfBm_sphere_H0.5_t2.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_05, t=ts[3], filename="$(directory)/QfBm_sphere_H0.5_t3.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_09, t=ts[1], filename="$(directory)/QfBm_sphere_H0.9_t1.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_09, t=ts[2], filename="$(directory)/QfBm_sphere_H0.9_t2.png")
    plot_sphere_q_fbm_makie(Θ, Φ, q_fbm_09, t=ts[3], filename="$(directory)/QfBm_sphere_H0.9_t3.png")
end

# Directory to save into
directory = "./"

# If a directory is provided, we save into that instead
if length(ARGS) > 0
    directory = ARGS[1]
end

# Simulate the sample that is in the manuscript and plot it
plot_QfBm_paper(seed=8)
