"""
Simple script to generate plots using matplotlib
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size':20})

figsize = (15, 6)

directory_errors = "./"
directory_runtimes = "./"

def CRMD_error_rates_all():
    """
    Computes the empirical error decay rates observed in CRMD for various values of H
    """
    Hs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
    levels = [9]

    # In this analysis, we focus on the behavior that is observed as mu grows
    # Hence, we ignore the first values
    ignore_firsts = [10, 20, 50]

    for ignore_first in ignore_firsts:
        print(f"Ignoring the first {ignore_first} error measurements, i.e., values for N = 1, ..., {ignore_first}, while computing the rate.")
        for H in Hs:
            for level in levels:
                # Read the errors that were computed in julia
                filename = f"{directory_errors}/errors_l2sup_comp_julia_N{level}_H{H}.h5"
                with h5py.File(filename, "r") as f:
                    errors = f["errors"][()]
                    m_s  = f["mu"][()]
                # The log of corresponding values of mu
                m_vals = np.log(m_s[ignore_first:-1])
                # Fit a line through the log-log plot, its slope is the rate of decay
                rate, b = np.polyfit(m_vals, np.log(errors[ignore_first:-1]), 1)
                # Simply print the result
                print("{:.2f}".format(round(-rate, 2)), end=" & ")
        print("\n")


def plot_CRMD_error(level, H, show_reference=False, marker='.', size=10):
    """
    Creates plots of the error incurred by CRMD
    Measured as supremum over all time steps of the L2(Omega) error
    """

    viridis = matplotlib.colormaps['viridis']
    color = viridis(H)

    markersize = 12

    # Load the errors file created using Julia
    filename = f"{directory_errors}/errors_l2sup_comp_julia_N{level}_H{H}.h5"
    with h5py.File(filename, "r") as f:
        errors = f["errors"][()]
        m_s  = f["mu"][()]

    # Indices of errors to consider
    # to declutter the plot
    indices = [1, 2, 3, 5, 7, 11, 15, 23, 31, 47, 63, 95, 126]
    m_vals = m_s

    # Plot the actual resulting errors
    plt.plot(m_vals[indices], errors[indices], label=r"$H={}$".format(H), marker=marker, linestyle='none', markersize=markersize, color=color)

    # Plot labels and legend
    plt.xlabel(r'$\mu$')
    plt.yscale("log")
    plt.ylabel(r'$\sup_{t\in [0, T]} || \beta^H(t) - \beta^{H, \mu}(t)||_{L^2(\Omega)}$')
    plt.xscale('log')


def plot_computation_times():
    """
    Creates a plot of the computation times of CE and CRMD for different mu
    """
    # Load runtime files
    filename_CE = f"{directory_runtimes}/runtimes_CE_julia.h5"
    with h5py.File(filename_CE, "r") as f:
        runtime_CE_Julia = f["runtimes"][()]

    filename_CRMD = f"{directory_runtimes}/runtimes_CRMD_julia.h5"
    with h5py.File(filename_CRMD, "r") as f:
        runtime_CRMD_Julia = f["runtimes"][()].T
        # The values of mu that were tested
        ms  = f["mu"][()]

    # Rescale from nanoseconds
    runtime_CE_Julia = runtime_CE_Julia / 1e9
    runtime_CRMD_Julia = runtime_CRMD_Julia.T / 1e9

    # Omit the first 5 values to make the plot nicer
    runtime_CE_Julia = runtime_CE_Julia[5:]
    runtime_CRMD_Julia = runtime_CRMD_Julia[:, 5:]

    # Values of N (in Python, arange excludes the endpoint)
    Ns = 2**np.arange(15, 25)

    # Size of markers
    size = 13

    plt.figure(figsize=figsize)

    # Reference lines
    plt.plot(Ns[:], 1.5*Ns[:]/4e7, label='$\mathcal{O}(N)$', linestyle='dashed')
    plt.plot(Ns, 1.2*Ns/1e9*np.log2(Ns), label='$\mathcal{O}(N\log(N))$', linestyle='dotted')

    # Iterate through all tested mu's, but plot only for selected few
    for j, m in enumerate(ms):
        # If it's one of those we want to have in the plot
        if m in [2, 5, 20]:
            # Plot
            markers = {0: '.',1: 'x', 2:'D', 5:'o', 10: '+', 20:'P'}
            plt.plot(Ns[:], runtime_CRMD_Julia[j][:], label=f'CRMD, $\mu={m}$', 
                     marker=markers[m], linewidth=0, markersize=size)

    # Error of CE
    plt.plot(Ns[:], runtime_CE_Julia[:], label='CE', marker='*', linewidth=0, markersize=size+1)

    # Labels for legend
    plt.ylabel("Computation time [s]", fontsize=23)
    plt.xlabel(r"$N$")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.04, 1))
    plt.savefig("perf.eps", bbox_inches='tight')
    plt.show()


def plot_CRMD_error_all_H():
    """Plot a selection of CRMD errors
    This function creates the figure that is in the manuscript and saves it.
    """
    # Initialize figure
    plt.figure(figsize=figsize)

    # Plot the reference line
    m_vals = np.arange(1,128)
    ref1 = 0.4752/(m_vals**(1))
    plt.plot(m_vals[1:], ref1[1:], label = r"$\mathcal{O}(\mu^{-1})$", markersize=10, linestyle='dashed')

    # Plot all the errors in the final figure
    plot_CRMD_error(level=9, H=0.05, marker='D', size=6)
    plot_CRMD_error(level=9, H=0.1, marker='o', size=3)
    plot_CRMD_error(level=9, H=0.2, marker='X', size=7)
    plot_CRMD_error(level=9, H=0.3, marker='v', size=5)
    plot_CRMD_error(level=9, H=0.4, marker='s', size=8)
    plot_CRMD_error(level=9, H=0.6, marker='P', size=9)
    plot_CRMD_error(level=9, H=0.7, marker='^', size=11)
    plot_CRMD_error(level=9, H=0.9, marker='*', size=13)
    plt.legend(bbox_to_anchor=(1.04, 1))
    # Write the figure into an eps file
    plt.savefig("errors.eps", bbox_inches='tight')
    plt.show()

# Do everything upon execution of the script
if __name__ == '__main__':

    # Parse command line arguments
    # First provide directory for errors, then runtimes
    import sys
    args = sys.argv[1:]

    # If directories provided, then we read from the given directories
    if len(args) >= 1:
        directory_errors = args[0]
        directory_runtimes = args[0]
    if len(args) >= 2:
        directory_runtimes = args[1]
    
    try:
        CRMD_error_rates_all()
        plot_CRMD_error_all_H()
        plot_computation_times()
    except Exception as e:
        raise(e)
        print("Usage: python generate_plots.py [error directory] [runtime directory]\n\
              Default directories are ./.\n Make sure that errors and runtimes are available with right filenames.")
