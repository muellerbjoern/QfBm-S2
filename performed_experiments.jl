"""
Runs the simulation experiments that are included in the manuscript
"""

include("evaluation.jl")

# Setup
directory = "./fbm_simulation/results/"
storage_directory = "$(directory)"

if length(ARGS) > 0
    directory = ARGS[1]
end
if length(ARGS) > 1
    storage_directory = ARGS[2]
end
print("Saving outputs into directory $(directory)\n")
print("Saving intermediate files into $storage_directory\n")

mkpath(directory, mode=0o777)
mkpath("$(storage_directory)/sample_paths/CRMD/Julia/", mode=0o777)
mkpath("$(storage_directory)/julia_generators/", mode=0o777)


# Execute all simulations
# Note that these require multiple GB of RAM
# And multiple hours of computation time

evaluate_performance_CE(directory)
evaluate_performance_CRMD(directory)

fBm_error_L2sup_nm(9, 0.01, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.05, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.1, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.2, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.3, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.4, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.45, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.49, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.51, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.55, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.6, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.7, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.8, 1000, 10, 128, directory, storage_directory)
fBm_error_L2sup_nm(9, 0.9, 1000, 10, 128, directory, storage_directory)
