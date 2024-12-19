# Isotropic $Q$-fractional Brownian motion on $\mathbb{S}^2$

Simulation and plotting of isotropic $Q$-fractional Brownian motion on the unit sphere $\mathbb{S}^2$ and evaluation of simulation methods.

This repository represents a snapshot of the code used for the creation of the manuscript <br><p align=center>*Isotropic Q-fractional Brownian motion on the sphere: regularity and fast simulation*.</p>
It allows simulating sample paths of fractional Brownian motion using the Circulant Embedding (CE) and Conditionalized Random Midpoint Displacement (CRMD) methods.
Further, it contains helper functions for computational performance evaluations of both methods and a Monte-Carlo error analysis of the CRMD method.
The Python script `generate_plots.py` was used to generate the plots for these evaluations.
Lastly, code is provided that allows plotting isotropic Q-fractional Brownian motion on the sphere for different points in time. 


## Usage
In order to use the tools provided here, clone this repository and enter it via the commands
```console
$ git clone https://github.com/muellerbjoern/QfBm-S2.git
$ cd QfBm-S2
```


### Evaluation of simulation methods of fractional Brownian motion sample paths

This section describes how to reproduce the performance comparison of CE and CRMD, and Monte-Carlo Error analysis of CRMD shown in the manuscript.

```console
$ julia --project=. -e "using Pkg; Pkg.instantiate();"
$ julia --project=. performed_experiments.jl [output_directory] [storage_directory]
```
will execute all experiments that are described in the manuscript.
It will first execute the performance evaluation of the Circulant Embedding and CRMD method, writing the results to a file in the folder `output_directory`.
It will then perform a Monte-Carlo error analysis of the CRMD method with the parameters used in the manuscript. The sample paths generated for this, as well as some precomputations, will be stored in `storage_directory`.
If `output_directory` or `storage_directory` is not provided, the current directory will be used.
Note that the first line only needs to be executed one time, when first running the script.

### Generating plots from the above evaluations

Calling
```console
$ python3 generate_plots.py
```
will first create a plot from the performance data and then the plot of the error analysis of CRMD.
Note that both performance evaluation and error analysis need to be performed first.

### Plotting Q-fractional Brownian motion on $\mathbb{S}^2$
To recreate the plots of samples of Q-fractional Brownian on $\mathbb{S}^2, execute
```console
$ julia --project=. -e "using Pkg; Pkg.instantiate();"
$ julia --project=. plot_QfBm.jl [output_directory]
```
Note that the first line only needs to be executed one time, when first running the script.

## Requirements
This code was run under Julia version 1.11, and Python 3.12 with corresponding versions of relevant libraries. In order for the results to be reproducible, the file Manifest.toml contains the exact Julia environment that was used.

## Authors and acknowledgment
This code was created by Annika Lang and Björn Müller.<br>
This work was supported in part by the European Union (ERC, StochMan, 101088589),
by the Swedish Research Council (VR) through grant no. 2020-04170, by the Wallenberg AI, Autonomous
Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation, and by the
Chalmers AI Research Centre (CHAIR).
Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

## License
This snapshot is published under the GPL v3.0 license.

## References
[1] Norros I, Mannersalo P, Wang J. 1999 Simulation of fractional Brownian motion with conditionalized random midpoint displacement. Advances in Performance Analysis 2, 77-101. <br>
[2] Dieker T. 2004 [Simulation of Fractional Brownian Motion](http://www.columbia.edu/~ad3217/fbm.html). Master's thesis Vrije Universiteit Amsterdam, Amsterdam. <br>
[3] Perrin E, Harba R, Jennane R, Iribarren I. 2002 [Fast and exact synthesis for 1-D fractional Brownian motion and fractional Gaussian noises](https://doi.org/10.1109/LSP.2002.805311). IEEE Signal Processing Letters 9, 382-384.

## Disclaimer
Funded by the European Union. Views and opinions expressed are however those of the author(s) only
and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither
the European Union nor the granting authority can be held responsible for them.

<img src="docs/images/LOGO_ERC-FLAG_FP.png" alt="ERC Logo" width= "400px" hspace="10px"/>

