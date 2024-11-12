# GaMD Variable Density Reweighting (VDR)

[![PyPI package](https://img.shields.io/badge/pip%20install-gamdvdr-brightgreen)](https://pypi.org/project/gamdvdr/) 
[![Version](https://img.shields.io/badge/version-1.0.0-yellow)](#)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Introduction
VDR is a python-based package for energetic reweighting of Gaussian Accelerated MD simulations. VDR provides a toolkit for calculating optimal boost parameters for GaMD simulations and performing post-hoc reweighting of GaMD simulation trajectories.

VDR serves an improvement to the original PyReweighting script by Yinglong Miao (2014).

# Installation
## From source
``` 
git clone https://github.com/sct1g15/GaMD_Variable_Density_Reweighting.git
cd GaMD_Variable_Density_Reweighting
python setup.py install
```
## Using pip
``` 
pip install gamdvdr
``` 

# Tutorial
This tutorial will take you through parameterisation and reweighting of a Gaussian Accelerated MD simulation.

## Calculate the GaMD parameters
The VDR_param command calculates the highest standard deviation limits applicable to the amount of simulation frames you plan to save to your GaMD output trajectory. Default parameters assume a 0.01 anharmonicity tolerance, 0.02 kcal/mol standard error and 100 generated local clusters.
``` 
VDR_param --frames 950000
``` 
## Run GaMD Simulation
Run your GaMD simulation, where the sum of the standard deviation limits used, should not exceed the value output by the VDR_param command.

## Calculate CV values
Calculate the values of your CV of interest from the GaMD trajectory. This will vary between simulations, but an example script for calculating phi and psi angles from an alanine dipeptide example simulation have been provided in tutorial/phi_psi_calc.py. Formatting should match that in tutorial/data_example.dat, i.e. white spaced deliminated with three columns for CV1, CV2, frame number.

## Combine Repeats (Optional)
VDR_comb supports multiple inputs if multiple repeats were used to concatenate results. This will output a data_concat.dat and gamd_concat.log file.
``` 
VDR_comb --data data1.dat data2.dat data3.dat data4.dat --gamd gamd1.log gamd2.log gamd3.log gamd4.log
``` 

## Run VDR
Below is a minimum example for running VDR reweighting, this generate a single PMF distribution using a VDR cut-off of 9500:
``` 
VDR --gamd output/gamd.log --data input/data_example.txt --mode single --conv_points 9500 --pbc True --output output_VDR
``` 
For a more customised reweighting (Testing the data convergence):
``` 
VDR --gamd output/gamd.log --data input/data_example.txt --cores 12 --emax 8 --mode convergence --conv_points 1000 100000 --conv_points_num 7 --conv_points_scale log --pbc True --output output_VDR
``` 
For details on all the arguments, you can use
``` 
VDR -h
```

## VDR Arguments

Required Parameters:
| Parameter | Description | Required | Default | 
| :--- | :--- | :--- | :--- |
| gamd | gamd weights .dat file location, generated from GaMD simulation. | yes | None |
| data | Datafile location containing CV values and timestep, formatted as in input/data_example.txt. | yes | None |
| conv_points | Cut-off values for VDR segmentation, use multiple values for convergence mode, one value for single mode, or define range of values with conv_points_num and conv_points_scale. | yes | None |
| mode | Whether to evaluate a single cut-off value (--mode single) or evaluate convergence across multiple cut-off values (--mode convergence). | yes | None |
| cores | Number of CPU cores to use for VDR. | yes | None |

Optional Parameters:
| Parameter | Description | Required | Default | 
| :--- | :--- | :--- | :--- |
| emax | kcal/mol value assigned for unsampled regions of CV-space. | no | 8 |
| pbc | Whether to add partial duplicated boundaries if the CV-limits loop around, i.e. phi/psi angles. | no | False |
| error_tol | Standard deviation convergence error tolerance. | no | 0.02 |
| anharm_error_tol | Anharmonicity convergence error tolerance. | no | 0.01 |
| conv_points_num | Number of cut-off data points to use between range specified in --conv_points, only required for --mode convergence. | no | None |
| conv_points_scale | Whether to use a 'linear' or 'log' scale to distribute points across conv_points range with conv_points_scale. | no | 'linear' |
| cluster | Whether to extract frames from local cluster centers. Must be used with --mode single. | no | False |
| cluster_frames | Number of frames to include per local minima during clustering. | no | 100 |
| xlab | X-axis label for plotting. | no | 'CV1' |
| ylab | Y-axis label for plotting. | no | 'CV2' |

## Outputs
### PMF
- **`2C_PMF_{cutoff}.png`**: Contains reweighted 2D PMF plots.

### Intermediates
- **`bias_{cutoff}.png`**: Bias potential projected across CV space.
- **`PMF_{cutoff}.png`**: Unreweighted PMF distribution.
- **`distribution_{cutoff}.png`**: Distribution of points used to generate the bias potential energy surface through interpolation:
  - **`Blue points`**: Represent averaged sampled regions.
  - **`Orange points`**: Represent unsampled regions, assigned a value of *emax*.

### Convergence (only with `--mode convergence`)
- **`anharm_plot_max.png`**: Shows the anharmonicity of the datapoint with the highest anharmonicity value.
- **`std_plot_max.png`**: Shows the standard deviation of the datapoint with the highest standard deviation.

### Clusters
- **Clusters**: Contains frames from 2D PMF local minima, generated when using `--cluster` (must be combined with `--mode single`).
 

## References
Miao Y, Sinko W, Pierce L, Bucher D, Walker RC, McCammon JA (2014) Improved reweighting of accelerated molecular dynamics simulations for free energy calculation. J Chemical Theory and Computation, 10(7): 2677-2689.
Miao, Y., et al. (2015). Gaussian Accelerated Molecular Dynamics: Unconstrained Enhanced Sampling and Free Energy Calculation. Journal of Chemical Theory and Computation 11(8): 3584-3595.
	


