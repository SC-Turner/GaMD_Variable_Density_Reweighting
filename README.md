# GaMD Variable Density Reweighting (VDR)
Reweighting of GaMD simulation trajectories in 2D collective variable space.

[![PyPI package](https://img.shields.io/badge/pip%20install-example--pypi--package-brightgreen)](https://pypi.org/project/example-pypi-package/) 
[![version number](https://img.shields.io/pypi/v/example-pypi-package?color=green&label=version)](https://github.com/tomchen/example_pypi_package/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Introduction
VDR is a reweighting methodology developed as an improvement to the original PyReweighting script by Yinglong Miao (2014).

# Installation
## Using pip
``` 
pip install <UPDATE PYPI>
``` 
## From tarball
``` 
git clone https://github.com/sct1g15/GaMD_Variable_Density_Reweighting.git
cd GaMD_Variable_Density_Reweighting
python setup.py install
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
For a more customised reweighting:
``` 
VDR --gamd output/gamd.log --data input/data_example.txt --cores 12 --emax 8 --mode convergence --conv_points 9500 --pbc True --output output_VDR
``` 
For details on all the arguments, you can use
``` 
VDR -h
``` 

## Requirements
- python3
- Numpy
- Scipy (>1.7.0)
- MDAnalysis
- multiprocessing
- pandas
- sklearn

## References
Miao Y, Sinko W, Pierce L, Bucher D, Walker RC, McCammon JA (2014) Improved reweighting of accelerated molecular dynamics simulations for free energy calculation. J Chemical Theory and Computation, 10(7): 2677-2689.
