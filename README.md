# GaMD_Variable_Density_Reweighting
Improved reweighting of GaMD simulation results in 2D collective variable space

## Introduction
VDR is a reweighting methodology developed as an improvement to the original PyReweighting script by Yinglong Miao (2014)

## Tutorial
This tutorial provides a basic script to run an alanine dipeptide simulation from a preconfigured topology, using the OpenMM GaMD implementation

If you have already run your GaMD simulations you may skip this step.
### To run a test alanine simulation from scratch using openmm:
Starting from the tutorial directory
``` 
python gamd_testsim.py
``` 

## To reweight your trajectories along a given CV space
This project has been designed so that users can call the VDR_Call.py script directly from the command line directly after a GaMD simulation with no further modification, e.g:
``` 
python ../VDR_Call.py --gamd output/gamd.log --data input/data_example.txt --cores 6 --emax 8 --mode convergence --conv_points 10 25 50 75 100 200 --pbc True --output output_VDR --engine OpenMM
``` 
For details on the arguments, you can use
``` 
python ../VDR_Call.py -help
``` 
Example formatting for the input files can be found in the tutorial/input folder

gamd.log files generated from GaMD simulation outputs are processed directly by the script and do not need to be modified, due to differences in trajectory outputs and recording of equilibration steps between OpenMM and Amber/NaMD simulations we require specification of which simulation engine (--engine) was used so that equilibration steps can be removed.

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
