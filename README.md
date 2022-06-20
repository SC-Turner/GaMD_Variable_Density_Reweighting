# GaMD_Variable_Density_Reweighting
Improved reweighting of GaMD simulation results in 2D collective variable space

## Introduction
VDR is an improved reweighting methodology developed from the original PyReweighting script by Yinglong Miao (2014)

## Tutorial
This project has been designed so that users can call the VDR_Call.py script directly from the command line, e.g:

'''
python3 VDR_Call.py --gamd weights.dat --data.dat --cores 6 --emax 8 --itermax 9999 --mode convergence --conv_points 100 1000 10000 100000 --pbc False --output output_dir
'''

### Arguments
gamd = weights.dat derived from GaMD simulation


## Requirements
python3
Numpy
Scipy (>1.7.0)
MDAnalysis
multiprocessing
pandas
sklearn

## References
Miao Y, Sinko W, Pierce L, Bucher D, Walker RC, McCammon JA (2014) Improved reweighting of accelerated molecular dynamics simulations for free energy calculation. J Chemical Theory and Computation, 10(7): 2677-2689.
