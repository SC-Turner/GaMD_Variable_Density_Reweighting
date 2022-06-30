python gamd_testsim.py #This will run a short openmm simulation, using parameters defined in input/test.xml, note that any existing output folders will be deleted and recreated

python phi_psi_calc.py -topol input/diala.ions.prmtop -traj output/output.dcd

python VDR_Call.py --gamd gamd_example.txt --data data_example.txt --cores 6 --emax 8 --mode convergence --conv_points_range 200 10000 --conv_points_num 20 --conv_points_scale log --pbc True --output output_E1_3 --step_multi True


