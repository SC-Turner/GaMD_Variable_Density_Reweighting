from VDR.VDR_Indep import Variable_Density_Reweighting as VDR
import argparse
import sys

#Example Input Script (Windows)
#python ..\..\VDR_Clean\VDR_Call.py --gamd Amber19SB_diala_E2_dual_600ns\weights_E2_concat.dat --data Amber19SB_diala_E2_dual_600ns\data_E2_concat.dat --cores 6 --emax 8 --itermax 6 --conv_points 10 100 1000 10000 100000 1000000 --mode convergence --output E2_dual_600ns
#

def parse_args():
    print('ARGS test')
    parser = argparse.ArgumentParser(description="Variable Density Reweighting of Gaussian Accelerated Molecular Dynamics Simulations")
    parser.add_argument("--gamd", help="gamd weights .dat file location, generated from GaMD simulation", required=True)
    parser.add_argument("--data", help="Datafile location containing CV values and timestep, formatted as in input/data_example.txt", required=True)
    parser.add_argument("--cores", type=int, help='Number of CPU cores to use for VDR', required=True)
    parser.add_argument("--emax", type=float, help='Kcal/mol value assigned for unsampled regions of CV-space', required=False, default=8)
    parser.add_argument("--itermax", type=int, help='Generally ignore, cutoff for how many segmentation iterations for VDR', required=False, default=9999)
    parser.add_argument("--conv_points", nargs='+', type=int, help='Cut-off values for VDR segmentation, use multiple values for convergence mode, one value for single mode', required=True)
    parser.add_argument("--output", help='Output directory')
    parser.add_argument("--mode", required=True, help='Whether to evaluate a single cut-off value (mode=single) or evaluate convergence across multiple cut-off values (mode=convergence)',  choices=['single', 'convergence'])
    parser.add_argument("--pbc", default=False, help='Whether to add partial duplicated boundaries if the CV-limits loop around, i.e. phi/psi angles', choices=['True', 'False'])
    parser.add_argument("--step_multi", default=True,
                        help='Whether to multiply timestep column in datafile by timestep identified in gamd weight input file',
                        choices=['True', 'False'])
    args, leftovers = parser.parse_known_args()

    print(args.mode)

    if args.mode == 'single':
        if args.conv_points is None:
            args.conv_points = 1000
        if len(args.conv_points) != 1:
            raise ValueError("--mode single, only supports one conv_points value")
    elif args.mode == 'convergence':
        if args.conv_points is None:
            args.conv_points = 10, 100, 1000, 10000, 100000
        if len(args.conv_points) == 1:
            raise ValueError("--mode convergence, only supports multiple conv_points value, use --mode single for a single cut-off")
    if args.output is None:
        args.output = 'output_VDR'
        
    return args

def main():
    a = VDR(gamd=args.gamd, data=args.data, step_multi=args.step_multi, cores=args.cores, Emax=args.emax, output_dir=args.output, pbc=args.pbc, step_multi=args.step_multi, maxiter=args.itermax)

    if args.mode == 'single':
        i = args.conv_points
        a.identify_segments(cutoff=i)
        a.reweight_segments(cutoff=i)
        a.interpolate_pmf(cutoff=i)
        a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title='')

    if args.mode == 'convergence':
        for count, i in enumerate(args.conv_points):
            print('Limit:', i)
            a.identify_segments(cutoff=i)
            a.reweight_segments(cutoff=i)
            if count == 0:
                a.calc_limdata(cutoff=i)
            a.interpolate_pmf(cutoff=i)
            a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title=f'PMF_cutoff_{i}')
        a.calc_conv(conv_points=args.conv_points)

if __name__ == '__main__':
  args = parse_args()
  main()

