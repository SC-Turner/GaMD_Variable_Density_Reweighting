from VDR.VDR_Indep import Variable_Density_Reweighting as VDR
import numpy as np
import argparse
import sys

#Example Input Script (Windows)
#python ..\..\VDR_Clean\VDR_Call.py --gamd Amber19SB_diala_E2_dual_600ns\weights_E2_concat.dat --data Amber19SB_diala_E2_dual_600ns\data_E2_concat.dat --cores 6 --emax 8 --itermax 6 --conv_points 10 100 1000 10000 100000 1000000 --mode convergence --output E2_dual_600ns
#

print('Running VDR:')

def parse_args():
    print('ARGS test')
    parser = argparse.ArgumentParser(description="Variable Density Reweighting of Gaussian Accelerated Molecular Dynamics Simulations")
    parser.add_argument("--gamd", help="gamd weights .dat file location, generated from GaMD simulation", required=True)
    parser.add_argument("--data", help="Datafile location containing CV values and timestep, formatted as in input/data_example.txt", required=True)
    parser.add_argument("--cores", type=int, help='Number of CPU cores to use for VDR', required=True)
    parser.add_argument("--emax", type=float, help='Kcal/mol value assigned for unsampled regions of CV-space', required=False, default=8)
    parser.add_argument("--itermax", type=int, help='Generally ignore, cutoff for how many segmentation iterations for VDR', required=False, default=9999)
    parser.add_argument("--conv_points", nargs='+', type=int, help='Cut-off values for VDR segmentation, use multiple values for convergence mode, one value for single mode', required=True)
    parser.add_argument("--conv_points_num", type=int, help='Number of cut-off data points to use between range specified in --conv_points, only required for --mode convergence')
    parser.add_argument("--conv_points_scale", type=str, default='linear', choices=['linear', 'log'], help='whether to use a linear or log scale to distribute points across conv_points range')
    parser.add_argument("--output", type=str, help='Output directory', default='output')
    parser.add_argument("--mode", type=str, required=True, help='Whether to evaluate a single cut-off value (mode=single) or evaluate convergence across multiple cut-off values (mode=convergence)',  choices=['single', 'convergence'])
    parser.add_argument("--pbc", default='False', help='Whether to add partial duplicated boundaries if the CV-limits loop around, i.e. phi/psi angles', choices=['True', 'False'])
    parser.add_argument("--step_multi", type=str, default='True',
                        help='Whether to multiply frames column in CV datafile by timestep identified in gamd weight input file, to match timesteps in gamd weight input file',
                        choices=['True', 'False'])
    args, leftovers = parser.parse_known_args()

    if args.mode == 'single':
        if len(args.conv_points) != 1:
            parser.error("--mode single, requires one value to --conv_points_range, defines single cutoff value")
        if args.conv_points_num is not None:
            parser.error("--mode single does not support --conv_points_num, use --mode convergence or remove")
        if args.conv_points_scale is not None:
            parser.error("--mode single does not support --conv_points_scale, use --mode convergence or remove")
    elif args.mode == 'convergence':
        if len(args.conv_points) != 2:
            parser.error("--mode convergence, requires two values to --conv_points, defines range of cutoff values to use in combination with --conv_points_num")
        if args.conv_points_num is None:
            parser.error("--conv_points_num, requires specification of how many cutoff values to use between defined range in --conv_points")
        
    return args

def main():
    a = VDR(gamd=args.gamd, data=args.data, step_multi=args.step_multi, cores=args.cores, Emax=args.emax, output_dir=args.output, pbc=args.pbc, maxiter=args.itermax)

    if args.mode == 'single':
        i = args.conv_points
        a.identify_segments(cutoff=i)
        a.reweight_segments(cutoff=i)
        a.interpolate_pmf(cutoff=i)
        a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title='')

    if args.mode == 'convergence':
        if args.conv_points_scale == 'linear':
            conv_points=range(args.conv_points[0], args.conv_points[1], args.conv_points_num)
        if args.conv_points_scale == 'log':
            conv_points = np.logspace(np.log10(args.conv_points[0]), np.log10(args.conv_points[1]), num=args.conv_points_num)
 
        for count, i in enumerate(conv_points):
            print('Limit:', str(int(i)))
            a.identify_segments(cutoff=i)
            a.reweight_segments(cutoff=i)
            if count == 0:
                a.calc_limdata(cutoff=i)
            a.interpolate_pmf(cutoff=i)
            a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title=f'PMF_cutoff_{i}')
        a.calc_conv(conv_points=conv_points)

if __name__ == '__main__':
  args = parse_args()
  main()

