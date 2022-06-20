from VDR_Indep import Variable_Density_Reweighting as VDR
import argparse
import sys

#Example Input Script (Windows)
#python ..\..\VDR_Clean\VDR_Call.py --gamd Amber19SB_diala_E2_dual_600ns\weights_E2_concat.dat --data Amber19SB_diala_E2_dual_600ns\data_E2_concat.dat --cores 6 --emax 8 --itermax 6 --conv_points 10 100 1000 10000 100000 1000000 --mode convergence --output E2_dual_600ns
#

def parse_args():
    print('ARGS test')
    parser = argparse.ArgumentParser(description="My Script")
    parser.add_argument("--gamd")
    parser.add_argument("--data")
    parser.add_argument("--cores", type=int)
    parser.add_argument("--emax", type=float)
    parser.add_argument("--itermax", type=int)
    parser.add_argument("--conv_points", nargs='+', type=int)
    parser.add_argument("--output")
    parser.add_argument("--mode")
    parser.add_argument("--pbc", type=bool, default=False)
    args, leftovers = parser.parse_known_args()

    print(args.mode)

    if args.gamd is None:
        args.gamd = 'Amber99SB_diala_E2_dihedral_90ns/weights_E1_concat.dat'
    if args.data is None:
        args.data = 'Amber99SB_diala_E2_dihedral_90ns/data_E1_concat.dat'
    if args.cores is None:
        args.cores = 10
    if args.emax is None:
        args.emax = 8
    if args.itermax is None:
        args.itermax = 6
    if args.mode == 'single':
        if args.conv_points is None:
            args.conv_points = 1000
    elif args.mode == 'convergence':
        if args.conv_points is None:
            args.conv_points = 10, 100, 1000, 10000, 100000
    else:
        print('please specify the calculation mode, single or convergence')
        sys.exit()
    if args.output is None:
        args.output = 'output' 
        
    return args

def main():
    print('MAIN')
    print(args)
    a = VDR(args.gamd, args.data, cores=args.cores, Emax=args.emax, output_dir=args.output, pbc=args.pbc)

    if args.mode == 'single':
        i = args.conv_points
        a.identify_segments(cutoff=i)
        a.reweight_segments(cutoff=i)
        a.interpolate_pmf(cutoff=i)
        a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title='')

    if args.mode == 'convergence':
        for count, i in enumerate(conv_points):
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

