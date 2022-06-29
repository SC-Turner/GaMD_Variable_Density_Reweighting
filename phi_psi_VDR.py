from MDAnalysis.analysis.dihedrals import Dihedral
import MDAnalysis as mda
import argparse
import numpy as np
import subprocess
import sys
from VDR.VDR_Indep import Variable_Density_Reweighting as VDR

parser = argparse.ArgumentParser()
parser.add_argument("-traj", help="trajectory", nargs='+', default='output.dcd')
parser.add_argument("-topol", help="topology", default='../diala.ions.pdb')
parser.add_argument("-gamd", help="gamd log file", nargs='+', default='gamd.log')
parser.add_argument("-i", default='', help='Optional suffix for file labelling, eg. _E1 or _E2')
parser.add_argument("-output", type=str, default="output", help='Output Directory')
args = parser.parse_args()

def main():
    u = mda.Universe(args.topol, args.traj, in_memory=True)

    ags1 = [res.phi_selection() for res in u.residues]
    ags2 = [res.psi_selection() for res in u.residues]
    ags = list((ags1[1], ags2[1]))
    output = Dihedral(ags).run()
    angles = output.angles
    data = np.column_stack((angles, range(0, len(u.trajectory))))
    #np.savetxt('data_example.txt', data)

    data = np.loadtxt('input/data_example.txt')
    a = VDR(args.gamd, data, cores=6, Emax=8, output_dir='output', pbc=True, maxiter=200, multistep=True)
    conv_points = np.logspace(np.log10(20), np.log10(10000), num=20)
    conv_points = conv_points[:-1]
    for count, i in enumerate(conv_points):
        print('Limit:', str(int(i)))
        a.identify_segments(cutoff=i)
        a.reweight_segments(cutoff=i)
        if count == 0:
            a.calc_limdata(cutoff=i)
        a.interpolate_pmf(cutoff=i)
        a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title=f'')
    a.calc_conv(conv_points=conv_points)

if __name__ == '__main__':
  main()
