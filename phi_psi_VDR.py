from MDAnalysis.analysis.dihedrals import Dihedral
import MDAnalysis as mda
import argparse
import numpy as np
import subprocess
import sys
sys.path.insert(1, 'C:/Users/sct1g15/Documents/Adaptive_GaMD_Dev/GaMD_openmm_dev')
from VDR_Indep_auto import Variable_Density_Reweighting as VDR

parser = argparse.ArgumentParser()
parser.add_argument("-traj", help="trajectory", nargs='+', default='output.dcd')
parser.add_argument("-topol", help="topology", default='../diala.ions.pdb')
parser.add_argument("-gamd", help="gamd log file", nargs='+', default='gamd.log')
parser.add_argument("-i", default='', help='Optional suffix for file labelling, eg. _E1 or _E2')
args = parser.parse_args()

def main():
    u = mda.Universe(args.topol, args.traj, in_memory=True)
    gamd = np.loadtxt(args.gamd, comments='#', usecols=(6, 7))
    gamd = gamd[:, 0] + gamd[:, 1]
    gamd = np.vstack((gamd, range(1, len(u.trajectory) + 1))).T
    # deletes gamd entries with 0 boost potential (equilibration steps), this is a bit of a hack, needs improving
    gamd = gamd[gamd[:, 0] != 0]

    ags1 = [res.phi_selection() for res in u.residues]
    ags2 = [res.psi_selection() for res in u.residues]
    ags = list((ags1[1], ags2[1]))
    output = Dihedral(ags).run()
    angles = output.angles
    data = np.column_stack((angles, range(0, len(u.trajectory))))

    a = VDR(gamd, data, cores=6, Emax=8, output_dir='output', pbc=False, maxiter=200)
    conv_points = np.logspace(np.log10(20), np.log10(10000), num=20)
    conv_points = conv_points[:-1]
    for count, i in enumerate(conv_points):
        print('Limit:', i)
        a.identify_segments(cutoff=i)
        a.reweight_segments(cutoff=i)
        if count == 0:
            a.calc_limdata(cutoff=i)
        a.interpolate_pmf(cutoff=i)
        a.plot_PMF(xlab='PC1', ylab='PC2', cutoff=i, title=f'PMF_cutoff_{i}')
    a.calc_conv(conv_points=conv_points)

if __name__ == '__main__':
  main()
