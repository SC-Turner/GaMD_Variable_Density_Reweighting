from MDAnalysis.analysis.dihedrals import Dihedral
import MDAnalysis as mda
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-traj", help="trajectory", default='output.dcd')
parser.add_argument("-topol", help="topology", default='../diala.ions.pdb')
parser.add_argument("-o", help="output", default='data.dat')
args = parser.parse_args()

def main():
    u = mda.Universe(args.topol, args.traj, in_memory=False)

    ags1 = [res.phi_selection() for res in u.residues]
    ags2 = [res.psi_selection() for res in u.residues]
    ags = list((ags1[1], ags2[1]))
    output = Dihedral(ags).run()
    angles = output.angles
    np.savetxt(args.o, angles)

if __name__ == '__main__':
  main()
