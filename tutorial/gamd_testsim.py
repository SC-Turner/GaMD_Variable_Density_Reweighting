import sys
sys.path.insert(0, 'C:/Users/sct1g15/Documents/gamd_openmm') #Insert path to gamd_openmm git repository
sys.path.insert(0, 'C:/Users/sct1g15/Documents/Adaptive_GaMD_Dev/VDR/GaMD_Variable_Density_Reweighting/VDR') #Insert path to VDR git repository
from gamd import gamdSimulation, parser
from gamd.runners import Runner
import shutil
import os
from MDAnalysis.analysis.dihedrals import Dihedral
import MDAnalysis as mda
import argparse
import numpy as np
import subprocess
import sys
from VDR_Indep import Variable_Density_Reweighting as VDR

def run_sim():
    # setup config
    parserFactory = parser.ParserFactory()  # creates a config class
    config_filename = 'input/test.xml'  # xml file
    config_file_type = 'xml'
    platform = 'CUDA'
    device_index = '0'
    debug = False
    config = parserFactory.parse_file(config_filename, config_file_type)  # returns a config file, XmlParser.config

    if os.path.isdir("output"):
        shutil.rmtree("output")
    config.outputs.directory = 'output'

    # setup gamd simulation
    # Sets openmm.app variables depending on config.system settings
    gamdSimulationFactory = gamdSimulation.GamdSimulationFactory()
    gamdSim = gamdSimulationFactory.createGamdSimulation(
        config, platform, device_index)

    runner = Runner(config, gamdSim, debug)
    restart = False
    runner.run(restart)

if __name__ == '__main__':
    run_sim()
