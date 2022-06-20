#Import GaMD_openmm
import sys
sys.path.insert(0,'C:/Users/sct1g15/Documents/Adaptive_GaMD_Dev/gamd-openmm2')
sys.path.insert(0,'C:/Users/sct1g15/Documents/Adaptive_GaMD_Dev/PCR_cov_khaled')

from gamd import gamdSimulation, parser
from gamd.runners import Runner
import openmm.unit as unit
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from tqdm import tqdm
import shutil
import os

#setup config
parserFactory = parser.ParserFactory() #creates a config class
config_filename = 'test.xml'#xml file
config_file_type = 'xml'
platform = 'CUDA'
device_index = '0'
debug=False
config = parserFactory.parse_file(config_filename, config_file_type) #returns a config file, XmlParser.config

#xml file example
#each e.g. <integrator> tag is parsed manually by XmlParser, which is called by ParserFactory.parse_file()
#modifies class variable in gamd/config
#paramater modifications must correspond to a self.config.x object

def main():
    if os.path.isdir("output2"):
        shutil.rmtree("output2")
    config.outputs.directory = 'output2'

    # setup gamd simulation
    # Sets openmm.app variables depending on config.system settings
    gamdSimulationFactory = gamdSimulation.GamdSimulationFactory()
    gamdSim = gamdSimulationFactory.createGamdSimulation(
        config, platform, device_index)

    runner = Runner(config, gamdSim, debug)
    restart = False
    runner.run(restart)

    exit()

    traj = "output/output.dcd"
    top = "diala.ions.pdb"
    ref = "diala.ions.pdb"
    gamd_log = np.loadtxt("output/gamd.log", skiprows=603)

    def load_traj(top, traj, ref):
        univ = mda.Universe(top, traj)  # , topology_format='PDB')
        univ_ref = mda.Universe(ref, topology_format='PDB')  # , topology_format='PDB')

        protein = univ.select_atoms('protein')
        protein_ref = univ_ref.select_atoms('protein')

        with mda.Writer("temp.xtc", protein.n_atoms) as W:
            for ts in univ.trajectory:
                if ts.frame in range(600, 1800):
                    print(ts)
                    W.write(protein)
        with mda.Writer("temp.pdb", protein.n_atoms) as W:
            W.write(protein)

        del univ
        univ_2 = mda.Universe("temp.pdb", "temp.xtc")
        univ_2_ref = mda.Universe("temp.pdb")

        alignment = align.AlignTraj(univ_2,
                                    univ_2_ref,
                                    select='protein and name CA',
                                    # select='protein',
                                    # in_memory=True,
                                    verbose=True,
                                    )
        alignment.run()

        print("Loaded universe with " + str(len(univ_2.atoms)) +
              " atoms and " + str(len(univ_2.trajectory)) + " aligned frames \n")

        return univ_2


    def movingaverage(arr, ncut):
        mov = []
        mov_sd = []

        arr = np.array(arr)

        for i in np.arange(ncut - 1, len(arr), ncut):
            mov.append(np.average(arr[i - (ncut - 1):i]))
            mov_sd.append(np.std(arr[i - (ncut - 1):i]))

        sma = np.zeros((len(mov), 2), dtype=np.float64)

        sma[:, 0] = [mov[i] for i in np.arange(0, len(mov))]
        sma[:, 1] = [mov_sd[i] for i in np.arange(0, len(mov_sd))]

        return sma


    def gen_movingaverage(sigarr, usecols, ncut=50, gen_framedata=False):
        # Calculate moving average of selected input columns

        if usecols is tuple and len(usecols) > 1:
            sma_dict = {}
            for idx in usecols:
                sma_dict[idx] = movingaverage(sigarr[:, idx], ncut=ncut)

            # Generate the output moving average data array
            sigarr_ma = np.hstack(list(sma_dict.values()))
        else:
            sigarr_ma = movingaverage(sigarr[:, usecols], ncut=ncut)

        if gen_framedata:
            framedata = [sigarr[i, 0] for i in range(ncut - 1, len(sigarr[:, 0]), ncut)]
            return sigarr_ma, framedata
        else:
            return sigarr_ma


    basename = "/hdd2/GB1_1.0_PBMetaD_1/md_folding_prot"

    inpuniv = load_traj(top, traj, ref)

    inpuniv.transfer_to_memory()

    # Load coordinate matrix of backbone Ca atoms for trajectory

    use_ma = True
    use_piecewise = False

    dt = 100  # Calculate PCA every ns
    nbounds, nbins = 10, 5

    tot_time = len(inpuniv.trajectory)
    ncomp = 2

    exvar_arr = np.zeros((int(tot_time / dt), ncomp + 1), dtype=np.float64)
    div1D_arr, div2D_arr = np.zeros((int(tot_time / dt), 3), dtype=np.float64), np.zeros((int(tot_time / dt), 3),
                                                                                         dtype=np.float64)

    div1D, div2D = np.zeros((int(tot_time / dt), nbounds - 1), dtype=np.float64), np.zeros(
        (int(tot_time / dt), nbounds - 1), dtype=np.float64)
    totdiv1D, totdiv2D = np.zeros(int(tot_time / dt), dtype=np.float64), np.zeros(int(tot_time / dt), dtype=np.float64)

    densities_2D, densities_1D = [], []

    i = 0

    for idx in tqdm(range(dt, tot_time, dt)):

        coords = inpuniv.trajectory.timeseries(inpuniv.select_atoms("name CA"),
                                               order="fac")[:idx, :, :]

        coordmatrix = np.reshape(coords, (coords.shape[0], coords.shape[1] * coords.shape[2]))

        # Running SVD/PCA analysis for linear dimensionality reduction of coordinate matrix

        pca_test = PCA(n_components=ncomp)
        pca_test.fit(coordmatrix)

        # Recording explained variance (%) by each principal component

        exvar_arr[i, 0] = idx
        exvar_arr[i, 1:] = 100.0 * pca_test.explained_variance_ratio_

        # print(f"Simulation Time {idx/dt} ns: \
        #      Explained Variance Ratios - {exvar_arr[i,:]}")

        model_matrix = pca_test.transform(coordmatrix)

        # Generate histogram density estimate for PC1 vs PC2 data
        print(model_matrix)

        model_2Ddens, bin_xedges, bin_yedges = np.histogram2d(x=model_matrix[:, 0], y=model_matrix[:, 1],
                                                              bins=[nbins, nbins], density=True)

        max_index = np.unravel_index(model_2Ddens.argmax(), model_2Ddens.shape)
        max_data_index = np.where(
            (model_matrix[:, 0] > bin_xedges[max_index[0]]) & (model_matrix[:, 1] > bin_yedges[max_index[1]]) &
            (model_matrix[:, 0] < bin_xedges[max_index[0] + 1]) & (model_matrix[:, 1] < bin_yedges[max_index[1] + 1]))


        # plt.figure()
        # myextent  =[bin_xedges[0],bin_xedges[-1],bin_yedges[0],bin_yedges[-1]]
        # plt.imshow(model_2Ddens.T, origin='lower', extent=myextent, interpolation='nearest', aspect='auto')
        # plt.colorbar()
        # plt.show()

        model_1Ddens, bin_edges = np.histogram(model_matrix[:, 0], bins=nbins, density=True)

        densities_2D.append(model_2Ddens)
        densities_1D.append(model_1Ddens)

    fig = plt.figure(figsize=(24, 12))
    ax_pca = fig.add_subplot(121)
    ax_exvar = fig.add_subplot(122)

    # Changing ax_exvar to plot GaMD convergence
    # ax_exvar.plot(exvar_arr[:, 0], exvar_arr[:, 1], lw=2, color="red", label="PC1")
    # ax_exvar.plot(exvar_arr[:, 0], exvar_arr[:, 2], lw=2, color="blue", label="PC2")
    #

    def anharm(data):
        var = np.var(data)
        hist, edges = np.histogram(data, 50, normed=True)
        hist = np.add(hist, 0.000000000000000001)  ###so that distrib
        dx = edges[1] - edges[0]
        S1 = -1 * np.trapz(np.multiply(hist, np.log(hist)), dx=dx)
        S2 = 0.5 * np.log(2.00 * np.pi * np.exp(1.0) * var + 0.000000000000000001)
        alpha = S2 - S1
        if np.isinf(alpha):
            alpha = 100
        return alpha


    # for openmm
    time = gamd_log[:, 1]
    dihedral_energy = gamd_log[:, -3]
    total_energy = gamd_log[:, -4]

    # # for amber
    # time = gamd_log[:, 1]
    # dihedral_energy = gamd_log[:, -1]
    # total_energy = gamd_log[:, -2]

    mean_dihedral = []
    std_dihedral = []
    array_dihedral = []
    time_dihedral = []
    anharm_dihedral = []

    for x in max_data_index[0]:
        time_dihedral = np.append(time_dihedral, time[x])
        array_dihedral = np.append(array_dihedral, dihedral_energy[x])
        mean_dihedral = np.append(mean_dihedral, np.mean(array_dihedral))
        std_dihedral = np.append(std_dihedral, np.std(array_dihedral))
        anharm_dihedral = np.append(anharm_dihedral, anharm(array_dihedral))

    anharm_total = []
    time_total = []
    mean_total = []
    array_total = []
    std_total = []
    for x in max_data_index[0]:
        time_total = np.append(time_total, time[x])
        array_total = np.append(array_total, total_energy[x])
        mean_total = np.append(mean_total, np.mean(array_total))
        std_total = np.append(std_total, np.std(array_total))
        anharm_total = np.append(anharm_total, anharm(array_total))

    # for x in range(0, len(time)):
    #     time_total = np.append(time_total, time[x])
    #     array_total = np.append(array_total, total_energy[x])
    #     mean_total = np.append(mean_total, np.mean(array_total))
    #     std_total = np.append(std_total, np.std(array_total))
    #     anharm_total = np.append(anharm_total, anharm(array_total))
    #
    # for x in range(0, len(time)):
    #     time_dihedral = np.append(time_dihedral, time[x])
    #     array_dihedral = np.append(array_dihedral, dihedral_energy[x])
    #     mean_dihedral = np.append(mean_dihedral, np.mean(array_dihedral))
    #     std_dihedral = np.append(std_dihedral, np.std(array_dihedral))
    #     anharm_dihedral = np.append(anharm_dihedral, anharm(array_dihedral))

    print(mean_total)
    print(time_total)
    print(anharm_dihedral)
    print(anharm_total)

    ax_exvar.plot(range(0, len(time_total)), mean_total, lw=2, color="blue", label="GaMD Mean Boost total")
    ax_exvar.plot(range(0, len(time_total)), anharm_total, lw=2, color="purple", label="GaMD Anharm Boost total")
    ax_exvar.plot(range(0, len(time_total)), std_total, lw=2, color="darkblue", label="GaMD StdDev Boost total")
    # # ax_exvar.scatter(range(0, len(time_total)), mean_total, lw=2, color="blue")
    ax_exvar.plot(range(0, len(time_dihedral)), mean_dihedral, lw=2, color="red", label="GaMD Mean Boost dihedral")
    ax_exvar.plot(range(0, len(time_dihedral)), anharm_dihedral, lw=2, color="black", label="GaMD Anharm Boost dihedral")
    ax_exvar.plot(range(0, len(time_dihedral)), std_dihedral, lw=2, color="darkred", label="GaMD StdDev Boost dihedral")
    ax_exvar.set_ylim(0.0)
    # ax_exvar.set_xscale('log')
    # # ax_exvar.plot(time_total, mean_total, lw=2, color="blue", label="GaMD Mean Boost total")
    # ax_exvar.plot(time_dihedral, anharm_dihedral, lw=2, color="black", label="GaMD Anharm Boost dihedral")
    # # ax_exvar.plot(time_total, anharm_total, lw=2, color="purple", label="GaMD Anharm Boost total")
    # # ax_exvar.scatter(range(0, len(time_total)), mean_total, lw=2, color="blue")
    # ax_exvar.plot(time_dihedral, mean_dihedral, lw=2, color="red", label="GaMD Mean Boost dihedral")
    # # ax_exvar.plot(time_total, std_total, lw=2, color="darkblue", label="GaMD StdDev Boost total")
    # ax_exvar.plot(time_dihedral, std_dihedral, lw=2, color="darkred", label="GaMD StdDev Boost dihedral")

    # ax_exvar.plot(time_total, mean_total, lw=2, color="blue", label="GaMD Mean Boost total")
    # ax_exvar.scatter(time_total, mean_total, lw=2, color="blue")
    # ax_exvar.plot(time_dihedral, mean_dihedral, lw=2, color="red", label="GaMD Mean Boost dihedral")
    # ax_exvar.plot(time_total, std_total, lw=2, color="darkblue", label="GaMD StdDev Boost total")
    # ax_exvar.plot(time_dihedral, std_dihedral, lw=2, color="darkred", label="GaMD StdDev Boost dihedral")

    ax_pca.set_xlim(0.0, int(tot_time / dt))
    ax_pca.set_ylim(0.0, 1.0)

    ax_pca.legend(loc="upper right", fontsize=20)
    ax_exvar.legend(loc="best", fontsize=20)

    fig.tight_layout()
    plt.savefig("output/svd_convergencetest.png", dpi=500, bbox_inches="tight")
    #plt.show()

if __name__ == '__main__':
    main()