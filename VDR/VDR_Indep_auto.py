import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, Manager, Queue
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin
import scipy
from functools import partial
from VDR.VDR_methods import *
from itertools import chain, repeat
import os

class Variable_Density_Reweighting:
    def __init__(self, gamd, data, cores, Emax=8, maxiter=6, pbc=False, temp=300, output_dir='output'):
        print('Initialising')
        self.output_dir = str(output_dir)+'/'
        if not os.path.exists(self.output_dir):
            os.mkdir(os.path.join(self.output_dir))

        # self.universe = data[(data[:, 0] <= max(data[:, 0])) & (data[:, 1] <= max(data[:, 1])) & (data[:, 0] > min(data[:, 0]))  & (data[:, 1] > min(data[:, 1]))]
        #Load CV Data
        #self.universe = np.loadtxt(data, usecols=[0,1,2])
        self.data = data
        print('Data Loading Complete')
        self.pbc=pbc

        #Load GaMD Weights
        self.gamd= gamd
        #self.weights = np.exp(self.gamd[:,0])
        self.dV = gamd
        print('GaMD Data Loading Complete')

        self.vertices = [(min(self.data[:, 0]), max(self.data[:, 1])), (max(self.data[:, 0]), max(self.data[:, 1])),
                         (min(self.data[:, 0]), min(self.data[:, 1])), (max(self.data[:, 0]), min(self.data[:, 1]))]
        self.cores = cores
        self.Emax = Emax
        self.T = temp

        self.dimensions = range(0, 3)
        self.iterations = maxiter  # no. of sequential segmentation iterations of the dataset to be performed (Stops once no changes occur)
        self.pmf_min_array = []
        self.pmf_max_array = []
        self.PMF = []
        self.pmf_array_convergence = []
        self.conv_check_array = []
        self.limdatapoints = None
        # self.convergence_points = 10, 100
        self.convergence_points = np.logspace(1, 7, num=13)
        self.convergence_points = np.logspace(3, 7, num=9)

        self.dv_avg_distribution_max = []
        self.dv_avg_distribution_min = []
        self.dv_std_distribution_max = []
        self.dv_std_distribution_min = []
        self.anharm_total_max = []
        self.anharm_total_min = []

    def identify_segments(self, cutoff=100000):
        self.cutoff = cutoff
        old_universe = []
        
        a = pd.DataFrame({'rc1': self.data[:, 0],
                            'rc2': self.data[:, 1],
                            'frame': self.data[:, 2]})  # match C
        b = pd.DataFrame({'frame': self.dV[:, 1],
                            'dV': self.dV[:, 0]})

        df = a.merge(b, how='inner', on = ['frame'])
        # df = pd.concat([a, b['dV']], axis=1)
        # df = pd.concat([a, b['dV']], axis=1) #possibly need to minimize the amount of information passed here

        anharm_total = anharm(np.array(df['dV']))
        with open(str(self.output_dir)+'anharm_total.txt', 'w') as f:
            f.write(str(anharm_total))
        self.conv_check = []
        self.universe = df.to_numpy()

        pos = [str(1)]
        for count, segment_iterations in enumerate(range(self.iterations)):
            print('iteration:', segment_iterations + 1)
            if segment_iterations == 0:
                PBCx_values = -360, 0, 360
                PBCy_valyes = -360, 0, 360
                if self.pbc:
                    xmax = 180
                    xmin = -180
                    ymax = 180
                    ymin = -180
                    for x in PBCx_values:
                        for y in PBCy_valyes:
                            temp_array = pd.concat([a, b['dV']], axis=1).to_numpy()
                            temp_array[:, 0] += x
                            temp_array[:, 1] += y
                            index = 0
                            del_index = []
                            for coord in temp_array:
                                if xmax+0.1*(xmax-xmin) < coord[0] or coord[0] < xmin-0.1*(xmax-xmin) or ymax+0.1*(ymax-ymin) < coord[1] or coord[1] < ymin-0.1*(ymax-ymin):
                                    del_index.append(index)
                                index+=1
                            temp_array = np.delete(temp_array, del_index, axis=0)
                            self.universe = np.concatenate((self.universe, temp_array), axis=0)
                x = [self.universe]
            else:
                x = [self.universe][0]
            # parallel
            pool = Pool(self.cores)
            output = pool.starmap(segment_data, zip(x, pos, repeat(cutoff)))
            output = list(chain.from_iterable(output))
            types = []
            output2 = []
            pos = []
            for x in output:
                types.append(type(x))
                if type(x) == tuple:
                    output2.append(x)
                if type(x) != tuple:
                    pos.append(x)

            pos = list(chain.from_iterable(pos))

            self.universe = list(chain.from_iterable(output2))
            pool.close()
            pool.join()


            universe = []
            for arr in self.universe:
                if isinstance(arr, str):
                    continue
                if isinstance(arr, float):
                    continue
                else:
                    universe.append(arr)
                    #plt.scatter(arr[..., 0], arr[..., 1])
            #plt.show()
            self.conv_check.append([universe, pos])
            self.universe = universe

            if np.array(old_universe).shape == np.array(self.universe).shape:
                #self.iterations = max(segment_iterations, 6)
                self.iterations = segment_iterations
                break
            else:
                old_universe = self.universe
        self.conv_check_array.append([self.conv_check, list(repeat(cutoff, len(self.conv_check)))])

    def reweight_segments(self, cutoff):
        beta = 1.0 / (0.001987 * self.T)
        plt.clf()
        pmf_array = []
        self.pmf_max_distribution = []
        self.pmf_min_distribution = []
        for i in self.universe:
            if isinstance(i, str):
                None
            else:
                # idx = np.round(np.linspace(0, len(i[:,0]) - 1, int(cutoff/5))).astype(int)
                # i = i[idx, :]
                atemp = np.asarray(i[:, 3])
                dV_avg = np.average(atemp)
                dV_std = np.std(atemp)
                c1 = beta * dV_avg
                c2 = 0.5 * beta ** 2 * dV_std ** 2
                c1 = -np.multiply(1.0 / beta, c1)
                c2 = -np.multiply(1.0 / beta, c2)
                c12 = np.add(c1, c2)

                pmf_val = -(0.001987 * self.T) * np.log(len(i) / len(self.data))  # -kbt*lnp
                pmf_val = pmf_val + c12  # reweighting applied
                # pmf_val = c12
                # pmf_val = pmf_val #reweighting not applied
                # pmf_array.append([pmf_val, min(i[:, 0])+(max(i[:, 0])-min(i[:, 0]))/2, min(i[:, 1])+(max(i[:, 1])-min(i[:, 1]))/2, dV_avg, dV_std])
                pmf_array.append(
                    [pmf_val, np.mean(i[:, 0]), np.mean(i[:, 1]), dV_avg, dV_std, i, atemp])

        pmf_array = np.array(pmf_array)
        self.pmf_min_array = np.append(self.pmf_min_array, np.min(pmf_array[:, 0]))
        self.pmf_max_array = np.append(self.pmf_max_array, np.max(pmf_array[:, 0][np.nonzero(pmf_array[:, 0])]))
        index = np.where(
            pmf_array[:, 0] == np.max(pmf_array[:, 0][np.nonzero(pmf_array[:, 0])]))  # index of max/min PMF universe
        self.pmf_max_distribution = np.append(self.pmf_max_distribution, pmf_array[index, 6])
        self.anharm_total_max = np.append(self.anharm_total_max, anharm(self.pmf_max_distribution[0]))
        self.dv_avg_distribution_max = np.append(self.dv_avg_distribution_max, pmf_array[index, 3])
        self.dv_std_distribution_max = np.append(self.dv_std_distribution_max, pmf_array[index, 4])

        index = np.where(
            pmf_array[:, 0] == np.min(pmf_array[:, 0][np.nonzero(pmf_array[:, 0])]))  # index of max/min PMF universe
        self.pmf_min_distribution = np.append(self.pmf_min_distribution, pmf_array[index, 6])

        self.anharm_total_min = np.append(self.anharm_total_min, anharm(self.pmf_max_distribution[0]))
        self.dv_avg_distribution_min = np.append(self.dv_avg_distribution_min, pmf_array[index, 3])
        self.dv_std_distribution_min = np.append(self.dv_std_distribution_min, pmf_array[index, 4])

        pmf_array[:, 0] -= np.min(pmf_array[:, 0])  # normalises the array to 0 minimum
        self.pmf_array = pmf_array
        self.pmf_array_convergence.append(pmf_array[:, 0:3])

        print('Average dV:', np.mean(pmf_array[:, 3]))
        print('Average Std:', np.mean(pmf_array[:, 4]))

    def interpolate_pmf(self, cutoff):
        print('iterval:', self.iterations)
        a = pd.DataFrame({'rc1': self.data[:, 0],
                          'rc2': self.data[:, 1],
                          'frame': self.data[:, 2]})  # match C
        b = pd.DataFrame({'frame': self.dV[:, 0],
                          'dV': self.dV[:, 1]})
        df = pd.concat([a, b['dV']], axis=1)
        df = df.to_numpy()

        self.x_dense_pmf, self.y_dense_pmf = np.meshgrid(
            np.linspace(min(df[:, 0]), max(df[:, 0]),
                        2 ** self.iterations),
            np.linspace(min(df[:, 1]), max(df[:, 1]),
                        2 ** self.iterations))

        xdim, ydim = (min(df[:, 0]), max(df[:, 0])), \
                     (min(df[:, 1]), max(df[:, 1]))
        bins = 2 ** self.iterations
        ydif = ((ydim[1] - ydim[0]) / bins) / 2
        xdif = ((xdim[1] - xdim[0]) / bins) / 2
        hist_unre, xedge, yedge = np.histogram2d(df[:, 0], df[:, 1], bins=bins, density=True, range=(xdim, ydim))
        hist_unre = 0.001987 * self.T * np.log(hist_unre)
        plt.imshow(hist_unre.T, interpolation='nearest', origin='lower',
                   extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]])
        plt.gca().set_aspect(1 / plt.gca().get_data_ratio())
        plt.savefig(str(self.output_dir)+'Histogram_PMF.png')
        plt.clf()
        dense_points = np.stack([self.x_dense_pmf.ravel() + xdif, self.y_dense_pmf.ravel() + ydif], -1)
        hist_unre = hist_unre.ravel()
        hist_unre = hist_unre - np.nanmax(hist_unre)
        hist_unre = np.absolute(hist_unre)
        hist_unre[hist_unre == np.inf] = self.Emax

        interpolation_unre = RBFInterpolator(dense_points, hist_unre, smoothing=0, kernel='linear', degree=0,
                                             neighbors=26)
        z_dense = interpolation_unre(dense_points).reshape(self.x_dense_pmf.shape)

        self.PMF = z_dense

        print('Starting Interpolation:')
        z_scattered = self.pmf_array[:, 0]

        if len(self.universe) == 1:
            return
        self.universe = np.delete(self.universe, [x == 'nan' for x in self.universe])

        # Add 8kcal/mol barrier around unsampled points
        minx = np.min(np.vstack(self.universe)[:, 0])
        maxx = np.max(np.vstack(self.universe)[:, 0])
        miny = np.min(np.vstack(self.universe)[:, 1])
        maxy = np.max(np.vstack(self.universe)[:, 1])
        x_denselim, y_denselim = np.meshgrid(
            np.linspace(minx - ((maxx - minx) * 0.4), maxx + ((maxx - minx) * 0.4), min(2 ** 8, 2 ** self.iterations)),
            np.linspace(miny - ((maxy - miny) * 0.4), maxy + ((maxy - miny) * 0.4),
                        min(2 ** 8, 2 ** self.iterations)))  # 40% boundaries


        datapoints = []
        for x, y in zip(self.pmf_array[:, 1], self.pmf_array[:, 2]):
            datapoints.append([x, y])
        datapoints = np.array(datapoints)
        datapoints_outergrid = []
        for x, y in zip(x_denselim, y_denselim):
            datapoints_outergrid.append([x, y])
        datapoints_outergrid = np.array(datapoints_outergrid)
        scaler = MinMaxScaler(feature_range=(0, 1))
        datapoints_norm = scaler.fit_transform(datapoints)

        if self.pbc:
            idx1 = np.where((datapoints[:, 0] <= 180) & (datapoints[:, 0] >= -180))
            datapoints_iter = datapoints[idx1]
            idx2 = np.where((datapoints_iter[:, 1] <= 180) & (datapoints_iter[:, 1] >= -180))
            datapoints_iter = datapoints_iter[idx2]
            datapoints_norm_iter = scaler.transform(datapoints_iter)
        else:
            datapoints_norm_iter = datapoints_norm

        pool = Pool(self.cores)
        func = partial(datamax_calc, datapoints_norm)
        output = pool.map(func, datapoints_norm_iter)
        pool.close()
        pool.join()
        datapointsmax = np.max(output)

        if self.limdatapoints is not None:
            limdatapoints_2 = self.limdatapoints
        else:
            limdatapoints = []
            datapoints_outergrid_norm = scaler.transform(np.column_stack(datapoints_outergrid).T)
            for i in datapoints_outergrid_norm:
                if (scipy.spatial.distance.cdist([i], np.array(datapoints_norm)) < datapointsmax).any():
                    continue
                else:
                    limdatapoints.append(i)
            limdatapoints_2 = scaler.inverse_transform(limdatapoints)
        limzval = np.repeat(self.Emax, len(limdatapoints_2))
        scattered_points = np.append(datapoints, limdatapoints_2, axis=0)
        plt.scatter(datapoints[:, 0], datapoints[:, 1])
        plt.scatter(limdatapoints_2[:, 0], limdatapoints_2[:, 1])
        plt.savefig(str(self.output_dir)+f'indep_{cutoff}.png')
        plt.clf()

        z_scattered = np.append(z_scattered, limzval, axis=0)

        scattered_points_norm = scaler.transform(scattered_points)
        dense_points_norm_noplc = scaler.transform(dense_points)

        interpolation = RBFInterpolator(scattered_points_norm, z_scattered.ravel(), smoothing=0, kernel='linear',
                                        degree=0, neighbors=26)

        self.z_dense = interpolation(dense_points_norm_noplc).reshape(self.x_dense_pmf.shape)

    def calc_limdata(self, cutoff=1000):
        z_scattered = self.pmf_array[:, 0]

        self.universe = np.delete(self.universe, [x == 'nan' for x in self.universe])

        # Add 8kcal/mol barrier around unsampled points
        minx = np.min(np.vstack(self.universe)[:, 0])
        maxx = np.max(np.vstack(self.universe)[:, 0])
        miny = np.min(np.vstack(self.universe)[:, 1])
        maxy = np.max(np.vstack(self.universe)[:, 1])
        x_denselim, y_denselim = np.meshgrid(np.linspace(minx - ((maxx - minx) * 0.4), maxx + ((maxx - minx) * 0.4),
                                                         min(2 ** 8, 2 ** self.iterations)),
                                             np.linspace(miny - ((maxy - miny) * 0.4), maxy + ((maxy - miny) * 0.4),
                                                         min(2 ** 8,
                                                             2 ** self.iterations)))  # 40% boundaries applied

        datapoints = []
        for x, y in zip(self.pmf_array[:, 1], self.pmf_array[:, 2]):
            datapoints.append([x, y])
        datapoints = np.array(datapoints)
        datapoints_outergrid = []
        for x, y in zip(x_denselim, y_denselim):
            datapoints_outergrid.append([x, y])
        datapoints_outergrid = np.array(datapoints_outergrid)

        scaler = MinMaxScaler(feature_range=(0, 1))
        datapoints_norm = scaler.fit_transform(datapoints)

        if self.pbc:
            idx1 = np.where((datapoints[:, 0] <= 180) & (datapoints[:, 0] >= -180))
            datapoints_iter = datapoints[idx1]
            idx2 = np.where((datapoints_iter[:, 1] <= 180) & (datapoints_iter[:, 1] >= -180))
            datapoints_iter = datapoints_iter[idx2]
            datapoints_norm_iter = scaler.transform(datapoints_iter)
        else:
            datapoints_norm_iter = datapoints_norm

        pool = Pool(self.cores)
        func = partial(datamax_calc,
                       datapoints_norm)  # ERROR, the iterable set needs to be non-pbc, whereas the reference non-iter set needs to be pbc
        output = pool.map(func, datapoints_norm_iter)
        pool.close()
        pool.join()
        datapointsmax = np.max(output)

        limdatapoints = []
        datapoints_outergrid_norm = scaler.transform(np.column_stack(datapoints_outergrid).T)
        for i in datapoints_outergrid_norm:
            if (scipy.spatial.distance.cdist([i], np.array(datapoints_norm)) < datapointsmax).any():
                continue
            else:
                limdatapoints.append(i)

        limdatapoints_2 = scaler.inverse_transform(limdatapoints)
        self.limdatapoints = limdatapoints_2

    def plot_PMF(self, xlab, ylab, cutoff, title):
        self.PMF = np.nan_to_num(self.PMF, nan=int(self.Emax)).T

        plt.contourf(self.x_dense_pmf, self.y_dense_pmf, self.z_dense, cmap='jet', levels=50)
        plt.colorbar()
        plt.savefig(str(self.output_dir)+f'bias_{cutoff}.png')
        plt.clf()

        plt.contourf(self.x_dense_pmf, self.y_dense_pmf, self.PMF, cmap='jet', levels=50)
        plt.colorbar()
        plt.savefig(str(self.output_dir)+f'PMF_{cutoff}.png')
        
        z_dense = np.add(self.z_dense, self.PMF)
        mymin = np.nanmin([np.nanmin(r) for r in z_dense])
        z_dense = z_dense - mymin
        z_dense = np.nan_to_num(np.array(z_dense), nan=self.Emax)
        z_dense[z_dense > self.Emax] = self.Emax

        # 2D Projection
        # fig = plt.figure(figsize=(15, 15), dpi=400)
        plt.clf()
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel(xlab, fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
        contourf_ = ax.contourf(self.x_dense_pmf, self.y_dense_pmf, z_dense, cmap='jet', levels=50)
        # contourf_ = ax.imshow((x_dense, y_dense), cmap='jet')
        # plt.xlim(-180, 180)
        # plt.ylim(-180, 180)
        cbar = fig.colorbar(contourf_)
        cbar.set_label('Kcal/mol\n', fontsize=16)
        plt.savefig(str(self.output_dir)+f'2C_PMF_{cutoff}_Limzmod_{title}.png')
        plt.clf()

        del self.universe

        # fig = plt.figure()
        # ax = plt.axes()
        # ax.set_xlabel(xlab, fontsize=16)
        # ax.set_ylabel(ylab, fontsize=16)
        # contourf_ = ax.contourf(self.x_dense_pmf, self.y_dense_pmf, self.z_dense, cmap='jet', levels=(0, 1, 2, 3, 4, 5, 6, 7, 8))
        # # contourf_ = ax.imshow((x_dense, y_dense), cmap='jet')
        # # plt.xlim(-180, 180)
        # # plt.ylim(-180, 180)
        # cbar = fig.colorbar(contourf_)
        # cbar.set_label('Kcal/mol\n', fontsize=16)
        # plt.savefig(str(self.output_dir)+f'2C_PMF_{cutoff}_Limzmod_simple_{title}.png')
        # plt.clf()

    def calc_conv(self, conv_points):
        print(self.dv_avg_distribution_min)

        plt.plot(conv_points, self.dv_avg_distribution_min)
        plt.savefig(str(self.output_dir)+'mean_plot_min.png')
        plt.clf()
        plt.plot(conv_points, self.dv_std_distribution_min)
        plt.savefig(str(self.output_dir)+'std_plot_min.png')
        plt.clf()
        plt.plot(conv_points, self.anharm_total_min)
        plt.savefig(str(self.output_dir)+'anharm_plot_min.png')
        plt.clf()

        plt.plot(conv_points, self.dv_avg_distribution_max)
        plt.savefig(str(self.output_dir)+'mean_plot_max.png')
        plt.clf()
        plt.plot(conv_points, self.dv_std_distribution_max)
        plt.savefig(str(self.output_dir)+'std_plot_max.png')
        plt.clf()
        plt.plot(conv_points, self.anharm_total_max)
        plt.savefig(str(self.output_dir)+'anharm_plot_max.png')
        plt.clf()




