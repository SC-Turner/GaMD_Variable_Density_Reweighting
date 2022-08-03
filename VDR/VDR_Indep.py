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


class VariableDensityReweighting:
    def __init__(self, gamd, data, cores, conv_points, step_multi, pbc='False', Emax=8, maxiter=9999, temp=300,
                 output_dir='output'):
        print('Initialising')
        self.output_dir = str(output_dir) + '/'
        if not os.path.exists(self.output_dir):
            os.mkdir(os.path.join(self.output_dir))

        if not os.path.exists(self.output_dir + '/PMF'):
            os.mkdir(os.path.join(self.output_dir + '/PMF'))

        if not os.path.exists(self.output_dir + '/convergence'):
            os.mkdir(os.path.join(self.output_dir + '/convergence'))

        if not os.path.exists(self.output_dir + '/intermediates'):
            os.mkdir(os.path.join(self.output_dir + '/intermediates'))
 
        if not os.path.exists(self.output_dir + '/clusters'):
            os.mkdir(os.path.join(self.output_dir + '/clusters'))


        self.pbc = pbc
        self.gamd, self.data = calc_inputs(step_multi=step_multi, gamd=gamd, data=data)
        self.dV = self.gamd[:, 0:2]

        self.vertices = [(min(self.data[:, 0]), max(self.data[:, 1])), (max(self.data[:, 0]), max(self.data[:, 1])),
                         (min(self.data[:, 0]), min(self.data[:, 1])), (max(self.data[:, 0]), min(self.data[:, 1]))]
        self.cores = cores
        self.Emax = Emax
        self.T = temp
        self.conv_points = conv_points  # Just for checking does not exceed no. frames
        self.iterations = maxiter  # No. of sequential segmentation iterations of the dataset to be performed (Stops
        # automatically once no changes occur)
        self.pmf_min_array = []
        self.pmf_max_distribution = []
        self.dv_avg_distribution = []
        self.dv_std_distribution = []
        self.pmf_max_array = []
        self.PMF = []
        self.pmf_array_convergence = []
        self.limdatapoints = None

        self.dv_avg_distribution_max = []
        self.dv_avg_distribution_min = []
        self.dv_std_distribution_max = []
        self.dv_std_distribution_min = []
        self.anharm_total_max = []
        self.anharm_total_min = []

    def identify_segments(self, cutoff):

        if type(cutoff) is list:
            self.cutoff = int(cutoff[0])
        else:
            self.cutoff = int(cutoff)
        old_universe = []

        a = pd.DataFrame({'rc1': self.data[:, 0],
                          'rc2': self.data[:, 1],
                          'frame': self.data[:, 2]})  # match C
        b = pd.DataFrame({'frame': self.dV[:, 1],
                          'dV': self.dV[:, 0]})

        df = a.merge(b, how='inner', on=['frame'])
        self.df = df #extract_minima_clusters

        print(self.data)
        print(self.dV)
        print(df)

        if df.shape[0] < self.cutoff:
            print(f"Cutoff {self.cutoff}: Cutoff exceeds number of frames, reduce --conv_points")

        self.whole_dv_avg = np.mean(df['dV'])
        with open(str(self.output_dir) + '/convergence/boost_potential_mean.txt', 'w') as f:
            f.write("%f" % np.mean(df['dV']))
        self.whole_dv_std = np.std(df['dV'])
        with open(str(self.output_dir) + '/convergence/boost_potential_std.txt', 'w') as f:
            f.write("%f" % np.std(df['dV']))

        self.conv_check = []
        self.universe = df.to_numpy()

        pos = [str(1)]
        for count, segment_iterations in enumerate(range(self.iterations)):
            print('iteration:', segment_iterations + 1)
            if segment_iterations == 0:
                PBCx_values = -360, 0, 360
                PBCy_valyes = -360, 0, 360
                if self.pbc == 'True':
                    xmax = 180
                    xmin = -180
                    ymax = 180
                    ymin = -180
                    for x in PBCx_values:
                        for y in PBCy_valyes:
                            # temp_array = pd.concat([a, b['dV']], axis=1).to_numpy()
                            temp_array = a.merge(b, how='inner', on=['frame']).to_numpy()
                            temp_array[:, 0] += x
                            temp_array[:, 1] += y
                            index = 0
                            del_index = []
                            for coord in temp_array:
                                if xmax + 0.1 * (xmax - xmin) < coord[0] or \
                                        coord[0] < xmin - 0.1 * (xmax - xmin) or \
                                        ymax + 0.1 * (ymax - ymin) < coord[1] or \
                                        coord[1] < ymin - 0.1 * (ymax - ymin):
                                    del_index.append(index)
                                index += 1
                            temp_array = np.delete(temp_array, del_index, axis=0)
                            self.universe = np.concatenate((self.universe, temp_array), axis=0)
                x = [self.universe]
            else:
                x = [self.universe][0]
            # parallel
            pool = Pool(self.cores)
            output = pool.starmap(segment_data, zip(x, pos, repeat(self.cutoff)))
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
                    # plt.scatter(arr[..., 0], arr[..., 1])
            #plt.show()
            self.universe = universe

            if np.array(old_universe).shape == np.array(self.universe).shape:
                self.iterations = max(segment_iterations, 6)
                # self.iterations = segment_iterations
                break
            else:
                old_universe = self.universe

    def reweight_segments(self):
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
                pmf_val = pmf_val + c12  # Delete c12 to not apply reweighting
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

        self.anharm_total_min = np.append(self.anharm_total_min, anharm(self.pmf_min_distribution[0]))
        self.dv_avg_distribution_min = np.append(self.dv_avg_distribution_min, pmf_array[index, 3])
        self.dv_std_distribution_min = np.append(self.dv_std_distribution_min, pmf_array[index, 4])

        pmf_array[:, 0] -= np.min(pmf_array[:, 0])  # normalises the array to 0 minimum
        self.pmf_array = pmf_array
        self.pmf_array_convergence.append(pmf_array[:, 0:3])

    def interpolate_pmf(self):
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
        plt.savefig(str(self.output_dir) + f'/intermediates/distribution_{self.cutoff}.png')
        plt.clf()

        z_scattered = np.append(z_scattered, limzval, axis=0)

        scattered_points_norm = scaler.transform(scattered_points)
        dense_points_norm_noplc = scaler.transform(dense_points)

        interpolation = RBFInterpolator(scattered_points_norm, z_scattered.ravel(), smoothing=0, kernel='linear',
                                        degree=0, neighbors=26)

        self.z_dense = interpolation(dense_points_norm_noplc).reshape(self.x_dense_pmf.shape)

        # rebuild the CV space using calculated PMF
        # for i in self.df:
        #     reweighted_datapoint = interpolation(i[0], i[1])
        #     print(reweighted_datapoint)

    def calc_limdata(self):

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

    def plot_PMF(self, xlab, ylab, title):
        self.PMF = np.nan_to_num(self.PMF, nan=int(self.Emax)).T

        plt.contourf(self.x_dense_pmf, self.y_dense_pmf, self.z_dense, cmap='jet', levels=50)
        plt.colorbar()
        plt.savefig(str(self.output_dir) + f'/intermediates/bias_{self.cutoff}.png')
        plt.clf()

        plt.contourf(self.x_dense_pmf, self.y_dense_pmf, self.PMF, cmap='jet', levels=50)
        plt.colorbar()
        plt.savefig(str(self.output_dir) + f'/intermediates/PMF_{self.cutoff}.png')

        z_dense = np.add(self.z_dense, self.PMF)
        mymin = np.nanmin([np.nanmin(r) for r in z_dense])
        z_dense = z_dense - mymin
        z_dense = np.nan_to_num(np.array(z_dense), nan=self.Emax)
        z_dense[z_dense > self.Emax] = self.Emax

        # 2D Projection
        plt.clf()
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel(xlab, fontsize=16)
        ax.set_ylabel(ylab, fontsize=16)
        contourf_ = ax.contourf(self.x_dense_pmf, self.y_dense_pmf, z_dense, cmap='jet', levels=50)
        cbar = fig.colorbar(contourf_)
        cbar.set_label('Kcal/mol\n', fontsize=16)
        plt.savefig(str(self.output_dir) + f'PMF/2C_PMF_{self.cutoff}.png')
        plt.clf()

        self.output_PMF = z_dense

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
        plt.plot(conv_points, self.dv_avg_distribution_min)
        plt.hlines(self.whole_dv_avg, np.min(conv_points), np.max(conv_points), color='red',
                   label='ΔV Mean:' + str(self.whole_dv_avg))
        plt.xscale('log')
        plt.ylabel('Mean (Kcal/mol)')
        plt.xlabel('VDR Cutoff (frames)')
        plt.savefig(str(self.output_dir) + '/convergence/mean_plot_min.png')
        np.savetxt(str(self.output_dir) + '/convergence/avg_min.dat',
                   np.column_stack((conv_points, self.dv_avg_distribution_min)))
        plt.clf()
        plt.plot(conv_points, self.dv_std_distribution_min)
        plt.hlines(self.whole_dv_std, np.min(conv_points), np.max(conv_points), color='red',
                   label='ΔV Standard Deviation:' + str(self.whole_dv_std))
        plt.xscale('log')
        plt.ylabel('Standard Deviation (Kcal/mol)')
        plt.xlabel('VDR Cutoff (frames)')
        plt.savefig(str(self.output_dir) + '/convergence/std_plot_min.png')
        np.savetxt(str(self.output_dir) + '/convergence/std_min.dat',
                   np.column_stack((conv_points, self.dv_std_distribution_min)))
        plt.clf()
        plt.plot(conv_points, self.anharm_total_min)
        plt.xscale('log')
        plt.ylabel('Anharmonicity')
        plt.xlabel('VDR Cutoff (frames)')
        plt.savefig(str(self.output_dir) + '/convergence/anharm_plot_min.png')
        np.savetxt(str(self.output_dir) + '/convergence/anharm_min.dat',
                   np.column_stack((conv_points, self.anharm_total_min)))
        plt.clf()

        plt.plot(conv_points, self.dv_avg_distribution_max)
        plt.hlines(self.whole_dv_avg, np.min(conv_points), np.max(conv_points), color='red',
                   label='ΔV Mean:' + str(self.whole_dv_avg))
        plt.xscale('log')
        plt.ylabel('Mean (Kcal/mol)')
        plt.xlabel('VDR Cutoff (frames)')
        plt.savefig(str(self.output_dir) + '/convergence/mean_plot_max.png')
        np.savetxt(str(self.output_dir) + '/convergence/avg_max.dat',
                   np.column_stack((conv_points, self.dv_avg_distribution_max)))
        plt.clf()
        plt.plot(conv_points, self.dv_std_distribution_max)
        plt.hlines(self.whole_dv_std, np.min(conv_points), np.max(conv_points), color='red',
                   label='ΔV Standard Deviation:' + str(self.whole_dv_std))
        plt.xscale('log')
        plt.ylabel('Standard Deviation (Kcal/mol)')
        plt.xlabel('VDR Cutoff (frames)')
        plt.savefig(str(self.output_dir) + '/convergence/std_plot_max.png')
        np.savetxt(str(self.output_dir) + '/convergence/std_max.dat',
                   np.column_stack((conv_points, self.dv_std_distribution_max)))
        plt.clf()
        plt.plot(conv_points, self.anharm_total_max)
        plt.xscale('log')
        plt.ylabel('Anharmonicity (Kcal/mol)')
        plt.xlabel('VDR Cutoff (frames)')
        plt.savefig(str(self.output_dir) + '/convergence/anharm_plot_max.png')
        np.savetxt(str(self.output_dir) + '/convergence/anharm_max.dat',
                   np.column_stack((conv_points, self.anharm_total_max)))
        plt.clf()

    def extract_minima_clusters(self, mode='mdanalysis'):
    #     if mode=='Inflecs':
    #         sys.path.append('/mainfs/scratch/sct1g15/software/InfleCS-free-energy-clustering-tutorial-master/')
    #         import free_energy_clustering as FEC

        if mode=='normal':
            print(self.output_PMF)
            from scipy.ndimage.filters import minimum_filter, maximum_filter
            neighborhood = 3
            arr2D = self.output_PMF

            threshold = 2
            data_min = minimum_filter(arr2D, neighborhood)
            minima = (arr2D == data_min) 
            data_max = maximum_filter(arr2D, neighborhood)
            maxima = (arr2D == data_max)
            boundary = (arr2D == self.Emax)
            
            peaks = (minima == True) & (boundary == False)
             
            plt.imshow(maxima, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/maxima.png')
            plt.clf()
            plt.imshow(minima, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/minima.png')
            plt.clf()
            plt.imshow(boundary, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/boundary.png')
            plt.clf()
            plt.imshow(peaks, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/peaks.png')
            plt.clf()

            peak_index = np.where(peaks == True)             
            print(self.x_dense_pmf, self.y_dense_pmf)
            clusters = np.column_stack((peak_index[1], peak_index[0], arr2D[peak_index]))
            clusters = clusters[clusters[:, 2].argsort()]
            print(self.df) 
            for count, i in enumerate(clusters):
                cluster_index = np.where((self.df['rc1'] > self.x_dense_pmf[0][int(i[0])]) & (self.df['rc1'] < self.x_dense_pmf[0][int(i[0])+1]) & (self.df['rc2'] > self.y_dense_pmf[int(i[1])][0]) & (self.df['rc2'] < self.y_dense_pmf[int(i[1])+1][0]))
                np.savetxt(f'{self.output_dir}clusters/Cluster_{count}_index.ndx', cluster_index, header='[frames]')

        if mode=='mdanalysis':
            import MDAnalysis as mda
            print(self.output_PMF)
            from scipy.ndimage.filters import minimum_filter, maximum_filter
            neighborhood = 3
            arr2D = self.output_PMF

            threshold = 2
            data_min = minimum_filter(arr2D, neighborhood)
            minima = (arr2D == data_min)
            data_max = maximum_filter(arr2D, neighborhood)
            maxima = (arr2D == data_max)
            boundary = (arr2D == self.Emax)

            peaks = (minima == True) & (boundary == False)

            plt.imshow(maxima, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/maxima.png')
            plt.clf()
            plt.imshow(minima, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/minima.png')
            plt.clf()
            plt.imshow(boundary, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/boundary.png')
            plt.clf()
            plt.imshow(peaks, origin='lower')
            plt.savefig(str(self.output_dir) + '/intermediates/peaks.png')
            plt.clf()

            peak_index = np.where(peaks == True)
            clusters = np.column_stack((peak_index[1], peak_index[0], arr2D[peak_index]))
            clusters = clusters[clusters[:, 2].argsort()]
            u = mda.Universe('../init_startfiles/PLP.leap.pdb', '../production_new/E2_12/protein_cat.xtc')
            protein = u.select_atoms("protein")
            for count, i in enumerate(clusters):
                cluster_index = np.where((self.df['rc1'] > self.x_dense_pmf[0][int(i[0])]) & (self.df['rc1'] < self.x_dense_pmf[0][int(i[0])+1]) & (self.df['rc2'] > self.y_dense_pmf[int(i[1])][0]) & (self.df['rc2'] < self.y_dense_pmf[int(i[1])+1][0]))
                cluster_mid_x = self.x_dense_pmf[0][int(i[0])] + (self.x_dense_pmf[0][int(i[0])+1] - self.x_dense_pmf[0][int(i[0])])/2
                cluster_mid_y = self.y_dense_pmf[int(i[1])][0] + (self.y_dense_pmf[int(i[1])+1][0] - self.y_dense_pmf[int(i[1])][0])/2
                np.savetxt(f'{self.output_dir}clusters/Cluster_{count}_index.ndx', cluster_index, header=f'rc1:{cluster_mid_x}, rc2:{cluster_mid_y}.txt')
                with mda.Writer(f'{self.output_dir}clusters/Cluster_{count}.pdb', protein.n_atoms) as W:
                    for ts in u.trajectory[np.array(cluster_index).flatten()]:
                        W.write(protein)






