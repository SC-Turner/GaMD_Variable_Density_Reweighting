import numpy as np
import scipy

def calc_inputs(gamd, data):
    weights_input = np.loadtxt(gamd, comments='#', usecols=(0, 1, 6, 7))
    weights = weights_input[:, 2] + weights_input[:, 3] 
    weights_output = np.vstack((weights, weights_input[:, 1])).T

    data = np.loadtxt(data)

    return weights_output, data

def segment_data(temp_universe, pos, cutoff, divcut=1):
    #Segment Function
    div_cut = divcut
    if isinstance(temp_universe, str)  == True:
        miniverse = 'nan'
    else:
        vertices_new = []
        vertices_new.append(
            ((np.amax(temp_universe[..., 0]) - (np.amax(temp_universe[..., 0]) - np.amin(temp_universe[..., 0])) / 2),
             (np.amax(temp_universe[..., 1]))))  # [middle x, top y]
        vertices_new.append(((np.amax(temp_universe[..., 0]), ((np.amax(temp_universe[..., 1]) - (
                np.amax(temp_universe[..., 1]) - np.amin(temp_universe[..., 1])) / 2)))))  # [top x, middle y]
        vertices_new.append(((np.amin(temp_universe[..., 0]), ((np.amax(temp_universe[..., 1]) - (
                np.amax(temp_universe[..., 1]) - np.amin(temp_universe[..., 1])) / 2)))))  # [bottom x, middle y]
        vertices_new.append(
            ((np.amax(temp_universe[..., 0]) - (np.amax(temp_universe[..., 0]) - np.amin(temp_universe[..., 0])) / 2),
             (np.amin(temp_universe[..., 1]))))  # [middle x, bottom y]
        vertices_new.append(
            ((np.amax(temp_universe[..., 0]) - (np.amax(temp_universe[..., 0]) - np.amin(temp_universe[..., 0])) / 2),
             ((np.amax(temp_universe[..., 1]) - (
                     np.amax(temp_universe[..., 1]) - np.amin(temp_universe[..., 1])) / 2))))  # [middle x, middle y]

        TL = temp_universe[
            (temp_universe[..., 0] < vertices_new[0][0]) & (temp_universe[..., 0] >= vertices_new[2][0])
            & (temp_universe[..., 1] >= vertices_new[1][1]) & (
                    temp_universe[..., 1] <= vertices_new[0][1])]
        TR = temp_universe[
            (temp_universe[..., 0] <= vertices_new[1][0]) & (temp_universe[..., 0] >= vertices_new[0][0])
            & (temp_universe[..., 1] >= vertices_new[1][1]) & (
                    temp_universe[..., 1] <= vertices_new[0][1])]
        BL = temp_universe[
            (temp_universe[..., 0] < vertices_new[0][0]) & (temp_universe[..., 0] >= vertices_new[2][0])
            & (temp_universe[..., 1] >= vertices_new[3][1]) & (
                    temp_universe[..., 1] <= vertices_new[1][1])]
        BR = temp_universe[
            (temp_universe[..., 0] <= vertices_new[1][0]) & (temp_universe[..., 0] > vertices_new[0][0])
            & (temp_universe[..., 1] >= vertices_new[3][1]) & (
                    temp_universe[..., 1] <= vertices_new[1][1])]

        # Selection Conditions
        pos_update = []
        if (TL[..., 0].shape[0] > cutoff) & \
                (TR[..., 0].shape[0] > cutoff) & \
                (BL[..., 0].shape[0] > cutoff) & \
                (BR[..., 0].shape[0] > cutoff):
            miniverse = TL, TR, BL, BR
            pos_update = ['{}.{}'.format(pos, 1), '{}.{}'.format(pos, 2), '{}.{}'.format(pos, 3), '{}.{}'.format(pos, 4)]
        # if one or more boxes have > datapoints condition
        elif (TL[..., 0].shape[0] > cutoff) or \
                (TR[..., 0].shape[0] > cutoff) or \
                (BL[..., 0].shape[0] > cutoff) or \
                (BR[..., 0].shape[0] > cutoff):
            if (TL[..., 0].shape[0] > cutoff / div_cut):
                pos_update.append('{}.{}'.format(pos, 1))
            elif (TL[..., 0].shape[0] < cutoff / div_cut):
                TL = 'nan'
            else:
                TL = 'nan'
            if (TR[..., 0].shape[0] > cutoff / div_cut):
                pos_update.append('{}.{}'.format(pos, 2))
            elif (TR[..., 0].shape[0] < cutoff / div_cut):
                TR = 'nan'
            else:
                TR = 'nan'
            if (BL[..., 0].shape[0] > cutoff / div_cut):
                pos_update.append('{}.{}'.format(pos, 3))
            elif (BL[..., 0].shape[0] < cutoff / div_cut):
                BL = 'nan'
            else:
                BL = 'nan'
            if (BR[..., 0].shape[0] > cutoff / div_cut):
                pos_update.append('{}.{}'.format(pos, 4))
            elif (BR[..., 0].shape[0] < cutoff / div_cut):
                BR = 'nan'
            else:
                BR = 'nan'
            miniverse = TL, TR, BL, BR
        # If no conditions met (all selections empty)
        else:
            miniverse = tuple([temp_universe])
            pos_update = ['{}.{}'.format(pos, 1)]

    return miniverse, pos_update

def datamax_calc(datapoint, datapoints_norm):
    distout = scipy.spatial.distance.cdist(np.array(datapoints_norm)[np.newaxis, :], np.array(datapoint))
    distout = np.min(distout[np.nonzero(distout)])
    return distout

def anharm(data):
    var=np.var(data)
    hist, edges=np.histogram(data, 50, normed=True)
    hist=np.add(hist,0.000000000000000001)  ###so that distrib
    dx=edges[1]-edges[0]
    S1=-1*np.trapz(np.multiply(hist, np.log(hist)),dx=dx)
    S2=0.5*np.log(2.00*np.pi*np.exp(1.0)*var+0.000000000000000001)
    alpha=S2-S1
    if np.isinf(alpha):
       alpha = 100
    return alpha
