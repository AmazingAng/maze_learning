#from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
import scipy.stats
from tqdm import tqdm
import pdb
import os
from matplotlib.backends.backend_pdf import PdfPages
import sklearn.preprocessing
import pylab as pl

from maze_graph import *
from maze_utils2 import *

# 函数区 ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def calc_SI(spikes, rate_map, t_total, t_nodes_frac):
    mean_rate = sum(spikes)/t_total # mean firing rate
    logArg = rate_map / mean_rate;
    logArg[logArg == 0] = 1; # keep argument in log non-zero

    IC = np.nansum(t_nodes_frac * rate_map * np.log2(logArg)) # information content
    SI = IC / mean_rate; # spatial information (bits/spike)
    return(SI)

def shuffle_test_isi(SI, spikes, spike_nodes, occu_time, _coords_range, _nbins = 144, shuffle_n = 1000):
    SI_rand = np.zeros(shuffle_n)
    t_total = np.nansum(occu_time)/1000
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6) 
    spike_ind = np.where(spikes==1)[0] # spike index
    isi = np.append(spike_ind[0], np.ediff1d(spike_ind)) # get interspike interval

    for i in range(shuffle_n):
        shuffle_isi = np.random.choice(isi, size = len(isi), replace = False) # shuffle interspike interval
        shuffle_spike_ind = np.cumsum(shuffle_isi) # shuffled spike index

        spikes_rand = np.zeros_like(spikes)        
        spikes_rand[shuffle_spike_ind] = 1
        spike_freq_rand, _, _ = scipy.stats.binned_statistic(
            spike_nodes,
            spikes_rand,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)
        rate_map_rand = spike_freq_rand/(occu_time/1000+ 1E-6)    
        SI_rand[i] = calc_SI(spikes = spikes_rand, rate_map = rate_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    is_placecell = SI > np.percentile(SI_rand, 95)
    return is_placecell

def plot_ratemap(ratemap, axes=None, title=None, extent = [0 , 12, 0 , 12],title_color = "black", *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot ratemaps."""
    if axes is None:
        axes = pl.gca()
    # Plot the ratemap
    axes.imshow(ratemap, interpolation='none', extent = extent, origin = "lower", alpha = 0.8, *args, **kwargs)
    # ax.pcolormesh(ratemap, *args, **kwargs)
    axes.axis('off')
    if title is not None:
        axes.set_title(title, color = title_color)
        
def node_to_run(node, runs):
    # transform node number (start from 1) to run number (start from 1)
    # for example node 1, to run 1
    run_ind = 0
    for run_i in runs:
        if node in run_i:
            run_ind = runs.index(run_i)+1
            return run_ind
    return run_ind
# ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

def Run_all_mice(mylist):
    # 主要参数赋值，参数来源Excel
    totalpath = "G:\YSY"
    date = mylist[0]
    NumOfMice = mylist[1]
    maze_type = mylist[2]
    file_path = mylist[3]
    print(file_path)
    
    path = os.path.join(file_path,"ms.mat")

    # behav_mat = loadmat('ms.mat'); ----------------------------------------------------------------------------------------------------------------------
    # read calcium data
    with h5py.File(path, 'r') as f:
        ms_mat = f['ms']
        FiltTraces = np.array(ms_mat['FiltTraces'])
        RawTraces = np.array(ms_mat['RawTraces'])
        DeconvSignal = np.array(ms_mat['DeconvSignals'])
        ms_time = np.array(ms_mat['time'])[0,]
    # read behav data
    # time in seconds
    print("     ms.mat file reading successfully!")

    with open(os.path.join(totalpath,NumOfMice,date,"behav_decision.pkl"), 'rb') as handle:
        correct_time, wrong_time, correct_time_percentage, decision_rate,_ = pickle.load(handle)
    with open(os.path.join(totalpath,NumOfMice,date,"behav_processed.pkl"), 'rb') as handle:
        behav_time_original, behav_nodes_interpolated, behav_dir = pickle.load(handle)
    
    # remove no behav recording period
    behav_mask = (ms_time>=behav_time_original[0]) & (ms_time<behav_time_original[-1])
    FiltTraces_behav = FiltTraces[:,behav_mask]
    RawTraces_behav = RawTraces[:,behav_mask]
    DeconvSignal_behav = DeconvSignal[:,behav_mask]
    ms_time_behav = ms_time[behav_mask]

    # plot spike data -------------------------------------------------------------------------------------------------------------------------------------------
    deconv_sd = np.std(DeconvSignal_behav, axis = 1) * 3
    Spikes = np.where(DeconvSignal_behav>np.repeat(deconv_sd[:,np.newaxis], DeconvSignal_behav.shape[1],1), 1, 0)
    plt.subplot(311)
    plt.plot(Spikes[1,:])
    plt.subplot(312)
    plt.plot(Spikes[2,:])
    plt.subplot(313)
    plt.plot(Spikes[3,:])

    path = os.path.join(totalpath,NumOfMice,date,"neural_activity",'Spike.png')
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity",'Spike.pdf')
    plt.savefig(path)
    print("     Figure 1 'Spikes' is done...")
    print("     initialze maze object...")
    # initialze maze object -------------------------------------------------------------------------------------------------------------------
    start_node = 1
    end_node = 144
    nx = 12
    ny = 12
    total_bin = nx*ny
    graph = maze1_graph if maze_type ==1 else maze2_graph
    test_maze = Maze(nx, ny, graph)
    new_maze = test_maze.make_maze()
    shortest_path = test_maze.BFS_SP(graph, start_node, end_node)
    # behav_bins = test_maze.idx_to_loc(behav_nodes_interpolated-1) # start from 0 ------------------------------------------------------------------

    # spike per bin
    spike_nodes = np.zeros_like(Spikes[0,:])
    for i in range(0, len(spike_nodes)):
        if ms_time_behav[i] < behav_time_original[0]:
            spike_nodes[i] = start_node;
        else:
            match_index = np.where(behav_time_original <= ms_time_behav[i])[0][-1]
            spike_nodes[i] = behav_nodes_interpolated[match_index]
    spike_bins = test_maze.idx_to_loc(spike_nodes-1)

    # occupancy map, in ms
    # Duration for each (X,Y) sample (clipped to maxGap) ----------------------------------------------------------------------------------------------
    maxGap = 100
    stay_time = np.append(np.ediff1d(ms_time_behav),0)
    stay_time[stay_time>maxGap] = maxGap
    _nbins = total_bin
    _coords_range = [0, _nbins +0.0001 ]
    occu_time, xbin_edges, bin_numbers = scipy.stats.binned_statistic(
        spike_nodes,
        stay_time,
        bins=_nbins,
        statistic="sum",
        range=_coords_range)

    assert((bin_numbers == spike_nodes).all())
    minimum_occu_thres = 50 # ms, occupancy time less than the threshold are set to NAN, preventing large spike rate ----------------------------------------------
    occu_time[occu_time<minimum_occu_thres] = np.nan

    ### sum rate map (for first neuron)
    _nbins = total_bin
    _coords_range = [0, _nbins +0.0001 ]
    spikes = Spikes[0,:]

    # spike_map: total spikes per nodes-------------------------------------------------------------------------------------------------------
    spike_freq, xbin_edges, bin_numbers = scipy.stats.binned_statistic(
        spike_nodes,
        spikes,
        bins=_nbins,
        statistic="sum",
        range=_coords_range)

    rate_map = spike_freq/(occu_time/1000+ 1E-6) # ratemap: spike_map / occupancy_map
    assert((bin_numbers == spike_nodes).all())
    print("     rate map data has been calculated successfully!")
    # rate map (for all neuron)
    n_neuron = Spikes.shape[0]
    spike_freq_all = np.zeros([n_neuron,total_bin])
    rate_map_all =  np.zeros_like(spike_freq_all)
    for i in range(n_neuron):
        spike_freq_all[i,] ,_ ,_= scipy.stats.binned_statistic(
            spike_nodes,
            Spikes[i,:],
            bins=_nbins,
            statistic="sum",
            range=_coords_range)
        rate_map_all[i,] = spike_freq_all[i,]/(occu_time/1000+ 1E-9)

    # spatial information (for one neuron) 
    # see DOI: 10.1126/science.aav9199
    t_total = np.nansum(occu_time)/1000 # total time of trial
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6)   # time spent in ith bin/total session time

    SI = calc_SI(spikes=spikes, rate_map=rate_map, t_total=t_total, t_nodes_frac=t_nodes_frac)
    print("     Here is the ISI shuffle. Several minutes will be taken...")
    # shuffle information for 1000 times
    shuffle_n = 1000
    SI_rand = np.zeros(shuffle_n)
    for i in range(shuffle_n):
        shuffle_tmp = np.random.randint(Spikes.shape[1])
        spikes_rand = np.roll(spikes, shuffle_tmp)
        spike_freq_rand, _, _ = scipy.stats.binned_statistic(
            spike_nodes,
            spikes_rand,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)
        rate_map_rand = spike_freq_rand/(occu_time/1000+ 1E-6)    
        SI_rand[i] = calc_SI(spikes = spikes_rand, rate_map = rate_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    SI_thres = np.percentile(SI_rand, 95)
    if SI<SI_thres:
        print("     This cell is rejected by 95% threshold, i.e. not place cell." )
    else:
        print("     Ye! This cell is a place cell!")
    
    # spatial information (all neurons) -------------------------------------------------------------------------------------------------------------------------
    # Shuffle ISI: DOI: 10.1523/JNEUROSCI.19-21-09497.1999
    SI_all = np.zeros(n_neuron)
    is_placecell_isi = np.zeros(n_neuron)

    for i in tqdm(range(n_neuron)):
        SI_all[i] = calc_SI(spikes=Spikes[i,], rate_map=rate_map_all[i,], t_total=t_total, t_nodes_frac=t_nodes_frac)
        is_placecell_isi[i] = shuffle_test_isi(SI = SI_all[i], spikes = Spikes[i,], spike_nodes=spike_nodes, occu_time=occu_time, _coords_range=_coords_range)
    # ISI shuffle ended
    print("     ISI Shuffle ended successfully!")
    # plot ratemap 2d:-----------------------------------------------------------------------------------------------------------------------------
    # np.where(is_placecell_isi)

    fig = plt.figure(figsize=[(12), (12)])
    axes = fig.add_subplot(1, 1, 1)
    # axes.invert_yaxis()

    # test_maze.maze_plot_num(axes = axes, mode = "cells")
    rate_map_2d = np.reshape(rate_map_all[2,], [12,12])
    plot_ratemap(rate_map_2d,  axes = axes)
    axes.invert_yaxis()
    # plt.scatter(1,1)

    path = os.path.join(totalpath,NumOfMice,date,"neural_activity",'ratemap.png')
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity",'ratemap.pdf')
    plt.savefig(path)
    print("     Figure 2 'rate map' is done... (Without normalization, for example)")
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    # plot ratemap 2d:
    # sorted by place cell first
    sort_by_place_cell = True
    # Sort by score if desired
    if sort_by_place_cell:
        ordering = np.argsort(-np.array(is_placecell_isi))
    else:
        ordering = range(n_neuron)
    # Plot
    cols = 8
    rows = int(np.ceil(n_neuron / cols))
    fig = plt.figure(figsize=(24, rows * 4))
    for i in range(n_neuron):
        rf = plt.subplot(rows * 2, cols, i + 1)
        if i < n_neuron:
            index = ordering[i]
            title = "%d (%.2f)" % (index, SI_all[index])
            # Plot maze
            # test_maze.maze_plot(axes = axes)
            # Plot the activation maps
            if is_placecell_isi[index] == 1:
                plot_ratemap(np.reshape(rate_map_all[index,], [12,12]), axes=rf, title=title, title_color = "red")
            else:
                plot_ratemap(np.reshape(rate_map_all[index,], [12,12]), axes=rf, title=title)
            # pl.colorbar()
            rf.invert_yaxis()
    print("     Figure 3 'all neuron's rate map' is done... (Without normalization)")
    path = os.path.join(totalpath,NumOfMice,date,"rate_map_all.pkl")
    with open(path, 'wb') as f:
        pickle.dump([is_placecell_isi, SI_all, rate_map_all], f)
    
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity",'rate_map.pdf')
    plt.savefig(path, format="pdf")
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity",'rate_map.png')
    plt.savefig(path, format="png")
    
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------
    # plot ratemap 1d (all place cells):
    sort_by_nodes = True
    num_pc = int(sum(is_placecell_isi))
    rate_map_pc= rate_map_all[is_placecell_isi==1,]
    place_fields = np.nanargmax(rate_map_pc,axis=1)+1 # place field

    if sort_by_nodes:
        ordering = np.argsort(-np.nanargmax(rate_map_pc,axis=1))
    else:
        ordering = range(num_pc)

    # normalize tuning curve to [0, 1]
    rate_map_pc_norm = sklearn.preprocessing.minmax_scale(rate_map_pc, feature_range=(0, 1), axis=1, copy=True)
    # plot
    plot_ratemap(rate_map_pc_norm[ordering,:][:,~np.isnan(rate_map_pc_norm[0,])],extent = [0 , 1000, 0 , 300])
    # save figure
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_1d.pdf")
    plt.savefig(path, format="pdf")
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_1d.png")
    plt.savefig(path, format="png")
    print("     Figure 4 'rate map' is done... (Normalized for all place cell)")
    # 备注: 该段代码与上一段几乎完全相同，只是排序方法不同。
    # ratemap 1d: orderd by shortest path --------------------------------------------------------------------------------------------------------------------------
    sort_by_path = True
    num_pc = int(sum(is_placecell_isi))
    rate_map_pc= rate_map_all[is_placecell_isi==1,]
    if sort_by_path:
        non_shortest_path = list(np.setdiff1d(list(graph.keys()), test_maze.shortest_path))
        preferred_nodes = np.nanargmax(rate_map_pc,axis=1)+1 # place field
        total_list = shortest_path + non_shortest_path
        tmp_index = np.zeros(num_pc)
        for i in range(num_pc):
            tmp_index[i] =  np.where(total_list == preferred_nodes[i])[0]
        
        ordering = np.argsort(-tmp_index)
    else:
        ordering = range(num_pc)

    # normalize tuning curve to [0, 1]
    rate_map_pc_norm = sklearn.preprocessing.minmax_scale(rate_map_pc, feature_range=(0, 1), axis=1, copy=True)
    # plot
    rate_map_ordered = rate_map_pc_norm[ordering,:][:,np.array(total_list)-1]
    plot_ratemap(rate_map_ordered[:,~np.isnan(rate_map_ordered[0,])],extent = [0 , 1000, 0 , 300])
    # save figure
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_1d_path.pdf")
    plt.savefig(path, format="pdf")
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_1d_path.png")
    plt.savefig(path, format="png")
    print("     Figure 5 'rate map' is done... (Normalized for all place cell, ranked in short path order)")
    # decision node selective %---------------------------------------------------------------------------------------------------------
    decision_prefer_num = 0
    for i in range(num_pc):
        if place_fields[i] in test_maze.decision_nodes:
            decision_prefer_num += 1
        
    decision_ratio = decision_prefer_num/num_pc
    decision_ratio_chance = len(test_maze.decision_nodes)/total_bin
    print("     % of place fields on decison nodes： " + "{:.2f}".format(decision_ratio*100) + "%\n Chance Level: " + "{:.2f}".format(decision_ratio_chance*100)+"%" )

    # branch selective % ----------------------------------------------------------------------------------------------------------------------------------
    ## spike node to runs
    spike_runs = np.zeros_like(spike_nodes)
    for i in range(len(spike_nodes)):
        spike_runs[i] = node_to_run(spike_nodes[i], test_maze.runs)


    # occupancy map, in ms
    _nruns = len(test_maze.runs)
    _coords_range = [0, _nruns +0.0001 ]
    occu_time_run, _, run_numbers = scipy.stats.binned_statistic(
            spike_runs,
            stay_time,
            bins=_nruns,
            statistic="sum",
            range=_coords_range)
    assert((run_numbers == spike_runs).all())
    minimum_occu_thres = 50 # ms, occupancy time less than the threshold are set to NAN, preventing large spike rate
    occu_time_run[occu_time_run<minimum_occu_thres] = np.nan

    # branch rate map 
    ### sum rate map (for first neuron)
    spikes = Spikes[0,:]
    # spike_map: total spikes per nodes
    spike_freq_run, xbin_edges, run_numbers = scipy.stats.binned_statistic(
            spike_runs,
            spikes,
            bins=_nruns,
            statistic="sum",
            range=_coords_range)
    rate_map_run = spike_freq_run/(occu_time_run/1000+ 1E-6) # ratemap: spike_map / occupancy_map
    assert((run_numbers == spike_runs).all())
    print("     Branch rate map has been generated successfully...")
    print("     Prepare for branche cell shuffle... (ISI method)")
    # rate map for branches (for all neuron)
    n_neuron = Spikes.shape[0]
    spike_freq_run_all = np.zeros([n_neuron, _nruns])
    rate_map_run_all =  np.zeros_like(spike_freq_run_all)
    for i in range(n_neuron):
        spike_freq_run_all[i,] ,_ ,_= scipy.stats.binned_statistic(
            spike_runs,
            Spikes[i,:],
            bins=_nruns,
            statistic="sum",
            range=_coords_range)
        rate_map_run_all[i,] = spike_freq_run_all[i,]/(occu_time_run/1000+ 1E-9)  

    # spatial information for branches (all neurons) ----------------------------------------------------------------------------------------------------------------------------
    t_runs_frac = occu_time_run/1000/ (t_total+ 1E-6)   # time spent in ith bin/total session time

    # Shuffle ISI: DOI: 10.1523/JNEUROSCI.19-21-09497.1999
    SI_run_all = np.zeros(n_neuron)
    is_runcell_isi = np.zeros(n_neuron)

    for i in tqdm(range(n_neuron)):
        SI_run_all[i] = calc_SI(spikes=Spikes[i,], rate_map=rate_map_run_all[i,], t_total=t_total, t_nodes_frac=t_runs_frac)
        is_runcell_isi[i] = shuffle_test_isi(SI = SI_run_all[i], spikes = Spikes[i,], spike_nodes=spike_runs, occu_time=occu_time_run, _coords_range=_coords_range, _nbins = _nruns)
    
    print("     # of run cell using time isi shuffle method:", np.sum(is_runcell_isi ))

    # plot ratemap 1d (all branches cells): ---------------------------------------------------------------------------------------------------------------------------------
    sort_by_nodes = True
    num_rc = int(sum(is_runcell_isi))
    rate_map_rc= rate_map_run_all[is_runcell_isi==1,]
    run_fields = np.nanargmax(rate_map_rc,axis=1) # run field

    if sort_by_nodes:
        ordering = np.argsort(-np.nanargmax(rate_map_rc,axis=1))
    else:
        ordering = range(num_rc)

    # normalize tuning curve to [0, 1]
    rate_map_rc_norm = sklearn.preprocessing.minmax_scale(rate_map_rc, feature_range=(0, 1), axis=1, copy=True)
    # plot
    plot_ratemap(rate_map_rc_norm[ordering,:][:,~np.isnan(rate_map_rc_norm[0,])],extent = [0 , 1000, 0 , 300])
    # save figure

    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_run_1d.pdf")
    plt.savefig(path, format="pdf")
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_run_1d.png")
    plt.savefig(path, format="png")
    print("     Figure 6 'rate map' is done... (Normalized for branch(run) cell)")
    
    # run ratemap 1d: orderd by shortest path ------------------------------------------------------------------------------------------------------
    sort_by_path = True
    num_rc = int(sum(is_runcell_isi))
    rate_map_rc= rate_map_run_all[is_runcell_isi==1,]
    run_fields = np.nanargmax(rate_map_rc,axis=1) # place field
    run_ind = np.arange(len(test_maze.runs))

    if sort_by_path:
        run_on_path = [test_maze.runs[i_run][0] in test_maze.shortest_path for i_run in range(_nruns)]
        on_path_runs = run_ind[run_on_path]
        non_shortest_path = list(np.setdiff1d(list(run_ind),on_path_runs))
        preferred_runs = np.nanargmax(rate_map_rc,axis=1) # run field
        total_list = list(on_path_runs) + non_shortest_path
        tmp_index = np.zeros(num_rc)
        for i in range(num_rc):
            tmp_index[i] =  np.where(total_list == preferred_runs[i])[0]
        
        ordering = np.argsort(-tmp_index)
    else:
        ordering = range(num_rc)

    # normalize tuning curve to [0, 1]
    rate_map_rc_norm = sklearn.preprocessing.minmax_scale(rate_map_rc, feature_range=(0, 1), axis=1, copy=True)
    # plot
    rate_map_rc_ordered = rate_map_rc_norm[ordering,:][:,np.array(total_list)]
    plot_ratemap(rate_map_rc_ordered[:,~np.isnan(rate_map_rc_ordered[0,])],extent = [0 , 1000, 0 , 300])
    # save figure
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_run_1d_path.pdf")
    plt.savefig(path, format="pdf")
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","norm_ratemap_run_1d_path.png")
    plt.savefig(path, format="png")
    print("     Figure 7 'rate map' is done... (Normalized for branch(run) cell, ranked in short path order)")
    
    # percent of run filed on shortest path--------------------------------------------------------------------------------------------------------
    num_shortpath_runfield = np.sum([preferred_run in on_path_runs for preferred_run in preferred_runs])
    run_path_ratio = num_shortpath_runfield/num_rc
    run_path_ratio_chance = np.sum(run_on_path)/len(test_maze.runs)
    print("     % of run fields on shortest path: " + "{:.2f}".format(run_path_ratio*100) + "%\n Chance Level: " + "{:.2f}".format(run_path_ratio_chance*100)+"%" )

    SI_rand = np.zeros(shuffle_n)
    t_total = np.nansum(occu_time)/1000
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6) 
    spike_ind = np.where(spikes==1)[0] # spike index
    isi = np.append(spike_ind[0], np.ediff1d(spike_ind)) # get interspike interval
    for i in range(shuffle_n):
        shuffle_isi = np.random.choice(isi, size = len(isi), replace = False) # shuffle interspike interval
        shuffle_spike_ind = np.cumsum(shuffle_isi) # shuffled spike index

        spikes_rand = np.zeros_like(spikes)        
        spikes_rand[shuffle_spike_ind] = 1
        spike_freq_rand, _, _ = scipy.stats.binned_statistic(
            spike_nodes,
            spikes_rand,
            bins=_nbins,
            statistic="sum",
            range=_coords_range)
        rate_map_rand = spike_freq_rand/(occu_time/1000+ 1E-6)    
        SI_rand[i] = calc_SI(spikes = spikes_rand, rate_map = rate_map_rand, t_total = t_total, t_nodes_frac=t_nodes_frac)
    is_placecell = SI > np.percentile(SI_rand, 95)

    plt.subplot(311)
    plt.plot(FiltTraces_behav[1,:])
    plt.subplot(312)
    plt.plot(FiltTraces_behav[2,:])
    plt.subplot(313)
    plt.plot(FiltTraces_behav[3,:])
    print("     Figure 8 'FiltTraces_behav' is done..")

    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","FiltTrace_behav.png")
    plt.savefig(path)
    path = os.path.join(totalpath,NumOfMice,date,"neural_activity","FiltTrace_behav.pdf")
    plt.savefig(path)
    
    path = os.path.join(totalpath,NumOfMice,date,"rate_map_pcrc.pkl")
    with open(path, 'wb') as f:
        pickle.dump([rate_map_pc, rate_map_pc_norm, rate_map_ordered, rate_map_rc, rate_map_rc_norm, is_runcell_isi, rate_map_rc_ordered], f)
    print("     All files have been saved successfully! This session was perfectly done.",end='\n\n\n')

import pandas as pd
file = pd.read_excel("G:\YSY\mice_maze_metadata_time_correction.xlsx", sheet_name = "training_recording_new")

for i in range(len(file)):
    if file['maze_type'][i] == 2 or i <= 2:
        continue #先跑1后跑2
    print(file['number'][i], file['date'][i], file['maze_type'][i],"is calculating...........................................................")
    mylist = [str(file['date'][i]), str(file['number'][i]), int(file['maze_type'][i]), str(file['recording_folder'][i])]
    Run_all_mice(mylist) 