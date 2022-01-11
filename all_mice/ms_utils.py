import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats
import pylab as pl

# calculate spatial information
def calc_SI(spikes, rate_map, t_total, t_nodes_frac):
    mean_rate = sum(spikes)/t_total # mean firing rate
    logArg = rate_map / mean_rate;
    logArg[logArg == 0] = 1; # keep argument in log non-zero

    IC = np.nansum(t_nodes_frac * rate_map * np.log2(logArg)) # information content
    SI = IC / mean_rate; # spatial information (bits/spike)
    return(SI)


# spatial information shuffle test: shuffle ISI (interspike interval)
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

# spatial information shuffle test: shuffle by a random time
def shuffle_test_shift(SI, spikes, spike_nodes, occu_time, _coords_range, _nbins = 144, shuffle_n = 1000):
    SI_rand = np.zeros(shuffle_n)
    t_total = np.nansum(occu_time)/1000
    t_nodes_frac = occu_time/1000/ (t_total+ 1E-6)
    for i in range(shuffle_n):
        shuffle_tmp = np.random.randint(spikes.shape[0])
        spikes_rand = np.roll(spikes, shuffle_tmp)
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

def calculate_ratemap(agent_poss, activation, statistic='mean', _nbins = 10, _coords_range = [[0,1], [0,1]]):
    xs = agent_poss[:,0]
    ys = agent_poss[:,1]

    return scipy.stats.binned_statistic_2d(
        xs,
        ys,
        activation,
        bins=_nbins,
        statistic=statistic,
        range=_coords_range,
        expand_binnumbers = True)

def plot_ratemap(ratemap, ax=None, title=None, axis_on = True, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot ratemaps."""
    if ax is None:
        ax = plt.gca()
    # Plot the ratemap
    ax.imshow(ratemap, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(ratemap, *args, **kwargs)
    if axis_on == False:
        ax.axis('off')
    else:
        ax.axis('auto')
    if title is not None:
        ax.set_title(title)
