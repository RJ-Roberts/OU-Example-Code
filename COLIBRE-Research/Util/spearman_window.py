import numpy as np
from scipy.stats import spearmanr

def spearman_track(c, y, x):
    # Define binning windows and empty metric lists    
    halo_total = len(y)
    delta_bin = 100 #200
    lower_bound = 0
    upper_bound = 500 #5000
    bin_means = []
    bins = []
    count = -1

    sr_arr = np.ones(5000)*np.nan
    p_arr = np.ones(5000)*np.nan


    # While the upper-bound of the bin edge is lower than the total number of haloes, continue walking bin process
    while upper_bound < halo_total:

        # Pull the correct values for the current bin range
        x_bin = x[lower_bound:upper_bound+delta_bin]
        c_bin = c[lower_bound:upper_bound+delta_bin]
        y_bin = y[lower_bound:upper_bound+delta_bin]

        # Record the x values at the edges of the bin, as well as the mean x value of the bin
        bins.append(np.min(x_bin))
        bins.append(np.max(x_bin))
        bin_means.append((np.min(x_bin)+np.max(x_bin))/2)
        
        # Calculate the Spearman coefficient and p-value for the bin for each input parameter
        sr_arr[count+1] = spearmanr(c_bin, y_bin)[0]
        p_arr[count+1] = spearmanr(c_bin, y_bin)[1]
        
        # Start reducing bin sixe at a given mass threshold
        if bin_means[count] > 13:
            delta_bin = int(halo_total/50)
            lower_bound = int(upper_bound - 1.25*delta_bin)
        else: 
            lower_bound += delta_bin
            
        upper_bound += delta_bin
        count += 1

    # Truncate arrays to remove empty space
    bins, bin_means = np.array(bins), np.array(bin_means)
    sr_arr = sr_arr[0:len(bin_means)]
    p_arr = p_arr[0:len(bin_means)]
    
    return bin_means, sr_arr, p_arr




def spearman_track_ebind(c, y, x):
    # Define binning windows and empty metric lists    
    halo_total = len(y)
    delta_bin = 100 #200
    lower_bound = 0
    upper_bound = 500 #5000
    bin_means = []
    bins = []
    count = -1
    bin_reduce = 25
    ebind_threshold = 62

    sr_arr = np.ones(5000)*np.nan
    p_arr = np.ones(5000)*np.nan


    # While the upper-bound of the bin edge is lower than the total number of haloes, continue walking bin process
    while upper_bound < halo_total:

        # Pull the correct values for the current bin range
        x_bin = x[lower_bound:upper_bound+delta_bin]
        c_bin = c[lower_bound:upper_bound+delta_bin]
        y_bin = y[lower_bound:upper_bound+delta_bin]

        # Record the x values at the edges of the bin, as well as the mean x value of the bin
        bins.append(np.min(x_bin))
        bins.append(np.max(x_bin))
        bin_means.append((np.min(x_bin)+np.max(x_bin))/2)
        
        # Calculate the Spearman coefficient and p-value for the bin for each input parameter
        sr_arr[count+1] = spearmanr(c_bin, y_bin)[0]
        p_arr[count+1] = spearmanr(c_bin, y_bin)[1]
        
        # Start reducing bin sixe at a given Ebind threshold
        if bin_means[count] > ebind_threshold:
            delta_bin = int(halo_total/bin_reduce)
            lower_bound = int(upper_bound - 1.25*delta_bin)
        else: 
            lower_bound += delta_bin
            
        upper_bound += delta_bin
        count += 1

    # Truncate arrays to remove empty space
    bins, bin_means = np.array(bins), np.array(bin_means)
    sr_arr = sr_arr[0:len(bin_means)]
    p_arr = p_arr[0:len(bin_means)]
    
    return bin_means, sr_arr, p_arr