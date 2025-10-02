import multiprocessing as mp
import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# =============================================================================
# ORIENTATION COMPUTATION FUNCTIONS
# =============================================================================
# These functions compute larval orientation based on tail-to-neck vector
# relative to the negative x-axis (flow direction in anemotaxis experiments).
# You can cehck xy_axis_characterisation.ipuynb
# 
# The orientation system:
# - 0° = larva facing downstream (-x direction) 
# - +90° = larva facing left (perpendicular to flow)
# - ±180° = larva facing upstream (+x direction)
# - -90° = larva facing right (perpendicular to flow)
# 
# This is particularly useful for anemotaxis analysis where the flow direction
# is typically along the negative x-axis, and we want to know if larvae are
# oriented with or against the flow.
# =============================================================================

def compute_orientation_tail_to_neck_wrt_negative_x(x_tail, y_tail, x_neck, y_neck):
    """
    Compute orientation angle between tail-to-neck vector and negative x-axis.
    Returns angle in degrees, where 0° = facing -x (downstream), ±180° = +x (upstream).
    """
    v_x = x_neck - x_tail
    v_y = y_neck - y_tail
    angle_rad = np.arctan2(v_y, -v_x)  # -v_x for -x axis
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 180) % 360 - 180
    return angle_deg

def get_larva_orientation_array(larva_data):
    """
    Always compute orientation as tail-to-neck relative to -x axis.
    Returns None if required keys are missing.
    """
    required_keys = ['x_tail', 'y_tail', 'x_neck', 'y_neck']
    if all(k in larva_data for k in required_keys):
        x_tail = np.array(larva_data['x_tail']).flatten()
        y_tail = np.array(larva_data['y_tail']).flatten()
        x_neck = np.array(larva_data['x_neck']).flatten()
        y_neck = np.array(larva_data['y_neck']).flatten()
        min_len = min(len(x_tail), len(y_tail), len(x_neck), len(y_neck))
        return compute_orientation_tail_to_neck_wrt_negative_x(
            x_tail[:min_len], y_tail[:min_len], x_neck[:min_len], y_neck[:min_len]
        )
    return None

### =============================================================================
### ORIENTATION ANALYSIS FUNCTIONS
### =============================================================================

def analyze_run_orientations(experiments_data, bin_width=10, sigma=2):
    """
    Analyze run orientations using tail-to-neck orientation definition with standard error across larvae.
    
    Args:
        experiments_data: Dictionary of larva data
        bin_width: Width of histogram bins in degrees
        sigma: Gaussian smoothing parameter
    
    Returns:
        Dictionary containing orientation data, histograms, and statistics
    """
    
    # Collect orientations per larva (not pooled across larvae)
    larva_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    # Extract run orientations for each larva separately
    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()

            # runs = np.logical_or(states == 1, states == 0.5)
            runs = states == 1 
             # Define runs as large runs (1) CHECK IF YOU WANT TO INCLUDE SMALL RUNS (0.5)
            if runs.sum() == 0:
                continue

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
                
            min_len = min(len(orientations), len(runs))
            orientations = orientations[:min_len]
            runs = runs[:min_len]
            
            larva_run_orientations = orientations[runs]
            if len(larva_run_orientations) > 0:
                larva_orientations.append(larva_run_orientations)
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_orientations:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': np.array([]),
            'n_larvae': 0
        }

    # Compute histograms for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    for larva_orients in larva_orientations:
        hist, _ = np.histogram(larva_orients, bins=bins, density=True)
        hist_smoothed = gaussian_filter1d(hist, sigma=sigma)
        hist_arrays.append(hist_smoothed)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae
    mean_hist = np.mean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

    return {
        'orientations': larva_orientations,  # List of arrays, one per larva
        'hist_arrays': hist_arrays,          # Individual histograms per larva
        'mean_hist': mean_hist,              # Mean across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations)
    }

def analyze_run_probability_by_orientation(experiments_data, bin_width=10, sigma=2):
    """
    Analyze run probability vs. orientation using tail-to-neck orientation.
    Returns analysis results that can be plotted with plot_orientation_histogram.
    """
    larva_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()

            is_run = states == 1 # Define runs as large runs (1) CHECK IF YOU WANT TO INCLUDE SMALL RUNS (0.5)

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
            min_len = min(len(orientations), len(is_run))
            orientations = orientations[:min_len]
            is_run = is_run[:min_len]
            
            # Store run probability data per larva
            larva_orientations.append({
                'orientations': orientations,
                'is_run': is_run
            })
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_orientations:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': np.array([]),
            'n_larvae': 0
        }

    # Compute histograms (run probabilities) for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    for larva_data in larva_orientations:
        orientations = larva_data['orientations']
        is_run = larva_data['is_run']
        
        run_probabilities = np.zeros_like(bin_centers, dtype=float)
        
        for i in range(len(bin_centers)):
            bin_mask = (orientations >= bins[i]) & (orientations < bins[i+1])
            if np.any(bin_mask):
                run_probabilities[i] = np.sum(is_run[bin_mask]) / np.sum(bin_mask)
        
        hist_smoothed = gaussian_filter1d(run_probabilities, sigma=sigma)
        hist_arrays.append(hist_smoothed)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae
    mean_hist = np.mean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

    return {
        'orientations': [data['orientations'][data['is_run']] for data in larva_orientations],  # Run orientations only
        'hist_arrays': hist_arrays,          # Individual run probability histograms per larva
        'mean_hist': mean_hist,              # Mean run probability across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations)
    }

def analyze_backup_probability_by_orientation(experiments_data, bin_width=10, sigma=2):
    """
    Analyze backup probability vs. orientation using tail-to-neck orientation.
    Returns analysis results that can be plotted with plot_orientation_histogram.
    """
    larva_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()

            is_backup = states == 5  # Define backups as backward runs (5)

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
            min_len = min(len(orientations), len(is_backup))
            orientations = orientations[:min_len]
            is_backup = is_backup[:min_len]
            
            # Store backup probability data per larva
            larva_orientations.append({
                'orientations': orientations,
                'is_backup': is_backup
            })
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_orientations:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': np.array([]),
            'n_larvae': 0
        }

    # Compute histograms (backup probabilities) for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    for larva_data in larva_orientations:
        orientations = larva_data['orientations']
        is_backup = larva_data['is_backup']
        
        backup_probabilities = np.zeros_like(bin_centers, dtype=float)
        
        for i in range(len(bin_centers)):
            bin_mask = (orientations >= bins[i]) & (orientations < bins[i+1])
            if np.any(bin_mask):
                backup_probabilities[i] = np.sum(is_backup[bin_mask]) / np.sum(bin_mask)
        
        hist_smoothed = gaussian_filter1d(backup_probabilities, sigma=sigma)
        hist_arrays.append(hist_smoothed)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae
    mean_hist = np.mean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

    return {
        'orientations': [data['orientations'][data['is_backup']] for data in larva_orientations],  # Backup orientations only
        'hist_arrays': hist_arrays,          # Individual backup probability histograms per larva
        'mean_hist': mean_hist,              # Mean backup probability across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations)
    }

def analyze_turn_rate_by_orientation(experiments_data, bin_width=10, sigma=3, min_turn_amplitude=45):
    """
    Analyze turn rate vs. orientation using tail-to-neck orientation.
    A turn is defined as a cast (state==2 or state==1.5, but check) that results in an orientation change >= min_turn_amplitude (deg).
    The orientation at which the turn occurs is the *initial* orientation at cast onset.
    Returns analysis results that can be plotted with plot_orientation_histogram.
    """
    larva_data_list = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data or 't' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            t = np.array(larva_data['t']).flatten()

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
                
            min_len = min(len(orientations), len(states), len(t))
            orientations = orientations[:min_len]
            states = states[:min_len]
            t = t[:min_len]

            # Store data for this larva
            larva_data_list.append({
                'orientations': orientations,
                'states': states,
                't': t,
                'min_turn_amplitude': min_turn_amplitude
            })
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_data_list:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': np.array([]),
            'n_larvae': 0
        }

    # Compute turn rates for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    turn_orientations_list = []
    
    for larva_data in larva_data_list:
        orientations = larva_data['orientations']
        states = larva_data['states']
        t = larva_data['t']
        min_turn_amplitude = larva_data['min_turn_amplitude']
        
        # Initialize turn rate array for this larva
        turn_counts = np.zeros_like(bin_centers, dtype=float)
        total_time = np.zeros_like(bin_centers, dtype=float)
        turn_orientations = []

        # Find all cast (state==2 or state==1.5) segments and detect turns
        i = 0
        while i < len(states):
            if states[i] == 2: # or states[i] == 1.5: # Define casts as large casts (2) CHECK IF YOU WANT TO INCLUDE SMALL CASTS (1.5)
                cast_start = i
                while i < len(states) and (states[i] == 2 or states[i] == 1.5):
                    i += 1
                cast_end = i - 1
                
                # Only consider casts with at least 2 frames
                if cast_end > cast_start:
                    orient_start = orientations[cast_start]
                    orient_end = orientations[cast_end]
                    
                    # Compute orientation change (handle wraparound)
                    delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                    
                    if np.abs(delta) >= min_turn_amplitude:
                        # This is a valid turn - record the initial orientation
                        turn_orientations.append(orient_start)
                        
                        # Bin by initial orientation
                        for j in range(len(bin_centers)):
                            if bins[j] <= orient_start < bins[j+1]:
                                turn_counts[j] += 1
                                break
            else:
                i += 1

        # Calculate total time spent in each orientation bin (all frames)
        for j in range(len(bin_centers)):
            bin_mask = (orientations >= bins[j]) & (orientations < bins[j+1])
            t_bin = t[bin_mask]
            if len(t_bin) > 1:
                total_time[j] += np.sum(np.diff(t_bin))
            elif len(t_bin) == 1 and len(t) > 1:
                dt = np.median(np.diff(t))
                total_time[j] += dt

        # Compute turn rate per second for this larva
        turn_rate_per_sec = np.zeros_like(bin_centers, dtype=float)
        for j in range(len(bin_centers)):
            if total_time[j] > 0:
                turn_rate_per_sec[j] = turn_counts[j] / total_time[j]

        # Apply smoothing
        hist_smoothed = gaussian_filter1d(turn_rate_per_sec, sigma=sigma)
        hist_arrays.append(hist_smoothed)
        turn_orientations_list.append(turn_orientations)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae
    mean_hist = np.mean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

    return {
        'orientations': turn_orientations_list,  # Turn orientations only (list of arrays per larva)
        'hist_arrays': hist_arrays,              # Individual turn rate histograms per larva
        'mean_hist': mean_hist,                  # Mean turn rate across larvae
        'se_hist': se_hist,                      # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_data_list)
    }

### The following is a calculation of turn probability that marks all frames in a turning cast as "turn" frames.
### This is consistent with how run and backup probabilities are calculated.

def analyze_turn_probability_by_orientation(experiments_data, bin_width=10, sigma=2, min_turn_amplitude=45):
    """
    Analyze turn probability vs. orientation using tail-to-neck orientation.
    A turn is defined as a cast that results in an orientation change >= min_turn_amplitude (deg).
    Returns analysis results that can be plotted with plot_orientation_histogram.
    """
    larva_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
                
            min_len = min(len(orientations), len(states))
            orientations = orientations[:min_len]
            states = states[:min_len]

            # Identify turns: casts that result in significant orientation change
            is_turn = np.zeros(len(states), dtype=bool)
            
            i = 0
            while i < len(states):
                if states[i] == 2:  # Large cast
                    cast_start = i
                    while i < len(states) and states[i] == 2:
                        i += 1
                    cast_end = i - 1
                    
                    # Check if this cast resulted in a turn
                    if cast_end > cast_start:
                        orient_start = orientations[cast_start]
                        orient_end = orientations[cast_end]
                        
                        # Compute orientation change (handle wraparound)
                        delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                        
                        if np.abs(delta) >= min_turn_amplitude:
                            # Mark all frames in this cast as turn frames
                            is_turn[cast_start:cast_end+1] = True
                else:
                    i += 1

            # Store turn probability data per larva
            larva_orientations.append({
                'orientations': orientations,
                'is_turn': is_turn
            })
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_orientations:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': np.array([]),
            'n_larvae': 0
        }

    # Compute histograms (turn probabilities) for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    for larva_data in larva_orientations:
        orientations = larva_data['orientations']
        is_turn = larva_data['is_turn']
        
        turn_probabilities = np.zeros_like(bin_centers, dtype=float)
        
        for i in range(len(bin_centers)):
            bin_mask = (orientations >= bins[i]) & (orientations < bins[i+1])
            if np.any(bin_mask):
                turn_probabilities[i] = np.sum(is_turn[bin_mask]) / np.sum(bin_mask)
        
        hist_smoothed = gaussian_filter1d(turn_probabilities, sigma=sigma)
        hist_arrays.append(hist_smoothed)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae
    mean_hist = np.mean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

    return {
        'orientations': [data['orientations'][data['is_turn']] for data in larva_orientations],  # Turn orientations only
        'hist_arrays': hist_arrays,          # Individual turn probability histograms per larva
        'mean_hist': mean_hist,              # Mean turn probability across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations)
    }

## The following definition of turn probability marks only the initial frame of a turning cast as a "turn" frame.
## This is not consistent with how run and backup probabilities are calculated, so it is commented out

# def analyze_turn_probability_by_orientation(experiments_data, bin_width=10, sigma=2, min_turn_amplitude=45):
#     """
#     Analyze turn probability vs. orientation using tail-to-neck orientation.
#     A turn is defined as a cast that results in an orientation change >= min_turn_amplitude (deg).
#     Returns analysis results that can be plotted with plot_orientation_histogram.
#     """
#     larva_orientations = []

#     if isinstance(experiments_data, dict) and 'data' in experiments_data:
#         data_to_process = experiments_data['data']
#     else:
#         data_to_process = experiments_data

#     for larva_id, larva_data in data_to_process.items():
#         try:
#             if 'global_state_small_large_state' not in larva_data:
#                 continue
#             states = np.array(larva_data['global_state_small_large_state']).flatten()

#             orientations = get_larva_orientation_array(larva_data)
#             if orientations is None:
#                 continue
                
#             min_len = min(len(orientations), len(states))
#             orientations = orientations[:min_len]
#             states = states[:min_len]

#             # Find turns (cast events that result in significant orientation change)
#             turn_orientations = []
#             cast_orientations = []
            
#             i = 0
#             while i < len(states):
#                 if states[i] == 2:  # Large cast
#                     cast_start = i
#                     while i < len(states) and states[i] == 2:
#                         i += 1
#                     cast_end = i - 1
                    
#                     # Record the initial orientation of this cast
#                     cast_orientations.append(orientations[cast_start])
                    
#                     # Check if this cast resulted in a turn
#                     if cast_end > cast_start:
#                         orient_start = orientations[cast_start]
#                         orient_end = orientations[cast_end]
                        
#                         # Compute orientation change (handle wraparound)
#                         delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                        
#                         if np.abs(delta) >= min_turn_amplitude:
#                             # This cast resulted in a turn
#                             turn_orientations.append(orient_start)
#                 else:
#                     i += 1

#             # Store turn probability data per larva
#             larva_orientations.append({
#                 'cast_orientations': cast_orientations,  # All cast orientations
#                 'turn_orientations': turn_orientations   # Only turning cast orientations
#             })
                
#         except Exception as e:
#             print(f"Error processing larva {larva_id}: {str(e)}")

#     if not larva_orientations:
#         return {
#             'orientations': [], 
#             'hist_arrays': np.array([]), 
#             'mean_hist': np.array([]), 
#             'se_hist': np.array([]), 
#             'bin_centers': np.array([]),
#             'n_larvae': 0
#         }

#     # Compute histograms (turn probabilities) for each larva
#     bins = np.arange(-180, 181, bin_width)
#     bin_centers = (bins[:-1] + bins[1:]) / 2
    
#     hist_arrays = []
#     for larva_data in larva_orientations:
#         cast_orientations = larva_data['cast_orientations']
#         turn_orientations = larva_data['turn_orientations']
        
#         turn_probabilities = np.zeros_like(bin_centers, dtype=float)
        
#         for i in range(len(bin_centers)):
#             # Count casts in this orientation bin
#             cast_count = sum(1 for orient in cast_orientations 
#                            if bins[i] <= orient < bins[i+1])
            
#             # Count turns in this orientation bin  
#             turn_count = sum(1 for orient in turn_orientations 
#                            if bins[i] <= orient < bins[i+1])
            
#             if cast_count > 0:
#                 turn_probabilities[i] = turn_count / cast_count
        
#         hist_smoothed = gaussian_filter1d(turn_probabilities, sigma=sigma)
#         hist_arrays.append(hist_smoothed)
    
#     # Convert to array for easier computation
#     hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
#     # Compute mean and standard error across larvae
#     mean_hist = np.mean(hist_arrays, axis=0)
#     se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

#     return {
#         'orientations': [data['turn_orientations'] for data in larva_orientations],  # Turn orientations only
#         'hist_arrays': hist_arrays,          # Individual turn probability histograms per larva
#         'mean_hist': mean_hist,              # Mean turn probability across larvae
#         'se_hist': se_hist,                  # Standard error across larvae
#         'bin_centers': bin_centers,
#         'n_larvae': len(larva_orientations)
#     }
def analyze_cast_probability_by_orientation(experiments_data, bin_width=10, sigma=2):
    """
    Analyze cast probability vs. orientation using tail-to-neck orientation.
    Returns analysis results that can be plotted with plot_orientation_histogram.
    """
    larva_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()

            is_cast = states == 2  # Define casts as large casts (2) - CHECK IF YOU WANT TO INCLUDE SMALL CASTS (1.5)

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
            min_len = min(len(orientations), len(is_cast))
            orientations = orientations[:min_len]
            is_cast = is_cast[:min_len]
            
            # Store cast probability data per larva
            larva_orientations.append({
                'orientations': orientations,
                'is_cast': is_cast
            })
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_orientations:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': np.array([]),
            'n_larvae': 0
        }

    # Compute histograms (cast probabilities) for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    for larva_data in larva_orientations:
        orientations = larva_data['orientations']
        is_cast = larva_data['is_cast']
        
        cast_probabilities = np.zeros_like(bin_centers, dtype=float)
        
        for i in range(len(bin_centers)):
            bin_mask = (orientations >= bins[i]) & (orientations < bins[i+1])
            if np.any(bin_mask):
                cast_probabilities[i] = np.sum(is_cast[bin_mask]) / np.sum(bin_mask)
        
        hist_smoothed = gaussian_filter1d(cast_probabilities, sigma=sigma)
        hist_arrays.append(hist_smoothed)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae
    mean_hist = np.mean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0)  # Standard error of the mean

    return {
        'orientations': [data['orientations'][data['is_cast']] for data in larva_orientations],  # Cast orientations only
        'hist_arrays': hist_arrays,          # Individual cast probability histograms per larva
        'mean_hist': mean_hist,              # Mean cast probability across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations)
    }

def analyze_turn_amplitudes_by_orientation(experiments_data, bin_width=10, sigma=1, min_turn_amplitude=60):
    """
    Analyze turn amplitudes vs. orientation using tail-to-neck orientation.
    A turn is defined as a cast that results in an orientation change >= min_turn_amplitude (deg).
    Amplitude is defined as the absolute start-to-end orientation change.
    Returns analysis results that can be plotted with plot_orientation_histogram.
    """
    larva_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data
    
    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
                
            min_len = min(len(orientations), len(states))
            orientations = orientations[:min_len]
            states = states[:min_len]

            # Find turns and their amplitudes
            turn_data = []
            
            i = 0
            while i < len(states):
                if states[i] == 2:  # Large cast
                    cast_start = i
                    while i < len(states) and states[i] == 2:
                        i += 1
                    cast_end = i - 1
                    
                    # Check if this cast resulted in a turn
                    if cast_end > cast_start:
                        orient_start = orientations[cast_start]
                        orient_end = orientations[cast_end]
                        
                        # Compute orientation change (handle wraparound)
                        delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                        
                        if np.abs(delta) >= min_turn_amplitude:
                            amplitude = np.abs(delta)
                            
                            turn_data.append({
                                'start_orientation': orient_start,
                                'amplitude': amplitude
                            })
                else:
                    i += 1

            # Store turn amplitude data per larva
            larva_orientations.append({
                'turn_data': turn_data
            })
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_orientations:
        bins = np.arange(-180, 181, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.full_like(bin_centers, np.nan), 
            'se_hist': np.full_like(bin_centers, np.nan), 
            'bin_centers': bin_centers,
            'n_larvae': 0
        }

    # Compute amplitude histograms for each larva
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    hist_arrays = []
    turn_orientations_list = []
    
    for larva_data in larva_orientations:
        turn_data = larva_data['turn_data']
        
        # Initialize amplitude array for this larva with NaN
        mean_amplitudes = np.full_like(bin_centers, np.nan, dtype=float)
        turn_orientations = []
        
        # Group turns by orientation bin for this larva
        bin_amplitudes = {i: [] for i in range(len(bin_centers))}
        
        for turn in turn_data:
            start_orient = turn['start_orientation']
            amplitude = turn['amplitude']
            turn_orientations.append(start_orient)
            
            # Find the correct bin
            for j in range(len(bin_centers)):
                if bins[j] <= start_orient < bins[j+1]:
                    bin_amplitudes[j].append(amplitude)
                    break
        
        # Calculate mean amplitude per bin for this larva (only if data exists)
        for j in range(len(bin_centers)):
            if bin_amplitudes[j]:  # Only if there are turns in this bin
                mean_amplitudes[j] = np.mean(bin_amplitudes[j])
            # else: leave as NaN

        # Apply smoothing only to non-NaN values
        if sigma > 0:
            # Create a mask for valid (non-NaN) values
            valid_mask = ~np.isnan(mean_amplitudes)
            if np.sum(valid_mask) > 1:  # Need at least 2 points to smooth
                # Create a temporary array for smoothing
                temp_amplitudes = np.copy(mean_amplitudes)
                # Interpolate NaN values for smoothing (simple approach)
                if np.sum(valid_mask) < len(mean_amplitudes):
                    # Fill NaN with interpolated values for smoothing
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = mean_amplitudes[valid_mask]
                    temp_amplitudes = np.interp(
                        np.arange(len(bin_centers)), 
                        valid_indices, 
                        valid_values
                    )
                
                # Apply smoothing
                hist_smoothed = gaussian_filter1d(temp_amplitudes, sigma=sigma)
                
                # Restore NaN values where original data was NaN
                hist_smoothed[~valid_mask] = np.nan
            else:
                hist_smoothed = mean_amplitudes
        else:
            hist_smoothed = mean_amplitudes

        hist_arrays.append(hist_smoothed)
        turn_orientations_list.append(turn_orientations)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae using nanmean and nanstd
    mean_hist = np.nanmean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0, nan_policy='omit')
    
    # Count number of larvae contributing to each bin
    n_larvae_per_bin = np.sum(~np.isnan(hist_arrays), axis=0)

    return {
        'orientations': turn_orientations_list,  # Turn orientations only (list of arrays per larva)
        'hist_arrays': hist_arrays,              # Individual turn amplitude histograms per larva (with NaNs)
        'mean_hist': mean_hist,                  # Mean turn amplitude across larvae (NaN where no data)
        'se_hist': se_hist,                      # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations),
        'n_larvae_per_bin': n_larvae_per_bin     # Number of larvae contributing to each bin
    }

def analyze_run_velocity_by_orientation(experiments_data, bin_width=10, sigma=2):
    """
    Analyze velocity by orientation ONLY during run states (states == 1.0 or 0.5).
    Uses tail-to-neck orientation and motion_velocity_norm_smooth_5.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        bin_width: Bin width in degrees for orientation binning
        sigma: Gaussian smoothing parameter
        
    Returns:
        Dictionary with analysis results compatible with plot_orientation_histogram
    """
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    larva_velocity_data = []

    for larva_id, larva_data in data_to_process.items():
        try:
            # Check for required data
            if 'global_state_large_state' not in larva_data:
                continue
                
            states = np.array(larva_data['global_state_large_state']).flatten()
            
            # Filter for RUN states only (1.0 = large run, 0.5 = small run)
            is_run = np.logical_or(states == 1.0, states == 0.5)
            if not np.any(is_run):
                continue

            # Get velocity data
            if 'motion_velocity_norm_smooth_5' not in larva_data:
                continue
                
            velocity = np.array(larva_data['motion_velocity_norm_smooth_5']).flatten()

            # Get orientations using tail-to-neck
            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue

            # Ensure all arrays have the same length
            min_len = min(len(orientations), len(is_run), len(velocity))
            if min_len <= 1:
                continue

            orientations = orientations[:min_len]
            is_run = is_run[:min_len]
            velocity = velocity[:min_len]

            # Extract run orientations and velocities
            run_orientations = orientations[is_run]
            run_velocities = velocity[is_run]

            # Store data for this larva
            larva_velocity_data.append({
                'orientations': run_orientations,
                'velocities': run_velocities
            })
                    
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not larva_velocity_data:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': bin_centers,
            'n_larvae': 0
        }

    # Calculate mean velocities per bin for each larva
    hist_arrays = []
    
    for larva_data in larva_velocity_data:
        run_orientations = larva_data['orientations']
        run_velocities = larva_data['velocities']
        
        velocity_means = np.full_like(bin_centers, np.nan, dtype=float)
        
        # Bin velocities by orientation for this larva
        for i, center in enumerate(bin_centers):
            bin_mask = (run_orientations >= bins[i]) & (run_orientations < bins[i+1])
            if np.any(bin_mask):
                velocity_means[i] = np.mean(run_velocities[bin_mask])
        
        # Apply Gaussian smoothing (handle NaNs)
        if sigma > 0:
            valid_mask = ~np.isnan(velocity_means)
            if np.sum(valid_mask) > 1:
                # Interpolate NaN values for smoothing
                velocity_means_interp = np.copy(velocity_means)
                if np.sum(valid_mask) < len(velocity_means):
                    valid_indices = np.where(valid_mask)[0]
                    valid_values = velocity_means[valid_mask]
                    velocity_means_interp = np.interp(
                        np.arange(len(bin_centers)), 
                        valid_indices, 
                        valid_values
                    )
                
                # Apply smoothing
                velocity_smooth = gaussian_filter1d(velocity_means_interp, sigma=sigma)
                
                # Restore NaN values where original data was NaN
                velocity_smooth[~valid_mask] = np.nan
            else:
                velocity_smooth = velocity_means
        else:
            velocity_smooth = velocity_means
        
        hist_arrays.append(velocity_smooth)
    
    # Convert to array for easier computation
    hist_arrays = np.array(hist_arrays)  # Shape: (n_larvae, n_bins)
    
    # Compute mean and standard error across larvae using nanmean and nanstd
    mean_hist = np.nanmean(hist_arrays, axis=0)
    se_hist = stats.sem(hist_arrays, axis=0, nan_policy='omit')

    return {
        'orientations': [data['orientations'] for data in larva_velocity_data],  # Run orientations per larva
        'hist_arrays': hist_arrays,          # Individual velocity histograms per larva
        'mean_hist': mean_hist,              # Mean run velocity across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_velocity_data)
    }


def analyze_head_casts_by_orientation(experiments_data, bin_width=20, peak_threshold=5.0, peak_prominence=3.0, smooth_sigma=4.0, large_casts_only=True, jump_threshold=15):
    """
    Analyze head cast frequency as a function of larva orientation at the beginning of cast events.
    Now uses upstream/downstream classification.
    """
    # Detect all cast events with head cast counts
    cast_events_data = detect_head_casts_in_casts(
        experiments_data, 
        peak_threshold=peak_threshold, 
        peak_prominence=peak_prominence,
        smooth_sigma=smooth_sigma,
        large_casts_only=large_casts_only,
        jump_threshold=jump_threshold
    )
    
    # Set up orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Collect data per larva for statistical analysis
    larva_data_list = []
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        # Count total head casts per orientation bin based on cast start orientation
        head_cast_counts = np.zeros_like(bin_centers)
        
        for cast_event in cast_events:
            cast_start_orientation = cast_event['cast_start_orientation']
            total_head_casts = cast_event['total_head_casts']
            
            # Find the correct bin for the cast start orientation
            for i, center in enumerate(bin_centers):
                if bins[i] <= cast_start_orientation < bins[i+1]:
                    head_cast_counts[i] += total_head_casts
                    break
        
        larva_data_list.append({
            'head_cast_counts': head_cast_counts,
            'larva_id': larva_id
        })
    
    if not larva_data_list:
        return {
            'orientations': [],
            'hist_arrays': np.array([]),
            'mean_hist': np.full_like(bin_centers, np.nan),
            'se_hist': np.full_like(bin_centers, np.nan),
            'bin_centers': bin_centers,
            'n_larvae': 0,
            'cast_events_data': cast_events_data
        }
    
    # Convert to arrays for analysis
    head_cast_arrays = np.array([data['head_cast_counts'] for data in larva_data_list])
    
    # Compute means and standard errors
    mean_head_casts = np.nanmean(head_cast_arrays, axis=0)
    se_head_casts = stats.sem(head_cast_arrays, axis=0, nan_policy='omit')
    
    return {
        'orientations': [[] for _ in larva_data_list],
        'hist_arrays': head_cast_arrays,
        'mean_hist': mean_head_casts,
        'se_hist': se_head_casts,
        'bin_centers': bin_centers,
        'n_larvae': len(larva_data_list),
        'cast_events_data': cast_events_data
    }
### =============================================================================
######## ANALYSIS OVER TIME ########
### =============================================================================

def analyze_run_probability_over_time(experiments_data, window=20, step=10):
    """
    Analyze run probability over time for each larva individually, then compute mean and SEM.
    
    Args:
        trx_data: dict of larvae data (or {'data': ...})
        window: window size in seconds for probability calculation
        step: step size in seconds for moving window
        
    Returns:
        dict with 'time_centers', 'metric_arrays', 'mean_metric', 'se_metric', 'n_larvae'
    """
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    # Find global time range
    all_times = []
    for larva in data_to_process.values():
        if 't' in larva:
            t_arr = np.array(larva['t']).flatten()
            all_times.extend(t_arr.tolist())
    
    if not all_times:
        return {
            'time_centers': np.array([]),
            'metric_arrays': np.array([]),
            'mean_metric': np.array([]),
            'se_metric': np.array([]),
            'n_larvae': 0
        }
    
    t_min, t_max = np.min(all_times), np.max(all_times)
    time_bins = np.arange(t_min, t_max - window + step, step)
    time_centers = time_bins + window/2
    
    # Compute run probability for each larva
    metric_arrays = []
    
    for larva_id, larva in data_to_process.items():
        if 't' not in larva or 'global_state_large_state' not in larva:
            continue
            
        t = np.array(larva['t']).flatten()
        states = np.array(larva['global_state_large_state']).flatten()
        
        # Initialize run probabilities for this larva
        run_probs = np.full_like(time_bins, np.nan, dtype=float)
        
        for i, t0 in enumerate(time_bins):
            t1 = t0 + window
            mask = (t >= t0) & (t < t1)
            
            if np.sum(mask) < 1:
                continue
                
            states_win = states[mask]
            n_run = np.sum(states_win == 1)
            n_total = len(states_win)
            
            if n_total > 0:
                run_probs[i] = n_run / n_total
        
        metric_arrays.append(run_probs)
    
    if not metric_arrays:
        return {
            'time_centers': time_centers,
            'metric_arrays': np.array([]),
            'mean_metric': np.full_like(time_centers, np.nan),
            'se_metric': np.full_like(time_centers, np.nan),
            'n_larvae': 0
        }
    
    # Convert to array for easier computation
    metric_arrays = np.array(metric_arrays)  # Shape: (n_larvae, n_time_bins)
    
    # Compute mean and standard error across larvae
    mean_metric = np.nanmean(metric_arrays, axis=0)
    se_metric = stats.sem(metric_arrays, axis=0, nan_policy='omit')
    
    return {
        'time_centers': time_centers,
        'metric_arrays': metric_arrays,
        'mean_metric': mean_metric,
        'se_metric': se_metric,
        'n_larvae': len(metric_arrays)
    }

def analyze_turn_probability_over_time(experiments_data, window=60, step=10, min_turn_amplitude=60):
    """
    Analyze turn probability over time for each larva individually, then compute mean and SEM.
    A turn is defined as a cast that results in an orientation change >= min_turn_amplitude (deg).
    
    Args:
        experiments_data: dict of larvae data (or {'data': ...})
        window: window size in seconds for probability calculation
        step: step size in seconds for moving window
        min_turn_amplitude: minimum orientation change in degrees to qualify as a turn
        
    Returns:
        dict with 'time_centers', 'metric_arrays', 'mean_metric', 'se_metric', 'n_larvae'
    """
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    # Find global time range
    all_times = []
    for larva in data_to_process.values():
        if 't' in larva:
            t_arr = np.array(larva['t']).flatten()
            all_times.extend(t_arr.tolist())
    
    if not all_times:
        return {
            'time_centers': np.array([]),
            'metric_arrays': np.array([]),
            'mean_metric': np.array([]),
            'se_metric': np.array([]),
            'n_larvae': 0
        }
    
    t_min, t_max = np.min(all_times), np.max(all_times)
    time_bins = np.arange(t_min, t_max - window + step, step)
    time_centers = time_bins + window/2
    
    # Compute turn probability for each larva
    metric_arrays = []
    
    for larva_id, larva in data_to_process.items():
        if 't' not in larva or 'global_state_small_large_state' not in larva:
            continue
            
        t = np.array(larva['t']).flatten()
        states = np.array(larva['global_state_small_large_state']).flatten()
        
        # Get orientations for this larva
        orientations = get_larva_orientation_array(larva)
        if orientations is None:
            continue
            
        min_len = min(len(orientations), len(states), len(t))
        orientations = orientations[:min_len]
        states = states[:min_len]
        t = t[:min_len]
        
        # Identify turns: casts that result in significant orientation change
        is_turn = np.zeros(len(states), dtype=bool)
        
        i = 0
        while i < len(states):
            if states[i] == 2:  # Large cast
                cast_start = i
                while i < len(states) and states[i] == 2:
                    i += 1
                cast_end = i - 1
                
                # Check if this cast resulted in a turn
                if cast_end > cast_start:
                    orient_start = orientations[cast_start]
                    orient_end = orientations[cast_end]
                    
                    # Compute orientation change (handle wraparound)
                    delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                    
                    if np.abs(delta) >= min_turn_amplitude:
                        # Mark all frames in this cast as turn frames
                        is_turn[cast_start:cast_end+1] = True
            else:
                i += 1
        
        # Initialize turn probabilities for this larva
        turn_probs = np.full_like(time_bins, np.nan, dtype=float)
        
        for i, t0 in enumerate(time_bins):
            t1 = t0 + window
            mask = (t >= t0) & (t < t1)
            
            if np.sum(mask) < 1:
                continue
                
            is_turn_win = is_turn[mask]
            n_turn = np.sum(is_turn_win)
            n_total = len(is_turn_win)
            
            if n_total > 0:
                turn_probs[i] = n_turn / n_total
        
        metric_arrays.append(turn_probs)
    
    if not metric_arrays:
        return {
            'time_centers': time_centers,
            'metric_arrays': np.array([]),
            'mean_metric': np.full_like(time_centers, np.nan),
            'se_metric': np.full_like(time_centers, np.nan),
            'n_larvae': 0
        }
    
    # Convert to array for easier computation
    metric_arrays = np.array(metric_arrays)  # Shape: (n_larvae, n_time_bins)
    
    # Compute mean and standard error across larvae
    mean_metric = np.nanmean(metric_arrays, axis=0)
    se_metric = stats.sem(metric_arrays, axis=0, nan_policy='omit')
    
    return {
        'time_centers': time_centers,
        'metric_arrays': metric_arrays,
        'mean_metric': mean_metric,
        'se_metric': se_metric,
        'n_larvae': len(metric_arrays)
    }

def analyze_backup_probability_over_time(experiments_data, window=60, step=10):
    """
    Analyze backup probability over time for each larva individually, then compute mean and SEM.
    
    Args:
        experiments_data: dict of larvae data (or {'data': ...})
        window: window size in seconds for probability calculation
        step: step size in seconds for moving window
        
    Returns:
        dict with 'time_centers', 'metric_arrays', 'mean_metric', 'se_metric', 'n_larvae'
    """
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    # Find global time range
    all_times = []
    for larva in data_to_process.values():
        if 't' in larva:
            t_arr = np.array(larva['t']).flatten()
            all_times.extend(t_arr.tolist())
    
    if not all_times:
        return {
            'time_centers': np.array([]),
            'metric_arrays': np.array([]),
            'mean_metric': np.array([]),
            'se_metric': np.array([]),
            'n_larvae': 0
        }
    
    t_min, t_max = np.min(all_times), np.max(all_times)
    time_bins = np.arange(t_min, t_max - window + step, step)
    time_centers = time_bins + window/2
    
    # Compute backup probability for each larva
    metric_arrays = []
    
    for larva_id, larva in data_to_process.items():
        if 't' not in larva or 'global_state_small_large_state' not in larva:
            continue
            
        t = np.array(larva['t']).flatten()
        states = np.array(larva['global_state_small_large_state']).flatten()
        
        # Initialize backup probabilities for this larva
        backup_probs = np.full_like(time_bins, np.nan, dtype=float)
        
        for i, t0 in enumerate(time_bins):
            t1 = t0 + window
            mask = (t >= t0) & (t < t1)
            
            if np.sum(mask) < 1:
                continue
                
            states_win = states[mask]
            n_backup = np.sum(states_win == 5)  # State 5 = backup
            n_total = len(states_win)
            
            if n_total > 0:
                backup_probs[i] = n_backup / n_total
        
        metric_arrays.append(backup_probs)
    
    if not metric_arrays:
        return {
            'time_centers': time_centers,
            'metric_arrays': np.array([]),
            'mean_metric': np.full_like(time_centers, np.nan),
            'se_metric': np.full_like(time_centers, np.nan),
            'n_larvae': 0
        }
    
    # Convert to array for easier computation
    metric_arrays = np.array(metric_arrays)  # Shape: (n_larvae, n_time_bins)
    
    # Compute mean and standard error across larvae
    mean_metric = np.nanmean(metric_arrays, axis=0)
    se_metric = stats.sem(metric_arrays, axis=0, nan_policy='omit')
    
    return {
        'time_centers': time_centers,
        'metric_arrays': metric_arrays,
        'mean_metric': mean_metric,
        'se_metric': se_metric,
        'n_larvae': len(metric_arrays)
    }

def analyze_turn_amplitudes_over_time(experiments_data, window=60, step=10, min_turn_amplitude=45):
    """
    Analyze turn amplitudes over time for each larva individually, then compute mean and SEM.
    A turn is defined as a cast that results in an orientation change >= min_turn_amplitude (deg).
    Amplitude is defined as the absolute start-to-end orientation change.
    
    Args:
        experiments_data: dict of larvae data (or {'data': ...})
        window: window size in seconds for amplitude calculation
        step: step size in seconds for moving window
        min_turn_amplitude: minimum orientation change in degrees to qualify as a turn
        
    Returns:
        dict with 'time_centers', 'metric_arrays', 'mean_metric', 'se_metric', 'n_larvae'
    """
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    # Find global time range
    all_times = []
    for larva in data_to_process.values():
        if 't' in larva:
            t_arr = np.array(larva['t']).flatten()
            all_times.extend(t_arr.tolist())
    
    if not all_times:
        return {
            'time_centers': np.array([]),
            'metric_arrays': np.array([]),
            'mean_metric': np.array([]),
            'se_metric': np.array([]),
            'n_larvae': 0
        }
    
    t_min, t_max = np.min(all_times), np.max(all_times)
    time_bins = np.arange(t_min, t_max - window + step, step)
    time_centers = time_bins + window/2
    
    # Compute turn amplitudes for each larva
    metric_arrays = []
    
    for larva_id, larva in data_to_process.items():
        if 't' not in larva or 'global_state_small_large_state' not in larva:
            continue
            
        t = np.array(larva['t']).flatten()
        states = np.array(larva['global_state_small_large_state']).flatten()
        
        # Get orientations for this larva
        orientations = get_larva_orientation_array(larva)
        if orientations is None:
            continue
            
        min_len = min(len(orientations), len(states), len(t))
        orientations = orientations[:min_len]
        states = states[:min_len]
        t = t[:min_len]
        
        # Find turns and their amplitudes with timestamps
        turn_data = []
        
        i = 0
        while i < len(states):
            if states[i] == 2:  # Large cast
                cast_start = i
                while i < len(states) and states[i] == 2:
                    i += 1
                cast_end = i - 1
                
                # Check if this cast resulted in a turn
                if cast_end > cast_start:
                    orient_start = orientations[cast_start]
                    orient_end = orientations[cast_end]
                    
                    # Compute orientation change (handle wraparound)
                    delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                    
                    if np.abs(delta) >= min_turn_amplitude:
                        # Amplitude is simply the absolute start-to-end orientation change
                        amplitude = np.abs(delta)
                        
                        turn_data.append({
                            'time': t[cast_start],
                            'amplitude': amplitude
                        })
            else:
                i += 1
        
        # Initialize turn amplitudes for this larva
        turn_amplitudes = np.full_like(time_bins, np.nan, dtype=float)
        
        for i, t0 in enumerate(time_bins):
            t1 = t0 + window
            
            # Find turns within this time window
            window_amplitudes = []
            for turn in turn_data:
                if t0 <= turn['time'] < t1:
                    window_amplitudes.append(turn['amplitude'])
            
            if window_amplitudes:
                turn_amplitudes[i] = np.mean(window_amplitudes)
        
        metric_arrays.append(turn_amplitudes)
    
    if not metric_arrays:
        return {
            'time_centers': time_centers,
            'metric_arrays': np.array([]),
            'mean_metric': np.full_like(time_centers, np.nan),
            'se_metric': np.full_like(time_centers, np.nan),
            'n_larvae': 0
        }
    
    # Convert to array for easier computation
    metric_arrays = np.array(metric_arrays)  # Shape: (n_larvae, n_time_bins)
    
    # Compute mean and standard error across larvae
    mean_metric = np.nanmean(metric_arrays, axis=0)
    se_metric = stats.sem(metric_arrays, axis=0, nan_policy='omit')
    
    return {
        'time_centers': time_centers,
        'metric_arrays': metric_arrays,
        'mean_metric': mean_metric,
        'se_metric': se_metric,
        'n_larvae': len(metric_arrays)
    }

def analyze_run_velocity_over_time(experiments_data, window=60, step=10):
    """
    Analyze run velocity over time for each larva individually, then compute mean and SEM.
    Only considers velocity during run states (1.0 and 0.5).
    
    Args:
        experiments_data: dict of larvae data (or {'data': ...})
        window: window size in seconds for velocity calculation
        step: step size in seconds for moving window
        
    Returns:
        dict with 'time_centers', 'metric_arrays', 'mean_metric', 'se_metric', 'n_larvae'
    """
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    # Find global time range
    all_times = []
    for larva in data_to_process.values():
        if 't' in larva:
            t_arr = np.array(larva['t']).flatten()
            all_times.extend(t_arr.tolist())
    
    if not all_times:
        return {
            'time_centers': np.array([]),
            'metric_arrays': np.array([]),
            'mean_metric': np.array([]),
            'se_metric': np.array([]),
            'n_larvae': 0
        }
    
    t_min, t_max = np.min(all_times), np.max(all_times)
    time_bins = np.arange(t_min, t_max - window + step, step)
    time_centers = time_bins + window/2
    
    # Compute run velocity for each larva
    metric_arrays = []
    
    for larva_id, larva in data_to_process.items():
        if ('t' not in larva or 
            'global_state_large_state' not in larva or 
            'motion_velocity_norm_smooth_5' not in larva):
            continue
            
        t = np.array(larva['t']).flatten()
        states = np.array(larva['global_state_large_state']).flatten()
        velocity = np.array(larva['motion_velocity_norm_smooth_5']).flatten()
        
        # Ensure all arrays have same length
        min_len = min(len(t), len(states), len(velocity))
        t = t[:min_len]
        states = states[:min_len]
        velocity = velocity[:min_len]
        
        # Filter for run states
        is_run = np.logical_or(states == 1.0, states == 0.5)
        
        # Initialize run velocities for this larva
        run_velocities = np.full_like(time_bins, np.nan, dtype=float)
        
        for i, t0 in enumerate(time_bins):
            t1 = t0 + window
            mask = (t >= t0) & (t < t1)
            
            if np.sum(mask) < 1:
                continue
                
            # Get run states and velocities in this window
            is_run_win = is_run[mask]
            velocity_win = velocity[mask]
            
            # Only consider velocities during run states
            run_velocity_win = velocity_win[is_run_win]
            
            if len(run_velocity_win) > 0:
                run_velocities[i] = np.mean(run_velocity_win)
        
        metric_arrays.append(run_velocities)
    
    if not metric_arrays:
        return {
            'time_centers': time_centers,
            'metric_arrays': np.array([]),
            'mean_metric': np.full_like(time_centers, np.nan),
            'se_metric': np.full_like(time_centers, np.nan),
            'n_larvae': 0
        }
    
    # Convert to array for easier computation
    metric_arrays = np.array(metric_arrays)  # Shape: (n_larvae, n_time_bins)
    
    # Compute mean and standard error across larvae
    mean_metric = np.nanmean(metric_arrays, axis=0)
    se_metric = stats.sem(metric_arrays, axis=0, nan_policy='omit')
    
    return {
        'time_centers': time_centers,
        'metric_arrays': metric_arrays,
        'mean_metric': mean_metric,
        'se_metric': se_metric,
        'n_larvae': len(metric_arrays)
    }

def analyze_head_casts_over_time(experiments_data, window=60, step=20, peak_threshold=5.0, peak_prominence=3.0, smooth_sigma=4.0, large_casts_only=True, jump_threshold=15):
    """
    Analyze head cast frequency over time using sliding windows.
    Now uses upstream/downstream classification.
    """
    # Detect all cast events with head cast counts
    cast_events_data = detect_head_casts_in_casts(
        experiments_data, 
        peak_threshold=peak_threshold, 
        peak_prominence=peak_prominence,
        smooth_sigma=smooth_sigma,
        large_casts_only=large_casts_only,
        jump_threshold=jump_threshold
    )
    
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data
    
    # Find the global time range across all larvae
    all_times = []
    for larva_data in data_to_process.values():
        if 't' in larva_data:
            t = np.array(larva_data['t']).flatten()
            all_times.extend(t)
    
    if not all_times:
        return {
            'time_centers': np.array([]),
            'metric_arrays': np.array([]),
            'mean_metric': np.array([]),
            'se_metric': np.array([]),
            'n_larvae': 0
        }
    
    global_t_min = np.min(all_times)
    global_t_max = np.max(all_times)
    
    # Create time windows
    window_starts = np.arange(global_t_min, global_t_max - window + step, step)
    time_centers = window_starts + window / 2
    
    # Collect data per larva
    larva_time_data = []
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        # Count head casts in each time window
        head_cast_rates = []
        
        for window_start in window_starts:
            window_end = window_start + window
            
            # Count total head casts from all cast events in this time window
            head_casts_in_window = 0
            for cast_event in cast_events:
                cast_start_time = cast_event['cast_start_time']
                cast_end_time = cast_event['cast_end_time']
                
                # If cast overlaps with window, count its head casts
                if not (cast_end_time < window_start or cast_start_time >= window_end):
                    head_casts_in_window += cast_event['total_head_casts']
            
            # Calculate rate (head casts per second)
            head_cast_rate = head_casts_in_window / window
            head_cast_rates.append(head_cast_rate)
        
        larva_time_data.append({
            'head_cast_rates': np.array(head_cast_rates),
            'larva_id': larva_id
        })
    
    if not larva_time_data:
        return {
            'time_centers': time_centers,
            'metric_arrays': np.array([]),
            'mean_metric': np.full_like(time_centers, np.nan),
            'se_metric': np.full_like(time_centers, np.nan),
            'n_larvae': 0
        }
    
    # Convert to array for statistical analysis
    rate_arrays = np.array([data['head_cast_rates'] for data in larva_time_data])
    
    # Compute means and standard errors
    mean_rates = np.nanmean(rate_arrays, axis=0)
    se_rates = stats.sem(rate_arrays, axis=0, nan_policy='omit')
    
    return {
        'time_centers': time_centers,
        'metric_arrays': rate_arrays,
        'mean_metric': mean_rates,
        'se_metric': se_rates,
        'n_larvae': len(larva_time_data),
        'cast_events_data': cast_events_data
    }

### =============================================================================
######## Analyze cast in details ########
### =============================================================================
def detect_head_casts_in_casts(experiments_data, peak_threshold=5.0, peak_prominence=3.0, min_cast_duration=3, smooth_sigma=4.0, large_casts_only=True, jump_threshold=15):
    """
    Detect head casts (cast peaks) during cast events and classify them as upstream or downstream.
    Uses bend angle direction + orientation to determine upstream/downstream, similar to compare_cast_directions_peaks.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        peak_threshold: Minimum absolute angle for a peak to be considered significant (degrees)
        peak_prominence: Minimum prominence for peak detection
        min_cast_duration: Minimum duration of cast event in frames
        smooth_sigma: Gaussian smoothing parameter for head angles
        large_casts_only: If True, only detect head casts in large casts (state == 2)
        jump_threshold: Threshold for detecting orientation jumps in degrees/frame
        
    Returns:
        Dictionary with cast event data for each larva, including head cast direction classifications
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data
    
    # Define perpendicular ranges (both left and right sides)
    left_perp_range = (-90 - 10, -90 + 10)  # Using ±10° around -90°
    right_perp_range = (90 - 10, 90 + 10)   # Using ±10° around +90°
    
    def is_perpendicular(angle):
        """Check if an angle is within the perpendicular ranges"""
        return ((left_perp_range[0] <= angle <= left_perp_range[1]) or 
                (right_perp_range[0] <= angle <= right_perp_range[1]))
    
    all_cast_events = {}
    
    for larva_id, larva_data in data_to_process.items():
        try:
            # Check for required data
            if ('global_state_small_large_state' not in larva_data or 
                'angle_upper_lower_smooth_5' not in larva_data or
                't' not in larva_data):
                continue
                
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            head_angles = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
            t = np.array(larva_data['t']).flatten()
            
            # Get orientation for reference
            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
                
            # Ensure all arrays have same length
            min_len = min(len(states), len(head_angles), len(t), len(orientations))
            states = states[:min_len]
            head_angles = head_angles[:min_len]
            t = t[:min_len]
            orientations = orientations[:min_len]
            
            # Convert head angles to degrees
            head_angles_deg = np.degrees(head_angles)
            
            # Apply orientation jump detection and smoothing
            orientation_raw = orientations.copy()
            orientation_diff = np.abs(np.diff(orientations))
            orientation_diff = np.insert(orientation_diff, 0, 0)
            
            # Find jumps bigger than threshold
            jumps = orientation_diff > jump_threshold
            orientation_masked = np.ma.array(orientations, mask=jumps)
            
            # Interpolate masked values for smoothing
            orientation_interp = orientation_masked.filled(np.nan)
            mask = np.isnan(orientation_interp)
            
            if np.sum(~mask) > 1:
                indices = np.arange(len(orientation_interp))
                valid_indices = indices[~mask]
                valid_values = orientation_interp[~mask]
                orientation_interp[mask] = np.interp(indices[mask], valid_indices, valid_values)
            
            # Apply smoothing to orientation
            orientation_smooth = gaussian_filter1d(orientation_interp, smooth_sigma/3.0)
            
            # Apply smoothing to bend angle
            bend_angle_smooth = gaussian_filter1d(head_angles_deg, smooth_sigma/3.0)
            
            # Find cast segments
            cast_segments = []
            i = 0
            while i < len(states):
                if large_casts_only:
                    if states[i] == 2:
                        cast_start = i
                        while i < len(states) and states[i] == 2:
                            i += 1
                        cast_end = i - 1
                        
                        if cast_end - cast_start + 1 >= min_cast_duration:
                            cast_segments.append((cast_start, cast_end))
                    else:
                        i += 1
                else:
                    if states[i] == 2 or states[i] == 1.5:
                        cast_start = i
                        while i < len(states) and (states[i] == 2 or states[i] == 1.5):
                            i += 1
                        cast_end = i - 1
                        
                        if cast_end - cast_start + 1 >= min_cast_duration:
                            cast_segments.append((cast_start, cast_end))
                    else:
                        i += 1
            
            # Process each cast segment
            larva_cast_events = []
            
            for start_idx, end_idx in cast_segments:
                cast_head_angles = bend_angle_smooth[start_idx:end_idx+1]
                cast_times = t[start_idx:end_idx+1]
                cast_orientations = orientation_smooth[start_idx:end_idx+1]
                
                # Store larva orientation at the BEGINNING of the cast event
                cast_start_orientation = orientation_smooth[start_idx]
                
                if len(cast_head_angles) < 3:
                    continue
                
                # Find peaks in absolute bend angles (like in compare_cast_directions_peaks)
                pos_peaks, _ = find_peaks(cast_head_angles, height=peak_threshold, prominence=peak_prominence, distance=3)
                neg_peaks, _ = find_peaks(-cast_head_angles, height=peak_threshold, prominence=peak_prominence, distance=3)
                
                # Combine and sort peaks by position
                all_peaks = sorted(list(pos_peaks) + list(neg_peaks))
                
                # Classify peaks as upstream/downstream and count by type
                head_cast_details = []
                upstream_count = 0
                downstream_count = 0
                
                for peak_idx in all_peaks:
                    global_peak_idx = start_idx + peak_idx
                    
                    # Check if larva is perpendicular at this peak
                    peak_orientation = cast_orientations[peak_idx]
                    if is_perpendicular(peak_orientation):
                        bend_angle = cast_head_angles[peak_idx]
                        
                        # Normalize orientation to -180 to 180 range
                        while peak_orientation > 180:
                            peak_orientation -= 360
                        while peak_orientation <= -180:
                            peak_orientation += 360
                        
                        # Classify as upstream or downstream
                        is_upstream = False
                        if (peak_orientation > 0 and peak_orientation < 180):  # Right side (positive orientation)
                            if bend_angle < 0:  # Negative bend is upstream
                                is_upstream = True
                            else:  # Positive bend is downstream
                                is_upstream = False
                        else:  # Left side (negative orientation)
                            if bend_angle > 0:  # Positive bend is upstream
                                is_upstream = True
                            else:  # Negative bend is downstream
                                is_upstream = False
                        
                        direction = 'upstream' if is_upstream else 'downstream'
                        
                        if is_upstream:
                            upstream_count += 1
                        else:
                            downstream_count += 1
                        
                        head_cast_details.append({
                            'direction': direction,
                            'amplitude': abs(bend_angle),
                            'peak_time': cast_times[peak_idx],
                            'peak_orientation': peak_orientation,
                            'bend_angle': bend_angle,
                            'cast_frame_idx': peak_idx,
                            'global_frame_idx': global_peak_idx
                        })
                
                # Sort head casts by time within the cast
                head_cast_details.sort(key=lambda x: x['peak_time'])
                
                total_head_casts = len(head_cast_details)
                
                # Create cast event record
                cast_event = {
                    'larva_id': larva_id,
                    'cast_start_time': cast_times[0],
                    'cast_end_time': cast_times[-1],
                    'cast_duration': len(cast_times),
                    'cast_start_orientation': cast_start_orientation,
                    'total_head_casts': total_head_casts,
                    'n_upstream_head_casts': upstream_count,
                    'n_downstream_head_casts': downstream_count,
                    'head_cast_details': head_cast_details,
                    'global_start_idx': start_idx,
                    'global_end_idx': end_idx
                }
                
                larva_cast_events.append(cast_event)
            
            all_cast_events[larva_id] = larva_cast_events
            
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    return all_cast_events

def analyze_first_head_cast_bias_perpendicular(cast_events_data, analysis_type='first'):
    """
    Analyze bias of head cast direction when larvae are perpendicular to flow.
    Uses upstream/downstream classification based on bend angle + orientation.
    
    Args:
        cast_events_data: Output from detect_head_casts_in_casts function
        analysis_type: 'first', 'last', or 'all' - which head casts to analyze
        
    Returns:
        Dictionary with bias analysis results
    """
    # Collect head cast data for perpendicular orientations
    head_cast_data = []
    larva_summaries = []
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        larva_upstream_count = 0
        larva_downstream_count = 0
        larva_total_casts = 0
        
        for cast_event in cast_events:
            if not cast_event['head_cast_details']:
                continue
                
            # Select which head casts to analyze based on analysis_type
            if analysis_type == 'first' and len(cast_event['head_cast_details']) > 0:
                head_casts_to_analyze = [cast_event['head_cast_details'][0]]
            elif analysis_type == 'last' and len(cast_event['head_cast_details']) > 0:
                head_casts_to_analyze = [cast_event['head_cast_details'][-1]]
            elif analysis_type == 'all':
                head_casts_to_analyze = cast_event['head_cast_details']
            else:
                continue
            
            for head_cast in head_casts_to_analyze:
                direction = head_cast['direction']
                
                # Record the head cast direction
                cast_data = {
                    'larva_id': larva_id,
                    'cast_orientation': cast_event['cast_start_orientation'],
                    'head_cast_direction': direction,  # 'upstream' or 'downstream'
                    'head_cast_amplitude': head_cast['amplitude'],
                    'peak_orientation': head_cast['peak_orientation'],
                    'bend_angle': head_cast['bend_angle'],
                    'total_head_casts_in_event': cast_event['total_head_casts'],
                    'cast_duration': cast_event['cast_duration']
                }
                
                head_cast_data.append(cast_data)
                larva_total_casts += 1
                
                if direction == 'upstream':
                    larva_upstream_count += 1
                else:
                    larva_downstream_count += 1
        
        # Store per-larva summary
        if larva_total_casts > 0:
            larva_summaries.append({
                'larva_id': larva_id,
                'upstream_count': larva_upstream_count,
                'downstream_count': larva_downstream_count,
                'total_count': larva_total_casts,
                'upstream_bias': larva_upstream_count / larva_total_casts,
                'downstream_bias': larva_downstream_count / larva_total_casts
            })
    
    if not head_cast_data:
        return {
            'head_cast_data': [],
            'larva_summaries': [],
            'total_upstream': 0,
            'total_downstream': 0,
            'total_casts': 0,
            'overall_upstream_bias': np.nan,
            'overall_downstream_bias': np.nan,
            'p_value_binomial': np.nan,
            'n_larvae': 0,
            'analysis_type': analysis_type
        }
    
    # Calculate overall statistics
    total_upstream = sum(1 for cast in head_cast_data if cast['head_cast_direction'] == 'upstream')
    total_downstream = sum(1 for cast in head_cast_data if cast['head_cast_direction'] == 'downstream')
    total_casts = len(head_cast_data)
    
    overall_upstream_bias = total_upstream / total_casts
    overall_downstream_bias = total_downstream / total_casts
    
    # Statistical test: binomial test for bias
    from scipy.stats import binomtest
    binom_result = binomtest(total_upstream, total_casts, 0.5, alternative='two-sided')
    p_value = binom_result.pvalue
    
    # Per-larva bias statistics
    larva_upstream_biases = [summary['upstream_bias'] for summary in larva_summaries]
    larva_downstream_biases = [summary['downstream_bias'] for summary in larva_summaries]
    
    mean_larva_upstream_bias = np.mean(larva_upstream_biases)
    se_larva_upstream_bias = stats.sem(larva_upstream_biases)
    mean_larva_downstream_bias = np.mean(larva_downstream_biases)
    se_larva_downstream_bias = stats.sem(larva_downstream_biases)
    
    return {
        'head_cast_data': head_cast_data,
        'larva_summaries': larva_summaries,
        'total_upstream': total_upstream,
        'total_downstream': total_downstream,
        'total_casts': total_casts,
        'overall_upstream_bias': overall_upstream_bias,
        'overall_downstream_bias': overall_downstream_bias,
        'mean_larva_upstream_bias': mean_larva_upstream_bias,
        'se_larva_upstream_bias': se_larva_upstream_bias,
        'mean_larva_downstream_bias': mean_larva_downstream_bias,
        'se_larva_downstream_bias': se_larva_downstream_bias,
        'p_value_binomial': p_value,
        'n_larvae': len(larva_summaries),
        'analysis_type': analysis_type
    }

### NAVIGATIONAL INDEX CALCULATION ###
def compute_velocity_and_axis(experiments_data):
    """
    Compute velocity along x and y axes for each larva and store normalized velocities.
    Adds: c_axis_x, c_axis_y, speed, c_axis_x_normalized, c_axis_y_normalized, speed_normalized to each larva.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        
    Returns:
        Updated experiments_data with velocity fields
    """
    import numpy as np

    # Accept both {'data': ...} and plain dict
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        # Require t, x_center, y_center, larva_length_smooth_5
        required_fields = ['t', 'x_center', 'y_center', 'larva_length_smooth_5']
        if not all(k in larva_data for k in required_fields):
            print(f"Skipping larva {larva_id} due to missing required fields.")
            continue
            
        t = np.array(larva_data['t']).flatten()
        x = np.array(larva_data['x_center']).flatten()
        y = np.array(larva_data['y_center']).flatten()
        length = np.array(larva_data['larva_length_smooth_5']).flatten()
        
        n = min(len(t), len(x), len(y), len(length))
        if n < 2:
            continue
            
        t = t[:n]
        x = x[:n]
        y = y[:n]
        length = length[:n]

        # Compute velocities (forward difference, last value is nan)
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)
        
        c_axis_x = np.full(n, np.nan)
        c_axis_y = np.full(n, np.nan)
        speed = np.full(n, np.nan)
        
        valid = dt > 0
        c_axis_x[:-1][valid] = dx[valid] / dt[valid]
        c_axis_y[:-1][valid] = dy[valid] / dt[valid]
        speed[:-1][valid] = np.sqrt(c_axis_x[:-1][valid]**2 + c_axis_y[:-1][valid]**2)

        # Normalize by mean larva length (ignoring nans)
        mean_size = np.nanmean(length)
        if mean_size > 0:
            c_axis_x_normalized = c_axis_x / mean_size
            c_axis_y_normalized = c_axis_y / mean_size
            speed_normalized = speed / mean_size
        else:
            c_axis_x_normalized = c_axis_x
            c_axis_y_normalized = c_axis_y
            speed_normalized = speed

        # Store computed values
        larva_data['c_axis_x'] = c_axis_x
        larva_data['c_axis_y'] = c_axis_y
        larva_data['speed'] = speed
        larva_data['c_axis_x_normalized'] = c_axis_x_normalized
        larva_data['c_axis_y_normalized'] = c_axis_y_normalized
        larva_data['speed_normalized'] = speed_normalized

    return experiments_data

def analyze_navigational_index_over_time(experiments_data, window=60, step=10, t_max=600):
    """
    Compute NI_x and NI_y (navigational index in x and y) across time for each larva.
    NI_x = <c_axis_x_normalized> / <speed_normalized> (mean over window)
    NI_y = <c_axis_y_normalized> / <speed_normalized>
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        window: Window size in seconds for averaging
        step: Step size in seconds for moving window
        t_max: Maximum time to analyze
        
    Returns:
        Dictionary with time-series NI data
    """
    import numpy as np

    # Ensure velocity fields are computed
    experiments_data = compute_velocity_and_axis(experiments_data)

    # Prepare data
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    larva_ids = list(data_to_process.keys())
    n_larvae = len(larva_ids)
    time_bins = np.arange(0, t_max - window + step, step) + window / 2
    
    NI_x_mat = np.full((n_larvae, len(time_bins)), np.nan)
    NI_y_mat = np.full((n_larvae, len(time_bins)), np.nan)

    for i, larva_id in enumerate(larva_ids):
        larva_data = data_to_process[larva_id]
        
        # Check for required velocity fields
        if not all(k in larva_data for k in ['t', 'c_axis_x_normalized', 'c_axis_y_normalized', 'speed_normalized']):
            continue
            
        t = np.array(larva_data['t']).flatten()
        c_x = np.array(larva_data['c_axis_x_normalized']).flatten()
        c_y = np.array(larva_data['c_axis_y_normalized']).flatten()
        speed = np.array(larva_data['speed_normalized']).flatten()
        
        if len(t) < 2 or len(c_x) != len(t):
            continue
            
        for j, t_center in enumerate(time_bins):
            t0 = t_center - window / 2
            t1 = t_center + window / 2
            mask = (t >= t0) & (t < t1)
            
            if np.sum(mask) < 2:
                continue
                
            mean_cx = np.nanmean(c_x[mask])
            mean_cy = np.nanmean(c_y[mask])
            mean_speed = np.nanmean(speed[mask])
            
            # Compute NI: mean projection divided by mean speed
            if mean_speed > 0:
                NI_x_mat[i, j] = mean_cx / mean_speed
                NI_y_mat[i, j] = mean_cy / mean_speed

    # Compute overall statistics
    mean_NI_x = np.nanmean(NI_x_mat, axis=0)
    mean_NI_y = np.nanmean(NI_y_mat, axis=0)
    se_NI_x = stats.sem(NI_x_mat, axis=0, nan_policy='omit')
    se_NI_y = stats.sem(NI_y_mat, axis=0, nan_policy='omit')

    return {
        'time_centers': time_bins,
        'metric_arrays': np.array([NI_x_mat, NI_y_mat]),  # Shape: (2, n_larvae, n_time_bins)
        'NI_x_arrays': NI_x_mat,
        'NI_y_arrays': NI_y_mat,
        'mean_metric': np.array([mean_NI_x, mean_NI_y]),  # Shape: (2, n_time_bins)
        'mean_NI_x': mean_NI_x,
        'mean_NI_y': mean_NI_y,
        'se_metric': np.array([se_NI_x, se_NI_y]),  # Shape: (2, n_time_bins)
        'se_NI_x': se_NI_x,
        'se_NI_y': se_NI_y,
        'larva_ids': larva_ids,
        'n_larvae': n_larvae
    }

def analyze_navigational_index_single_values(experiments_data, window=60, step=10, t_max=600):
    """
    Compute single NI_x and NI_y values per larva by averaging across time.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        window: Window size in seconds for averaging
        step: Step size in seconds for moving window
        t_max: Maximum time to analyze
        
    Returns:
        Dictionary with single-value NI data per larva
    """
    import numpy as np
    from scipy import stats
    
    # Get the time-series NI data first
    time_result = analyze_navigational_index_over_time(experiments_data, window=window, step=step, t_max=t_max)
    
    NI_x_mat = time_result['NI_x_arrays']
    NI_y_mat = time_result['NI_y_arrays']
    larva_ids = time_result['larva_ids']
    
    # Average across time for each larva
    NI_x_single = np.nanmean(NI_x_mat, axis=1)
    NI_y_single = np.nanmean(NI_y_mat, axis=1)
    
    # Combine into single NI_x_y metric (vector magnitude)
    NI_x_y_single = np.sqrt(NI_x_single**2 + NI_y_single**2)
    
    # Remove NaN values for statistical testing
    NI_x_clean = NI_x_single[~np.isnan(NI_x_single)]
    NI_y_clean = NI_y_single[~np.isnan(NI_y_single)]
    NI_xy_clean = NI_x_y_single[~np.isnan(NI_x_y_single)]
    
    # Perform one-sample t-tests against 0
    def test_significance(values, test_value=0):
        if len(values) < 3:
            return np.nan, "insufficient data"
        t_stat, p_value = stats.ttest_1samp(values, test_value)
        if p_value < 0.001:
            return p_value, "***"
        elif p_value < 0.01:
            return p_value, "**"
        elif p_value < 0.05:
            return p_value, "*"
        else:
            return p_value, "ns"
    
    p_NI_x, sig_NI_x = test_significance(NI_x_clean)
    p_NI_y, sig_NI_y = test_significance(NI_y_clean)
    
    return {
        'larva_ids': larva_ids,
        'NI_x_single': NI_x_single,
        'NI_y_single': NI_y_single,
        'NI_x_y_single': NI_x_y_single,
        'NI_x_clean': NI_x_clean,
        'NI_y_clean': NI_y_clean,
        'NI_xy_clean': NI_xy_clean,
        'p_values': {'NI_x': p_NI_x, 'NI_y': p_NI_y},
        'significances': {'NI_x': sig_NI_x, 'NI_y': sig_NI_y},
        'means': {'NI_x': np.nanmean(NI_x_clean), 'NI_y': np.nanmean(NI_y_clean)},
        'n_larvae': len(NI_x_clean)
    }