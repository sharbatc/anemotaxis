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
# - 0° = larva facing downstream or AWAY FROM WIND SOURCE (-x direction) 
# - +90° = larva facing left (perpendicular to flow)
# - ±180° = larva facing upstream or TOWARDS WIND SOURCE (+x direction)
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

    # Add tracking variables for debugging
    total_casts_processed = 0
    total_turns_detected = 0
    orientation_changes = []  # Store all orientation changes
    turns_per_bin_debug = {}  # Track turns per orientation bin
    all_turn_events = []  # Store detailed turn information
    all_cast_events = []  # Store ALL cast events with orientations

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_small_large_state' not in larva_data or 't' not in larva_data:
                continue
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            times = np.array(larva_data['t']).flatten()  # Use actual timing data

            orientations = get_larva_orientation_array(larva_data)
            if orientations is None:
                continue
                
            min_len = min(len(orientations), len(states), len(times))
            orientations = orientations[:min_len]
            states = states[:min_len]
            times = times[:min_len]

            # Identify turns: casts that result in significant orientation change
            is_turn = np.zeros(len(states), dtype=bool)
            larva_turns = 0
            larva_casts = 0
            
            print(f"\n🐛 Larva {larva_id} cast orientations:")
            
            i = 0
            while i < len(states):
                if states[i] == 2:  # Large cast
                    cast_start = i
                    while i < len(states) and states[i] == 2:
                        i += 1
                    cast_end = i - 1
                    
                    larva_casts += 1
                    total_casts_processed += 1
                    
                    # Record ALL cast events with their orientations
                    if cast_end > cast_start:
                        orient_start = orientations[cast_start]
                        orient_end = orientations[cast_end]
                        time_start = times[cast_start]
                        time_end = times[cast_end]
                        
                        # Compute orientation change (handle wraparound)
                        delta = np.angle(np.exp(1j * np.deg2rad(orient_end - orient_start)), deg=True)
                        
                        # Store ALL cast events
                        cast_event = {
                            'larva_id': larva_id,
                            'cast_number': larva_casts,
                            'time_start': time_start,
                            'time_end': time_end,
                            'duration': time_end - time_start,
                            'start_orientation': orient_start,
                            'end_orientation': orient_end,
                            'orientation_change': delta,
                            'abs_orientation_change': np.abs(delta),
                            'is_turn': np.abs(delta) >= min_turn_amplitude
                        }
                        all_cast_events.append(cast_event)
                        
                        # Print ALL cast details
                        turn_marker = "TURN" if np.abs(delta) >= min_turn_amplitude else "cast"
                        print(f"   {turn_marker:4s} {larva_casts:2d}: t={time_start:6.1f}-{time_end:5.1f}s, "
                              f"θ={orient_start:6.1f}°→{orient_end:6.1f}° (Δ={delta:+6.1f}°)")
                        
                        # Store all orientation changes for analysis
                        orientation_changes.append({
                            'larva_id': larva_id,
                            'start_orientation': orient_start,
                            'end_orientation': orient_end,
                            'delta': delta,
                            'abs_delta': np.abs(delta),
                            'is_turn': np.abs(delta) >= min_turn_amplitude,
                            'time_start': time_start,
                            'time_end': time_end
                        })
                        
                        if np.abs(delta) >= min_turn_amplitude:
                            # Mark all frames in this cast as turn frames
                            is_turn[cast_start:cast_end+1] = True
                            larva_turns += 1
                            total_turns_detected += 1
                            
                            # Store detailed turn event
                            turn_event = {
                                'larva_id': larva_id,
                                'turn_number': larva_turns,
                                'time_start': time_start,
                                'time_end': time_end,
                                'duration': time_end - time_start,
                                'start_orientation': orient_start,
                                'end_orientation': orient_end,
                                'orientation_change': delta,
                                'abs_orientation_change': np.abs(delta)
                            }
                            all_turn_events.append(turn_event)
                            
                            # Track which orientation bin this turn belongs to
                            bin_idx = int((orient_start + 180) // bin_width)
                            if bin_idx not in turns_per_bin_debug:
                                turns_per_bin_debug[bin_idx] = []
                            turns_per_bin_debug[bin_idx].append({
                                'larva_id': larva_id,
                                'start_orientation': orient_start,
                                'delta': delta,
                                'time': time_start
                            })
                else:
                    i += 1

            # Store turn probability data per larva
            larva_orientations.append({
                'orientations': orientations,
                'is_turn': is_turn
            })
            
            print(f"   Summary: {larva_casts:2d} casts, {larva_turns:2d} turns ({100*larva_turns/max(1,larva_casts):4.1f}%)")
                
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

    # Print summary statistics
    print(f"\n🎯 Cast & Turn Analysis Summary:")
    print(f"   Total casts processed: {total_casts_processed}")
    print(f"   Total turns detected: {total_turns_detected}")
    print(f"   Overall turn rate: {100*total_turns_detected/max(1,total_casts_processed):.1f}%")
    print(f"   Min turn amplitude: {min_turn_amplitude}°")
    
    # Print orientation change statistics
    if orientation_changes:
        all_deltas = [change['abs_delta'] for change in orientation_changes]
        print(f"   Orientation change stats:")
        print(f"     Mean: {np.mean(all_deltas):.1f}°")
        print(f"     Median: {np.median(all_deltas):.1f}°")
        print(f"     Range: {np.min(all_deltas):.1f}° - {np.max(all_deltas):.1f}°")
    
    # Print ALL cast orientations chronologically
    print(f"\n📍 All cast orientations (chronological):")
    all_cast_events.sort(key=lambda x: x['time_start'])
    for event in all_cast_events:
        turn_marker = "TURN" if event['is_turn'] else "cast"
        print(f"   Larva {event['larva_id']:2d}, {turn_marker:4s}: "
              f"t={event['time_start']:6.1f}s, θ={event['start_orientation']:6.1f}°, "
              f"Δ={event['orientation_change']:+6.1f}°")
    
    # Print turns per orientation bin
    print(f"\n📊 Turns per orientation bin:")
    for i, center in enumerate(bin_centers):
        if i in turns_per_bin_debug:
            n_turns = len(turns_per_bin_debug[i])
            deltas = [turn['delta'] for turn in turns_per_bin_debug[i]]
            mean_delta = np.mean(np.abs(deltas))
            print(f"   {center:6.1f}°: {n_turns:2d} turns (mean Δ = {mean_delta:5.1f}°)")

    return {
        'orientations': [data['orientations'][data['is_turn']] for data in larva_orientations],  # Turn orientations only
        'hist_arrays': hist_arrays,          # Individual turn probability histograms per larva
        'mean_hist': mean_hist,              # Mean turn probability across larvae
        'se_hist': se_hist,                  # Standard error across larvae
        'bin_centers': bin_centers,
        'n_larvae': len(larva_orientations),
        # Add debugging information to return dict
        'debug_info': {
            'total_casts': total_casts_processed,
            'total_turns': total_turns_detected,
            'orientation_changes': orientation_changes,
            'turns_per_bin': turns_per_bin_debug,
            'turn_rate': total_turns_detected/max(1,total_casts_processed),
            'all_turn_events': all_turn_events,
            'all_cast_events': all_cast_events  # NEW: All cast events with orientations
        }
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

# The following is the per larva histogram approach

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
            # Filter for RUN states only (1.0 = large run)
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

# The following is the pooled approach for run velocity by orientation
def analyze_run_velocity_by_orientation_pooled(experiments_data, bin_width=10, sigma=2):
    """
    Analyze velocity by orientation ONLY during run states (states == 1.0 or 0.5).
    Uses tail-to-neck orientation and motion_velocity_norm_smooth_5.
    POOLS velocities across all larvae for each orientation bin.
    
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
    
    # Pool velocities across all larvae for each orientation bin
    velocity_by_bin = {center: [] for center in bin_centers}
    n_larvae = 0

    for larva_id, larva_data in data_to_process.items():
        try:
            # Check for required data
            if 'global_state_large_state' not in larva_data:
                continue
                
            states = np.array(larva_data['global_state_large_state']).flatten()
            # Filter for RUN states only (1.0 = large run)
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

            # Pool velocities by orientation bin across all larvae
            for i, center in enumerate(bin_centers):
                bin_mask = (run_orientations >= bins[i]) & (run_orientations < bins[i+1])
                if np.any(bin_mask):
                    velocity_by_bin[center].extend(run_velocities[bin_mask])
            
            n_larvae += 1
                    
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if n_larvae == 0:
        return {
            'orientations': [], 
            'hist_arrays': np.array([]), 
            'mean_hist': np.array([]), 
            'se_hist': np.array([]), 
            'bin_centers': bin_centers,
            'n_larvae': 0
        }

    # Calculate mean velocities and standard errors per bin from pooled data
    mean_velocities = np.full_like(bin_centers, np.nan, dtype=float)
    se_velocities = np.full_like(bin_centers, np.nan, dtype=float)
    
    for i, center in enumerate(bin_centers):
        if velocity_by_bin[center]:  # If there are velocities in this bin
            velocities = np.array(velocity_by_bin[center])
            mean_velocities[i] = np.mean(velocities)
            # Calculate standard error from pooled raw data
            se_velocities[i] = stats.sem(velocities)
    
    # Apply Gaussian smoothing to means (handle NaNs)
    if sigma > 0:
        valid_mask = ~np.isnan(mean_velocities)
        if np.sum(valid_mask) > 1:
            # Interpolate NaN values for smoothing
            velocity_means_interp = np.copy(mean_velocities)
            se_velocities_interp = np.copy(se_velocities)
            
            if np.sum(valid_mask) < len(mean_velocities):
                valid_indices = np.where(valid_mask)[0]
                valid_means = mean_velocities[valid_mask]
                valid_ses = se_velocities[valid_mask]
                
                velocity_means_interp = np.interp(
                    np.arange(len(bin_centers)), 
                    valid_indices, 
                    valid_means
                )
                se_velocities_interp = np.interp(
                    np.arange(len(bin_centers)), 
                    valid_indices, 
                    valid_ses
                )
            
            # Apply smoothing to both mean and SE
            velocity_smooth = gaussian_filter1d(velocity_means_interp, sigma=sigma)
            se_smooth = gaussian_filter1d(se_velocities_interp, sigma=sigma)
            
            # Restore NaN values where original data was NaN
            velocity_smooth[~valid_mask] = np.nan
            se_smooth[~valid_mask] = np.nan
        else:
            velocity_smooth = mean_velocities
            se_smooth = se_velocities
    else:
        velocity_smooth = mean_velocities
        se_smooth = se_velocities

    # For compatibility with plot_orientation_histogram, create single-row array
    hist_arrays = velocity_smooth.reshape(1, -1)  # Shape: (1, n_bins)

    return {
        'orientations': [[] for _ in range(n_larvae)],  # Empty list per larva for compatibility
        'hist_arrays': hist_arrays,          # Single histogram from pooled data
        'mean_hist': velocity_smooth,        # Mean run velocity from pooled data
        'se_hist': se_smooth,                # Standard error from pooled raw data
        'bin_centers': bin_centers,
        'n_larvae': n_larvae,
        'pooled_data': velocity_by_bin,      # Raw pooled data for reference
        'sample_sizes': {center: len(velocity_by_bin[center]) for center in bin_centers}  # Sample sizes per bin
    }

def analyze_head_casts_by_orientation(experiments_data, bin_width=20, peak_threshold=5.0, peak_prominence=3.0, smooth_sigma=4.0, large_casts_only=True, turns_only=False, min_turn_amplitude=45, separate_by_turn_success=False):
    """
    Analyze head cast frequency as a function of larva orientation at the beginning of cast events.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        bin_width: Bin width in degrees for orientation binning
        peak_threshold: Minimum absolute angle for head cast detection
        peak_prominence: Minimum prominence for peak detection
        smooth_sigma: Gaussian smoothing parameter
        large_casts_only: If True, only analyze large casts
        turns_only: If True, only count head casts during cast events that result in turns
        min_turn_amplitude: Minimum orientation change to qualify as a turn (only used if turns_only=True)
        separate_by_turn_success: If True, separate analysis by successful vs unsuccessful turns
        
    Returns:
        Dictionary with analysis results compatible with plot_orientation_histogram
    """
    # Detect all cast events with head cast counts
    cast_events_data = detect_head_casts_in_casts(
        experiments_data, 
        peak_threshold=peak_threshold, 
        peak_prominence=peak_prominence,
        smooth_sigma=smooth_sigma,
        large_casts_only=large_casts_only,
    )
    
    # Set up orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Collect data per larva for statistical analysis
    larva_data_list = []
    
    if separate_by_turn_success:
        print(f"\n🎯 Head Casts by Turn Success Analysis (Min turn: {min_turn_amplitude}°)")
        print("=" * 80)
    elif turns_only:
        print(f"\n🎯 Head Casts in Turning Events Analysis (Min turn: {min_turn_amplitude}°)")
        print("=" * 60)
    
    total_cast_events = 0
    total_turning_events = 0
    total_non_turning_events = 0
    total_head_casts_successful_turns = 0
    total_head_casts_unsuccessful_turns = 0
    total_head_casts_all = 0
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        if separate_by_turn_success:
            # Three separate arrays for successful turns, unsuccessful turns, and all
            head_cast_counts_successful = np.zeros_like(bin_centers)
            head_cast_counts_unsuccessful = np.zeros_like(bin_centers)
            head_cast_counts_all = np.zeros_like(bin_centers)
        else:
            # Single array for the requested analysis
            head_cast_counts = np.zeros_like(bin_centers)
            
        larva_cast_events = 0
        larva_turning_events = 0
        larva_non_turning_events = 0
        larva_head_casts_successful = 0
        larva_head_casts_unsuccessful = 0
        larva_head_casts_all = 0
        
        for cast_event in cast_events:
            larva_cast_events += 1
            total_cast_events += 1
            
            cast_start_orientation = cast_event['cast_start_orientation']
            total_head_casts = cast_event['total_head_casts']
            
            # Determine if this cast resulted in a turn
            cast_end_orientation = cast_event['cast_end_orientation']
            orientation_change = np.angle(np.exp(1j * np.deg2rad(cast_end_orientation - cast_start_orientation)), deg=True)
            is_turn = np.abs(orientation_change) >= min_turn_amplitude
            
            if is_turn:
                larva_turning_events += 1
                total_turning_events += 1
            else:
                larva_non_turning_events += 1
                total_non_turning_events += 1
            
            # Find the correct bin for the cast start orientation
            bin_idx = None
            for i, center in enumerate(bin_centers):
                if bins[i] <= cast_start_orientation < bins[i+1]:
                    bin_idx = i
                    break
            
            if bin_idx is not None:
                if separate_by_turn_success:
                    # Add to all casts
                    head_cast_counts_all[bin_idx] += total_head_casts
                    larva_head_casts_all += total_head_casts
                    total_head_casts_all += total_head_casts
                    
                    # Add to successful or unsuccessful based on turn outcome
                    if is_turn:
                        head_cast_counts_successful[bin_idx] += total_head_casts
                        larva_head_casts_successful += total_head_casts
                        total_head_casts_successful_turns += total_head_casts
                    else:
                        head_cast_counts_unsuccessful[bin_idx] += total_head_casts
                        larva_head_casts_unsuccessful += total_head_casts
                        total_head_casts_unsuccessful_turns += total_head_casts
                        
                elif turns_only:
                    # Only include if this cast resulted in a turn
                    if is_turn:
                        head_cast_counts[bin_idx] += total_head_casts
                        larva_head_casts_all += total_head_casts
                        total_head_casts_all += total_head_casts
                        
                else:
                    # Include all casts
                    head_cast_counts[bin_idx] += total_head_casts
                    larva_head_casts_all += total_head_casts
                    total_head_casts_all += total_head_casts
        
        if larva_cast_events > 0:
            if separate_by_turn_success:
                turn_rate = 100 * larva_turning_events / larva_cast_events
                print(f"Larva {larva_id:2d}: {larva_cast_events:2d} casts ({larva_turning_events:2d} turns, {larva_non_turning_events:2d} non-turns, {turn_rate:4.1f}%)")
                print(f"           Head casts: {larva_head_casts_successful:2d} in successful turns, "
                      f"{larva_head_casts_unsuccessful:2d} in unsuccessful, {larva_head_casts_all:2d} total")
                
                larva_data_list.append({
                    'head_cast_counts_successful': head_cast_counts_successful,
                    'head_cast_counts_unsuccessful': head_cast_counts_unsuccessful,
                    'head_cast_counts_all': head_cast_counts_all,
                    'larva_id': larva_id
                })
            else:
                if turns_only:
                    turn_rate = 100 * larva_turning_events / larva_cast_events
                    print(f"Larva {larva_id:2d}: {larva_cast_events:2d} casts, {larva_turning_events:2d} turns ({turn_rate:4.1f}%), "
                          f"{larva_head_casts_all:2d} head casts in turns")
                
                larva_data_list.append({
                    'head_cast_counts': head_cast_counts if not separate_by_turn_success else head_cast_counts_all,
                    'larva_id': larva_id
                })
    
    if not larva_data_list:
        analysis_type = "cast events" if not turns_only else "turning cast events"
        print(f"No {analysis_type} found for analysis.")
        empty_result = {
            'orientations': [],
            'hist_arrays': np.array([]),
            'mean_hist': np.full_like(bin_centers, np.nan),
            'se_hist': np.full_like(bin_centers, np.nan),
            'bin_centers': bin_centers,
            'n_larvae': 0
        }
        if separate_by_turn_success:
            empty_result.update({
                'hist_arrays_successful': np.array([]),
                'hist_arrays_unsuccessful': np.array([]),
                'hist_arrays_all': np.array([]),
                'mean_hist_successful': np.full_like(bin_centers, np.nan),
                'mean_hist_unsuccessful': np.full_like(bin_centers, np.nan),
                'mean_hist_all': np.full_like(bin_centers, np.nan),
                'se_hist_successful': np.full_like(bin_centers, np.nan),
                'se_hist_unsuccessful': np.full_like(bin_centers, np.nan),
                'se_hist_all': np.full_like(bin_centers, np.nan),
                'separate_by_turn_success': True
            })
        return empty_result
    
    if separate_by_turn_success:
        # Convert to arrays for analysis - three separate datasets
        successful_arrays = np.array([data['head_cast_counts_successful'] for data in larva_data_list])
        unsuccessful_arrays = np.array([data['head_cast_counts_unsuccessful'] for data in larva_data_list])
        all_arrays = np.array([data['head_cast_counts_all'] for data in larva_data_list])
        
        # Compute means and standard errors for each category
        mean_successful = np.nanmean(successful_arrays, axis=0)
        mean_unsuccessful = np.nanmean(unsuccessful_arrays, axis=0)
        mean_all = np.nanmean(all_arrays, axis=0)
        
        se_successful = stats.sem(successful_arrays, axis=0, nan_policy='omit')
        se_unsuccessful = stats.sem(unsuccessful_arrays, axis=0, nan_policy='omit')
        se_all = stats.sem(all_arrays, axis=0, nan_policy='omit')
        
        # Calculate summary statistics
        overall_turn_rate = 100 * total_turning_events / total_cast_events if total_cast_events > 0 else 0
        
        print("-" * 80)
        print(f"SUMMARY:")
        print(f"  Total cast events: {total_cast_events}")
        print(f"  Successful turns: {total_turning_events} ({overall_turn_rate:.1f}%)")
        print(f"  Unsuccessful casts: {total_non_turning_events} ({100-overall_turn_rate:.1f}%)")
        print(f"  Head casts in successful turns: {total_head_casts_successful_turns}")
        print(f"  Head casts in unsuccessful casts: {total_head_casts_unsuccessful_turns}")
        print(f"  Total head casts: {total_head_casts_all}")
        if total_turning_events > 0:
            print(f"  Mean head casts per successful turn: {total_head_casts_successful_turns/total_turning_events:.1f}")
        if total_non_turning_events > 0:
            print(f"  Mean head casts per unsuccessful cast: {total_head_casts_unsuccessful_turns/total_non_turning_events:.1f}")
        
        return {
            'orientations': [[] for _ in larva_data_list],
            'hist_arrays_successful': successful_arrays,
            'hist_arrays_unsuccessful': unsuccessful_arrays,
            'hist_arrays_all': all_arrays,
            'hist_arrays': all_arrays,  # Default for compatibility
            'mean_hist_successful': mean_successful,
            'mean_hist_unsuccessful': mean_unsuccessful,
            'mean_hist_all': mean_all,
            'mean_hist': mean_all,  # Default for compatibility
            'se_hist_successful': se_successful,
            'se_hist_unsuccessful': se_unsuccessful,
            'se_hist_all': se_all,
            'se_hist': se_all,  # Default for compatibility
            'bin_centers': bin_centers,
            'n_larvae': len(larva_data_list),
            'separate_by_turn_success': True,
            'summary_stats': {
                'total_casts': total_cast_events,
                'successful_turns': total_turning_events,
                'unsuccessful_casts': total_non_turning_events,
                'turn_rate': overall_turn_rate,
                'head_casts_successful': total_head_casts_successful_turns,
                'head_casts_unsuccessful': total_head_casts_unsuccessful_turns,
                'head_casts_total': total_head_casts_all
            }
        }
    else:
        # Original single analysis
        head_cast_arrays = np.array([data['head_cast_counts'] for data in larva_data_list])
        mean_head_casts = np.nanmean(head_cast_arrays, axis=0)
        se_head_casts = stats.sem(head_cast_arrays, axis=0, nan_policy='omit')
        
        if turns_only:
            overall_turn_rate = 100 * total_turning_events / total_cast_events if total_cast_events > 0 else 0
            
            print("-" * 60)
            print(f"SUMMARY:")
            print(f"  Total cast events: {total_cast_events}")
            print(f"  Turning events: {total_turning_events} ({overall_turn_rate:.1f}%)")
            print(f"  Head casts in turns: {total_head_casts_all}")
            print(f"  Mean head casts per turn: {total_head_casts_all/max(1,total_turning_events):.1f}")
        
        return {
            'orientations': [[] for _ in larva_data_list],
            'hist_arrays': head_cast_arrays,
            'mean_hist': mean_head_casts,
            'se_hist': se_head_casts,
            'bin_centers': bin_centers,
            'n_larvae': len(larva_data_list),
            'separate_by_turn_success': False
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

def analyze_head_casts_over_time(experiments_data, window=60, step=20, peak_threshold=5.0, peak_prominence=3.0, smooth_sigma=4.0, large_casts_only=True):
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
def detect_head_casts_in_casts(experiments_data, peak_threshold=3.0, peak_prominence=2.0, min_cast_duration=3, smooth_sigma=4.0, large_casts_only=True, min_turn_amplitude=45):
    """
    Detect ALL head casts during cast events and classify them as towards/away from wind only when perpendicular.
    Also detect if each cast event is a turn.
    
    Args:
        experiments_data: Dictionary of larva data (or {'data': ...})
        peak_threshold: Minimum absolute angle for a peak to be considered significant (degrees)
        peak_prominence: Minimum prominence for peak detection
        min_cast_duration: Minimum duration of cast event in frames
        smooth_sigma: Gaussian smoothing parameter for head angles
        large_casts_only: If True, only detect head casts in large casts (state == 2)
        min_turn_amplitude: Minimum orientation change in degrees to qualify as a turn
        
    Returns:
        Dictionary with cast event data for each larva, including head cast direction classifications and turn detection
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data
    
    # Define perpendicular ranges (both left and right sides)
    # HARDCODED - THIS IS AT 20° TOLERANCE - TO CHANGE
    left_perp_range = (-90 - 20, -90 + 20)  # Using ±20° around -90°
    right_perp_range = (90 - 20, 90 + 20)   # Using ±20° around +90°
    
    def is_perpendicular(angle):
        """Check if an angle is within the perpendicular ranges"""
        return ((left_perp_range[0] <= angle <= left_perp_range[1]) or 
                (right_perp_range[0] <= angle <= right_perp_range[1]))
    
    all_cast_events = {}
    
    # Print summary header
    print("🎯 Head Cast Detection Summary (with Turn Detection)")
    print("=" * 80)
    
    total_larvae = 0
    total_cast_periods = 0
    total_head_casts = 0
    total_towards_wind = 0
    total_away_from_wind = 0
    total_perpendicular_head_casts = 0
    total_turns = 0
    total_turn_casts = 0
    
    # Store per-larva probabilities for mean calculation
    larva_towards_probs = []
    larva_away_probs = []
    
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
            
            # Convert head angles to degrees and apply smoothing
            head_angles_deg = np.degrees(head_angles)
            orientation_deg = np.degrees(orientations)
            
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
            larva_cast_periods = len(cast_segments)
            larva_head_casts = 0
            larva_towards_wind = 0
            larva_away_from_wind = 0
            larva_perpendicular_head_casts = 0
            larva_turns = 0
            
            for start_idx, end_idx in cast_segments:
                cast_head_angles = bend_angle_smooth[start_idx:end_idx+1]
                cast_times = t[start_idx:end_idx+1]
                cast_orientations = orientations[start_idx:end_idx+1]
                
                # Store larva orientation at the BEGINNING and END of the cast event
                cast_start_orientation = orientations[start_idx]
                cast_end_orientation = orientations[end_idx]
                
                # Detect if this is a turn
                orientation_change = np.angle(np.exp(1j * np.deg2rad(cast_end_orientation - cast_start_orientation)), deg=True)
                is_turn = np.abs(orientation_change) >= min_turn_amplitude
                
                if is_turn:
                    larva_turns += 1
                    total_turns += 1
                
                if len(cast_head_angles) < 3:
                    # Still record the cast event even if no head casts detected
                    cast_event = {
                        'larva_id': larva_id,
                        'cast_start_time': cast_times[0],
                        'cast_end_time': cast_times[-1],
                        'cast_duration': len(cast_times),
                        'cast_start_orientation': cast_start_orientation,
                        'cast_end_orientation': cast_end_orientation,
                        'orientation_change': orientation_change,
                        'is_turn': is_turn,
                        'turn_direction': 'towards_wind' if is_turn and is_perpendicular(cast_start_orientation) else 'unclassified',
                        'total_head_casts': 0,
                        'n_towards_wind_head_casts': 0,
                        'n_away_from_wind_head_casts': 0,
                        'n_perpendicular_head_casts': 0,
                        'head_cast_details': [],
                        'global_start_idx': start_idx,
                        'global_end_idx': end_idx
                    }
                    larva_cast_events.append(cast_event)
                    continue
                
                # Find ALL peaks in absolute bend angles (regardless of orientation)
                pos_peaks, _ = find_peaks(cast_head_angles, height=peak_threshold, prominence=peak_prominence, distance=3)
                neg_peaks, _ = find_peaks(-cast_head_angles, height=peak_threshold, prominence=peak_prominence, distance=3)
                
                # Combine and sort peaks by position
                all_peaks = sorted(list(pos_peaks) + list(neg_peaks))
                
                # Process ALL peaks and classify only perpendicular ones
                head_cast_details = []
                towards_wind_count = 0
                away_from_wind_count = 0
                perpendicular_count = 0
                
                for peak_idx in all_peaks:
                    global_peak_idx = start_idx + peak_idx
                    bend_angle = cast_head_angles[peak_idx]
                    peak_orientation = cast_orientations[peak_idx]
                    
                    # Normalize orientation to -180 to 180 range
                    while peak_orientation > 180:
                        peak_orientation -= 360
                    while peak_orientation <= -180:
                        peak_orientation += 360
                    
                    # Default classification
                    direction = 'unclassified'
                    is_perp = is_perpendicular(peak_orientation)
                    
                    # Only classify direction if larva is perpendicular
                    if is_perp:
                        perpendicular_count += 1
                        
                        # Classify as towards or away from wind
                        if (peak_orientation > 0 and peak_orientation < 180):  # Right side (positive orientation)
                            if bend_angle < 0:  # Negative bend is towards wind
                                direction = 'towards_wind'
                                towards_wind_count += 1
                            else:  # Positive bend is away from wind
                                direction = 'away_from_wind'
                                away_from_wind_count += 1
                        else:  # Left side (negative orientation)
                            if bend_angle > 0:  # Positive bend is towards wind
                                direction = 'towards_wind'
                                towards_wind_count += 1
                            else:  # Negative bend is away from wind
                                direction = 'away_from_wind'
                                away_from_wind_count += 1
                    
                    head_cast_details.append({
                        'direction': direction,
                        'amplitude': abs(bend_angle),
                        'peak_time': cast_times[peak_idx],
                        'peak_orientation': peak_orientation,
                        'bend_angle': bend_angle,
                        'cast_frame_idx': peak_idx,
                        'global_frame_idx': global_peak_idx,
                        'is_perpendicular': is_perp
                    })
                
                # Sort head casts by time within the cast
                head_cast_details.sort(key=lambda x: x['peak_time'])
                
                # Determine turn direction for this cast (only if it's a turn and larva is perpendicular)
                turn_direction = 'unclassified'
                if is_turn and is_perpendicular(cast_start_orientation):
                    # Determine if turn is towards or away from wind based on orientation change
                    # For perpendicular orientations, positive change means turning towards wind from right side
                    # or away from wind from left side
                    if cast_start_orientation > 0:  # Right side
                        turn_direction = 'towards_wind' if orientation_change > 0 else 'away_from_wind'
                    else:  # Left side
                        turn_direction = 'away_from_wind' if orientation_change > 0 else 'towards_wind'
                
                total_head_casts_in_cast = len(head_cast_details)
                larva_head_casts += total_head_casts_in_cast
                larva_towards_wind += towards_wind_count
                larva_away_from_wind += away_from_wind_count
                larva_perpendicular_head_casts += perpendicular_count
                
                # Create cast event record
                cast_event = {
                    'larva_id': larva_id,
                    'cast_start_time': cast_times[0],
                    'cast_end_time': cast_times[-1],
                    'cast_duration': len(cast_times),
                    'cast_start_orientation': cast_start_orientation,
                    'cast_end_orientation': cast_end_orientation,
                    'orientation_change': orientation_change,
                    'is_turn': is_turn,
                    'turn_direction': turn_direction,
                    'total_head_casts': total_head_casts_in_cast,
                    'n_towards_wind_head_casts': towards_wind_count,
                    'n_away_from_wind_head_casts': away_from_wind_count,
                    'n_perpendicular_head_casts': perpendicular_count,
                    'head_cast_details': head_cast_details,
                    'global_start_idx': start_idx,
                    'global_end_idx': end_idx
                }
                
                larva_cast_events.append(cast_event)
            
            all_cast_events[larva_id] = larva_cast_events
            
            # Update totals
            if larva_cast_periods > 0:
                total_larvae += 1
                total_cast_periods += larva_cast_periods
                total_head_casts += larva_head_casts
                total_towards_wind += larva_towards_wind
                total_away_from_wind += larva_away_from_wind
                total_perpendicular_head_casts += larva_perpendicular_head_casts
                total_turn_casts += sum(1 for event in larva_cast_events if event['is_turn'])
                
                # Calculate probabilities for this larva (only if there are perpendicular head casts)
                if larva_perpendicular_head_casts > 0:
                    larva_towards_prob = larva_towards_wind / larva_perpendicular_head_casts
                    larva_away_prob = larva_away_from_wind / larva_perpendicular_head_casts
                    larva_towards_probs.append(larva_towards_prob)
                    larva_away_probs.append(larva_away_prob)
                    
                    print(f"Larva {larva_id:2d}: {larva_cast_periods:2d} cast periods ({larva_turns:2d} turns), "
                          f"{larva_head_casts:3d} head casts ({larva_perpendicular_head_casts:2d} perpendicular: "
                          f"{larva_towards_wind:2d} towards [{larva_towards_prob:.1%}], "
                          f"{larva_away_from_wind:2d} away [{larva_away_prob:.1%}])")
                else:
                    print(f"Larva {larva_id:2d}: {larva_cast_periods:2d} cast periods ({larva_turns:2d} turns), "
                          f"{larva_head_casts:3d} head casts (0 perpendicular)")
            
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    # Calculate mean probabilities across larvae
    if larva_towards_probs:
        mean_towards_prob = np.mean(larva_towards_probs)
        mean_away_prob = np.mean(larva_away_probs)
        se_towards_prob = stats.sem(larva_towards_probs)
        se_away_prob = stats.sem(larva_away_probs)
    else:
        mean_towards_prob = np.nan
        mean_away_prob = np.nan
        se_towards_prob = np.nan
        se_away_prob = np.nan
    
    # Print summary statistics
    print("-" * 80)
    print(f"TOTAL:     {total_cast_periods:2d} cast periods ({total_turn_casts:2d} turns), "
          f"{total_head_casts:3d} head casts ({total_perpendicular_head_casts:2d} perpendicular)")
    
    if total_perpendicular_head_casts > 0:
        overall_towards_prob = total_towards_wind / total_perpendicular_head_casts
        overall_away_prob = total_away_from_wind / total_perpendicular_head_casts
        print(f"Overall:   {total_towards_wind:2d} towards [{overall_towards_prob:.1%}], "
              f"{total_away_from_wind:2d} away [{overall_away_prob:.1%}]")
    
    if larva_towards_probs:
        print(f"Mean across larvae: {mean_towards_prob:.1%} ± {se_towards_prob:.1%} towards, "
              f"{mean_away_prob:.1%} ± {se_away_prob:.1%} away (n={len(larva_towards_probs)} larvae)")
    
    if total_larvae > 0:
        print(f"Average head casts per larva: {total_head_casts/total_larvae:.1f}")
    if total_cast_periods > 0:
        print(f"Average head casts per cast period: {total_head_casts/total_cast_periods:.1f}")
        print(f"Turn rate: {100*total_turn_casts/total_cast_periods:.1f}% ({total_turn_casts}/{total_cast_periods} casts)")
    
    return all_cast_events

def analyze_head_cast_bias(cast_events_data, analysis_type='first'):
    """
    Analyze bias of head cast direction when larvae are perpendicular to flow.
    Uses already calculated values from detect_head_casts_in_casts.
    
    Args:
        cast_events_data: Output from detect_head_casts_in_casts function
        analysis_type: 'first', 'last', 'all', or 'turn' - which head casts/events to analyze
        
    Returns:
        Dictionary with bias analysis results
    """
    # Collect head cast data for perpendicular orientations
    head_cast_data = []
    larva_summaries = []
    
    print(f"\n🎯 {analysis_type.title()} Head Cast Bias Analysis")
    print("=" * 60)
    
    # Minimum number of perpendicular head casts per larva to include in statistical analysis
    MIN_CASTS_FOR_STATS = 2  # Adjustable threshold
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        larva_towards_count = 0
        larva_away_count = 0
        larva_total_perpendicular_casts = 0
        
        for cast_event in cast_events:
            # Handle 'turn' analysis type - analyze turn direction instead of head casts
            if analysis_type == 'turn':
                # Only consider cast events that are turns and from perpendicular orientations
                if not cast_event['is_turn']:
                    continue
                    
                cast_start_orientation = cast_event['cast_start_orientation']
                # Normalize orientation to -180 to 180 range
                while cast_start_orientation > 180:
                    cast_start_orientation -= 360
                while cast_start_orientation <= -180:
                    cast_start_orientation += 360
                
                # Check if starting from perpendicular orientation
                left_perp_range = (-90 - 20, -90 + 20)
                right_perp_range = (90 - 20, 90 + 20)
                is_perp = ((left_perp_range[0] <= cast_start_orientation <= left_perp_range[1]) or 
                          (right_perp_range[0] <= cast_start_orientation <= right_perp_range[1]))
                
                if not is_perp:
                    continue
                
                turn_direction = cast_event['turn_direction']
                if turn_direction not in ['towards_wind', 'away_from_wind']:
                    continue
                
                # Record the turn direction
                cast_data = {
                    'larva_id': larva_id,
                    'cast_orientation': cast_start_orientation,
                    'head_cast_direction': turn_direction,  # Using same key for consistency
                    'orientation_change': cast_event['orientation_change'],
                    'cast_duration': cast_event['cast_duration'],
                    'total_head_casts_in_event': cast_event['total_head_casts']
                }
                
                head_cast_data.append(cast_data)
                larva_total_perpendicular_casts += 1
                
                if turn_direction == 'towards_wind':
                    larva_towards_count += 1
                elif turn_direction == 'away_from_wind':
                    larva_away_count += 1
                    
            else:
                # Original head cast analysis logic
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
                    # Only include head casts that are perpendicular AND have direction classification
                    if not head_cast['is_perpendicular']:
                        continue
                        
                    direction = head_cast['direction']
                    if direction not in ['towards_wind', 'away_from_wind']:
                        continue
                    
                    # Record the head cast direction
                    cast_data = {
                        'larva_id': larva_id,
                        'cast_orientation': cast_event['cast_start_orientation'],
                        'head_cast_direction': direction,
                        'head_cast_amplitude': head_cast['amplitude'],
                        'peak_orientation': head_cast['peak_orientation'],
                        'bend_angle': head_cast['bend_angle'],
                        'total_head_casts_in_event': cast_event['total_head_casts'],
                        'cast_duration': cast_event['cast_duration']
                    }
                    
                    head_cast_data.append(cast_data)
                    larva_total_perpendicular_casts += 1
                    
                    if direction == 'towards_wind':
                        larva_towards_count += 1
                    elif direction == 'away_from_wind':
                        larva_away_count += 1
        
        # Store per-larva summary (only if there are perpendicular head casts/turns)
        if larva_total_perpendicular_casts > 0:
            larva_bias_towards = larva_towards_count / larva_total_perpendicular_casts
            larva_bias_away = larva_away_count / larva_total_perpendicular_casts
            
            larva_summaries.append({
                'larva_id': larva_id,
                'towards_count': larva_towards_count,
                'away_count': larva_away_count,
                'total_count': larva_total_perpendicular_casts,
                'towards_bias': larva_bias_towards,
                'away_bias': larva_bias_away
            })
            
            event_type = "turns" if analysis_type == 'turn' else f"{analysis_type} perpendicular head casts"
            print(f"Larva {larva_id:2d}: {larva_total_perpendicular_casts:2d} {event_type} "
                  f"({larva_towards_count:2d} towards [{larva_bias_towards:.1%}], "
                  f"{larva_away_count:2d} away [{larva_bias_away:.1%}])")
    
    if not head_cast_data:
        event_type = "perpendicular turns" if analysis_type == 'turn' else "perpendicular head casts"
        print(f"No {event_type} found for analysis.")
        return {
            'head_cast_data': [],
            'larva_summaries': [],
            'total_towards': 0,
            'total_away': 0,
            'total_casts': 0,
            'overall_towards_bias': np.nan,
            'overall_away_bias': np.nan,
            'mean_larva_towards_bias': np.nan,
            'se_larva_towards_bias': np.nan,
            'mean_larva_away_bias': np.nan,
            'se_larva_away_bias': np.nan,
            'p_value_wilcoxon': np.nan,
            'p_value_ttest': np.nan,
            'p_value_fisher_combined': np.nan,
            'n_larvae': 0,
            'n_larvae_for_stats': 0,
            'analysis_type': analysis_type
        }
    
    # Calculate overall statistics
    total_towards = sum(1 for cast in head_cast_data if cast['head_cast_direction'] == 'towards_wind')
    total_away = sum(1 for cast in head_cast_data if cast['head_cast_direction'] == 'away_from_wind')
    total_casts = len(head_cast_data)
    
    overall_towards_bias = total_towards / total_casts
    overall_away_bias = total_away / total_casts
    
    # Per-larva bias statistics
    larva_towards_biases = [summary['towards_bias'] for summary in larva_summaries]
    larva_away_biases = [summary['away_bias'] for summary in larva_summaries]
    
    mean_larva_towards_bias = np.mean(larva_towards_biases)
    se_larva_towards_bias = stats.sem(larva_towards_biases)
    mean_larva_away_bias = np.mean(larva_away_biases)
    se_larva_away_bias = stats.sem(larva_away_biases)
    
    # === STATISTICAL TESTS ===
    # Filter larvae with sufficient data for robust statistical testing
    larvae_for_stats = [summary for summary in larva_summaries 
                       if summary['total_count'] >= MIN_CASTS_FOR_STATS]
    
    if len(larvae_for_stats) >= 3:  # Need at least 3 larvae for meaningful stats
        filtered_towards_biases = [summary['towards_bias'] for summary in larvae_for_stats]
        
        # PRIMARY TEST: Wilcoxon signed-rank test (non-parametric)
        from scipy.stats import wilcoxon
        # Test if median bias differs from 0.5
        deviations_from_chance = np.array(filtered_towards_biases) - 0.5
        if np.any(deviations_from_chance != 0):  # Avoid all-zeros case
            w_stat, p_wilcoxon = wilcoxon(deviations_from_chance, alternative='two-sided')
        else:
            p_wilcoxon = 1.0
        
        # Alternative tests
        from scipy.stats import ttest_1samp
        t_stat, p_ttest = ttest_1samp(filtered_towards_biases, 0.5)
        
        # Per-larva binomial tests combined (Fisher's method)
        from scipy.stats import combine_pvalues, binomtest
        individual_pvals = []
        for summary in larvae_for_stats:
            # Binomial test for each individual larva
            binom_result = binomtest(summary['towards_count'], 
                                   summary['total_count'], 
                                   0.5, alternative='two-sided')
            individual_pvals.append(binom_result.pvalue)
        
        # Combine p-values using Fisher's method
        if individual_pvals:
            fisher_stat, p_fisher = combine_pvalues(individual_pvals, method='fisher')
        else:
            p_fisher = np.nan
        
        print(f"\n📊 Statistical Tests (n={len(larvae_for_stats)} larvae with ≥{MIN_CASTS_FOR_STATS} events):")
        print(f"   Wilcoxon signed-rank test: p={p_wilcoxon:.4f}")
        
        # Interpretation
        alpha = 0.05
        if p_wilcoxon < alpha:
            direction = "towards" if mean_larva_towards_bias > 0.5 else "away from"
            print(f"   ✓ Significant bias {direction} wind (Wilcoxon p < {alpha})")
        else:
            print(f"   ✗ No significant bias detected (Wilcoxon p ≥ {alpha})")
            
    else:
        print(f"\n⚠️  Insufficient data for robust statistical testing")
        print(f"   Only {len(larvae_for_stats)} larvae with ≥{MIN_CASTS_FOR_STATS} events")
        p_wilcoxon = np.nan
        p_ttest = np.nan
        p_fisher = np.nan
    
    print("-" * 60)
    event_type = "perpendicular turns" if analysis_type == 'turn' else f"{analysis_type} perpendicular head casts"
    print(f"TOTAL: {total_casts} {event_type}")
    print(f"Overall: {total_towards} towards [{overall_towards_bias:.1%}], {total_away} away [{overall_away_bias:.1%}]")
    print(f"Mean across larvae: {mean_larva_towards_bias:.1%} ± {se_larva_towards_bias:.1%} towards, "
          f"{mean_larva_away_bias:.1%} ± {se_larva_away_bias:.1%} away (n={len(larva_summaries)} larvae)")
    
    return {
        'head_cast_data': head_cast_data,
        'larva_summaries': larva_summaries,
        'larvae_for_stats': larvae_for_stats,
        'total_towards': total_towards,
        'total_away': total_away,
        'total_casts': total_casts,
        'overall_towards_bias': overall_towards_bias,
        'overall_away_bias': overall_away_bias,
        'mean_larva_towards_bias': mean_larva_towards_bias,
        'se_larva_towards_bias': se_larva_towards_bias,
        'mean_larva_away_bias': mean_larva_away_bias,
        'se_larva_away_bias': se_larva_away_bias,
        'p_value_wilcoxon': p_wilcoxon if 'p_wilcoxon' in locals() else np.nan,
        'p_value_ttest': p_ttest if 'p_ttest' in locals() else np.nan,
        'p_value_fisher_combined': p_fisher if 'p_fisher' in locals() else np.nan,
        'min_casts_threshold': MIN_CASTS_FOR_STATS,
        'n_larvae': len(larva_summaries),
        'n_larvae_for_stats': len(larvae_for_stats) if 'larvae_for_stats' in locals() else 0,
        'analysis_type': analysis_type
    }
    
    # Calculate overall statistics
    total_towards = sum(1 for cast in head_cast_data if cast['head_cast_direction'] == 'towards_wind')
    total_away = sum(1 for cast in head_cast_data if cast['head_cast_direction'] == 'away_from_wind')
    total_casts = len(head_cast_data)
    
    overall_towards_bias = total_towards / total_casts
    overall_away_bias = total_away / total_casts
    
    # Per-larva bias statistics
    larva_towards_biases = [summary['towards_bias'] for summary in larva_summaries]
    larva_away_biases = [summary['away_bias'] for summary in larva_summaries]
    
    mean_larva_towards_bias = np.mean(larva_towards_biases)
    se_larva_towards_bias = stats.sem(larva_towards_biases)
    mean_larva_away_bias = np.mean(larva_away_biases)
    se_larva_away_bias = stats.sem(larva_away_biases)
    
    # === STATISTICAL TESTS ===
    # Filter larvae with sufficient data for robust statistical testing
    larvae_for_stats = [summary for summary in larva_summaries 
                       if summary['total_count'] >= MIN_CASTS_FOR_STATS]
    
    if len(larvae_for_stats) >= 3:  # Need at least 3 larvae for meaningful stats
        filtered_towards_biases = [summary['towards_bias'] for summary in larvae_for_stats]
        
        # PRIMARY TEST: Wilcoxon signed-rank test (non-parametric)
        from scipy.stats import wilcoxon
        # Test if median bias differs from 0.5
        deviations_from_chance = np.array(filtered_towards_biases) - 0.5
        if np.any(deviations_from_chance != 0):  # Avoid all-zeros case
            w_stat, p_wilcoxon = wilcoxon(deviations_from_chance, alternative='two-sided')
        else:
            p_wilcoxon = 1.0
        
        # # ALTERNATIVE TESTS (commented out)
        # 1. One-sample t-test: Are per-larva towards biases significantly different from 0.5?
        from scipy.stats import ttest_1samp
        t_stat, p_ttest = ttest_1samp(filtered_towards_biases, 0.5)
        
        # 2. Per-larva binomial tests combined (Fisher's method)
        from scipy.stats import combine_pvalues, binomtest
        individual_pvals = []
        for summary in larvae_for_stats:
            # Binomial test for each individual larva
            binom_result = binomtest(summary['towards_count'], 
                                   summary['total_count'], 
                                   0.5, alternative='two-sided')
            individual_pvals.append(binom_result.pvalue)
        
        # Combine p-values using Fisher's method
        if individual_pvals:
            fisher_stat, p_fisher = combine_pvalues(individual_pvals, method='fisher')
        else:
            p_fisher = np.nan
        
        print(f"\n📊 Statistical Tests (n={len(larvae_for_stats)} larvae with ≥{MIN_CASTS_FOR_STATS} casts):")
        print(f"   Wilcoxon signed-rank test: p={p_wilcoxon:.4f}")
        # print(f"   One-sample t-test vs 0.5: t={t_stat:.3f}, p={p_ttest:.4f}")
        # print(f"   Combined binomial tests (Fisher): p={p_fisher:.4f}")
        
        # Interpretation
        alpha = 0.05
        if p_wilcoxon < alpha:
            direction = "towards" if mean_larva_towards_bias > 0.5 else "away from"
            print(f"   ✓ Significant bias {direction} wind (Wilcoxon p < {alpha})")
        else:
            print(f"   ✗ No significant bias detected (Wilcoxon p ≥ {alpha})")
            
    else:
        print(f"\n⚠️  Insufficient data for robust statistical testing")
        print(f"   Only {len(larvae_for_stats)} larvae with ≥{MIN_CASTS_FOR_STATS} perpendicular head casts")
        p_wilcoxon = np.nan
        p_ttest = np.nan
        p_fisher = np.nan
    
    print("-" * 60)
    print(f"TOTAL: {total_casts} {analysis_type} perpendicular head casts")
    print(f"Overall: {total_towards} towards [{overall_towards_bias:.1%}], {total_away} away [{overall_away_bias:.1%}]")
    print(f"Mean across larvae: {mean_larva_towards_bias:.1%} ± {se_larva_towards_bias:.1%} towards, "
          f"{mean_larva_away_bias:.1%} ± {se_larva_away_bias:.1%} away (n={len(larva_summaries)} larvae)")
    
    return {
        'head_cast_data': head_cast_data,
        'larva_summaries': larva_summaries,
        'larvae_for_stats': larvae_for_stats,
        'total_towards': total_towards,
        'total_away': total_away,
        'total_casts': total_casts,
        'overall_towards_bias': overall_towards_bias,
        'overall_away_bias': overall_away_bias,
        'mean_larva_towards_bias': mean_larva_towards_bias,
        'se_larva_towards_bias': se_larva_towards_bias,
        'mean_larva_away_bias': mean_larva_away_bias,
        'se_larva_away_bias': se_larva_away_bias,
        'p_value_wilcoxon': p_wilcoxon if 'p_wilcoxon' in locals() else np.nan,
        'p_value_ttest': p_ttest if 'p_ttest' in locals() else np.nan,
        'p_value_fisher_combined': p_fisher if 'p_fisher' in locals() else np.nan,
        'min_casts_threshold': MIN_CASTS_FOR_STATS,
        'n_larvae': len(larva_summaries),
        'n_larvae_for_stats': len(larvae_for_stats) if 'larvae_for_stats' in locals() else 0,
        'analysis_type': analysis_type
    }




def analyze_head_cast_bias_pooled(cast_events_data, analysis_type='first'):
    """
    Analyze bias of head cast direction when larvae are perpendicular to flow.
    POOLS all head casts across larvae for analysis (similar to velocity pooled approach).
    
    Args:
        cast_events_data: Output from detect_head_casts_in_casts function
        analysis_type: 'first', 'last', 'all', or 'turn' - which head casts/events to analyze
        
    Returns:
        Dictionary with pooled bias analysis results
    """
    # Collect ALL head cast data for perpendicular orientations (pooled across larvae)
    all_head_casts = []
    larva_summaries = []
    
    print(f"\n🎯 {analysis_type.title()} Head Cast Bias Analysis (POOLED)")
    print("=" * 60)
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        larva_towards_count = 0
        larva_away_count = 0
        larva_total_perpendicular_casts = 0
        
        for cast_event in cast_events:
            # Handle 'turn' analysis type - analyze turn direction instead of head casts
            if analysis_type == 'turn':
                # Only consider cast events that are turns and from perpendicular orientations
                if not cast_event['is_turn']:
                    continue
                    
                cast_start_orientation = cast_event['cast_start_orientation']
                # Normalize orientation to -180 to 180 range
                while cast_start_orientation > 180:
                    cast_start_orientation -= 360
                while cast_start_orientation <= -180:
                    cast_start_orientation += 360
                
                # Check if starting from perpendicular orientation
                left_perp_range = (-90 - 20, -90 + 20)
                right_perp_range = (90 - 20, 90 + 20)
                is_perp = ((left_perp_range[0] <= cast_start_orientation <= left_perp_range[1]) or 
                          (right_perp_range[0] <= cast_start_orientation <= right_perp_range[1]))
                
                if not is_perp:
                    continue
                
                turn_direction = cast_event['turn_direction']
                if turn_direction not in ['towards_wind', 'away_from_wind']:
                    continue
                
                # Add to pooled data
                all_head_casts.append({
                    'larva_id': larva_id,
                    'direction': turn_direction,
                    'cast_orientation': cast_start_orientation,
                    'event_type': 'turn'
                })
                
                larva_total_perpendicular_casts += 1
                if turn_direction == 'towards_wind':
                    larva_towards_count += 1
                elif turn_direction == 'away_from_wind':
                    larva_away_count += 1
                    
            else:
                # Original head cast analysis logic
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
                    # Only include head casts that are perpendicular AND have direction classification
                    if not head_cast['is_perpendicular']:
                        continue
                        
                    direction = head_cast['direction']
                    if direction not in ['towards_wind', 'away_from_wind']:
                        continue
                    
                    # Add to pooled data
                    all_head_casts.append({
                        'larva_id': larva_id,
                        'direction': direction,
                        'cast_orientation': cast_event['cast_start_orientation'],
                        'amplitude': head_cast['amplitude'],
                        'peak_orientation': head_cast['peak_orientation'],
                        'bend_angle': head_cast['bend_angle'],
                        'event_type': 'head_cast'
                    })
                    
                    larva_total_perpendicular_casts += 1
                    if direction == 'towards_wind':
                        larva_towards_count += 1
                    elif direction == 'away_from_wind':
                        larva_away_count += 1
        
        # Store per-larva summary for reference - INCLUDE BOTH towards_bias AND away_bias
        if larva_total_perpendicular_casts > 0:
            larva_bias_towards = larva_towards_count / larva_total_perpendicular_casts
            larva_bias_away = larva_away_count / larva_total_perpendicular_casts  # ADD THIS LINE
            
            larva_summaries.append({
                'larva_id': larva_id,
                'towards_count': larva_towards_count,
                'away_count': larva_away_count,
                'total_count': larva_total_perpendicular_casts,
                'towards_bias': larva_bias_towards,
                'away_bias': larva_bias_away  # ADD THIS LINE
            })
            
            event_type = "turns" if analysis_type == 'turn' else f"{analysis_type} perpendicular head casts"
            print(f"Larva {larva_id:2d}: {larva_total_perpendicular_casts:2d} {event_type} "
                  f"({larva_towards_count:2d} towards [{larva_bias_towards:.1%}], "
                  f"{larva_away_count:2d} away [{1-larva_bias_towards:.1%}])")
    
    if not all_head_casts:
        event_type = "perpendicular turns" if analysis_type == 'turn' else "perpendicular head casts"
        print(f"No {event_type} found for analysis.")
        return {
            'head_cast_data': [],
            'larva_summaries': [],
            'total_towards': 0,
            'total_away': 0,
            'total_casts': 0,
            'pooled_towards_bias': np.nan,
            'pooled_away_bias': np.nan,
            'confidence_interval': (np.nan, np.nan),
            'p_value_binomial': np.nan,
            'n_larvae': 0,
            'analysis_type': analysis_type,
            'method': 'pooled'  # ADD THIS LINE
        }
    
    # === POOLED ANALYSIS ===
    # Count total across all larvae
    total_towards = sum(1 for cast in all_head_casts if cast['direction'] == 'towards_wind')
    total_away = sum(1 for cast in all_head_casts if cast['direction'] == 'away_from_wind')
    total_casts = len(all_head_casts)
    
    pooled_towards_bias = total_towards / total_casts
    pooled_away_bias = total_away / total_casts
    
    # === STATISTICAL TEST (POOLED) ===
    from scipy.stats import binomtest
    
    # Primary test: Binomial test on pooled data
    # H0: probability of towards_wind = 0.5 (no bias)
    # H1: probability ≠ 0.5 (bias exists)
    binom_result = binomtest(total_towards, total_casts, 0.5, alternative='two-sided')
    p_binomial = binom_result.pvalue
    
    # Calculate confidence interval for the bias
    confidence_interval = binom_result.proportion_ci(confidence_level=0.95)
    
    # === RESULTS ===
    print("-" * 60)
    event_type = "perpendicular turns" if analysis_type == 'turn' else f"{analysis_type} perpendicular head casts"
    print(f"POOLED ANALYSIS: {total_casts} {event_type}")
    print(f"Total: {total_towards} towards [{pooled_towards_bias:.1%}], {total_away} away [{pooled_away_bias:.1%}]")
    print(f"95% CI for towards bias: [{confidence_interval.low:.1%}, {confidence_interval.high:.1%}]")
    
    print(f"\n📊 Statistical Test (pooled binomial test):")
    print(f"   Binomial test: p={p_binomial:.4f}")
    
    # Interpretation
    alpha = 0.05
    if p_binomial < alpha:
        direction = "towards" if pooled_towards_bias > 0.5 else "away from"
        print(f"   ✓ Significant bias {direction} wind (p < {alpha})")
    else:
        print(f"   ✗ No significant bias detected (p ≥ {alpha})")
    
    # Compare with per-larva means for reference
    if larva_summaries:
        larva_towards_biases = [summary['towards_bias'] for summary in larva_summaries]
        mean_larva_bias = np.mean(larva_towards_biases)
        se_larva_bias = stats.sem(larva_towards_biases)
        print(f"\nComparison - Mean per-larva bias: {mean_larva_bias:.1%} ± {se_larva_bias:.1%} "
              f"(n={len(larva_summaries)} larvae)")
    
    return {
        'head_cast_data': all_head_casts,           # All individual head casts (pooled)
        'larva_summaries': larva_summaries,         # Per-larva summaries for reference
        'total_towards': total_towards,
        'total_away': total_away,
        'total_casts': total_casts,
        'pooled_towards_bias': pooled_towards_bias,
        'pooled_away_bias': pooled_away_bias,
        'confidence_interval': confidence_interval,
        'p_value_binomial': p_binomial,
        'n_larvae': len(larva_summaries),
        'analysis_type': analysis_type,
        'method': 'pooled'
    }


def analyze_head_cast_bias_categorical_bootstrap(cast_events_data, analysis_type='turn', n_bootstrap=1000):
    """
    Analyze head cast bias using categorical approach with bootstrap confidence intervals.
    Each larva is classified as: biased_towards, biased_away, or unbiased.
    Uses Fisher's exact test and bootstrap for robust inference with sparse data.
    
    Args:
        cast_events_data: Output from detect_head_casts_in_casts function
        analysis_type: 'first', 'last', 'all', or 'turn'
        n_bootstrap: Number of bootstrap samples for confidence intervals
        
    Returns:
        Dictionary with categorical bias analysis results
    """
    from scipy.stats import fisher_exact, binomtest
    import numpy as np
    
    print(f"\n🎯 {analysis_type.title()} Head Cast Bias Analysis (CATEGORICAL + BOOTSTRAP)")
    print("=" * 80)
    
    larva_classifications = []
    larva_raw_data = []
    
    for larva_id, cast_events in cast_events_data.items():
        if not cast_events:
            continue
            
        larva_towards_count = 0
        larva_away_count = 0
        larva_total_perpendicular_casts = 0
        
        # [Same data collection logic as before]
        for cast_event in cast_events:
            if analysis_type == 'turn':
                if not cast_event['is_turn']:
                    continue
                    
                cast_start_orientation = cast_event['cast_start_orientation']
                while cast_start_orientation > 180:
                    cast_start_orientation -= 360
                while cast_start_orientation <= -180:
                    cast_start_orientation += 360
                
                left_perp_range = (-90 - 20, -90 + 20)
                right_perp_range = (90 - 20, 90 + 20)
                is_perp = ((left_perp_range[0] <= cast_start_orientation <= left_perp_range[1]) or 
                          (right_perp_range[0] <= cast_start_orientation <= right_perp_range[1]))
                
                if not is_perp:
                    continue
                
                turn_direction = cast_event['turn_direction']
                if turn_direction not in ['towards_wind', 'away_from_wind']:
                    continue
                
                larva_total_perpendicular_casts += 1
                if turn_direction == 'towards_wind':
                    larva_towards_count += 1
                elif turn_direction == 'away_from_wind':
                    larva_away_count += 1
                    
            else:
                # Head cast analysis logic
                if not cast_event['head_cast_details']:
                    continue
                    
                if analysis_type == 'first' and len(cast_event['head_cast_details']) > 0:
                    head_casts_to_analyze = [cast_event['head_cast_details'][0]]
                elif analysis_type == 'last' and len(cast_event['head_cast_details']) > 0:
                    head_casts_to_analyze = [cast_event['head_cast_details'][-1]]
                elif analysis_type == 'all':
                    head_casts_to_analyze = cast_event['head_cast_details']
                else:
                    continue
                
                for head_cast in head_casts_to_analyze:
                    if not head_cast['is_perpendicular']:
                        continue
                        
                    direction = head_cast['direction']
                    if direction not in ['towards_wind', 'away_from_wind']:
                        continue
                    
                    larva_total_perpendicular_casts += 1
                    if direction == 'towards_wind':
                        larva_towards_count += 1
                    elif direction == 'away_from_wind':
                        larva_away_count += 1
        
        # Store raw data for bootstrap
        if larva_total_perpendicular_casts >= 1:  # Minimum 1 event to classify
            larva_raw_data.append({
                'larva_id': larva_id,
                'towards_count': larva_towards_count,
                'away_count': larva_away_count,
                'total_count': larva_total_perpendicular_casts
            })
            
            # Classify larva based on majority
            if larva_towards_count > larva_away_count:
                classification = 'biased_towards'
            elif larva_away_count > larva_towards_count:
                classification = 'biased_away'
            else:
                classification = 'unbiased'
            
            larva_classifications.append({
                'larva_id': larva_id,
                'classification': classification,
                'towards_count': larva_towards_count,
                'away_count': larva_away_count,
                'total_count': larva_total_perpendicular_casts,
                'towards_bias': larva_towards_count / larva_total_perpendicular_casts
            })
            
            print(f"Larva {larva_id:2d}: {larva_total_perpendicular_casts:2d} events → {classification:15s} "
                  f"({larva_towards_count:2d} towards, {larva_away_count:2d} away)")
    
    if not larva_classifications:
        return {
            'larva_classifications': [],
            'n_biased_towards': 0,
            'n_biased_away': 0,
            'n_unbiased': 0,
            'p_value_categorical': np.nan,
            'analysis_type': analysis_type,
            'method': 'categorical_bootstrap'
        }
    
    # === CATEGORICAL ANALYSIS ===
    n_towards = sum(1 for l in larva_classifications if l['classification'] == 'biased_towards')
    n_away = sum(1 for l in larva_classifications if l['classification'] == 'biased_away')
    n_unbiased = sum(1 for l in larva_classifications if l['classification'] == 'unbiased')
    n_total_larvae = len(larva_classifications)
    
    # === STATISTICAL TESTS ===
    
    # 1. Binomial test: Are more larvae biased towards than expected by chance?
    # Under null hypothesis: P(biased_towards) = P(biased_away) = equal
    # We test: n_towards vs (n_towards + n_away), expected p = 0.5
    if n_towards + n_away > 0:
        p_binomial = binomtest(n_towards, n_towards + n_away, 0.5, alternative='two-sided').pvalue
    else:
        p_binomial = np.nan
    
    # 2. Fisher's exact test: biased_towards vs (biased_away + unbiased)
    if n_total_larvae > 2:
        table = [[n_towards, n_away + n_unbiased], 
                [n_total_larvae, n_total_larvae]]  # Expected vs observed
        try:
            _, p_fisher = fisher_exact([[n_towards, n_away + n_unbiased], 
                                       [n_total_larvae//2, n_total_larvae//2]], 
                                      alternative='two-sided')
        except:
            p_fisher = np.nan
    else:
        p_fisher = np.nan
    
    # === BOOTSTRAP CONFIDENCE INTERVALS ===
    def bootstrap_categorical_proportion(raw_data, n_bootstrap=1000):
        """Bootstrap the proportion of larvae biased towards wind"""
        bootstrap_proportions = []
        n_larvae = len(raw_data)
        
        for _ in range(n_bootstrap):
            # Resample larvae with replacement
            bootstrap_sample = np.random.choice(range(n_larvae), size=n_larvae, replace=True)
            
            bootstrap_towards = 0
            bootstrap_away = 0
            
            for idx in bootstrap_sample:
                larva = raw_data[idx]
                if larva['towards_count'] > larva['away_count']:
                    bootstrap_towards += 1
                elif larva['away_count'] > larva['towards_count']:
                    bootstrap_away += 1
                # Unbiased larvae contribute to neither
            
            if bootstrap_towards + bootstrap_away > 0:
                proportion = bootstrap_towards / (bootstrap_towards + bootstrap_away)
                bootstrap_proportions.append(proportion)
        
        return np.array(bootstrap_proportions)
    
    if len(larva_raw_data) >= 3:
        bootstrap_props = bootstrap_categorical_proportion(larva_raw_data, n_bootstrap)
        if len(bootstrap_props) > 0:
            ci_lower = np.percentile(bootstrap_props, 2.5)
            ci_upper = np.percentile(bootstrap_props, 97.5)
            bootstrap_mean = np.mean(bootstrap_props)
        else:
            ci_lower = ci_upper = bootstrap_mean = np.nan
    else:
        ci_lower = ci_upper = bootstrap_mean = np.nan
    
    # === RESULTS ===
    print("-" * 80)
    print(f"CATEGORICAL CLASSIFICATION:")
    print(f"  Biased towards wind: {n_towards:2d} larvae ({100*n_towards/n_total_larvae:4.1f}%)")
    print(f"  Biased away from wind: {n_away:2d} larvae ({100*n_away/n_total_larvae:4.1f}%)")
    print(f"  Unbiased: {n_unbiased:2d} larvae ({100*n_unbiased/n_total_larvae:4.1f}%)")
    
    if n_towards + n_away > 0:
        observed_proportion = n_towards / (n_towards + n_away)
        print(f"\nAmong biased larvae: {observed_proportion:.1%} towards wind")
        print(f"Bootstrap 95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
    
    print(f"\n📊 Statistical Tests:")
    print(f"   Binomial test: p={p_binomial:.4f}")
    
    # Interpretation
    alpha = 0.05
    if not np.isnan(p_binomial) and p_binomial < alpha:
        direction = "towards" if n_towards > n_away else "away from"
        print(f"   ✓ Significant categorical bias {direction} wind (p < {alpha})")
    else:
        print(f"   ✗ No significant categorical bias detected (p ≥ {alpha})")
    
    return {
        'larva_classifications': larva_classifications,
        'larva_raw_data': larva_raw_data,
        'n_biased_towards': n_towards,
        'n_biased_away': n_away,
        'n_unbiased': n_unbiased,
        'n_total_larvae': n_total_larvae,
        'observed_proportion_towards': n_towards / (n_towards + n_away) if n_towards + n_away > 0 else np.nan,
        'p_value_categorical': p_binomial,
        'p_value_fisher': p_fisher if 'p_fisher' in locals() else np.nan,
        'bootstrap_ci': (ci_lower, ci_upper),
        'bootstrap_mean': bootstrap_mean,
        'bootstrap_proportions': bootstrap_props if 'bootstrap_props' in locals() else [],
        'analysis_type': analysis_type,
        'method': 'categorical_bootstrap'
    }

def choose_appropriate_bias_test(cast_events_data, analysis_type='turn', min_events_for_continuous=5):
    """
    Automatically choose between categorical and continuous bias tests based on data characteristics.
    
    Args:
        cast_events_data: Output from detect_head_casts_in_casts function
        analysis_type: 'first', 'last', 'all', or 'turn'
        min_events_for_continuous: Minimum events per larva to use continuous tests
        
    Returns:
        Dictionary with bias analysis results using the most appropriate method
    """
    # Quick scan of data characteristics
    larvae_with_sufficient_data = 0
    total_events = 0
    event_counts = []
    
    for larva_id, cast_events in cast_events_data.items():
        larva_event_count = 0
        
        for cast_event in cast_events:
            if analysis_type == 'turn':
                if cast_event['is_turn']:
                    # Check if perpendicular (same logic as in analysis functions)
                    cast_start_orientation = cast_event['cast_start_orientation']
                    while cast_start_orientation > 180:
                        cast_start_orientation -= 360
                    while cast_start_orientation <= -180:
                        cast_start_orientation += 360
                    
                    left_perp_range = (-90 - 20, -90 + 20)
                    right_perp_range = (90 - 20, 90 + 20)
                    is_perp = ((left_perp_range[0] <= cast_start_orientation <= left_perp_range[1]) or 
                              (right_perp_range[0] <= cast_start_orientation <= right_perp_range[1]))
                    
                    if is_perp and cast_event['turn_direction'] in ['towards_wind', 'away_from_wind']:
                        larva_event_count += 1
            else:
                # Count perpendicular head casts
                for head_cast in cast_event.get('head_cast_details', []):
                    if (head_cast['is_perpendicular'] and 
                        head_cast['direction'] in ['towards_wind', 'away_from_wind']):
                        larva_event_count += 1
        
        if larva_event_count > 0:
            event_counts.append(larva_event_count)
            total_events += larva_event_count
            if larva_event_count >= min_events_for_continuous:
                larvae_with_sufficient_data += 1
    
    # Decision criteria
    mean_events_per_larva = np.mean(event_counts) if event_counts else 0
    proportion_with_sufficient_data = larvae_with_sufficient_data / len(event_counts) if event_counts else 0
    
    print(f"\n🤖 Automatic Method Selection for {analysis_type} analysis:")
    print(f"   Mean events per larva: {mean_events_per_larva:.1f}")
    print(f"   Larvae with ≥{min_events_for_continuous} events: {proportion_with_sufficient_data:.1%}")
    
    # Use categorical approach if:
    # 1. Few events per larva on average (< 5), OR
    # 2. Few larvae have sufficient data (< 50%), OR
    # 3. This is turn analysis (typically sparse)
    use_categorical = (mean_events_per_larva < min_events_for_continuous or 
                      proportion_with_sufficient_data < 0.5 or 
                      analysis_type == 'turn')
    
    if use_categorical:
        print(f"   → Using CATEGORICAL + BOOTSTRAP approach")
        return analyze_head_cast_bias_categorical_bootstrap(cast_events_data, analysis_type)
    else:
        print(f"   → Using CONTINUOUS (Wilcoxon) approach")
        return analyze_head_cast_bias(cast_events_data, analysis_type)
    
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