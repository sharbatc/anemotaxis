import multiprocessing as mp
import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
plt.style.use('../anemotaxis.mplstyle')

def get_behavior_data(f, field, i):
    """Extract behavior-related cell arrays from MATLAB struct.
    
    Args:
        f: HDF5 file object
        field: Field name (e.g., 'duration_large', 't_start_stop_large', 'duration_large_small')
        i: Larva index
        
    Returns:
        list: List of arrays, one for each behavior type
            - For 'duration_large' fields: List of 7 arrays for standard behavior types
            - For 'duration_large_small' fields: List of 12 arrays for detailed behavior types
    """
    try:
        # Get the reference to the cell array
        cell_ref = f['trx'][field][0][i]
        if not isinstance(cell_ref, h5py.h5r.Reference):
            return None
            
        # Get the 1xN cell array (either 1x7 or 1x12)
        cell_array = f[cell_ref]
        
        # Determine if this is a large_small field (has 12 elements) or regular field (7 elements)
        num_elements = cell_array.shape[0]
        
        # Initialize list to store arrays for each behavior
        behavior_data = []
        
        # Extract each behavior's data (7 or 12 behaviors total)
        for j in range(num_elements):
            try:
                # Get reference to data for this behavior
                behavior_ref = cell_array[j,0]  # MATLAB stores in column-major order
                if isinstance(behavior_ref, h5py.h5r.Reference):
                    behavior_array = f[behavior_ref]
                    behavior_data.append(np.array(behavior_array))
                else:
                    behavior_data.append(np.array([]))
            except Exception as e:
                tqdm.write(f"Error getting behavior {j+1} data: {str(e)}")
                behavior_data.append(np.array([]))
        
        return behavior_data
        
    except Exception as e:
        tqdm.write(f"Error getting behavior data for field {field}: {str(e)}")
        return None

def process_single_file(file_path, show_progress=False):
    """Process a single trx.mat file containing larval tracking data.
    
    This function extracts behavioral and tracking data from a MATLAB-generated trx.mat file.
    The file contains tracking information for multiple larvae including:
    - Time series data
    - Spine and contour coordinates 
    - Body part positions (head, tail, etc.)
    - Behavioral state information
    
    Shape information for key arrays:
    - t: (steps,) - Time points for each frame
    - x_spine, y_spine: (steps, 11) - 11 points along the larva's spine
    - x_contour, y_contour: (steps, 500) - 500 points defining the larva's outline
    - x_center, y_center, etc.: (steps,) - Track points for body parts
    - global_state_large_state: (steps,) - Behavioral state per frame
        States: 1=run, 2=cast, 3=stop, 4=hunch, 5=backup, 6=roll, 7=small actions
    - t_start_stop_large: List of 7 arrays - Start/stop times for each behavior type
    - duration_large: List of 7 arrays - Duration of each behavior type
    - nb_action_large: List of 7 arrays - Number of events per behavior type
    
    Args:
        file_path (str): Path to the trx.mat file
        show_progress (bool): Whether to show progress messages
        
    Returns:
        tuple: (date_str, extracted_data, metadata)
            - date_str (str): Experiment date from folder name
            - extracted_data (dict): Dictionary of larva_id -> tracking data
            - metadata (dict): File info including path, date, number of larvae
    """
    try:
        with h5py.File(file_path, 'r') as f:
            fields = list(f['trx'].keys())
            nb_larvae = f['trx'][fields[0]].shape[1]
            
            if show_progress:
                print(f"\nProcessing file: {file_path}")
                print(f"Number of larvae: {nb_larvae}")
            
            extracted_data = {}
            for i in tqdm(range(nb_larvae), desc="Processing larvae"):
                larva = {}
                try:
                    # Helper function to safely extract array data
                    def get_array(field):
                        ref = f['trx'][field][0][i]
                        if isinstance(ref, h5py.Dataset):
                            return np.array(ref)
                        return np.array(f[ref])
                    
                    # Time series data
                    larva['t'] = get_array('t')  # (steps,)

                    # Head & Tail Velocity data
                    larva['head_velocity_norm_smooth_5'] = get_array('head_velocity_norm_smooth_5')  # (steps,)
                    larva['tail_velocity_norm_smooth_5'] = get_array('tail_velocity_norm_smooth_5')  # (steps,)
                    larva['angle_upper_lower_smooth_5'] = get_array('angle_upper_lower_smooth_5')  # (steps,)
                    larva['angle_downer_upper_smooth_5'] = get_array('angle_downer_upper_smooth_5')  # (steps,)
                    
                    # Spine coordinates
                    larva['x_spine'] = get_array('x_spine')  # (steps, 11)
                    larva['y_spine'] = get_array('y_spine')  # (steps, 11)
                    
                    # Contour coordinates  
                    larva['x_contour'] = get_array('x_contour')  # (steps, 500)
                    larva['y_contour'] = get_array('y_contour')  # (steps, 500)
                    
                    # Single point tracking coordinates
                    for point in ['center', 'neck', 'head', 'tail', 'neck_down', 'neck_top']:
                        larva[f'x_{point}'] = get_array(f'x_{point}')  # (steps,)
                        larva[f'y_{point}'] = get_array(f'y_{point}')  # (steps,)
                    
                    # Behavioral state data
                    larva['global_state_large_state'] = get_array('global_state_large_state')
                    larva['global_state_small_large_state'] = get_array('global_state_small_large_state')
                    
                    # Get behavior metrics - each will be list of 7 arrays
                    larva['duration_large'] = get_behavior_data(f, 'duration_large', i)
                    larva['t_start_stop_large'] = get_behavior_data(f, 't_start_stop_large', i)
                    larva['start_stop_large'] = get_behavior_data(f, 'start_stop_large', i)
                    larva['nb_action_large'] = get_behavior_data(f, 'nb_action_large', i)

                    # Get behavior metrics - each will be list of 12 arrays
                    larva['duration_large_small'] = get_behavior_data(f, 'duration_large_small', i)
                    larva['t_start_stop_large_small'] = get_behavior_data(f, 't_start_stop_large_small', i)
                    larva['start_stop_large_small'] = get_behavior_data(f, 'start_stop_large_small', i)
                    larva['nb_action_large_small'] = get_behavior_data(f, 'nb_action_large_small', i)
                    
                    # Get larva ID
                    larva_id_ref = f['trx']['numero_larva_num'][0][i]
                    if isinstance(larva_id_ref, h5py.Dataset):
                        larva_id = int(np.array(larva_id_ref))
                    else:
                        larva_id = int(np.array(f[larva_id_ref]))
                    
                    # Only add larva if we have valid behavior data
                    if all(x is not None for x in [
                        larva['duration_large'],
                        larva['t_start_stop_large'],
                        larva['nb_action_large']
                    ]):
                        extracted_data[larva_id] = larva
                    
                except Exception as e:
                    tqdm.write(f"Error extracting data for larva {i}: {str(e)}")
                    continue
            
            date_str = os.path.basename(os.path.dirname(file_path))
            
            return date_str, extracted_data, {
                'path': file_path,
                'date': datetime.strptime(date_str.split('_')[0], '%Y%m%d'),
                'n_larvae': len(extracted_data),
                'date_str': date_str
            }
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_all_trx_files(base_path):
    """Process all trx.mat files in a directory tree sequentially with progress bar."""
    
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    
    print(f"Found {len(file_list)} trx.mat files")
    
    # Initialize results
    all_data = {}
    metadata = {
        'files_processed': [],
        'total_larvae': 0,
        'experiments': {}
    }
    
    # Process files sequentially with progress bar
    for file_path in tqdm(file_list, desc="Processing trx files"):
        result = process_single_file(file_path)
        if result is not None:
            date_str, trx_extracted, exp_info = result
            metadata['experiments'][date_str] = exp_info
            metadata['files_processed'].append(exp_info['path'])
            metadata['total_larvae'] += exp_info['n_larvae']
            
            for larva_id, larva_data in trx_extracted.items():
                unique_id = f"{date_str}_{larva_id}"
                larva_data['experiment_date'] = date_str
                all_data[unique_id] = larva_data
    
    print(f"\nProcessed {len(metadata['files_processed'])} files")
    print(f"Total larvae: {metadata['total_larvae']}")
    
    return {'data': all_data, 'metadata': metadata}

def process_all_trx_files_parallel(base_path, n_processes=None):

    """Process all trx.mat files in a directory tree using parallel processing."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    # Find all trx files
    file_list = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'trx.mat':
                file_path = os.path.join(root, file)
                file_list.append(file_path)  # Only append the path, not a tuple
    
    print(f"Found {len(file_list)} trx.mat files")
    print(f"Using {n_processes} processes")
    
    # Initialize results
    all_data = {}
    metadata = {
        'files_processed': [],
        'total_larvae': 0,
        'experiments': {}
    }
    
    # Process files in parallel
    with mp.get_context('spawn').Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, file_list),
            total=len(file_list),
            desc="Processing files"
        ))
    
    # Combine results
    for result in results:
        if result is not None:
            date_str, trx_extracted, exp_info = result
            metadata['experiments'][date_str] = exp_info
            metadata['files_processed'].append(exp_info['path'])
            metadata['total_larvae'] += exp_info['n_larvae']
            
            for larva_id, larva_data in trx_extracted.items():
                unique_id = f"{date_str}_{larva_id}"
                larva_data['experiment_date'] = date_str
                all_data[unique_id] = larva_data
    
    print(f"\nProcessed {len(metadata['files_processed'])} files")
    print(f"Total larvae: {metadata['total_larvae']}")
    
    return {'data': all_data, 'metadata': metadata}

def filter_larvae_by_duration(data, min_total_duration=None, percentile=10):
    """Filter larvae based on their total tracked duration.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        min_total_duration: Minimum total duration in seconds. If None, uses percentile
        percentile: Percentile threshold (0-100) to use if min_total_duration is None
        
    Returns:
        dict: Filtered data with same structure as input, excluding larvae below threshold
    """
    # Handle data type
    if 'data' in data:
        extracted_data = data['data']
        is_multi_exp = True
    else:
        extracted_data = data
        is_multi_exp = False
    
    # Calculate total duration for each larva
    larva_durations = {}
    for larva_id, larva_data in extracted_data.items():
        total = 0
        for durations in larva_data['duration_large_small']:
            if durations is not None:
                total += float(np.nansum(durations.flatten()))
        larva_durations[larva_id] = total
    
    # Determine threshold
    if min_total_duration is None:
        min_total_duration = np.percentile(list(larva_durations.values()), percentile)
    
    # Filter larvae
    filtered_data = {}
    for larva_id, total_duration in larva_durations.items():
        if total_duration >= min_total_duration:
            filtered_data[larva_id] = extracted_data[larva_id]
    
    # Return with appropriate structure
    if is_multi_exp:
        return {
            'data': filtered_data,
            'metadata': {
                **data['metadata'],
                'total_larvae': len(filtered_data),
                'duration_threshold': min_total_duration,
                'original_total_larvae': len(extracted_data)
            }
        }
    else:
        return filtered_data

def analyze_behavior_durations(data, show_plot=True, title=None):
    """Analyze behavior durations from processed trx data with separate analysis for large, small, and total behaviors.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        show_plot (bool): Whether to show visualization
        title (str): Optional plot title override
        
    Returns:
        dict: Statistics for each behavior type (large, small, total)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Define behavior mappings for large_small arrays
    # Even indices (0, 2, 4, 6, 8, 10) are large behaviors
    # Odd indices (1, 3, 5, 7, 9, 11) are small behaviors
    behavior_mapping = {
        0: ('large_run', 'run'),
        1: ('small_run', 'run'),
        2: ('large_cast', 'cast'),
        3: ('small_cast', 'cast'),
        4: ('large_stop', 'stop'),
        5: ('small_stop', 'stop'),
        6: ('large_hunch', 'hunch'),
        7: ('small_hunch', 'hunch'),
        8: ('large_backup', 'backup'),
        9: ('small_backup', 'backup'),
        10: ('large_roll', 'roll'),
        11: ('small_roll', 'roll')
    }
    
    # Define base color scheme
    base_behavior_colors = {
        'run': [0.0, 0.0, 0.0],      # Black (for Crawl)
        'cast': [1.0, 0.0, 0.0],     # Red (for Bend)
        'stop': [0.0, 1.0, 0.0],     # Green (for Stop)
        'hunch': [0.0, 0.0, 1.0],    # Blue (for Hunch)
        'backup': [1.0, 0.5, 0.0],   # Orange
        'roll': [0.5, 0.0, 0.5]      # Purple
    }
    
    # Create color mapping with different shades for large, small, and total behaviors
    behavior_colors = {}
    
    # Add colors for large behaviors (darker)
    for idx, (behavior_key, base_name) in behavior_mapping.items():
        if 'large' in behavior_key:
            behavior_colors[behavior_key] = base_behavior_colors[base_name]
    
    # Add colors for small behaviors (pastel versions)
    for idx, (behavior_key, base_name) in behavior_mapping.items():
        if 'small' in behavior_key:
            # Create pastel version (mix with white)
            color = base_behavior_colors[base_name]
            pastel_color = [0.7 + 0.3 * c for c in color]  # Mix with white
            behavior_colors[behavior_key] = pastel_color

    
    # Add colors for total behaviors (dotted pattern via hatching in plot)
    for base_name in base_behavior_colors:
        # Create intermediate shade for total
        color = base_behavior_colors[base_name]
        # Make slightly darker than the base color
        total_color = [max(0, c * 0.8) for c in color]
        behavior_colors[f"{base_name}_total"] = total_color
    
    # Handle data type
    if 'data' in data:
        extracted_data = data['data']
        n_larvae = data['metadata']['total_larvae']
        if title is None:
            title = 'All Experiments'
    else:
        extracted_data = data
        n_larvae = len(extracted_data)
        if title is None:
            title = 'Multiple Experiments'
    
    # Initialize statistics for each behavior group
    behavior_stats = {}
    
    # Initialize all behaviors - large, small, and total
    for idx, (behavior_key, _) in behavior_mapping.items():
        behavior_stats[behavior_key] = {
            'durations': [],
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'per_larva_total': []  # Total duration per larva
        }
    
    # Initialize total behaviors (combined large and small)
    for base_name in base_behavior_colors:
        behavior_stats[f"{base_name}_total"] = {
            'durations': [],
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'per_larva_total': []  # Total duration per larva
        }
    
    # Process each larva
    total_actions = {'large': 0, 'small': 0, 'total': 0}
    
    for larva_id, larva_data in extracted_data.items():
        # Skip if large_small data is not available
        if 'duration_large_small' not in larva_data or 'nb_action_large_small' not in larva_data:
            continue
            
        # Initialize per-larva totals
        larva_totals = {behavior_key: 0 for behavior_key, _ in behavior_mapping.values()}
        
        # Get the large_small arrays
        duration_large_small = larva_data['duration_large_small']
        nb_action_large_small = larva_data['nb_action_large_small']
        
        # Process all indices from the arrays
        for idx in range(len(duration_large_small)):
            if idx >= len(nb_action_large_small):
                continue
                
            # Get behavior mapping for this index
            if idx not in behavior_mapping:
                continue
                
            behavior_key, base_name = behavior_mapping[idx]
            
            # Determine whether this is a large or small behavior
            if 'large_' in behavior_key:
                size_group = 'large'
            else:
                size_group = 'small'
            
            # Get durations and counts
            durations = duration_large_small[idx]
            n_actions = nb_action_large_small[idx]
            
            if durations is not None and n_actions is not None:
                # Clean data - remove NaN values
                n = int(np.nansum(n_actions.flatten()))
                clean_durations = durations.flatten()[~np.isnan(durations.flatten())]
                
                if n > 0 and len(clean_durations) > 0:
                    # Add to behavior stats
                    behavior_stats[behavior_key]['n_actions'] += n
                    behavior_stats[behavior_key]['durations'].extend(clean_durations)
                    
                    # Track total duration for this larva
                    larva_totals[behavior_key] += float(np.nansum(clean_durations))
                    
                    # Update action counts by size
                    total_actions[size_group] += n
                    total_actions['total'] += n
        
        # Store per-larva totals
        for behavior_key, total_duration in larva_totals.items():
            if behavior_key in behavior_stats:
                behavior_stats[behavior_key]['per_larva_total'].append(total_duration)
        
        # Calculate combined totals for each behavior type
        for base_name in base_behavior_colors:
            large_key = f"large_{base_name}"
            small_key = f"small_{base_name}"
            total_key = f"{base_name}_total"
            
            # Add up durations for this larva
            total_duration = larva_totals.get(large_key, 0) + larva_totals.get(small_key, 0)
            
            # Store combined total for this larva
            behavior_stats[total_key]['per_larva_total'].append(total_duration)
            
            # Combine durations arrays
            if large_key in larva_totals and larva_totals[large_key] > 0:
                behavior_stats[total_key]['durations'].extend(behavior_stats[large_key]['durations'])
            if small_key in larva_totals and larva_totals[small_key] > 0:
                behavior_stats[total_key]['durations'].extend(behavior_stats[small_key]['durations'])
    
    # Calculate statistics for each behavior
    for behavior in behavior_stats:
        stats = behavior_stats[behavior]
        if stats['durations']:
            # Determine the behavior group
            if 'large_' in behavior:
                group_total = total_actions['large']
            elif 'small_' in behavior:
                group_total = total_actions['small']
            else:  # This is a _total behavior
                group_total = total_actions['total']
            
            # Calculate n_actions for total behaviors
            if behavior.endswith('_total'):
                base_name = behavior.replace('_total', '')
                large_key = f"large_{base_name}"
                small_key = f"small_{base_name}"
                if large_key in behavior_stats and small_key in behavior_stats:
                    stats['n_actions'] = (behavior_stats[large_key]['n_actions'] + 
                                         behavior_stats[small_key]['n_actions'])
                
            # Calculate statistics
            stats.update({
                'total_duration': float(np.nansum(stats['durations'])),
                'mean_duration': float(np.nanmean(stats['durations'])),
                'std_duration': float(np.nanstd(stats['durations'])),
                'percent_of_total': 100 * stats['n_actions'] / group_total if group_total > 0 else 0
            })
    
    if show_plot:
        # Define the base behavior types to plot
        base_behavior_names = ['run', 'cast', 'stop', 'hunch', 'backup', 'roll']
        
        # Filter to only include behaviors with data
        active_behavior_types = []
        for base_name in base_behavior_names:
            large_key = f"large_{base_name}"
            small_key = f"small_{base_name}"
            
            # Check if any variant has data
            if (large_key in behavior_stats and behavior_stats[large_key]['n_actions'] > 0 or 
                small_key in behavior_stats and behavior_stats[small_key]['n_actions'] > 0):
                active_behavior_types.append(base_name)
        
        if active_behavior_types:
            # Create figure with shared x-axis
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
            
            # Create subplots with shared x-axis
            ax1 = fig.add_subplot(gs[0])  # Event durations
            ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Total durations per larva
            ax3 = fig.add_subplot(gs[2], sharex=ax1)  # Event counts
            
            # Set up x positions for grouped bars/boxes
            n_groups = len(active_behavior_types)
            n_bars = 3  # large, small, total
            group_width = 0.5  # Make the groups slightly thinner
            bar_width = group_width / n_bars
            
            # Set positions for each group of bars
            indices = np.arange(n_groups)
            
            # First plot: Event durations box plot
            for i, base_name in enumerate(active_behavior_types):
                # Define the three variants for this behavior
                large_key = f"large_{base_name}"
                small_key = f"small_{base_name}"
                total_key = f"{base_name}_total"
                
                # Define positions for this group
                large_pos = indices[i] - bar_width
                small_pos = indices[i]
                total_pos = indices[i] + bar_width
                
                # Plot box for large behaviors
                if large_key in behavior_stats and behavior_stats[large_key]['durations']:
                    large_durations = behavior_stats[large_key]['durations']
                    bp_large = ax1.boxplot([large_durations], positions=[large_pos], 
                                         widths=bar_width*0.7, patch_artist=True,
                                         showfliers=False)
                    
                    # Style the boxes
                    for patch in bp_large['boxes']:
                        patch.set_facecolor(behavior_colors.get(large_key, [0, 0, 0]))
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_large[element]:
                            item.set_color(behavior_colors.get(large_key, [0, 0, 0]))
                            item.set_alpha(0.9)
                
                # Plot box for small behaviors
                if small_key in behavior_stats and behavior_stats[small_key]['durations']:
                    small_durations = behavior_stats[small_key]['durations']
                    bp_small = ax1.boxplot([small_durations], positions=[small_pos], 
                                         widths=bar_width*0.7, patch_artist=True,
                                         showfliers=False)
                    
                    # Style the boxes
                    for patch in bp_small['boxes']:
                        patch.set_facecolor(behavior_colors.get(small_key, [0.7, 0.7, 0.7]))
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_small[element]:
                            item.set_color(behavior_colors.get(small_key, [0.7, 0.7, 0.7]))
                            item.set_alpha(0.9)
                
                # Plot box for total behaviors
                if total_key in behavior_stats and behavior_stats[total_key]['durations']:
                    total_durations = behavior_stats[total_key]['durations']
                    bp_total = ax1.boxplot([total_durations], positions=[total_pos], 
                                         widths=bar_width*0.7, patch_artist=True,
                                         showfliers=False)
                    
                    # Style the boxes
                    for patch in bp_total['boxes']:
                        patch.set_facecolor(behavior_colors.get(total_key, [0.5, 0.5, 0.5]))
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_total[element]:
                            item.set_color(behavior_colors.get(total_key, [0.5, 0.5, 0.5]))
                            item.set_alpha(0.9)
            
            ax1.set_title('Event Durations Distribution')
            ax1.set_ylabel('Average Event Duration (s)')
            
            # Second plot: Total duration per larva box plot
            for i, base_name in enumerate(active_behavior_types):
                # Define the three variants for this behavior
                large_key = f"large_{base_name}"
                small_key = f"small_{base_name}"
                total_key = f"{base_name}_total"
                
                # Define positions for this group
                large_pos = indices[i] - bar_width
                small_pos = indices[i]
                total_pos = indices[i] + bar_width
                
                # Plot box for large behaviors
                if large_key in behavior_stats and behavior_stats[large_key]['per_larva_total']:
                    large_totals = behavior_stats[large_key]['per_larva_total']
                    bp_large = ax2.boxplot([large_totals], positions=[large_pos], 
                                         widths=bar_width*0.7, patch_artist=True,
                                         showfliers=False)
                    
                    # Style the boxes
                    for patch in bp_large['boxes']:
                        patch.set_facecolor(behavior_colors.get(large_key, [0, 0, 0]))
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_large[element]:
                            item.set_color(behavior_colors.get(large_key, [0, 0, 0]))
                            item.set_alpha(0.9)
                
                # Plot box for small behaviors
                if small_key in behavior_stats and behavior_stats[small_key]['per_larva_total']:
                    small_totals = behavior_stats[small_key]['per_larva_total']
                    bp_small = ax2.boxplot([small_totals], positions=[small_pos], 
                                         widths=bar_width*0.7, patch_artist=True,
                                         showfliers=False)
                    
                    # Style the boxes
                    for patch in bp_small['boxes']:
                        patch.set_facecolor(behavior_colors.get(small_key, [0.7, 0.7, 0.7]))
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_small[element]:
                            item.set_color(behavior_colors.get(small_key, [0.7, 0.7, 0.7]))
                            item.set_alpha(0.9)
                
                # Plot box for total behaviors
                if total_key in behavior_stats and behavior_stats[total_key]['per_larva_total']:
                    total_totals = behavior_stats[total_key]['per_larva_total']
                    bp_total = ax2.boxplot([total_totals], positions=[total_pos], 
                                         widths=bar_width*0.7, patch_artist=True,
                                         showfliers=False)
                    
                    # Style the boxes
                    for patch in bp_total['boxes']:
                        patch.set_facecolor(behavior_colors.get(total_key, [0.5, 0.5, 0.5]))
                        patch.set_alpha(0.7)
                    for element in ['whiskers', 'caps', 'medians']:
                        for item in bp_total[element]:
                            item.set_color(behavior_colors.get(total_key, [0.5, 0.5, 0.5]))
                            item.set_alpha(0.9)
            
            ax2.set_ylabel('Total Duration (s)')
            
            # Third plot: Action counts bar chart
            for i, base_name in enumerate(active_behavior_types):
                # Define the three variants for this behavior
                large_key = f"large_{base_name}"
                small_key = f"small_{base_name}"
                total_key = f"{base_name}_total"
                
                # Define positions for this group
                large_pos = indices[i] - bar_width
                small_pos = indices[i]
                total_pos = indices[i] + bar_width
                
                # Get counts
                large_count = behavior_stats[large_key]['n_actions'] if large_key in behavior_stats else 0
                small_count = behavior_stats[small_key]['n_actions'] if small_key in behavior_stats else 0
                total_count = large_count + small_count
                
                # Create bars
                large_bar = ax3.bar(large_pos, large_count, width=bar_width*0.7, 
                                  color=behavior_colors.get(large_key, [0, 0, 0]), alpha=0.7)
                small_bar = ax3.bar(small_pos, small_count, width=bar_width*0.7, 
                                  color=behavior_colors.get(small_key, [0.7, 0.7, 0.7]), alpha=0.7)
                total_bar = ax3.bar(total_pos, total_count, width=bar_width*0.7, 
                                  color=behavior_colors.get(total_key, [0.5, 0.5, 0.5]), alpha=0.7)
                
                # Add percentage labels only for large and small (not total)
                if large_count > 0 and large_key in behavior_stats:
                    large_pct = behavior_stats[large_key]['percent_of_total']
                    ax3.text(large_pos, large_count, f'{large_pct:.1f}%', 
                           ha='center', va='bottom', rotation=90, fontsize=8)
                
                if small_count > 0 and small_key in behavior_stats:
                    small_pct = behavior_stats[small_key]['percent_of_total']
                    ax3.text(small_pos, small_count, f'{small_pct:.1f}%', 
                           ha='center', va='bottom', rotation=90, fontsize=8)
            
            ax3.set_ylabel('Number of Events')
            
            # Set x-ticks in the center of each group
            ax3.set_xticks(indices)
            ax3.set_xticklabels(active_behavior_types)
            
            # Hide x-axis for top two plots
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            ax1.set_xlabel('')
            ax2.set_xlabel('')
            
            # Remove top spines
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            # Create legend patches
            legend_elements = [
                Patch(facecolor=behavior_colors['large_run'], alpha=0.7, label='Large behavior'),
                Patch(facecolor=behavior_colors['small_run'], alpha=0.7, label='Small behavior'),
                Patch(facecolor=behavior_colors['run_total'], alpha=0.7, label='Total (large + small)')
            ]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            fig.suptitle(f"Behavior Analysis - {title}\n(n = {n_larvae} larvae)", y=1)
            plt.tight_layout()
            plt.show()
    
    # Print summary statistics for each group
    for group_name, prefix in [
        ('Large Behaviors', 'large_'), 
        ('Small Behaviors', 'small_'),
        ('Total Behaviors', '_total')
    ]:
        # Get relevant behaviors for this group
        if prefix == '_total':
            behaviors = [f"{base}_total" for base in base_behavior_names]
        else:
            behaviors = [f"{prefix}{base}" for base in base_behavior_names]
            
        # Filter behaviors with data
        behaviors = [b for b in behaviors if b in behavior_stats and behavior_stats[b]['durations']]
        
        if behaviors:
            # Calculate group total actions
            group_key = 'large' if prefix == 'large_' else 'small' if prefix == 'small_' else 'total'
            group_total_actions = total_actions.get(group_key, 0)
            
            print(f"\n{group_name} analysis for {title}")
            print(f"Number of larvae: {n_larvae}")
            print(f"Total actions: {group_total_actions}\n")
            print(f"{'Behavior':>12} {'Events':>8} {'%Total':>7} {'Mean(s)':>12} {'Median(s)':>10}")
            print("-" * 60)
            
            for behavior in behaviors:
                stats = behavior_stats[behavior]
                if stats['durations']:
                    median = float(np.median(stats['durations']))
                    
                    # Get readable name
                    if prefix == '_total':
                        display_name = behavior.replace('_total', '')
                    else:
                        display_name = behavior.replace(prefix, '')
                    
                    print(f"{display_name:>12}: {stats['n_actions']:8d} {stats['percent_of_total']:6.1f}%"
                          f"{stats['mean_duration']:10.2f} {median:10.2f}")
    
    return behavior_stats

def plot_global_behavior_matrix(trx_data, show_separate_totals=True):
    """
    Plot global behavior using the global state.
    
    This function visualizes behavioral states across time for all larvae.
    It processes both large_state (1-6) and small_large_state (0.5-5.5) values.
    
    Args:
        trx_data: Dictionary of larva data
        show_separate_totals: If True, show large, small, and total behaviors as separate rows
                             If False, show only a single row per larva
    
    State mapping:
    - 1, 0.5: run (large/small)
    - 2, 1.5: cast (large/small)
    - 3, 2.5: stop (large/small)
    - 4, 3.5: hunch (large/small)
    - 5, 4.5: backup (large/small)
    - 6, 5.5: roll (large/small)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Get sorted larva IDs
    larva_ids = sorted(trx_data.keys())
    n_larvae = len(larva_ids)
    
    # Compute time range
    tmins = [np.min(trx_data[lid]['t']) for lid in larva_ids if len(trx_data[lid]['t']) > 0]
    tmaxs = [np.max(trx_data[lid]['t']) for lid in larva_ids if len(trx_data[lid]['t']) > 0]
    if not tmins or not tmaxs:
        raise ValueError("No time data found!")
    min_time = float(min(tmins))
    max_time = float(max(tmaxs))
    
    resolution = 1000
    
    # Define state value mapping (both integer and half-integer values)
    # Maps state values to behavior names and colors
    state_mapping = {
        # Large behaviors (integer values)
        1.0: {'name': 'large_run', 'base': 'run', 'color': [0.0, 0.0, 0.0]},      # Black
        2.0: {'name': 'large_cast', 'base': 'cast', 'color': [1.0, 0.0, 0.0]},     # Red
        3.0: {'name': 'large_stop', 'base': 'stop', 'color': [0.0, 1.0, 0.0]},     # Green
        4.0: {'name': 'large_hunch', 'base': 'hunch', 'color': [0.0, 0.0, 1.0]},    # Blue
        5.0: {'name': 'large_backup', 'base': 'backup', 'color': [1.0, 0.5, 0.0]},   # Orange
        6.0: {'name': 'large_roll', 'base': 'roll', 'color': [0.5, 0.0, 0.5]},     # Purple
        
        # Small behaviors (half-integer values)
        0.5: {'name': 'small_run', 'base': 'run', 'color': [0.7, 0.7, 0.7]},      # Light gray
        1.5: {'name': 'small_cast', 'base': 'cast', 'color': [1.0, 0.7, 0.7]},     # Light red
        2.5: {'name': 'small_stop', 'base': 'stop', 'color': [0.7, 1.0, 0.7]},     # Light green
        3.5: {'name': 'small_hunch', 'base': 'hunch', 'color': [0.7, 0.7, 1.0]},    # Light blue
        4.5: {'name': 'small_backup', 'base': 'backup', 'color': [1.0, 0.8, 0.6]},   # Light orange
        5.5: {'name': 'small_roll', 'base': 'roll', 'color': [0.8, 0.6, 0.8]}      # Light purple
    }
    
    # Define total behavior colors (medium shade between large and small)
    total_behavior_colors = {
        'run': [0.4, 0.4, 0.4],       # Medium gray
        'cast': [0.8, 0.3, 0.3],      # Medium red
        'stop': [0.3, 0.8, 0.3],      # Medium green
        'hunch': [0.3, 0.3, 0.8],     # Medium blue
        'backup': [0.8, 0.6, 0.3],    # Medium orange
        'roll': [0.6, 0.3, 0.6]       # Medium purple
    }

    if show_separate_totals:
        # Create 3 rows per larva: large, small, total
        behavior_matrix = np.full((n_larvae * 3, resolution, 3), fill_value=1.0)  # white background
    else:
        # Create 1 row per larva
        behavior_matrix = np.full((n_larvae, resolution, 3), fill_value=1.0)  # white background

    # Process each larva
    for larva_idx, lid in enumerate(larva_ids):
        # Get time and state data
        larva_time = np.array(trx_data[lid]['t']).flatten()
        
        # Use global_state_small_large_state if available, otherwise use global_state_large_state
        if 'global_state_small_large_state' in trx_data[lid]:
            states = np.array(trx_data[lid]['global_state_small_large_state']).flatten()
        else:
            states = np.array(trx_data[lid]['global_state_large_state']).flatten()
        
        # Convert times to indices
        time_indices = np.floor(
            ((larva_time - min_time) / (max_time - min_time) * (resolution - 1))
        ).astype(int)
        time_indices = np.clip(time_indices, 0, resolution - 1)

        # Arrays to store large, small and total behaviors
        large_behaviors = np.full((resolution, 3), fill_value=1.0)  # white background
        small_behaviors = np.full((resolution, 3), fill_value=1.0)  # white background
        total_behaviors = np.full((resolution, 3), fill_value=1.0)  # white background
        
        # For each unique time index, use the corresponding state
        unique_indices = np.unique(time_indices)
        for t_idx in unique_indices:
            mask = time_indices == t_idx
            if np.any(mask):
                state_val = float(states[mask][0])  # Take first state if multiple exist
                
                # Round to nearest 0.5 to handle potential floating point issues
                state_val = round(state_val * 2) / 2
                
                # Determine if this is a large or small behavior
                is_large = state_val.is_integer()
                is_small = not is_large
                
                # Assign colors based on state
                if state_val in state_mapping:
                    behavior_info = state_mapping[state_val]
                    
                    if is_large:
                        large_behaviors[t_idx] = behavior_info['color']
                        total_behaviors[t_idx] = total_behavior_colors[behavior_info['base']]
                    elif is_small:
                        small_behaviors[t_idx] = behavior_info['color']
                        total_behaviors[t_idx] = total_behavior_colors[behavior_info['base']]
                        
                    # If not showing separate totals, just use the state color directly
                    if not show_separate_totals:
                        behavior_matrix[larva_idx, t_idx] = behavior_info['color']
        
        # If showing separate totals, assign the arrays to the behavior matrix
        if show_separate_totals:
            row_large = larva_idx * 3
            row_small = larva_idx * 3 + 1
            row_total = larva_idx * 3 + 2
            
            behavior_matrix[row_large] = large_behaviors
            behavior_matrix[row_small] = small_behaviors
            behavior_matrix[row_total] = total_behaviors

    # Create the plot
    plt.figure(figsize=(12, max(6, n_larvae * (3 if show_separate_totals else 1) * 0.4)))
    plt.imshow(behavior_matrix, aspect='auto', interpolation='nearest', alpha=0.8,
               extent=[min_time, max_time, behavior_matrix.shape[0], 0])
    
    # Create y-tick labels
    if show_separate_totals:
        ytick_positions = []
        ytick_labels = []
        
        for i, lid in enumerate(larva_ids):
            base_pos = i * 3
            # Position ticks in the middle of each row
            ytick_positions.extend([base_pos + 0.5, base_pos + 1.5, base_pos + 2.5])
            ytick_labels.extend([f"{lid} (large)", f"{lid} (small)", f"{lid} (total)"])
    else:
        ytick_positions = np.arange(0.5, n_larvae + 0.5)
        ytick_labels = larva_ids
    
    plt.yticks(ytick_positions, ytick_labels, fontsize='small')
    plt.title('Global Behavior States', pad=20)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Larva ID')
    
    # Add horizontal lines to separate larvae if showing separate totals
    if show_separate_totals:
        for i in range(1, n_larvae):
            y_pos = i * 3
            plt.axhline(y=y_pos, color='black', linestyle='-', linewidth=0.5)
    
    # Add legend with all states
    legend_elements = []
    
    # Add large behaviors to legend
    for state_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        if state_val in state_mapping:
            info = state_mapping[state_val]
            legend_elements.append(
                Patch(facecolor=info['color'], 
                      label=f"{info['base']} (large)")
            )
    
    # Add a separator in the legend
    legend_elements.append(Patch(facecolor='white', label=''))
    
    # Add small behaviors to legend
    for state_val in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        if state_val in state_mapping:
            info = state_mapping[state_val]
            legend_elements.append(
                Patch(facecolor=info['color'], 
                      label=f"{info['base']} (small)")
            )
    
    # Add a separator in the legend
    legend_elements.append(Patch(facecolor='white', label=''))
    
    # Add total behaviors to legend
    for base_name, color in total_behavior_colors.items():
        legend_elements.append(
            Patch(facecolor=color, label=f"{base_name} (total)")
        )
    
    # Add "Other" category
    legend_elements.append(Patch(facecolor=[1, 1, 1], label='Other'))
    
    plt.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1, 0.5), title='Behavioral States')
    plt.tight_layout()
    plt.show()
    
    return behavior_matrix

def plot_behavioral_contour_with_global_trajectory(trx_data, larva_id):
    """
    Plot a filled larva contour with spine and neck points, its global trajectory,
    and angle dynamics over time with behavior state coloring.
    
    Colors based on global_state_large_state value:
    1 -> Black (run/crawl)
    2 -> Red (cast/bend)
    3 -> Green (stop)
    4 -> Blue (hunch)
    5 -> Orange (backup)
    6 -> Purple (roll)
    7 -> Light gray (small actions)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from ipywidgets import Play, IntSlider, HBox, jslink
    from IPython.display import display
    from matplotlib.patches import Patch

    # Define visualization parameters
    WINDOW_SIZE = 2  # Size of zoom window
    FIGURE_SIZE = (12, 6)  # Original size
    LINE_WIDTH = 2
    MARKER_SIZE = 8
    SPINE_MARKER_SIZE = 4  # Smaller marker size for spine points
    ALPHA = 0.6
    TIME_WINDOW = 50  # Show 50 seconds of angle data

    # Get data and ensure proper shapes
    larva = trx_data[larva_id]
    x_contour = np.atleast_2d(np.array(larva['x_contour']))
    y_contour = np.atleast_2d(np.array(larva['y_contour']))
    x_center = np.array(larva['x_center']).flatten()
    y_center = np.array(larva['y_center']).flatten()
    x_spine = np.array(larva['x_spine'])
    y_spine = np.array(larva['y_spine'])
    x_neck = np.array(larva['x_neck']).flatten()
    y_neck = np.array(larva['y_neck']).flatten()
    x_neck_down = np.array(larva['x_neck_down']).flatten()
    y_neck_down = np.array(larva['y_neck_down']).flatten()
    x_neck_top = np.array(larva['x_neck_top']).flatten()
    y_neck_top = np.array(larva['y_neck_top']).flatten()
    time = np.array(larva['t']).flatten()
    global_state = np.array(larva['global_state_large_state'])
    
    # Get the angle data and convert to degrees
    try:
        angle_upper_lower = np.array(larva['angle_upper_lower_smooth_5']).flatten()
        # Convert to degrees and reverse
        angle_upper_lower_deg = np.degrees(angle_upper_lower)

        # angle_downer_upper = np.array(larva['angle_downer_upper_smooth_5']).flatten()
        # # Convert to degrees and reverse
        # angle_downer_upper_deg = -1 * np.degrees(angle_downer_upper)
    except KeyError:
        print("Warning: angle_downer_upper_smooth_5 not found, using zeros")
        angle_downer_upper_deg = np.zeros_like(time)

    # Define colors for each state
    state_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Crawl)
        2: [1.0, 0.0, 0.0],      # Red (for Bend)
        3: [0.0, 1.0, 0.0],      # Green (for Stop)
        4: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],      # Orange
        6: [0.5, 0.0, 0.5],      # Purple
        7: [0.7, 0.7, 0.7]       # Light gray (for Small action)
    }
    
    # Define behavior names for legend
    behavior_labels = {
        1: 'Run/Crawl',
        2: 'Cast/Bend',
        3: 'Stop',
        4: 'Hunch',
        5: 'Backup',
        6: 'Roll',
        7: 'Small Actions'
    }

    # Create trajectory colors
    trajectory_colors = []
    for state in global_state.flatten():
        try:
            state_val = int(state)
            trajectory_colors.append(state_colors.get(state_val, [1, 1, 1]))
        except:
            trajectory_colors.append([1, 1, 1])

    # Create figure and axes with the requested layout
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0])  # Larva contour (takes full height of left column)
    ax2 = fig.add_subplot(gs[0, 1])  # Global trajectory (top right)
    ax3 = fig.add_subplot(gs[1, 1])  # Angle plot (bottom right)

    # Initialize left panel plots
    contour_line, = ax1.plot([], [], 'k-', linewidth=LINE_WIDTH)
    contour_fill = ax1.fill([], [], color='gray', alpha=ALPHA)[0]
    
    # Initialize spine visualization with all points
    spine_line, = ax1.plot([], [], '-', lw=LINE_WIDTH, alpha=ALPHA)
    spine_points = []
    
    # Create placeholder points for each spine point
    if x_spine.ndim > 1:
        num_spine_points = x_spine.shape[0]
    else:
        num_spine_points = 1
    
    for i in range(num_spine_points):
        point, = ax1.plot([], [], 'o', ms=SPINE_MARKER_SIZE)
        spine_points.append(point)
    
    # Initialize specific body part points
    head_point, = ax1.plot([], [], 'o', ms=MARKER_SIZE)
    tail_point, = ax1.plot([], [], 's', ms=MARKER_SIZE)
    center_point, = ax1.plot([], [], '^', ms=MARKER_SIZE)
    neck_point, = ax1.plot([], [], 'D', ms=MARKER_SIZE)
    neck_down_point, = ax1.plot([], [], 'X', ms=MARKER_SIZE)
    neck_top_point, = ax1.plot([], [], '*', ms=MARKER_SIZE)

    # Initialize right top panel with colored segments
    for i in range(len(x_center)-1):
        ax2.plot(x_center[i:i+2], y_center[i:i+2], 
                color=trajectory_colors[i], 
                linewidth=LINE_WIDTH, 
                alpha=ALPHA)
    current_pos, = ax2.plot([], [], 'o', ms=MARKER_SIZE)
    
    # Prepare the angle plot (bottom right)
    # Plot the full angle data with behavior shading
    ax3.plot(time, angle_upper_lower_deg, 'k-', linewidth=1.0)
    
    # Add behavior state shading to angle plot for the full time range
    for state_val in range(1, 8):
        segments = []
        in_segment = False
        segment_start = 0
        states_flat = global_state.flatten()
        
        for i, s in enumerate(states_flat):
            if s == state_val and not in_segment:
                in_segment = True
                segment_start = i
            elif s != state_val and in_segment:
                in_segment = False
                segments.append((segment_start, i))
                
        # Handle case when segment extends to end of data
        if in_segment:
            segments.append((segment_start, len(states_flat)-1))
            
        # Shade each segment
        for seg_start, seg_end in segments:
            if seg_start < seg_end:  # Only shade non-empty segments
                ax3.axvspan(time[seg_start], time[seg_end], 
                         color=state_colors[state_val], 
                         alpha=0.3)
    
    # Initialize time marker line that will be updated
    time_marker, = ax3.plot([0, 0], [ax3.get_ylim()[0], ax3.get_ylim()[1]], 
                           'r-', linewidth=2.0)

    def update(frame):
        # Extract current frame data
        x_frame = x_contour[:, frame]
        y_frame = y_contour[:, frame]
        current_time = float(time[frame])

        # Get state and color
        try:
            state = int(global_state.flatten()[frame])
            color = state_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]

        # Update contour and points
        contour_line.set_data(x_frame, y_frame)
        contour_line.set_color(color)
        contour_fill.set_xy(np.column_stack((x_frame, y_frame)))
        contour_fill.set_facecolor(color)
        
        # Extract all spine point coordinates for this frame
        if x_spine.ndim > 1:  # 2D array (multiple spine points)
            spine_x = x_spine[:, frame]
            spine_y = y_spine[:, frame]
        else:  # 1D array (single spine point)
            spine_x = np.array([x_spine[frame]])
            spine_y = np.array([y_spine[frame]])
        
        # Update the full spine line
        spine_line.set_data(spine_x, spine_y)
        spine_line.set_color(color)
        
        # Update each spine point
        for i, point in enumerate(spine_points):
            if i < len(spine_x):
                point.set_data([spine_x[i]], [spine_y[i]])
                point.set_color(color)
            else:
                point.set_data([], [])  # Hide extra points
        
        # Get specific body part coordinates
        head_x = spine_x[0] if len(spine_x) > 0 else None
        head_y = spine_y[0] if len(spine_y) > 0 else None
        tail_x = spine_x[-1] if len(spine_x) > 0 else None
        tail_y = spine_y[-1] if len(spine_y) > 0 else None
        center_x = x_center[frame]
        center_y = y_center[frame]
        neck_x = x_neck[frame]
        neck_y = y_neck[frame]
        neck_down_x = x_neck_down[frame]
        neck_down_y = y_neck_down[frame]
        neck_top_x = x_neck_top[frame]
        neck_top_y = y_neck_top[frame]
        
        # Update body part points
        head_point.set_data([head_x], [head_y])
        tail_point.set_data([tail_x], [tail_y])
        center_point.set_data([center_x], [center_y])
        neck_point.set_data([neck_x], [neck_y])
        neck_down_point.set_data([neck_down_x], [neck_down_y])
        neck_top_point.set_data([neck_top_x], [neck_top_y])
        
        # Update colors
        head_point.set_color(color)
        tail_point.set_color(color)
        center_point.set_color(color)
        neck_point.set_color(color)
        neck_down_point.set_color(color)
        neck_top_point.set_color(color)
        current_pos.set_data([center_x], [center_y])
        current_pos.set_color(color)
        
        # Update time marker in angle plot
        time_marker.set_data([current_time, current_time], 
                            [ax3.get_ylim()[0], ax3.get_ylim()[1]])
        
        # Update visible window in angle plot - only move the window, don't redraw
        if time[-1] - time[0] > TIME_WINDOW:
            time_start = max(time[0], current_time - TIME_WINDOW/2)
            time_end = min(time[-1], current_time + TIME_WINDOW/2)
            ax3.set_xlim(time_start, time_end)
        else:
            # If recording is shorter than TIME_WINDOW, show the full range
            ax3.set_xlim(time[0], time[-1])

        # Update zoom window for contour
        ax1.set_xlim(center_x - WINDOW_SIZE, center_x + WINDOW_SIZE)
        ax1.set_ylim(center_y - WINDOW_SIZE, center_y + WINDOW_SIZE)
        ax1.set_title(f"Time: {current_time:.2f} s")

        fig.canvas.draw_idle()
        return (contour_line, contour_fill, spine_line, *spine_points, head_point, 
                tail_point, center_point, neck_point, neck_down_point, neck_top_point, 
                current_pos, time_marker)

    # Set up interactive controls
    play = Play(value=0, min=0, max=len(time)-1, step=1, interval=50, description="Play")
    slider = IntSlider(min=0, max=len(time)-1, value=0, description="Frame:",
                      continuous_update=True, layout={'width': '1000px'})
    jslink((play, 'value'), (slider, 'value'))

    def on_value_change(change):
        if change['name'] == 'value':
            update(change['new'])
    slider.observe(on_value_change)

    # Configure axes
    ax1.set_aspect("equal")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax1.set_title("Larva Contour")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X position")
    ax2.set_ylabel("Y position")
    ax2.set_title("Global Trajectory")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Angle Upper-Lower (°)")
    ax3.set_title("Angle Dynamics")
    
    # Configure initial view for angle plot
    if len(time) > 0:
        time_start = time[0]
        time_end = min(time[-1], time[0] + TIME_WINDOW)
        ax3.set_xlim(time_start, time_end)

    # Create behavior state legend for bottom of contour plot
    behavior_legend_elements = [
        Patch(facecolor=state_colors[i], alpha=0.6, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in state_colors
    ]
    
    # Add marker legend for contour plot (top right)
    marker_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Head'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Tail'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Center'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Neck'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Neck Down'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=MARKER_SIZE, label='Neck Top'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=SPINE_MARKER_SIZE, label='Spine Points'),
    ]
    
    # Add marker legend to top right of contour plot
    ax1.legend(handles=marker_legend, loc='upper right', title='Body parts')
    
    # Add behavior legend to bottom of contour plot
    # We place it outside the axis area
    ax1.figure.legend(handles=behavior_legend_elements, 
                     loc='lower center', 
                     bbox_to_anchor=(0.27, 0), 
                     ncol=4,
                     fontsize=8, 
                     title='Behaviors')

    # Display controls and figure
    display(HBox([play, slider]))

    update(0)
    plt.tight_layout()
    # Make room for the bottom legend
    plt.subplots_adjust(bottom=0.2)


def save_behavioral_contour_video(trx_data, larva_id, output_path=None, fps=20):
    """
    Save the behavioral contour visualization as a video file.
    
    Args:
        trx_data: Dictionary containing larva tracking data
        larva_id: ID of the larva to visualize
        output_path: Path where to save the video (default: './larva_behavior_{larva_id}.mp4')
        fps: Frames per second for the output video (default: 20)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.animation as animation
    
    if output_path is None:
        output_path = f'./larva_behavior_{larva_id}.mp4'

    # Define visualization parameters
    WINDOW_SIZE = 2
    FIGURE_SIZE = (12, 6)
    LINE_WIDTH = 2
    MARKER_SIZE = 8
    ALPHA = 0.6

    # Get data and ensure proper shapes
    larva = trx_data[larva_id]
    x_contour = np.atleast_2d(np.array(larva['x_contour']))
    y_contour = np.atleast_2d(np.array(larva['y_contour']))
    x_center = np.array(larva['x_center']).flatten()
    y_center = np.array(larva['y_center']).flatten()
    x_spine = np.array(larva['x_spine'])
    y_spine = np.array(larva['y_spine'])
    x_neck = np.array(larva['x_neck']).flatten()
    y_neck = np.array(larva['y_neck']).flatten()
    time = np.array(larva['t']).flatten()
    global_state = np.array(larva['global_state_large_state'])

    # Define colors for each state
    state_colors = {
        1: [1, 0.7, 0.7],    # Light red (run)
        2: [0.7, 0.7, 1],    # Light blue (cast)
        3: [0.7, 1, 0.7],    # Light green (stop)
        4: [0.9, 0.7, 0.9],  # Light purple (head cast)
        5: [1, 1, 0.7],      # Light yellow (backup)
        6: [1, 0.8, 0.6],    # Light orange (collision)
        7: [1, 0.8, 0.9],    # Light pink (small actions)
    }

    # Create figure and axes
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Initialize plots
    contour_line, = ax1.plot([], [], 'k-', linewidth=LINE_WIDTH)
    contour_fill = ax1.fill([], [], color='gray', alpha=ALPHA)[0]
    spine_line, = ax1.plot([], [], '-', lw=LINE_WIDTH, alpha=ALPHA)
    head_point, = ax1.plot([], [], 'o', ms=MARKER_SIZE)
    tail_point, = ax1.plot([], [], 's', ms=MARKER_SIZE)
    center_point, = ax1.plot([], [], '^', ms=MARKER_SIZE)
    neck_point, = ax1.plot([], [], 'D', ms=MARKER_SIZE)

    # Plot full trajectory with colors
    for i in range(len(x_center)-1):
        try:
            state = int(global_state.flatten()[i])
            color = state_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]
        ax2.plot(x_center[i:i+2], y_center[i:i+2], 
                color=color, linewidth=LINE_WIDTH, alpha=ALPHA)
    current_pos, = ax2.plot([], [], 'o', ms=MARKER_SIZE)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=state_colors[1], label='Run'),
        Line2D([0], [0], color=state_colors[2], label='Cast'),
        Line2D([0], [0], color=state_colors[3], label='Stop'),
        Line2D([0], [0], color=state_colors[4], label='Head Cast'),
        Line2D([0], [0], color=state_colors[5], label='Backup'),
        Line2D([0], [0], color=state_colors[6], label='Collision'),
        Line2D([0], [0], color=state_colors[7], label='Small Actions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Head'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Tail'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Center'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='k', 
               markersize=MARKER_SIZE, label='Neck'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    def animate(frame):
        # Extract current frame data
        x_frame = x_contour[:, frame]
        y_frame = y_contour[:, frame]
        head_x, head_y = x_spine[0, frame], y_spine[0, frame]
        tail_x, tail_y = x_spine[-1, frame], y_spine[-1, frame]
        center_x, center_y = x_center[frame], y_center[frame]
        neck_x, neck_y = x_neck[frame], y_neck[frame]
        current_time = float(time[frame])

        # Get state and color
        try:
            state = int(global_state.flatten()[frame])
            color = state_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]

        # Update plots
        contour_line.set_data(x_frame, y_frame)
        contour_line.set_color(color)
        contour_fill.set_xy(np.column_stack((x_frame, y_frame)))
        contour_fill.set_facecolor(color)
        
        spine_line.set_data([head_x, center_x, tail_x], [head_y, center_y, tail_y])
        head_point.set_data([head_x], [head_y])
        tail_point.set_data([tail_x], [tail_y])
        center_point.set_data([center_x], [center_y])
        neck_point.set_data([neck_x], [neck_y])
        
        # Update colors
        spine_line.set_color(color)
        head_point.set_color(color)
        tail_point.set_color(color)
        center_point.set_color(color)
        neck_point.set_color(color)
        current_pos.set_data([center_x], [center_y])
        current_pos.set_color(color)

        # Update zoom window and title
        ax1.set_xlim(center_x - WINDOW_SIZE, center_x + WINDOW_SIZE)
        ax1.set_ylim(center_y - WINDOW_SIZE, center_y + WINDOW_SIZE)
        ax1.set_title(f"Time: {current_time:.2f} s")

        return (contour_line, contour_fill, spine_line, head_point, 
                tail_point, center_point, neck_point, current_pos)

    # Configure axes
    ax1.set_aspect("equal")
    ax1.set_xlabel("X position")
    ax1.set_ylabel("Y position")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X position")
    ax2.set_ylabel("Y position")
    ax2.set_title("Global Trajectory")

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time),
                                 interval=1000/fps, blit=True)
    
    # Save animation
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close()

    print(f"Video saved to: {output_path}")


def analyze_run_orientations_single(trx_data):
    """
    Analyze run orientations for all larvae in a single experiment,
    separating large runs, small runs, and total runs.
    
    Args:
        trx_data: Dictionary containing larva tracking data
        
    Returns:
        dict: Contains orientation data for large runs, small runs, and total runs
    """
    import numpy as np
    
    # Initialize separate storage for large, small, and total runs
    large_run_orientations = []
    small_run_orientations = []
    all_run_orientations = []  # Combined large and small
    
    # Process each larva
    for larva_id, larva in trx_data.items():
        try:
            # Get position data - handle both nested and flat structures
            if 'data' in larva:
                larva = larva['data']
            
            # Extract and validate arrays
            x_center = np.asarray(larva['x_center']).flatten()
            y_center = np.asarray(larva['y_center']).flatten()
            x_spine = np.asarray(larva['x_spine'])
            y_spine = np.asarray(larva['y_spine'])
            
            # Get global state data - use small_large_state if available
            if 'global_state_small_large_state' in larva:
                states = np.asarray(larva['global_state_small_large_state']).flatten()
                # Define masks for large runs (state = 1.0) and small runs (state = 0.5)
                large_run_mask = states == 1.0
                small_run_mask = states == 0.5
                all_run_mask = large_run_mask | small_run_mask
            else:
                # Fall back to regular large_state if small_large_state isn't available
                states = np.asarray(larva['global_state_large_state']).flatten()
                # With just large state, only state 1 is run
                large_run_mask = states == 1
                small_run_mask = np.zeros_like(states, dtype=bool)  # No small runs
                all_run_mask = large_run_mask
            
            # Get tail positions (last spine point)
            x_tail = x_spine[-1].flatten() if x_spine.ndim > 1 else x_spine.flatten()
            y_tail = y_spine[-1].flatten() if y_spine.ndim > 1 else y_spine.flatten()
            
            # Calculate tail-to-center vectors
            tail_to_center = np.column_stack([
                x_center - x_tail,
                y_center - y_tail
            ])
            
            # Calculate orientations in degrees CHECK NEGATIVE AXIS IS ZERO
            orientations = np.degrees(np.arctan2(
                tail_to_center[:, 1],
                -tail_to_center[:, 0]
            ))
            
            # Add orientations to respective lists
            large_run_orientations.extend(orientations[large_run_mask])
            small_run_orientations.extend(orientations[small_run_mask])
            all_run_orientations.extend(orientations[all_run_mask])
            
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    return {
        'large_run_orientations': np.array(large_run_orientations),
        'small_run_orientations': np.array(small_run_orientations),
        'all_run_orientations': np.array(all_run_orientations),
        'n_larvae': len(trx_data),
        'n_large_runs': len(large_run_orientations),
        'n_small_runs': len(small_run_orientations),
        'n_total_runs': len(all_run_orientations)
    }

def analyze_run_orientations_all(experiments_data):
    """
    Analyze run orientations across all experiments,
    separating large runs, small runs, and total runs.
    
    Args:
        experiments_data: Dict containing all experiments data
        
    Returns:
        dict: Contains orientation data and statistics for large, small, and total runs
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Initialize separate storage for large, small, and total runs
    large_run_orientations = []
    small_run_orientations = []
    all_run_orientations = []
    total_larvae = 0
    
    # Handle nested data structure
    if 'data' in experiments_data:
        data_to_process = experiments_data['data']
        total_larvae = experiments_data['metadata']['total_larvae']
    else:
        data_to_process = experiments_data
        total_larvae = len(data_to_process)
    
    # Process the data
    if isinstance(data_to_process, dict):
        results = analyze_run_orientations_single(data_to_process)
        large_run_orientations.extend(results['large_run_orientations'])
        small_run_orientations.extend(results['small_run_orientations'])
        all_run_orientations.extend(results['all_run_orientations'])
    else:
        for exp_data in data_to_process.values():
            results = analyze_run_orientations_single(exp_data)
            large_run_orientations.extend(results['large_run_orientations'])
            small_run_orientations.extend(results['small_run_orientations'])
            all_run_orientations.extend(results['all_run_orientations'])
    
    # Convert to numpy arrays
    large_run_orientations = np.array(large_run_orientations)
    small_run_orientations = np.array(small_run_orientations)
    all_run_orientations = np.array(all_run_orientations)
    
    # Create histograms for each type
    bins = np.linspace(-180, 180, 31)  # 30 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate histograms
    hist_large = np.histogram(large_run_orientations, bins=bins, density=True)[0] if len(large_run_orientations) > 0 else np.zeros(36)
    hist_small = np.histogram(small_run_orientations, bins=bins, density=True)[0] if len(small_run_orientations) > 0 else np.zeros(36)
    hist_all = np.histogram(all_run_orientations, bins=bins, density=True)[0] if len(all_run_orientations) > 0 else np.zeros(36)
    
    # Apply smoothing
    smoothed_large = gaussian_filter1d(hist_large, sigma=1)
    smoothed_small = gaussian_filter1d(hist_small, sigma=1)
    smoothed_all = gaussian_filter1d(hist_all, sigma=1)
    
    # Create figure with three subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Large runs
    ax1.plot(bin_centers, hist_large, 'k-', alpha=0.3, linewidth=1)
    ax1.plot(bin_centers, smoothed_large, 'r-', linewidth=2)
    ax1.set_xlabel('Body orientation (°)')
    ax1.set_ylabel('Relative probability')
    ax1.set_xlim(-180, 180)
    ax1.set_title(f'Large Runs (n={len(large_run_orientations)})')
    
    # Plot 2: Small runs
    ax2.plot(bin_centers, hist_small, 'k-', alpha=0.3, linewidth=1)
    ax2.plot(bin_centers, smoothed_small, 'b-', linewidth=2)
    ax2.set_xlabel('Body orientation (°)')
    ax2.set_ylabel('Relative probability')
    ax2.set_xlim(-180, 180)
    ax2.set_title(f'Small Runs (n={len(small_run_orientations)})')
    
    # Plot 3: All runs combined
    ax3.plot(bin_centers, hist_all, 'k-', alpha=0.3, linewidth=1)
    ax3.plot(bin_centers, smoothed_all, 'g-', linewidth=2)
    ax3.set_xlabel('Body orientation (°)')
    ax3.set_ylabel('Relative probability')
    ax3.set_xlim(-180, 180)
    ax3.set_title(f'All Runs Combined (n={len(all_run_orientations)})')
    
    # Add super title
    plt.suptitle(f'Run Orientation Distributions (Total larvae: {total_larvae})', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create comparison plot - all distributions on one plot
    plt.figure(figsize=(6, 4))
    
    if len(large_run_orientations) > 0:
        plt.plot(bin_centers, smoothed_large, 'r-', linewidth=2, label='Large runs')
    
    if len(small_run_orientations) > 0:
        plt.plot(bin_centers, smoothed_small, 'b-', linewidth=2, label='Small runs')
    
    if len(all_run_orientations) > 0:
        plt.plot(bin_centers, smoothed_all, 'g-', linewidth=2, label='All runs')
    
    plt.xlabel('Body orientation (°)')
    plt.ylabel('Relative probability')
    plt.xlim(-180, 180)
    plt.title(f'Run Orientation Comparison (n={total_larvae} larvae)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'large_run_orientations': large_run_orientations,
        'small_run_orientations': small_run_orientations,
        'all_run_orientations': all_run_orientations,
        'hist_large': hist_large,
        'hist_small': hist_small,
        'hist_all': hist_all,
        'bin_centers': bin_centers,
        'smoothed_large': smoothed_large,
        'smoothed_small': smoothed_small,
        'smoothed_all': smoothed_all,
        'n_larvae': total_larvae,
        'n_large_runs': len(large_run_orientations),
        'n_small_runs': len(small_run_orientations),
        'n_total_runs': len(all_run_orientations)
    }




def analyze_turn_rate_by_orientation(trx_data, larva_id=None, bin_width=10):
    """
    Calculate turn rate as a function of orientation for large turns, small turns, and all turns.
    
    Args:
        trx_data: Dictionary containing tracking data
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae
        bin_width: Width of orientation bins in degrees
        
    Returns:
        dict: Contains turn rates and orientation bins for large, small, and all turns
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    
    def get_orientations_and_states(larva_data):
        """Extract orientations and turn states for large and small turns."""
        # Calculate orientation
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        # CHECK NEGATIVE AXIS IS ZERO
        # Calculate orientations in degrees
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
        
        # Get casting/turning states
        # Check if small_large_state is available
        if 'global_state_small_large_state' in larva_data:
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            # Define masks for large turns (state = 2.0) and small turns (state = 1.5)
            is_large_turn = states == 2.0
            is_small_turn = states == 1.5
            is_any_turn = is_large_turn | is_small_turn
        else:
            # Fall back to regular large_state if small_large_state isn't available
            states = np.array(larva_data['global_state_large_state']).flatten()
            # With just large state, only state 2 is turn/cast
            is_large_turn = states == 2
            is_small_turn = np.zeros_like(states, dtype=bool)  # No small turns
            is_any_turn = is_large_turn
        
        return orientations, is_large_turn, is_small_turn, is_any_turn
    
    # Initialize storage for large, small, and all turns
    all_orientations = []
    large_turn_states = []
    small_turn_states = []
    all_turn_states = []
    
    # Process data
    if larva_id is not None:
        # Single larva analysis
        larva_data = trx_data[larva_id]
        orientations, is_large, is_small, is_any = get_orientations_and_states(larva_data)
        all_orientations.extend(orientations)
        large_turn_states.extend(is_large)
        small_turn_states.extend(is_small)
        all_turn_states.extend(is_any)
        n_larvae = 1
        title = f'Larva {larva_id} - Cast Probability'
    else:
        # All larvae analysis
        if 'data' in trx_data:
            data_to_process = trx_data['data']
            n_larvae = trx_data['metadata']['total_larvae']
        else:
            data_to_process = trx_data
            n_larvae = len(data_to_process)
            
        for larva_data in data_to_process.values():
            orientations, is_large, is_small, is_any = get_orientations_and_states(larva_data)
            all_orientations.extend(orientations)
            large_turn_states.extend(is_large)
            small_turn_states.extend(is_small)
            all_turn_states.extend(is_any)
        title = f'Cast Probability (n={n_larvae})'
    
    # Convert to numpy arrays
    all_orientations = np.array(all_orientations)
    large_turn_states = np.array(large_turn_states)
    small_turn_states = np.array(small_turn_states)
    all_turn_states = np.array(all_turn_states)
    
    # Create orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate turn rates for each bin and turn type
    large_turn_rates = []
    small_turn_rates = []
    all_turn_rates = []
    
    for i in range(len(bins)-1):
        mask = (all_orientations >= bins[i]) & (all_orientations < bins[i+1])
        if np.sum(mask) > 0:
            large_turn_rates.append(np.mean(large_turn_states[mask]))
            small_turn_rates.append(np.mean(small_turn_states[mask]))
            all_turn_rates.append(np.mean(all_turn_states[mask]))
        else:
            large_turn_rates.append(0)
            small_turn_rates.append(0)
            all_turn_rates.append(0)
    
    large_turn_rates = np.array(large_turn_rates)
    small_turn_rates = np.array(small_turn_rates)
    all_turn_rates = np.array(all_turn_rates)
    
    # Apply smoothing
    large_smoothed = gaussian_filter1d(large_turn_rates, sigma=1)
    small_smoothed = gaussian_filter1d(small_turn_rates, sigma=1)
    all_smoothed = gaussian_filter1d(all_turn_rates, sigma=1)
    
    # Create figure with three subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Large turns
    ax1.plot(bin_centers, large_turn_rates, 'k-', alpha=0.3, linewidth=1)
    ax1.plot(bin_centers, large_smoothed, 'r-', linewidth=2)
    ax1.set_xlabel('Orientation (°)')
    ax1.set_ylabel('Cast probability')
    ax1.set_xlim(-180, 180)
    ax1.set_title('Cast Turns')
    
    # Plot 2: Small turns
    ax2.plot(bin_centers, small_turn_rates, 'k-', alpha=0.3, linewidth=1)
    ax2.plot(bin_centers, small_smoothed, 'b-', linewidth=2)
    ax2.set_xlabel('Orientation (°)')
    ax2.set_ylabel('Cast probability')
    ax2.set_xlim(-180, 180)
    ax2.set_title('Small Casts')
    
    # Plot 3: All turns combined
    ax3.plot(bin_centers, all_turn_rates, 'k-', alpha=0.3, linewidth=1)
    ax3.plot(bin_centers, all_smoothed, 'g-', linewidth=2)
    ax3.set_xlabel('Orientation (°)')
    ax3.set_ylabel('Cast probability')
    ax3.set_xlim(-180, 180)
    ax3.set_title('All Casts Combined')
    
    # Add super title
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create comparison plot - all distributions on one plot
    plt.figure(figsize=(6, 4))
    
    # Count non-zero values to check if we have data
    n_large_nonzero = np.count_nonzero(large_turn_rates)
    n_small_nonzero = np.count_nonzero(small_turn_rates)
    n_all_nonzero = np.count_nonzero(all_turn_rates)
    
    if n_large_nonzero > 0:
        plt.plot(bin_centers, large_smoothed, 'r-', linewidth=2, label='Large casts')
    
    if n_small_nonzero > 0:
        plt.plot(bin_centers, small_smoothed, 'b-', linewidth=2, label='Small casts')
    
    if n_all_nonzero > 0:
        plt.plot(bin_centers, all_smoothed, 'g-', linewidth=2, label='All casts')
    
    plt.xlabel('Orientation (°)')
    plt.ylabel('Cast probability')
    plt.xlim(-180, 180)
    plt.title(f'Cast Probability Comparison (n={n_larvae} larvae)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate counts and frequencies for statistical comparison
    large_turn_count = np.sum(large_turn_states)
    small_turn_count = np.sum(small_turn_states)
    all_turn_count = np.sum(all_turn_states)
    
    return {
        'orientations': all_orientations,
        'large_turn_states': large_turn_states,
        'small_turn_states': small_turn_states,
        'all_turn_states': all_turn_states,
        'bin_centers': bin_centers,
        'large_turn_rates': large_turn_rates,
        'small_turn_rates': small_turn_rates,
        'all_turn_rates': all_turn_rates,
        'large_smoothed': large_smoothed,
        'small_smoothed': small_smoothed,
        'all_smoothed': all_smoothed,
        'n_larvae': n_larvae,
        'large_turn_count': int(large_turn_count),
        'small_turn_count': int(small_turn_count),
        'all_turn_count': int(all_turn_count)
    }

def analyze_turn_amplitudes_by_orientation(trx_data, larva_id=None, bin_width=10):
    """
    Calculate turn amplitudes as a function of orientation for large turns, small turns, and all turns.
    
    Turn amplitude is defined as the absolute change in orientation between 
    successive frames when a turn is detected.
    
    Args:
        trx_data: Dictionary containing tracking data
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae
        bin_width: Width of orientation bins in degrees
        
    Returns:
        dict: Contains turn amplitude data for large, small, and all turns
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt

    def get_orientations_and_states(larva_data):
        """Extract orientations and turn states for large and small turns."""
        # Calculate orientation using tail-to-center vector
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        # CHECK NEGATIVE AXIS IS ZERO
        # Calculate orientations in degrees
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
        
        # Get turn states - handle both large and small turns
        if 'global_state_small_large_state' in larva_data:
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            # Define masks for large turns (state = 2.0) and small turns (state = 1.5)
            is_large_turn = states == 2.0
            is_small_turn = states == 1.5
            is_any_turn = is_large_turn | is_small_turn
        else:
            # Fall back to regular large_state if small_large_state isn't available
            states = np.array(larva_data['global_state_large_state']).flatten()
            # With just large state, only state 2 is turn/cast
            is_large_turn = states == 2
            is_small_turn = np.zeros_like(states, dtype=bool)  # No small turns
            is_any_turn = is_large_turn
            
        return orientations, is_large_turn, is_small_turn, is_any_turn

    def circular_diff(a, b):
        """
        Compute minimal difference between two angles a and b (in degrees).
        Result is in [-180, 180] and we take the absolute value.
        """
        diff = a - b
        diff = (diff + 180) % 360 - 180
        return np.abs(diff)
    
    # Storage for base orientation values and corresponding turn amplitudes
    all_base_orientations = []
    large_turn_amplitudes = []
    small_turn_amplitudes = []
    all_turn_amplitudes = []
    
    large_base_orientations = []
    small_base_orientations = []
    
    # Process data either for single larva or all larvae
    if larva_id is not None:
        # Single larva analysis
        larva_data = trx_data[larva_id]
        orientations, is_large_turn, is_small_turn, is_any_turn = get_orientations_and_states(larva_data)
        n_larvae = 1
        title = f'Larva {larva_id} - Cast Amplitudes'
        
        # Compute turn amplitude for frames where turning occurs (skip the first frame)
        for i in range(1, len(orientations)):
            # For large turns
            if is_large_turn[i]:
                amp = circular_diff(orientations[i], orientations[i-1])
                large_turn_amplitudes.append(amp)
                large_base_orientations.append(orientations[i-1])
                
            # For small turns
            if is_small_turn[i]:
                amp = circular_diff(orientations[i], orientations[i-1])
                small_turn_amplitudes.append(amp)
                small_base_orientations.append(orientations[i-1])
                
            # For any turn
            if is_any_turn[i]:
                amp = circular_diff(orientations[i], orientations[i-1])
                all_turn_amplitudes.append(amp)
                all_base_orientations.append(orientations[i-1])
    else:
        # Multiple larvae analysis
        if 'data' in trx_data:
            data_to_process = trx_data['data']
            n_larvae = trx_data['metadata']['total_larvae']
        else:
            data_to_process = trx_data
            n_larvae = len(data_to_process)
        
        title = f'Cast Amplitudes (n={n_larvae})'
        
        for larva in data_to_process.values():
            orientations, is_large_turn, is_small_turn, is_any_turn = get_orientations_and_states(larva)
            
            for i in range(1, len(orientations)):
                # For large turns
                if is_large_turn[i]:
                    amp = circular_diff(orientations[i], orientations[i-1])
                    large_turn_amplitudes.append(amp)
                    large_base_orientations.append(orientations[i-1])
                    
                # For small turns
                if is_small_turn[i]:
                    amp = circular_diff(orientations[i], orientations[i-1])
                    small_turn_amplitudes.append(amp)
                    small_base_orientations.append(orientations[i-1])
                    
                # For any turn
                if is_any_turn[i]:
                    amp = circular_diff(orientations[i], orientations[i-1])
                    all_turn_amplitudes.append(amp)
                    all_base_orientations.append(orientations[i-1])
    
    # Convert to numpy arrays
    large_base_orientations = np.array(large_base_orientations)
    small_base_orientations = np.array(small_base_orientations)
    all_base_orientations = np.array(all_base_orientations)
    
    large_turn_amplitudes = np.array(large_turn_amplitudes)
    small_turn_amplitudes = np.array(small_turn_amplitudes)
    all_turn_amplitudes = np.array(all_turn_amplitudes)
    
    # Create orientation bins from -180 to 180 degrees
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the turn amplitudes for each turn type
    large_mean_amplitudes = []
    small_mean_amplitudes = []
    all_mean_amplitudes = []
    
    for i in range(len(bins)-1):
        # For large turns
        if len(large_base_orientations) > 0:
            mask = (large_base_orientations >= bins[i]) & (large_base_orientations < bins[i+1])
            if np.sum(mask) > 0:
                large_mean_amplitudes.append(np.mean(large_turn_amplitudes[mask]))
            else:
                large_mean_amplitudes.append(np.nan)
        else:
            large_mean_amplitudes.append(np.nan)
            
        # For small turns
        if len(small_base_orientations) > 0:
            mask = (small_base_orientations >= bins[i]) & (small_base_orientations < bins[i+1])
            if np.sum(mask) > 0:
                small_mean_amplitudes.append(np.mean(small_turn_amplitudes[mask]))
            else:
                small_mean_amplitudes.append(np.nan)
        else:
            small_mean_amplitudes.append(np.nan)
        
        # For all turns
        if len(all_base_orientations) > 0:
            mask = (all_base_orientations >= bins[i]) & (all_base_orientations < bins[i+1])
            if np.sum(mask) > 0:
                all_mean_amplitudes.append(np.mean(all_turn_amplitudes[mask]))
            else:
                all_mean_amplitudes.append(np.nan)
        else:
            all_mean_amplitudes.append(np.nan)
    
    # Convert to numpy arrays
    large_mean_amplitudes = np.array(large_mean_amplitudes)
    small_mean_amplitudes = np.array(small_mean_amplitudes)
    all_mean_amplitudes = np.array(all_mean_amplitudes)
    
    # Apply smoothing (handling NaNs)
    large_smooth_input = np.nan_to_num(large_mean_amplitudes, nan=0)
    small_smooth_input = np.nan_to_num(small_mean_amplitudes, nan=0)
    all_smooth_input = np.nan_to_num(all_mean_amplitudes, nan=0)
    
    large_smoothed = gaussian_filter1d(large_smooth_input, sigma=1)
    small_smoothed = gaussian_filter1d(small_smooth_input, sigma=1)
    all_smoothed = gaussian_filter1d(all_smooth_input, sigma=1)
    
    # Create figure with three subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Large turn amplitudes - only smoothed curve without raw data points
    ax1.plot(bin_centers, large_smoothed, 'r-', linewidth=2)
    ax1.set_xlabel('Orientation (°)')
    ax1.set_ylabel('Mean Cast Amplitude (°)')
    ax1.set_xlim(-180, 180)
    ax1.set_title(f'Large Casts (n={len(large_turn_amplitudes)})')
    
    # Plot 2: Small turn amplitudes - only smoothed curve without raw data points
    ax2.plot(bin_centers, small_smoothed, 'b-', linewidth=2)
    ax2.set_xlabel('Orientation (°)')
    ax2.set_ylabel('Mean Cast Amplitude (°)')
    ax2.set_xlim(-180, 180)
    ax2.set_title(f'Small Casts (n={len(small_turn_amplitudes)})')
    
    # Plot 3: All turn amplitudes - only smoothed curve without raw data points
    ax3.plot(bin_centers, all_smoothed, 'g-', linewidth=2)
    ax3.set_xlabel('Orientation (°)')
    ax3.set_ylabel('Mean Cast Amplitude (°)')
    ax3.set_xlim(-180, 180)
    ax3.set_title(f'All Casts Combined (n={len(all_turn_amplitudes)})')
    
    # Add super title
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create comparison plot - all curves on one plot
    plt.figure(figsize=(6, 4))
    
    # Check if we have data to plot
    has_large_data = len(large_turn_amplitudes) > 0 and not np.all(np.isnan(large_mean_amplitudes))
    has_small_data = len(small_turn_amplitudes) > 0 and not np.all(np.isnan(small_mean_amplitudes))
    has_all_data = len(all_turn_amplitudes) > 0 and not np.all(np.isnan(all_mean_amplitudes))
    
    if has_large_data:
        plt.plot(bin_centers, large_smoothed, 'r-', linewidth=2, label='Large casts')
    
    if has_small_data:
        plt.plot(bin_centers, small_smoothed, 'b-', linewidth=2, label='Small casts')
    
    if has_all_data:
        plt.plot(bin_centers, all_smoothed, 'g-', linewidth=2, label='All casts')
    
    plt.xlabel('Orientation (°)')
    plt.ylabel('Mean Cast Amplitude (°)')
    plt.xlim(-180, 180)
    plt.title(f'Cast Amplitude Comparison (n={n_larvae} larvae)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return the computed data and statistics
    return {
        'large_base_orientations': large_base_orientations,
        'small_base_orientations': small_base_orientations,
        'all_base_orientations': all_base_orientations,
        'large_turn_amplitudes': large_turn_amplitudes,
        'small_turn_amplitudes': small_turn_amplitudes,
        'all_turn_amplitudes': all_turn_amplitudes,
        'bin_centers': bin_centers,
        'large_mean_amplitudes': large_mean_amplitudes,
        'small_mean_amplitudes': small_mean_amplitudes,
        'all_mean_amplitudes': all_mean_amplitudes,
        'large_smoothed': large_smoothed,
        'small_smoothed': small_smoothed,
        'all_smoothed': all_smoothed,
        'n_larvae': n_larvae,
        'n_large_turns': len(large_turn_amplitudes),
        'n_small_turns': len(small_turn_amplitudes),
        'n_all_turns': len(all_turn_amplitudes)
    }




def analyze_lateral_turn_rates(trx_data, angle_width=15, bin_width=5):
    """
    Analyze turn rates around ±90° orientations with confidence intervals.
    
    Args:
        trx_data: Dictionary containing tracking data
        angle_width: Width of lateral quadrants around ±90°
        bin_width: Width of orientation bins in degrees
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    def get_orientations_and_states(larva_data):
        """Extract orientation and casting state data from a single larva."""
        try:
            # Handle nested data structure
            if 'data' in larva_data:
                larva_data = larva_data['data']
                
            # Get position data
            x_center = np.array(larva_data['x_center']).flatten()
            y_center = np.array(larva_data['y_center']).flatten()
            x_spine = np.array(larva_data['x_spine'])
            y_spine = np.array(larva_data['y_spine'])
            
            # Get tail positions
            x_tail = x_spine[-1].flatten() if x_spine.ndim > 1 else x_spine.flatten()
            y_tail = y_spine[-1].flatten() if y_spine.ndim > 1 else y_spine.flatten()
            
            # Calculate tail-to-center vectors
            tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
            orientations = np.degrees(np.arctan2(tail_to_center[:, 1], tail_to_center[:, 0]))
            
            # Get casting states
            states = np.array(larva_data['global_state_large_state']).flatten()
            is_casting = states == 2
            
            return orientations, is_casting
            
        except Exception as e:
            print(f"Error processing larva data: {str(e)}")
            return np.array([]), np.array([])
    
    # Define lateral regions
    left_range = (-90-angle_width, -90+angle_width)
    right_range = (90-angle_width, 90+angle_width)
    
    # Initialize storage
    all_orientations = []
    all_casting_states = []
    left_rates = []
    right_rates = []
    
    # Process all larvae
    if 'data' in trx_data:
        data_to_process = trx_data['data']
        n_larvae = trx_data['metadata']['total_larvae']
    else:
        data_to_process = trx_data
        n_larvae = len(data_to_process)
        
    for larva_id, larva_data in data_to_process.items():
        orientations, is_casting = get_orientations_and_states(larva_data)
        
        if len(orientations) > 0 and len(is_casting) > 0:
            # Add to overall distributions
            all_orientations.extend(orientations)
            all_casting_states.extend(is_casting)
            
            # Calculate rates for lateral quadrants
            left_mask = ((orientations >= left_range[0]) & 
                        (orientations < left_range[1]))
            right_mask = ((orientations >= right_range[0]) & 
                         (orientations < right_range[1]))
            
            if np.any(left_mask):
                left_rates.append(np.mean(is_casting[left_mask]))
            if np.any(right_mask):
                right_rates.append(np.mean(is_casting[right_mask]))
    
    # Convert to arrays
    all_orientations = np.array(all_orientations)
    all_casting_states = np.array(all_casting_states)
    left_rates = np.array(left_rates)
    right_rates = np.array(right_rates)
    
    # Create orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate turn rates and confidence intervals for each bin
    turn_rates = []
    turn_rates_std = []
    turn_rates_sem = []
    n_samples = []
    
    for i in range(len(bins)-1):
        mask = (all_orientations >= bins[i]) & (all_orientations < bins[i+1])
        bin_data = all_casting_states[mask]
        if len(bin_data) > 0:
            turn_rates.append(np.mean(bin_data))
            turn_rates_std.append(np.std(bin_data))
            turn_rates_sem.append(stats.sem(bin_data))
            n_samples.append(len(bin_data))
        else:
            turn_rates.append(0)
            turn_rates_std.append(0)
            turn_rates_sem.append(0)
            n_samples.append(0)
    
    turn_rates = np.array(turn_rates)
    turn_rates_std = np.array(turn_rates_std)
    turn_rates_sem = np.array(turn_rates_sem)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Full distribution with confidence intervals
    smoothed = gaussian_filter1d(turn_rates, sigma=1)
    smoothed_sem = gaussian_filter1d(turn_rates_sem, sigma=1)
    
    # Plot raw data points with size proportional to sample size
    sizes = np.array(n_samples) / max(n_samples) * 100
    ax1.scatter(bin_centers, turn_rates, c='k', alpha=0.3, s=sizes)
    
    # Plot smoothed line with shaded confidence interval
    ax1.plot(bin_centers, smoothed, 'r-', linewidth=2, label='Mean turn rate')
    ax1.fill_between(bin_centers, 
                     smoothed - 1.96*smoothed_sem,
                     smoothed + 1.96*smoothed_sem,
                     color='r', alpha=0.2, label='95% CI')
    
    # Highlight lateral regions
    ax1.axvspan(left_range[0], left_range[1], color='blue', alpha=0.1)
    ax1.axvspan(right_range[0], right_range[1], color='blue', alpha=0.1)
    
    ax1.set_xlabel('Orientation (°)')
    ax1.set_ylabel('Turn probability')
    ax1.set_xlim(-180, 180)
    ax1.set_title('Turn Rate Distribution')
    ax1.legend()
    
    # Plot 2: Lateral quadrants comparison
    bp = ax2.boxplot([left_rates, right_rates],
                     positions=[0, 1],
                     labels=['Left\n(-90°)', 'Right\n(90°)'],
                     notch=True,
                     patch_artist=True)
    
    # Add individual points with jitter
    for i, data in enumerate([left_rates, right_rates]):
        if len(data) > 0:
            jitter = np.random.normal(0, 0.02, size=len(data))
            ax2.scatter(np.full_like(data, i+1) + jitter, data,
                       alpha=0.2, color='blue', label='Individual larvae' if i==0 else "")
    
    # Add mean ± SEM for each quadrant
    for i, data in enumerate([left_rates, right_rates]):
        if len(data) > 0:
            mean = np.mean(data)
            sem = stats.sem(data)
            ax2.errorbar(i+1, mean, yerr=sem, color='red', 
                        capsize=5, capthick=2, label='Mean ± SEM' if i==0 else "")
    
    # Add statistics
    stats_text = []
    for name, data in zip(['Left', 'Right'], [left_rates, right_rates]):
        if len(data) > 0:
            stats_text.append(f"{name}:\n"
                            f"mean = {np.mean(data):.3f}\n"
                            f"SEM = {stats.sem(data):.3f}\n"
                            f"n = {len(data)}")
    
    if len(left_rates) > 0 and len(right_rates) > 0:
        t_stat, p_val = stats.ttest_ind(left_rates, right_rates)
        stats_text.append(f"\nt-test p-value: {p_val:.3e}")
    
    ax2.text(1.2, ax2.get_ylim()[0], '\n'.join(stats_text),
             va='bottom', ha='left')
    
    ax2.set_ylabel('Turn probability')
    ax2.set_title('Lateral Turn Rates')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'left_rates': left_rates,
        'right_rates': right_rates,
        'bin_centers': bin_centers,
        'turn_rates': turn_rates,
        'turn_rates_sem': turn_rates_sem,
        'smoothed': smoothed,
        'n_samples': n_samples,
        'angle_width': angle_width,
        'n_larvae': n_larvae
    }

def analyze_cast_directions(trx_data, fps=3):
    """
    Analyze if casts are upstream or downstream relative to wind direction.
    
    Args:
        trx_data: Dictionary containing tracking data
        fps: Frames per second for time calculation
    """
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    
    def get_cast_events(larva_data):
        """
        Get cast events and their angles from larva data.
        Returns list of angles during cast initiation.
        """
        try:
            # Get states and find cast start indices
            states = np.array(larva_data['global_state_large_state']).flatten()
            cast_starts = np.where((states[1:] == 2) & (states[:-1] != 2))[0] + 1
            
            # For each cast, calculate average orientation in first second
            cast_directions = []
            frames_per_sec = int(fps)
            
            for start in cast_starts:
                end = min(start + frames_per_sec, len(states))
                if end - start < frames_per_sec/2:  # Skip if cast is too short
                    print('too short ')
                    continue
                
                try:
                    # Get head vector relative to center
                    x_center = larva_data['x_center'][start:end]
                    y_center = larva_data['y_center'][start:end]
                    
                    # Extract spine data correctly
                    x_spine = np.array(larva_data['x_spine'])
                    y_spine = np.array(larva_data['y_spine'])
                    
                    # Get head coordinates (first spine point)
                    if x_spine.ndim > 1:  # If spine data is 2D
                        x_head = x_spine[0][start:end]
                        y_head = y_spine[0][start:end]
                    else:  # If spine data is 1D
                        x_head = x_spine[start:end]
                        y_head = y_spine[start:end]
                    
                    # Ensure all arrays have data
                    if (len(x_center) == 0 or len(y_center) == 0 or 
                        len(x_head) == 0 or len(y_head) == 0):
                        continue
                    
                    # Convert to numpy arrays if they aren't already
                    x_center = np.array(x_center).flatten()
                    y_center = np.array(y_center).flatten()
                    x_head = np.array(x_head).flatten()
                    y_head = np.array(y_head).flatten()
                    
                    # Calculate head vector angle
                    head_vectors = np.column_stack([x_head - x_center, y_head - y_center])
                    angles = np.degrees(np.arctan2(head_vectors[:, 1], -head_vectors[:, 0]))
                    mean_angle = np.mean(angles)
                    print(f'mean angle {mean_angle}')
                    
                    cast_directions.append(mean_angle)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing cast at frame {start}: {str(e)}")
                    continue
                    
            return cast_directions
            
        except Exception as e:
            print(f"Error in get_cast_events: {str(e)}")
            return []
    
    # Process all larvae
    upstream_casts = []
    downstream_casts = []
    
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
        
    for larva_id, larva_data in data_to_process.items():
        try:
            if 'data' in larva_data:
                larva_data = larva_data['data']
                
            cast_angles = get_cast_events(larva_data)
            # Classify casts
            for angle in cast_angles:
                print(angle)
                # Normalize angle to [-180, 180]
                angle = ((angle + 180) % 360) - 180
                if -90 <= angle <= 90:
                    upstream_casts.append(angle)
                else:
                    downstream_casts.append(angle)
                    
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    # Check if we have any data to plot
    if len(upstream_casts) == 0 and len(downstream_casts) == 0:
        print("No valid cast events found in the data")
        return None
        
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create box plots instead of violin plots (more robust with small datasets)
    positions = [1, 2]
    data = [upstream_casts, downstream_casts]
    labels = ['Upstream\ncasts', 'Downstream\ncasts']
    
    # Create box plots
    bp = plt.boxplot(data, positions=positions, 
                    notch=True, 
                    patch_artist=True,
                    showfliers=False)
    
    # Customize colors
    colors = ['#4DBBD5', '#00A087']
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    
    # Add individual points with jitter
    for i, d in enumerate(data):
        if len(d) > 0:
            pos = positions[i]
            jitter = np.random.normal(0, 0.02, size=len(d))
            plt.scatter(np.full_like(d, pos) + jitter, d,
                       alpha=0.2, color=colors[i], s=20)
    
    # Add statistics
    stats_text = []
    for name, d in zip(['Upstream', 'Downstream'], data):
        if len(d) > 0:
            stats_text.append(f"{name}:\n"
                            f"mean = {np.mean(d):.1f}°\n"
                            f"std = {np.std(d):.1f}°\n"
                            f"n = {len(d)}")
    
    # Perform statistical test
    if len(upstream_casts) > 0 and len(downstream_casts) > 0:
        t_stat, p_val = stats.ttest_ind(upstream_casts, downstream_casts)
        stats_text.append(f"\nt-test p-value: {p_val:.3e}")
    
    plt.text(2.3, plt.ylim()[0], '\n'.join(stats_text),
             va='bottom', ha='left')
    
    # Customize plot
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xticks(positions, labels)
    plt.ylabel('Cast angle (degrees)')
    plt.title('Distribution of Cast Directions\nRelative to Wind (-x axis)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'upstream_angles': np.array(upstream_casts),
        'downstream_angles': np.array(downstream_casts),
        'upstream_mean': np.mean(upstream_casts) if len(upstream_casts) > 0 else np.nan,
        'downstream_mean': np.mean(downstream_casts) if len(downstream_casts) > 0 else np.nan,
        'upstream_std': np.std(upstream_casts) if len(upstream_casts) > 0 else np.nan,
        'downstream_std': np.std(downstream_casts) if len(downstream_casts) > 0 else np.nan,
        'n_upstream': len(upstream_casts),
        'n_downstream': len(downstream_casts)
    }


def analyze_perpendicular_cast_directions(trx_data, angle_width=15, min_frame=3):
    """
    Analyze the probability of upstream vs downstream casts when larvae are perpendicular to flow.
    Shows per-larva casting biases and separates analysis for small casts, large casts, and all casts.
    
    Args:
        trx_data (dict): Tracking data dictionary containing larvae data
        angle_width (int): Width of perpendicular orientation sector in degrees
        min_frame (int): Minimum number of frames to consider for a cast
    
    Returns:
        dict: Cast direction metrics including probabilities and statistical analysis
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import pandas as pd
    
    # Determine which data structure we're working with
    if 'data' in trx_data:
        data_to_process = trx_data['data']
        n_larvae = trx_data['metadata']['total_larvae']
    else:
        data_to_process = trx_data
        n_larvae = len(data_to_process)
    
    # Store raw counts for analysis, separated by cast type
    total_counts = {
        'small': {'upstream': 0, 'downstream': 0},
        'large': {'upstream': 0, 'downstream': 0},
        'all': {'upstream': 0, 'downstream': 0}
    }
    
    # Store per-larva probabilities for box plot
    larva_probabilities = {
        'small': {'upstream': [], 'downstream': []},
        'large': {'upstream': [], 'downstream': []},
        'all': {'upstream': [], 'downstream': []}
    }
    
    # Store per-larva counts (for plotting)
    per_larva_counts = {
        'small': {'upstream': [], 'downstream': [], 'larva_ids': [], 'total': []},
        'large': {'upstream': [], 'downstream': [], 'larva_ids': [], 'total': []},
        'all': {'upstream': [], 'downstream': [], 'larva_ids': [], 'total': []}
    }
    
    # Define perpendicular angle ranges (both left and right sides)
    # With wind blowing towards negative x, perpendicular is ±90°
    left_perp_range = (-90 - angle_width, -90 + angle_width)
    right_perp_range = (90 - angle_width, 90 + angle_width)
    
    def is_perpendicular(angle):
        """Check if angle is in perpendicular range"""
        return ((left_perp_range[0] <= angle <= left_perp_range[1]) or 
                (right_perp_range[0] <= angle <= right_perp_range[1]))
    
    def determine_cast_direction(init_angle, cast_angle):
        """
        Determine if cast is upstream or downstream when larva is perpendicular.
        
        Wind is blowing towards negative x direction.
        When oriented at -90° (left perpendicular), casting towards positive x is upstream
        When oriented at +90° (right perpendicular), casting towards positive x is upstream
        """
        angle_diff = (cast_angle - init_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
            
        # If larva is around -90° (left perpendicular)
        if left_perp_range[0] <= init_angle <= left_perp_range[1]:
            # Positive angle_diff means turning clockwise (towards positive x/upstream)
            return 'upstream' if angle_diff > 0 else 'downstream'
        # If larva is around +90° (right perpendicular)
        else:
            # Negative angle_diff means turning counter-clockwise (towards positive x/upstream)
            return 'upstream' if angle_diff < 0 else 'downstream'
    
    # Process each larva
    larvae_processed = 0
    for larva_id, larva_data in data_to_process.items():
        try:
            # Extract nested data if needed
            if 'data' in larva_data:
                larva_data = larva_data['data']
            
            # Check for small_large_state or fall back to large_state
            has_small_large_state = 'global_state_small_large_state' in larva_data
            
            if has_small_large_state:
                # Extract both small and large cast states
                cast_states = np.array(larva_data['global_state_small_large_state']).flatten()
                large_cast_mask = cast_states == 2.0  # Large casts = 2.0
                small_cast_mask = cast_states == 1.5  # Small casts = 1.5
                any_cast_mask = large_cast_mask | small_cast_mask
            else:
                # Fall back to just large state
                cast_states = np.array(larva_data.get('cast', larva_data.get('global_state_large_state', []))).flatten()
                large_cast_mask = cast_states == 2  # Only large casts available
                small_cast_mask = np.zeros_like(large_cast_mask, dtype=bool)  # No small casts
                any_cast_mask = large_cast_mask
                
            if len(cast_states) == 0:
                continue
                
            # Find start frames for different cast types
            large_cast_starts = np.where((large_cast_mask[1:]) & (~large_cast_mask[:-1]))[0] + 1
            small_cast_starts = np.where((small_cast_mask[1:]) & (~small_cast_mask[:-1]))[0] + 1
            any_cast_starts = np.where((any_cast_mask[1:]) & (~any_cast_mask[:-1]))[0] + 1
            
            # Alternative: include all frames in a cast state
            if len(large_cast_starts) == 0:
                large_cast_starts = np.where(large_cast_mask)[0]
            if len(small_cast_starts) == 0 and has_small_large_state:
                small_cast_starts = np.where(small_cast_mask)[0]
            if len(any_cast_starts) == 0:
                any_cast_starts = np.where(any_cast_mask)[0]
                
            if len(any_cast_starts) == 0:
                continue
            
            # Extract coordinates
            x_spine = np.array(larva_data['x_spine'])
            y_spine = np.array(larva_data['y_spine'])
            x_center = np.array(larva_data['x_center']).flatten()
            y_center = np.array(larva_data['y_center']).flatten()
            
            # Handle different spine data shapes
            if x_spine.ndim == 1:  # 1D array
                x_tail = x_spine
                y_tail = y_spine
                x_head = x_spine
                y_head = y_spine
            else:  # 2D array with shape (spine_points, frames)
                x_tail = x_spine[-1, :]
                y_tail = y_spine[-1, :]
                x_head = x_spine[0, :]
                y_head = y_spine[0, :]
            
            # Calculate vectors
            tail_to_center = np.column_stack([
                x_center - x_tail,
                y_center - y_tail
            ])
            
            center_to_head = np.column_stack([
                x_head - x_center,
                y_head - y_center
            ])
            
            # Calculate angles
            # Note: arctan2(y, -x) is correct for wind in negative x direction
            orientation_angles = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
            cast_angles = np.degrees(np.arctan2(center_to_head[:, 1], -center_to_head[:, 0]))
            
            # Initialize counts for this larva
            larva_counts = {
                'small': {'upstream': 0, 'downstream': 0},
                'large': {'upstream': 0, 'downstream': 0},
                'all': {'upstream': 0, 'downstream': 0}
            }
            
            # Process large casts
            for start in large_cast_starts:
                try:
                    if start >= len(orientation_angles):
                        continue
                        
                    # Get initial orientation (at start of cast)
                    init_orientation = orientation_angles[start]
                    
                    # Only analyze casts when orientation is perpendicular
                    if not is_perpendicular(init_orientation):
                        continue
                        
                    # Get maximum cast angle
                    end = min(start + 6, len(cast_angles))  # Look at first 6 frames of cast
                    if end <= start or end >= len(cast_angles):
                        continue
                        
                    cast_sequence = cast_angles[start:end]
                    if len(cast_sequence) < 3:  # Need at least 3 frames
                        continue
                    
                    # Find frame with maximum deviation
                    angle_diffs = np.abs(cast_sequence - init_orientation)
                    max_deviation_idx = np.argmax(angle_diffs)
                    max_cast_angle = cast_sequence[max_deviation_idx]
                    
                    # Determine if cast is upstream or downstream
                    cast_direction = determine_cast_direction(init_orientation, max_cast_angle)
                    
                    # Update counts for large casts
                    total_counts['large'][cast_direction] += 1
                    larva_counts['large'][cast_direction] += 1
                    
                    # Update total counts too
                    total_counts['all'][cast_direction] += 1
                    larva_counts['all'][cast_direction] += 1
                        
                except (IndexError, ValueError):
                    continue
            
            # Process small casts if available
            if has_small_large_state:
                for start in small_cast_starts:
                    try:
                        if start >= len(orientation_angles):
                            continue
                            
                        # Get initial orientation (at start of cast)
                        init_orientation = orientation_angles[start]
                        
                        # Only analyze casts when orientation is perpendicular
                        if not is_perpendicular(init_orientation):
                            continue
                            
                        # Get maximum cast angle
                        end = min(start + 6, len(cast_angles))  # Look at first 6 frames of cast
                        if end <= start or end >= len(cast_angles):
                            continue
                            
                        cast_sequence = cast_angles[start:end]
                        if len(cast_sequence) < 3:  # Need at least 3 frames
                            continue
                        
                        # Find frame with maximum deviation
                        angle_diffs = np.abs(cast_sequence - init_orientation)
                        max_deviation_idx = np.argmax(angle_diffs)
                        max_cast_angle = cast_sequence[max_deviation_idx]
                        
                        # Determine if cast is upstream or downstream
                        cast_direction = determine_cast_direction(init_orientation, max_cast_angle)
                        
                        # Update counts for small casts
                        total_counts['small'][cast_direction] += 1
                        larva_counts['small'][cast_direction] += 1
                        
                        # Update total counts too
                        total_counts['all'][cast_direction] += 1
                        larva_counts['all'][cast_direction] += 1
                            
                    except (IndexError, ValueError):
                        continue
            
            # Calculate per-larva probabilities if enough casts
            # For each cast type, require at least 3 casts to include in the analysis
            for cast_type in ['small', 'large', 'all']:
                larva_total = sum(larva_counts[cast_type].values())
                if larva_total >= 3:
                    upstream_prob = larva_counts[cast_type]['upstream'] / larva_total
                    downstream_prob = larva_counts[cast_type]['downstream'] / larva_total
                    
                    # Add probability to list
                    larva_probabilities[cast_type]['upstream'].append(upstream_prob)
                    larva_probabilities[cast_type]['downstream'].append(downstream_prob)
                    
                    # Save the raw counts for per-larva plotting
                    per_larva_counts[cast_type]['upstream'].append(larva_counts[cast_type]['upstream'])
                    per_larva_counts[cast_type]['downstream'].append(larva_counts[cast_type]['downstream'])
                    per_larva_counts[cast_type]['larva_ids'].append(str(larva_id))
                    per_larva_counts[cast_type]['total'].append(larva_total)
            
            # Count larva as processed if it had enough data for any cast type
            if (sum(larva_counts['small'].values()) >= 3 or 
                sum(larva_counts['large'].values()) >= 3 or 
                sum(larva_counts['all'].values()) >= 3):
                larvae_processed += 1
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    # Calculate overall probabilities for each cast type
    probabilities = {}
    for cast_type in ['small', 'large', 'all']:
        total_type_casts = sum(total_counts[cast_type].values())
        if total_type_casts > 0:
            probabilities[cast_type] = {
                'upstream': total_counts[cast_type]['upstream'] / total_type_casts,
                'downstream': total_counts[cast_type]['downstream'] / total_type_casts
            }
        else:
            probabilities[cast_type] = {'upstream': 0, 'downstream': 0}
    
    # Calculate mean probabilities from per-larva data
    mean_probabilities = {}
    sem_probabilities = {}  # Standard error of mean
    for cast_type in ['small', 'large', 'all']:
        up_probs = np.array(larva_probabilities[cast_type]['upstream'])
        down_probs = np.array(larva_probabilities[cast_type]['downstream'])
        
        if len(up_probs) > 0:
            mean_probabilities[cast_type] = {
                'upstream': np.mean(up_probs),
                'downstream': np.mean(down_probs)
            }
            sem_probabilities[cast_type] = {
                'upstream': np.std(up_probs) / np.sqrt(len(up_probs)) if len(up_probs) > 1 else 0,
                'downstream': np.std(down_probs) / np.sqrt(len(down_probs)) if len(down_probs) > 1 else 0
            }
        else:
            mean_probabilities[cast_type] = {'upstream': 0, 'downstream': 0}
            sem_probabilities[cast_type] = {'upstream': 0, 'downstream': 0}
    
    # Statistical tests for each cast type
    stats_results = {}
    for cast_type in ['small', 'large', 'all']:
        # Chi-square on raw counts
        observed = np.array([total_counts[cast_type]['upstream'], total_counts[cast_type]['downstream']])
        if np.sum(observed) > 0:
            expected = np.sum(observed) / 2  # Expected equal distribution
            chi2, p_chi2 = stats.chisquare(observed)
        else:
            chi2, p_chi2 = None, None
        
        # Paired t-test on larva probabilities
        up_probs = np.array(larva_probabilities[cast_type]['upstream'])
        down_probs = np.array(larva_probabilities[cast_type]['downstream'])
        
        if len(up_probs) > 1:
            t_stat, p_val = stats.ttest_rel(up_probs, down_probs)
        else:
            t_stat, p_val = None, None
            
        stats_results[cast_type] = {
            'chi2_result': (chi2, p_chi2) if chi2 is not None else None,
            'ttest_result': (t_stat, p_val) if t_stat is not None else None
        }
    
    # Create figures for the per-larva analysis
    plot_types = []
    for cast_type in ['all', 'large', 'small']:  # Order changed to put 'all' first
        if len(larva_probabilities[cast_type]['upstream']) > 0:
            plot_types.append(cast_type)
    
    # 1. First figure: standard box plots for each cast type
    fig, axes = plt.subplots(1, len(plot_types), figsize=(3 * len(plot_types), 4))
    
    # Handle case where only one subplot is created
    if len(plot_types) == 1:
        axes = [axes]
    
    # Create each box plot
    for i, cast_type in enumerate(plot_types):
        ax = axes[i]
        
        # Get data for this cast type
        data = [larva_probabilities[cast_type]['upstream'], 
                larva_probabilities[cast_type]['downstream']]
        
        # Create box plot - use consistent color scheme
        bp = ax.boxplot(data, positions=[0, 1], labels=['Upstream', 'Downstream'], 
                       notch=True, patch_artist=True,
                       widths=0.4, showfliers=True)
        
        # Customize colors - use same color for all box plots
        for box in bp['boxes']:
            box.set_facecolor('lightgray')
            box.set_alpha(0.8)
        
        # Add individual data points with jitter
        for j, d in enumerate(data):
            if len(d) > 0:
                pos = j
                jitter = np.random.normal(0, 0.02, size=len(d))
                ax.scatter(np.full_like(d, pos) + jitter, d,
                          alpha=0.5, color='black', s=20)
        
        # Add significance bar if p-value is significant
        p_val = stats_results[cast_type]['ttest_result'][1] if stats_results[cast_type]['ttest_result'] else None
        if p_val is not None and p_val < 0.05:
            # Get y positions
            y_max = max(max(data[0]) if len(data[0]) > 0 else 0, 
                        max(data[1]) if len(data[1]) > 0 else 0) + 0.05
            
            # Plot the line
            x1, x2 = 0, 1
            ax.plot([x1, x1, x2, x2], [y_max, y_max + 0.03, y_max + 0.03, y_max], lw=1.5, c='black')
            
            # Add stars based on significance level
            stars = '*' * sum([p_val < p for p in [0.05, 0.01, 0.001]])
            ax.text((x1 + x2) * 0.5, y_max + 0.04, stars, ha='center', va='bottom', color='black', fontsize=14)
            
            # Add p-value
            ax.text((x1 + x2) * 0.5, y_max + 0.06, f'p = {p_val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Add reference line for 0.5 probability (chance level)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        
        # Add raw counts as text on plot
        ax.text(0, 0.05, f'n = {total_counts[cast_type]["upstream"]}', ha='center', fontsize=8)
        ax.text(1, 0.05, f'n = {total_counts[cast_type]["downstream"]}', ha='center', fontsize=8)
        
        # Format plot
        if i == 0:
            ax.set_ylabel('Probability', fontsize=10)
        ax.set_ylim(0, 1.1)
        
        # Add type label as title
        type_labels = {
            'large': 'Large Casts',
            'small': 'Small Casts',
            'all': 'All Casts'
        }
        
        ax.set_title(f'{type_labels[cast_type]}\n(n={len(larva_probabilities[cast_type]["upstream"])} larvae)', 
                   fontsize=10)
        
        # Add mean probabilities as text
        upstream_mean = mean_probabilities[cast_type]['upstream']
        downstream_mean = mean_probabilities[cast_type]['downstream']
        
        ax.text(0, upstream_mean + 0.03, f"Mean: {upstream_mean:.2f}", 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(1, downstream_mean + 0.03, f"Mean: {downstream_mean:.2f}", 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add super title
    fig.suptitle(f'Cast Direction When Perpendicular to Flow (±{angle_width}°)', 
              fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)  # Make room for the suptitle
    plt.show()
    
    # 2. Second figure: per-larva bar plots for ALL cast types
    # Create a 1x3 subplot layout (or fewer if some types have no data)
    if len(plot_types) > 0:
        fig, axes = plt.subplots(len(plot_types), 1, figsize=(8, 3 * len(plot_types)))
        
        # Handle case where only one subplot is created
        if len(plot_types) == 1:
            axes = [axes]
        
        # Process each cast type
        for i, cast_type in enumerate(plot_types):
            ax = axes[i]
            
            # Find number of larvae with data
            n_larvae_with_data = len(per_larva_counts[cast_type]['larva_ids'])
            
            if n_larvae_with_data > 0:
                # Calculate downstream probability for each larva and store with other data
                larva_data = []
                for j in range(n_larvae_with_data):
                    up = per_larva_counts[cast_type]['upstream'][j]
                    down = per_larva_counts[cast_type]['downstream'][j]
                    total = per_larva_counts[cast_type]['total'][j]
                    down_prob = down / total
                    
                    larva_data.append({
                        'id': per_larva_counts[cast_type]['larva_ids'][j],
                        'upstream': up,
                        'downstream': down,
                        'total': total,
                        'downstream_prob': down_prob
                    })
                
                # Sort by downstream probability (ascending)
                sorted_larvae = sorted(larva_data, key=lambda x: x['downstream_prob'])
                
                # Create normalized stacked bar chart
                larva_ids = [larva['id'] for larva in sorted_larvae]
                upstream_normalized = [larva['upstream']/larva['total'] for larva in sorted_larvae]
                downstream_normalized = [larva['downstream']/larva['total'] for larva in sorted_larvae]
                
                # Create positions
                y_pos = np.arange(len(larva_ids))
                
                # Create normalized stacked bar chart
                ax.barh(y_pos, upstream_normalized, color='green', alpha=0.7, label='Upstream')
                ax.barh(y_pos, downstream_normalized, left=upstream_normalized, color='orange', alpha=0.7, label='Downstream')
                
                # Add downstream probability as text
                for j, larva in enumerate(sorted_larvae):
                    ax.text(1.02, j, f"{larva['downstream_prob']*100:.0f}%", va='center', fontsize=8)
                    
                    # Add total count at the start of each bar
                    ax.text(-0.05, j, f"{larva['total']}", va='center', ha='right', fontsize=8)
                
                # Mark 50% line
                ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
                
                # Format plot
                ax.set_yticks(y_pos)
                ax.set_yticklabels(larva_ids)
                ax.set_xlim(0, 1.15)  # Make room for percentage labels
                
                # Add labels
                if i == len(plot_types) - 1:
                    ax.set_xlabel('Proportion of Casts')
                
                ax.set_ylabel('Larva ID')
                
                # Add type label as title
                type_labels = {
                    'large': 'Large Casts',
                    'small': 'Small Casts',
                    'all': 'All Casts'
                }
                
                # Add type label title with annotations for number of larvae and casts
                total_casts = sum(larva['total'] for larva in sorted_larvae)
                ax.set_title(f'{type_labels[cast_type]} - Per Larva Breakdown (n={n_larvae_with_data} larvae, {total_casts} casts)', 
                           fontsize=10)
                
                # Add legend if this is the first subplot
                if i == 0:
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
                             ncol=2, frameon=False)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3)
        plt.show()
    
    # Return comprehensive results
    return {
        'total_counts': total_counts,
        'probabilities': probabilities,
        'mean_probabilities': mean_probabilities,
        'sem_probabilities': sem_probabilities,
        'larva_probabilities': larva_probabilities,
        'per_larva_counts': per_larva_counts,
        'stats_results': stats_results,
        'larvae_processed': larvae_processed,
        'total_larvae': n_larvae,
        'angle_width': angle_width
    }


def analyze_cast_head_dynamics(trx_data, larva_id=None, max_events=10):
    """
    Analyze head angle dynamics during casting events.
    
    Args:
        trx_data (dict): Tracking data dictionary
        larva_id (str, optional): ID of specific larva to analyze, if None, selects a random larva
        max_events (int): Maximum number of cast events to plot
    
    Returns:
        dict: Statistics about cast events and head angle dynamics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import random
    
    # Helper function to extract cast events from a single larva
    def extract_cast_events(larva_data):
        # Get behavioral states
        states = np.array(larva_data['global_state_large_state']).flatten()
        
        # Find cast bout starts and ends
        cast_starts = np.where((states[1:] == 2) & (states[:-1] != 2))[0] + 1
        cast_ends = []
        
        for start in cast_starts:
            end_idx = np.where(states[start:] != 2)[0]
            if len(end_idx) > 0:
                cast_ends.append(start + end_idx[0])
            else:
                cast_ends.append(len(states) - 1)  # End of recording
        
        # Ensure we have the same number of starts and ends
        n_events = min(len(cast_starts), len(cast_ends))
        cast_starts = cast_starts[:n_events]
        cast_ends = cast_ends[:n_events]
        
        # Ensure each cast has a valid duration
        valid_casts = []
        for i in range(n_events):
            if cast_ends[i] - cast_starts[i] >= 3:  # At least 3 frames
                valid_casts.append((cast_starts[i], cast_ends[i]))
        
        return valid_casts
    
    # Helper function to calculate angle between vectors
    def calculate_angle_between_vectors(v1, v2):
        """Calculate angle between two vectors in degrees"""
        dot_product = np.sum(v1 * v2, axis=1)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        
        # Avoid division by zero
        valid_mask = (norm_v1 > 0) & (norm_v2 > 0)
        if not np.any(valid_mask):
            return np.zeros(len(v1))
            
        cos_angle = np.zeros(len(v1))
        cos_angle[valid_mask] = dot_product[valid_mask] / (norm_v1[valid_mask] * norm_v2[valid_mask])
        
        # Clip to handle floating point errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        
        return angles
    
    # Determine which larva to analyze
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
    
    if larva_id is None:
        # Select a random larva with good data
        larvae_with_casts = []
        for lid, larva_data in data_to_process.items():
            try:
                # Extract larva data if nested
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                
                # Check if larva has spine data and state data
                if ('x_spine' in larva_data and 'y_spine' in larva_data and
                    'x_center' in larva_data and 'y_center' in larva_data and
                    'global_state_large_state' in larva_data):
                    events = extract_cast_events(larva_data)
                    if len(events) > 0:
                        larvae_with_casts.append(lid)
            except Exception as e:
                print(f"Error checking larva {lid}: {str(e)}")
                continue
        
        if not larvae_with_casts:
            raise ValueError("No larvae with valid cast and angle data found")
        
        larva_id = random.choice(larvae_with_casts)
        print(f"Selected random larva: {larva_id}")
    
    # Get data for the selected larva
    larva_data = data_to_process[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract data arrays
    try:
        x_spine = np.array(larva_data['x_spine'])
        y_spine = np.array(larva_data['y_spine'])
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        time = np.array(larva_data['t']).flatten()
        states = np.array(larva_data['global_state_large_state']).flatten()
        angle_downer_upper = np.array(larva_data['angle_downer_upper_smooth_5']).flatten()
    except KeyError as e:
        raise KeyError(f"Missing required data field: {str(e)}")
    
    # Convert angle_downer_upper to degrees and reverse direction
    # (assuming it's in radians, multiply by -1 to reverse)
    angle_downer_upper_deg = -1 * np.degrees(angle_downer_upper)
    
    # Handle different spine data shapes
    if x_spine.ndim == 1:  # 1D array
        x_tail = np.copy(x_spine)
        y_tail = np.copy(y_spine)
        x_head = np.copy(x_spine)
        y_head = np.copy(y_spine)
    else:  # 2D array with shape (spine_points, frames)
        x_tail = x_spine[-1, :] if len(x_spine) > 1 else x_spine.flatten()
        y_tail = y_spine[-1, :] if len(y_spine) > 1 else y_spine.flatten()
        x_head = x_spine[0, :] if len(x_spine) > 1 else x_spine.flatten()
        y_head = y_spine[0, :] if len(y_spine) > 1 else y_spine.flatten()
    
    # Calculate vectors
    tail_to_center = np.column_stack([
        x_center - x_tail,
        y_center - y_tail
    ])
    
    center_to_head = np.column_stack([
        x_head - x_center,
        y_head - y_center
    ])
    
    # Calculate angle between tail-to-center and center-to-head vectors
    bend_angle = calculate_angle_between_vectors(tail_to_center, center_to_head)
    
    # Extract cast events
    cast_events = extract_cast_events(larva_data)
    
    # Limit number of events to plot
    if len(cast_events) > max_events:
        # Select a diverse sample of cast events (short and long)
        durations = [end - start for start, end in cast_events]
        # Sort events by duration
        sorted_indices = np.argsort(durations)
        # Select events spread across duration range
        indices_to_use = sorted_indices[::len(sorted_indices)//max_events][:max_events]
        plot_events = [cast_events[i] for i in indices_to_use]
    else:
        plot_events = cast_events
    
    # Create figure
    n_events = len(plot_events)
    if n_events == 0:
        print("No cast events found for this larva")
        return None
    
    fig = plt.figure(figsize=(12, 2 * n_events))
    gs = GridSpec(n_events, 2, figure=fig)
    
    # Plot each cast event
    for i, (start, end) in enumerate(plot_events):
        # Ensure we have a complete window
        window_start = max(0, start - 5)  # 5 frames before cast
        window_end = min(len(time), end + 5)  # 5 frames after cast
        
        # Extract data for this window
        t_window = time[window_start:window_end] - time[window_start]
        angle_du_window = angle_downer_upper_deg[window_start:window_end]
        bend_angle_window = bend_angle[window_start:window_end]
        states_window = states[window_start:window_end]
        
        # Create cast mask for highlighting
        cast_mask = states_window == 2
        
        # Plot angle_downer_upper (converted to degrees and reversed)
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t_window, angle_du_window, 'b-')
        
        # Highlight cast period
        cast_times = t_window[cast_mask]
        if len(cast_times) > 0:
            ax1.axvspan(cast_times[0], cast_times[-1], alpha=0.2, color='red')
        
        ax1.set_ylabel('Angle downer-upper (°)\n(reversed)')
        if i == n_events - 1:
            ax1.set_xlabel('Time (s)')
        else:
            ax1.set_xticklabels([])
        
        # Plot bend angle between tail-to-center and center-to-head
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(t_window, bend_angle_window, 'g-')
        
        # Highlight cast period
        if len(cast_times) > 0:
            ax2.axvspan(cast_times[0], cast_times[-1], alpha=0.2, color='red')
        
        ax2.set_ylabel('Bend angle (°)\n(tail-center-head)')
        if i == n_events - 1:
            ax2.set_xlabel('Time (s)')
        else:
            ax2.set_xticklabels([])
        
        # Add event details as title
        duration_sec = (time[end] - time[start])
        ax1.set_title(f'Event {i+1}: Duration = {duration_sec:.2f}s')
    
    plt.tight_layout()
    plt.suptitle(f"Cast Angle Dynamics - Larva {larva_id}", y=1.02, fontsize=14)
    plt.show()
    
    # Compute statistics
    all_durations = [(time[end] - time[start]) for start, end in cast_events]
    max_angle_du = []
    max_bend_angle = []
    angle_amplitude_du = []  # Range of angles during cast
    bend_amplitude = []
    
    for start, end in cast_events:
        du_during_cast = angle_downer_upper_deg[start:end]
        bend_during_cast = bend_angle[start:end]
        
        if len(du_during_cast) > 0 and len(bend_during_cast) > 0:
            max_angle_du.append(np.max(np.abs(du_during_cast)))
            max_bend_angle.append(np.max(bend_during_cast))
            
            # Calculate range/amplitude of angles during cast
            angle_amplitude_du.append(np.max(du_during_cast) - np.min(du_during_cast))
            bend_amplitude.append(np.max(bend_during_cast) - np.min(bend_during_cast))
    
    # Return statistics
    return {
        'larva_id': larva_id,
        'n_events': len(cast_events),
        'event_durations': all_durations,
        'mean_duration': np.mean(all_durations) if all_durations else None,
        'max_angle_du': max_angle_du,
        'max_bend_angle': max_bend_angle,
        'mean_max_angle_du': np.mean(max_angle_du) if max_angle_du else None,
        'mean_max_bend_angle': np.mean(max_bend_angle) if max_bend_angle else None,
        'mean_angle_amplitude_du': np.mean(angle_amplitude_du) if angle_amplitude_du else None,
        'mean_bend_amplitude': np.mean(bend_amplitude) if bend_amplitude else None
    }


def plot_larva_angle_dynamics(trx_data, larva_id=None, smooth_window=5, highlight_peaks=True, peak_prominence=None, peak_distance=20):
    """
    Plot complete angle dynamics over time for a larva with behavior states highlighted,
    including several angle calculations:
    1. Neck angle: signed angle between tail-to-neck and neck-to-head vectors
       (positive = bend to the right, negative = bend to the left)
    2. Center angle: signed angle between tail-to-center and center-to-head vectors
       (positive = bend to the right, negative = bend to the left)
    3. Orientation angle: angle between tail-to-center vector and negative x-axis (0-360°)
       0° = aligned with negative x-axis, 180° = aligned with positive x-axis
       +90° = pointing right (y-axis), -90° = pointing left (negative y-axis)
    4. Upper-lower angle: pre-calculated angle between upper and lower body segments
    
    Args:
        trx_data (dict): Tracking data dictionary
        larva_id (str, optional): ID of specific larva to analyze, if None, selects a random larva
        smooth_window (int, optional): Window size for smoothing angles, defaults to 5
        highlight_peaks (bool, optional): Whether to highlight peaks in the upper-lower angle during cast events
        peak_prominence (float, optional): Minimum prominence for peak detection, if None uses auto-detection
        peak_distance (int, optional): Minimum distance between peaks in frames, defaults to 20
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from matplotlib.patches import Patch
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    
    # Define behavior color scheme
    behavior_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Run/Crawl)
        2: [1.0, 0.0, 0.0],      # Red (for Cast/Bend)
        3: [0.0, 1.0, 0.0],      # Green (for Stop)
        4: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],      # Orange (for Backup)
        6: [0.5, 0.0, 0.5],      # Purple (for Roll)
        7: [0.7, 0.7, 0.7]       # Light gray (for Small Actions)
    }
    
    # Behavior labels for legend
    behavior_labels = {
        1: 'Run/Crawl',
        2: 'Cast/Bend',
        3: 'Stop',
        4: 'Hunch',
        5: 'Backup',
        6: 'Roll',
        7: 'Small Actions'
    }
    
    # Helper function to calculate signed angle between vectors
    def calculate_signed_angle(v1, v2):
        """
        Calculate signed angle between two vectors in degrees.
        Positive angle means v2 is to the right of v1, negative means to the left.
        """
        # Cross product for direction (sign)
        cross_products = np.cross(v1, v2)
        
        # Dot product for angle
        dot_products = np.sum(v1 * v2, axis=1)
        norms_v1 = np.linalg.norm(v1, axis=1)
        norms_v2 = np.linalg.norm(v2, axis=1)
        
        # Avoid division by zero
        valid_mask = (norms_v1 > 0) & (norms_v2 > 0)
        
        # Calculate unsigned angle
        cos_angles = np.zeros(len(v1))
        cos_angles[valid_mask] = dot_products[valid_mask] / (norms_v1[valid_mask] * norms_v2[valid_mask])
        
        # Clip to handle floating point errors
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angles))
        
        # Apply sign based on cross product
        # If cross product is negative, angle is negative (bend to the left)
        # If cross product is positive, angle is positive (bend to the right)
        signed_angles = angles * np.sign(cross_products)
        
        return signed_angles
    
    # Helper function to calculate orientation angle (relative to negative x-axis)
    def calculate_orientation_angle(v):
        """
        Calculate angle between vector and negative x-axis in degrees.
        0° = aligned with negative x-axis, 180° = aligned with positive x-axis
        +90° = pointing right (y-axis), -90° = pointing left (negative y-axis)
        """
        # Negative x-axis unit vector
        neg_x_axis = np.array([-1.0, 0.0])
        
        # Calculate angles for each vector
        orientation_angles = np.zeros(len(v))
        
        for i in range(len(v)):
            if np.linalg.norm(v[i]) == 0:
                # Handle zero vectors
                orientation_angles[i] = np.nan
                continue
                
            # Normalize the vector
            normalized_v = v[i] / np.linalg.norm(v[i])
            
            # Calculate angle with negative x-axis using dot product
            dot_product = np.dot(normalized_v, neg_x_axis)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            # Determine sign based on y-component
            # If y > 0, vector points up/right of the negative x-axis: positive angle
            # If y < 0, vector points down/left of the negative x-axis: negative angle
            if normalized_v[1] < 0:
                angle_deg = -angle_deg
                
            orientation_angles[i] = angle_deg
            
        return orientation_angles
    
    # Helper function to smooth data
    def smooth_data(data, window_size):
        """Apply Gaussian smoothing to data"""
        sigma = window_size / 3.0  # Approximately equivalent to moving average
        return gaussian_filter1d(data, sigma)
    
    # Helper function to find global peaks during specific behavior states
    def find_global_peaks_during_behavior(data, states, behavior_id, time, min_prominence=None, min_distance=20):
        """
        Find significant peaks (both positive and negative) during specified behavior state.
        For negative peaks, we invert the data, find peaks, then invert the results.
        """
        # Create a mask for the specified behavior
        behavior_mask = states == behavior_id
        
        # If no data points with this behavior, return empty arrays
        if not np.any(behavior_mask):
            return {
                "positive": {"times": np.array([]), "values": np.array([])},
                "negative": {"times": np.array([]), "values": np.array([])}
            }
            
        # Create a copy of data for masking
        masked_data = data.copy()
        
        # If min_prominence is not specified, auto-calculate based on data range during behavior
        if min_prominence is None:
            behavior_data = data[behavior_mask]
            if len(behavior_data) > 0:
                data_range = np.max(behavior_data) - np.min(behavior_data)
                min_prominence = data_range * 0.15  # 15% of range as default
            else:
                min_prominence = np.std(data) * 0.8  # Fallback
        
        # For negative peaks, we'll invert the data
        inverted_data = -1 * masked_data
        
        # Find positive peaks (only during behavior state)
        # First, mask out non-behavior segments with very low values
        pos_masked_data = masked_data.copy()
        pos_masked_data[~behavior_mask] = np.min(masked_data) - 100
        pos_peak_indices, pos_properties = find_peaks(
            pos_masked_data, 
            prominence=min_prominence, 
            distance=min_distance
        )
        
        # Find negative peaks (only during behavior state)
        # First, mask out non-behavior segments with very low values
        neg_masked_data = inverted_data.copy()
        neg_masked_data[~behavior_mask] = np.min(inverted_data) - 100
        neg_peak_indices, neg_properties = find_peaks(
            neg_masked_data, 
            prominence=min_prominence, 
            distance=min_distance
        )
        
        # Prepare results
        results = {
            "positive": {
                "times": time[pos_peak_indices],
                "values": data[pos_peak_indices],
                "prominence": pos_properties.get("prominences", []),
                "indices": pos_peak_indices
            },
            "negative": {
                "times": time[neg_peak_indices],
                "values": data[neg_peak_indices],
                "prominence": neg_properties.get("prominences", []),
                "indices": neg_peak_indices
            }
        }
        
        return results
    
    # Determine which data structure we're working with
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
    
    # Select a larva if not specified
    if larva_id is None:
        valid_larvae = []
        for lid, larva_data in data_to_process.items():
            try:
                # Extract larva data if nested
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                
                # Check if larva has required data
                if ('x_spine' in larva_data and 
                    'y_spine' in larva_data and
                    'x_center' in larva_data and
                    'y_center' in larva_data and
                    'x_neck' in larva_data and
                    'y_neck' in larva_data and
                    'x_neck_down' in larva_data and
                    'y_neck_down' in larva_data and
                    'x_neck_top' in larva_data and
                    'y_neck_top' in larva_data and
                    'angle_upper_lower_smooth_5' in larva_data and
                    'global_state_large_state' in larva_data):
                    valid_larvae.append(lid)
            except:
                continue
        
        if not valid_larvae:
            raise ValueError("No larvae with valid data found")
        
        # Select exactly one random larva
        larva_id = random.choice(valid_larvae)
        print(f"Selected random larva: {larva_id}")
    
    # Get data for the selected larva
    larva_data = data_to_process[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract data arrays
    try:
        # Coordinates for angle calculations
        x_spine = np.array(larva_data['x_spine'])
        y_spine = np.array(larva_data['y_spine'])
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_neck = np.array(larva_data['x_neck']).flatten()
        y_neck = np.array(larva_data['y_neck']).flatten()
        x_neck_down = np.array(larva_data['x_neck_down']).flatten()
        y_neck_down = np.array(larva_data['y_neck_down']).flatten()
        x_neck_top = np.array(larva_data['x_neck_top']).flatten()
        y_neck_top = np.array(larva_data['y_neck_top']).flatten()
        
        # Pre-calculated angles
        angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
        
        # Time and behavioral states
        time = np.array(larva_data['t']).flatten()
        states = np.array(larva_data['global_state_large_state']).flatten()
        
        # Check for sufficient data
        if len(time) < 10:  # Require at least 10 frames
            raise ValueError(f"Insufficient data for larva {larva_id}")
            
    except KeyError as e:
        raise KeyError(f"Missing required data field for larva {larva_id}: {str(e)}")
    
    # Handle different spine data shapes to get head and tail coordinates
    if x_spine.ndim == 1:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
        x_head = x_spine
        y_head = y_spine
    else:  # 2D array with shape (spine_points, frames)
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
        x_head = x_spine[0].flatten()
        y_head = y_spine[0].flatten()
    
    # 1. Calculate vectors for neck angle (signed angle between tail-to-neck and neck-to-head)
    tail_to_neck = np.column_stack([
        x_neck - x_tail,
        y_neck - y_tail
    ])
    
    neck_to_head = np.column_stack([
        x_head - x_neck,
        y_head - y_neck
    ])
    
    # Calculate signed neck angle
    angle_neck = calculate_signed_angle(tail_to_neck, neck_to_head)
    # Create smoothed version
    angle_neck_smooth = smooth_data(angle_neck, smooth_window)
    
    # 2. Calculate vectors for center angle (signed angle between tail-to-center and center-to-head)
    tail_to_center = np.column_stack([
        x_center - x_tail,
        y_center - y_tail
    ])
    
    center_to_head = np.column_stack([
        x_head - x_center,
        y_head - y_center
    ])
    
    # Calculate signed center angle
    angle_center = calculate_signed_angle(tail_to_center, center_to_head)
    # Create smoothed version
    angle_center_smooth = smooth_data(angle_center, smooth_window)
    
    # 3. Calculate orientation angle (angle between tail-to-center vector and negative x-axis)
    # First, we need vectors from tail to center
    orientation_vectors = tail_to_center
    
    # Calculate orientation angles
    orientation_angle = calculate_orientation_angle(orientation_vectors)
    # Create smoothed version
    orientation_angle_smooth = smooth_data(orientation_angle, smooth_window)
    
    # 4. Convert other angles to degrees
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Ensure all arrays have the same length
    min_length = min(len(time), len(angle_neck), len(angle_center), 
                    len(orientation_angle), len(angle_upper_lower_deg), len(states))
    
    time = time[:min_length]
    angle_neck = angle_neck[:min_length]
    angle_neck_smooth = angle_neck_smooth[:min_length]
    angle_center = angle_center[:min_length]
    angle_center_smooth = angle_center_smooth[:min_length]
    orientation_angle = orientation_angle[:min_length]
    orientation_angle_smooth = orientation_angle_smooth[:min_length]
    angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
    states = states[:min_length]
    
    # Find global peaks in the upper-lower angle during cast/bend events (state 2)
    if highlight_peaks:
        # Find both positive and negative peaks only during cast/bend behavior (state 2)
        peaks_results = find_global_peaks_during_behavior(
            angle_upper_lower_deg, states, 2, time, 
            min_prominence=peak_prominence, 
            min_distance=peak_distance
        )
        
        pos_peak_times = peaks_results["positive"]["times"]
        pos_peak_values = peaks_results["positive"]["values"]
        neg_peak_times = peaks_results["negative"]["times"]
        neg_peak_values = peaks_results["negative"]["values"]
    
    # Create figure with 4 stacked plots sharing x-axis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Neck angle with both raw and smoothed data
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Reference line at y=0
    ax1.plot(time, angle_neck, 'k-', linewidth=0.8, alpha=0.5, label='Raw')
    ax1.plot(time, angle_neck_smooth, 'b-', linewidth=1.5, label=f'Smoothed (window={smooth_window})')
    ax1.set_ylabel('Neck angle (°)\n(tail-neck-head)', fontsize=12)
    ax1.set_title(f'Angle Dynamics for Larva {larva_id}', fontsize=14)
    ax1.legend(loc='upper right')
    
    # Plot 2: Center angle with both raw and smoothed data
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Reference line at y=0
    ax2.plot(time, angle_center, 'k-', linewidth=0.8, alpha=0.5, label='Raw')
    ax2.plot(time, angle_center_smooth, 'r-', linewidth=1.5, label=f'Smoothed (window={smooth_window})')
    ax2.set_ylabel('Center angle (°)\n(tail-center-head)', fontsize=12)
    ax2.legend(loc='upper right')
    
    # Plot 3: Orientation angle
    # Add reference lines for 0, 90, -90, and 180 degrees
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)     # 0 degrees (negative x-axis)
    ax3.axhline(y=90, color='gray', linestyle='--', alpha=0.3)   # 90 degrees (up)
    ax3.axhline(y=-90, color='gray', linestyle='--', alpha=0.3)  # -90 degrees (down)
    ax3.axhline(y=180, color='gray', linestyle='-.', alpha=0.3)  # 180 degrees (positive x-axis)
    ax3.axhline(y=-180, color='gray', linestyle='-.', alpha=0.3) # -180 degrees (also positive x-axis)
    
    ax3.plot(time, orientation_angle, 'k-', linewidth=0.8, alpha=0.5, label='Raw')
    ax3.plot(time, orientation_angle_smooth, 'g-', linewidth=1.5, label=f'Smoothed (window={smooth_window})')
    ax3.set_ylabel('Orientation angle (°)\n(relative to negative x-axis)', fontsize=12)
    ax3.set_ylim(-200, 200)  # Set y-limits to show full range of angles
    ax3.legend(loc='upper right')
    
    # Plot 4: Angle upper-lower
    ax4.plot(time, angle_upper_lower_deg, 'k-', linewidth=1.5)
    ax4.set_xlabel('Time (seconds)', fontsize=12)
    ax4.set_ylabel('Angle upper-lower (°)', fontsize=12)
    
    # Add peaks as stars on upper-lower angle plot if requested
    if highlight_peaks:
        # Plot positive peaks as upward-pointing stars
        if len(pos_peak_times) > 0:
            ax4.plot(pos_peak_times, pos_peak_values, '*', color='red', markersize=12, 
                    label=f'Right bend peaks (n={len(pos_peak_times)})')
        
        # Plot negative peaks as downward-pointing stars
        if len(neg_peak_times) > 0:
            ax4.plot(neg_peak_times, neg_peak_values, '*', color='blue', markersize=12, 
                    label=f'Left bend peaks (n={len(neg_peak_times)})')
        
        if len(pos_peak_times) > 0 or len(neg_peak_times) > 0:
            ax4.legend(loc='upper right')
    
    # Add behavioral state highlighting to all plots
    for i in range(1, 8):  # For each behavior state (1-7)
        # Find continuous segments of this behavior
        behavior_segments = []
        in_segment = False
        segment_start = 0
        
        for j in range(len(states)):
            if states[j] == i and not in_segment:
                in_segment = True
                segment_start = j
            elif states[j] != i and in_segment:
                in_segment = False
                behavior_segments.append((segment_start, j))
        
        # If we're still in a segment at the end
        if in_segment:
            behavior_segments.append((segment_start, len(states)-1))
        
        # Highlight each segment on all plots
        for start, end in behavior_segments:
            if end > start:  # Check for valid segment
                color = behavior_colors[i]
                alpha = 0.3
                
                # Highlight on all plots
                ax1.axvspan(time[start], time[end], color=color, alpha=alpha)
                ax2.axvspan(time[start], time[end], color=color, alpha=alpha)
                ax3.axvspan(time[start], time[end], color=color, alpha=alpha)
                ax4.axvspan(time[start], time[end], color=color, alpha=alpha)
    
    # Add explanatory annotations
    ax1.text(time[0], ax1.get_ylim()[1]*0.9, "Positive = bend to right\nNegative = bend to left", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    ax2.text(time[0], ax2.get_ylim()[1]*0.9, "Positive = bend to right\nNegative = bend to left", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    ax3.text(time[0], 150, "0° = neg. x-axis\n±180° = pos. x-axis\n+90° = right\n-90° = left", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add behavior legend
    legend_elements = [
        Patch(facecolor=behavior_colors[i], alpha=0.3, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in behavior_colors
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              title='Behaviors', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for the legend
    plt.show()
    
    print(f"Plotted angle dynamics for larva {larva_id} over {time[-1] - time[0]:.1f} seconds")
    if highlight_peaks:
        print(f"Detected {len(pos_peak_times)} right bend peaks and {len(neg_peak_times)} left bend peaks during cast/bend events")
    
    # Return calculated angles and detected peaks for further analysis
    return {
        "larva_id": larva_id,
        "time": time,
        "angle_neck": angle_neck,
        "angle_neck_smooth": angle_neck_smooth,
        "angle_center": angle_center,
        "angle_center_smooth": angle_center_smooth,
        "orientation_angle": orientation_angle,
        "orientation_angle_smooth": orientation_angle_smooth,
        "angle_upper_lower": angle_upper_lower_deg,
        "states": states,
        "peaks": {
            "positive": {
                "times": pos_peak_times,
                "values": pos_peak_values
            },
            "negative": {
                "times": neg_peak_times,
                "values": neg_peak_values
            }
        } if highlight_peaks else None
    }

def plot_larva_angle_vectors(trx_data, larva_id=None, smooth_window=5, time_window=50):
    """
    Plot a dynamic visualization of larva orientation and body angles:
    - Left panel: Polar plot showing two vectors:
      1. Orientation vector (tail-to-center, showing larva heading)
      2. Bend angle vector (showing upper-lower body angle)
    - Right panels: Time series plots of:
      1. Orientation angle over time (top)
      2. Upper-lower angle over time (bottom)
    
    Both plots include behavior state coloring and a time slider for animation.
    
    Args:
        trx_data (dict): Tracking data dictionary
        larva_id (str, optional): ID of specific larva to analyze, if None, selects a random larva
        smooth_window (int): Window size for angle smoothing (default: 5)
        time_window (int): Number of seconds to show in time plots (default: 50)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from matplotlib.patches import Patch, FancyArrow
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter1d
    from ipywidgets import Play, IntSlider, HBox, jslink
    from IPython.display import display
    
    # Define visualization parameters
    FIGURE_SIZE = (14, 7)
    LINE_WIDTH = 2
    ARROW_WIDTH = 0.03
    ARROW_HEAD_WIDTH = 0.15
    ARROW_HEAD_LENGTH = 0.2
    ALPHA = 0.9
    TIME_WINDOW = time_window
    
    # Define behavior color scheme
    behavior_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Run/Crawl)
        2: [1.0, 0.0, 0.0],      # Red (for Cast/Bend)
        3: [0.0, 1.0, 0.0],      # Green (for Stop)
        4: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],      # Orange (for Backup)
        6: [0.5, 0.0, 0.5],      # Purple (for Roll)
        7: [0.7, 0.7, 0.7]       # Light gray (for Small Actions)
    }
    
    # Behavior labels for legend
    behavior_labels = {
        1: 'Run/Crawl',
        2: 'Cast/Bend',
        3: 'Stop',
        4: 'Hunch',
        5: 'Backup',
        6: 'Roll',
        7: 'Small Actions'
    }
    
    # Vector colors (fixed)
    ORIENTATION_COLOR = 'blue'
    BEND_COLOR = 'red'
    
    # Helper function to calculate orientation angle (relative to negative x-axis)
    def calculate_orientation_angle(vector):
        """
        Calculate angle between vector and negative x-axis in degrees.
        0° = aligned with negative x-axis (downstream), 180° = aligned with positive x-axis (upstream)
        +90° = pointing right (y-axis), -90° = pointing left (negative y-axis)
        """
        # Negative x-axis unit vector
        neg_x_axis = np.array([-1.0, 0.0])
        
        # Calculate angle with negative x-axis
        if np.linalg.norm(vector) == 0:
            return np.nan
            
        # Normalize the vector
        normalized_v = vector / np.linalg.norm(vector)
        
        # Calculate angle with negative x-axis using dot product
        dot_product = np.dot(normalized_v, neg_x_axis)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Determine sign based on y-component
        if normalized_v[1] < 0:
            angle_deg = -angle_deg
            
        return angle_deg
    
    # Helper function to smooth data
    def smooth_data(data, window_size):
        """Apply Gaussian smoothing to data"""
        sigma = window_size / 3.0  # Approximately equivalent to moving average
        return gaussian_filter1d(data, sigma)
    
    # Determine which data structure we're working with
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
    
    # Select a larva if not specified
    if larva_id is None:
        valid_larvae = []
        for lid, larva_data in data_to_process.items():
            try:
                # Extract larva data if nested
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                
                # Check if larva has required data
                if ('x_spine' in larva_data and 
                    'y_spine' in larva_data and
                    'x_center' in larva_data and
                    'y_center' in larva_data and
                    'angle_upper_lower_smooth_5' in larva_data and
                    'global_state_large_state' in larva_data):
                    valid_larvae.append(lid)
            except:
                continue
        
        if not valid_larvae:
            raise ValueError("No larvae with valid data found")
        
        # Select exactly one random larva
        larva_id = random.choice(valid_larvae)
        print(f"Selected random larva: {larva_id}")
    
    # Get data for the selected larva
    larva_data = data_to_process[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract data arrays
    try:
        # Coordinates for angle calculations
        x_spine = np.array(larva_data['x_spine'])
        y_spine = np.array(larva_data['y_spine'])
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        
        # Pre-calculated angles and behavioral states
        angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
        time = np.array(larva_data['t']).flatten()
        states = np.array(larva_data['global_state_large_state']).flatten()
        
        # Check for sufficient data
        if len(time) < 10:  # Require at least 10 frames
            raise ValueError(f"Insufficient data for larva {larva_id}")
            
    except KeyError as e:
        raise KeyError(f"Missing required data field for larva {larva_id}: {str(e)}")
    
    # Handle different spine data shapes
    if x_spine.ndim == 1:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
        x_head = x_spine
        y_head = y_spine
    else:  # 2D array with shape (spine_points, frames)
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
        x_head = x_spine[0].flatten()
        y_head = y_spine[0].flatten()
    
    # Calculate tail-to-center vectors and orientation angles
    tail_to_center_vectors = []
    orientation_angles = []
    for i in range(len(x_center)):
        vector = np.array([x_center[i] - x_tail[i], y_center[i] - y_tail[i]])
        tail_to_center_vectors.append(vector)
        orientation_angles.append(calculate_orientation_angle(vector))
    
    # Convert to numpy arrays
    orientation_angles = np.array(orientation_angles)
    
    # Convert upper-lower angle to degrees
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Apply smoothing
    orientation_angles_smooth = smooth_data(orientation_angles, smooth_window)
    angle_upper_lower_deg_smooth = smooth_data(angle_upper_lower_deg, smooth_window)
    
    # Create figure and axes with the requested layout
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[:, 0], projection='polar')  # Polar plot (takes full height of left column)
    ax2 = fig.add_subplot(gs[0, 1])  # Orientation angle plot (top right)
    ax3 = fig.add_subplot(gs[1, 1])  # Upper-lower angle plot (bottom right)
    
    # Initialize arrows for the polar plot
    # We'll replace these in the update function
    orientation_arrow = None
    bend_arrow = None
    
    # Set up polar plot
    ax1.set_theta_zero_location('N')  # 0 degrees at the top (North)
    ax1.set_theta_direction(-1)  # Clockwise
    ax1.set_rlabel_position(45)  # Move radial labels away from plotted line
    ax1.set_rticks([0.5, 1.0])  # Less radial ticks
    ax1.set_rlim(0, 1.2)  # Set radius limit
    
    # Add cardinal direction labels with upstream/downstream
    ax1.set_xticklabels(['90°\n(Right)', '45°', '0°\n(Downstream)', '315°', '270°\n(Left)', '225°', '180°\n(Upstream)', '135°'])
    
    # Prepare the orientation angle plot (top right)
    ax2.plot(time, orientation_angles_smooth, 'b-', linewidth=1.5)
    ax2.set_ylabel('Orientation angle (°)')
    ax2.set_title('Larva Orientation Angle')
    
    # Prepare the upper-lower angle plot (bottom right)
    ax3.plot(time, angle_upper_lower_deg_smooth, 'r-', linewidth=1.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Angle upper-lower (°)')
    ax3.set_title('Body Bend Angle')
    
    # Add reference line at 0 degrees for orientation angle
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Downstream')
    ax2.axhline(y=180, color='gray', linestyle='--', alpha=0.3, label='Upstream')
    ax2.axhline(y=-180, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=90, color='gray', linestyle=':', alpha=0.3, label='Right')
    ax2.axhline(y=-90, color='gray', linestyle=':', alpha=0.3, label='Left')
    
    # Add behavior state shading to both angle plots
    for state_val in range(1, 8):
        segments = []
        in_segment = False
        segment_start = 0
        states_flat = states.flatten()
        
        for i, s in enumerate(states_flat):
            if s == state_val and not in_segment:
                in_segment = True
                segment_start = i
            elif s != state_val and in_segment:
                in_segment = False
                segments.append((segment_start, i))
                
        # Handle case when segment extends to end of data
        if in_segment:
            segments.append((segment_start, len(states_flat)-1))
            
        # Shade each segment on both plots
        for seg_start, seg_end in segments:
            if seg_start < seg_end:  # Only shade non-empty segments
                ax2.axvspan(time[seg_start], time[seg_end], 
                          color=behavior_colors[state_val], 
                          alpha=0.3)
                ax3.axvspan(time[seg_start], time[seg_end], 
                          color=behavior_colors[state_val], 
                          alpha=0.3)
    
    # Initialize time marker lines
    time_marker_top, = ax2.plot([0, 0], [ax2.get_ylim()[0], ax2.get_ylim()[1]], 
                               'r-', linewidth=2.0)
    time_marker_bottom, = ax3.plot([0, 0], [ax3.get_ylim()[0], ax3.get_ylim()[1]], 
                                  'r-', linewidth=2.0)
    
    def update(frame):
        nonlocal orientation_arrow, bend_arrow
        current_time = float(time[frame])
        
        # Get current angles
        orientation_angle = orientation_angles[frame]
        bend_angle = angle_upper_lower_deg[frame]
        
        # Convert angles to radians for polar plot
        orientation_rad = np.radians(orientation_angle)
        
        # Remove previous arrows if they exist
        if orientation_arrow:
            orientation_arrow.remove()
        if bend_arrow:
            bend_arrow.remove()
        
        # Create orientation arrow with fixed length of 1
        orientation_arrow = ax1.arrow(orientation_rad, 0, 0, 1.0,
                                     alpha=ALPHA, 
                                     width=ARROW_WIDTH,
                                     head_width=ARROW_HEAD_WIDTH, 
                                     head_length=ARROW_HEAD_LENGTH,
                                     color=ORIENTATION_COLOR,
                                     zorder=10)
        
        # Calculate bend angle direction
        # Adjust bend vector direction based on sign of bend angle
        if bend_angle >= 0:
            bend_direction = orientation_rad + np.radians(30)  # 30 degrees clockwise
        else:
            bend_direction = orientation_rad - np.radians(30)  # 30 degrees counterclockwise
        
        # Create bend angle arrow with fixed length of 1
        bend_arrow = ax1.arrow(bend_direction, 0, 0, 1.0,
                              alpha=ALPHA, 
                              width=ARROW_WIDTH,
                              head_width=ARROW_HEAD_WIDTH, 
                              head_length=ARROW_HEAD_LENGTH,
                              color=BEND_COLOR,
                              zorder=10)
        
        # Update time marker in angle plots
        time_marker_top.set_data([current_time, current_time], 
                                [ax2.get_ylim()[0], ax2.get_ylim()[1]])
        time_marker_bottom.set_data([current_time, current_time], 
                                   [ax3.get_ylim()[0], ax3.get_ylim()[1]])
        
        # Update visible window in angle plots - only move the window, don't redraw
        if time[-1] - time[0] > TIME_WINDOW:
            time_start = max(time[0], current_time - TIME_WINDOW/2)
            time_end = min(time[-1], current_time + TIME_WINDOW/2)
            ax2.set_xlim(time_start, time_end)
            ax3.set_xlim(time_start, time_end)
        else:
            # If recording is shorter than TIME_WINDOW, show the full range
            ax2.set_xlim(time[0], time[-1])
            ax3.set_xlim(time[0], time[-1])
        
        # Update polar plot title with current angles
        ax1.set_title(f"Orientation: {orientation_angle:.1f}°, Bend: {bend_angle:.1f}°\nTime: {current_time:.2f}s", 
                     fontsize=10)
        
        fig.canvas.draw_idle()
        return (orientation_arrow, bend_arrow, 
                time_marker_top, time_marker_bottom)
    
    # Set up interactive controls
    play = Play(value=0, min=0, max=len(time)-1, step=1, interval=50, description="Play")
    slider = IntSlider(min=0, max=len(time)-1, value=0, description="Frame:",
                      continuous_update=True, layout={'width': '1000px'})
    jslink((play, 'value'), (slider, 'value'))
    
    def on_value_change(change):
        if change['name'] == 'value':
            update(change['new'])
    slider.observe(on_value_change)
    
    # Create behavior state legend
    behavior_legend_elements = [
        Patch(facecolor=behavior_colors[i], alpha=0.6, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in behavior_colors
    ]
    
    # Add reference line legend
    reference_legend = [
        Line2D([0], [0], color='gray', linestyle='-', alpha=0.5, label='Downstream (0°)'),
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Upstream (180°)'),
        Line2D([0], [0], color='gray', linestyle=':', alpha=0.5, label='Left/Right (±90°)')
    ]
    
    # Add vector legend
    vector_legend = [
        Line2D([0], [0], color=ORIENTATION_COLOR, linewidth=LINE_WIDTH, alpha=ALPHA, label='Orientation Vector'),
        Line2D([0], [0], color=BEND_COLOR, linewidth=LINE_WIDTH, alpha=ALPHA, label='Bend Angle Vector')
    ]
    
    # Add behavior legend to bottom of figure
    fig.legend(handles=behavior_legend_elements, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 0), 
              ncol=4, 
              fontsize=8, 
              title='Behaviors')
    
    # Add vector legend to polar plot
    ax1.legend(handles=vector_legend, loc='lower left', fontsize=9)
    
    # Add reference line legend to orientation plot
    ax2.legend(handles=reference_legend, loc='upper right', fontsize=8)
    
    # Configure initial view for angle plots
    if len(time) > 0:
        time_start = time[0]
        time_end = min(time[-1], time[0] + TIME_WINDOW)
        ax2.set_xlim(time_start, time_end)
        ax3.set_xlim(time_start, time_end)
    
    # Set y-limits for orientation angle plot
    ax2.set_ylim(-200, 200)
    
    # Add label annotations to orientation angle plot to clarify directions
    ax2.text(time_start + 0.05*(time_end-time_start), 0, "Downstream", 
             verticalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    ax2.text(time_start + 0.05*(time_end-time_start), 180, "Upstream", 
             verticalalignment='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    # Display title
    fig.suptitle(f'Larva {larva_id} - Angle Vectors and Dynamics', fontsize=14)
    
    # Display interactive controls
    display(HBox([play, slider]))
    
    # Initialize with first frame
    update(0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the behavior legend
    plt.show()
    
    return {
        'larva_id': larva_id,
        'time': time,
        'orientation_angles': orientation_angles,
        'orientation_angles_smooth': orientation_angles_smooth,
        'angle_upper_lower_deg': angle_upper_lower_deg,
        'angle_upper_lower_deg_smooth': angle_upper_lower_deg_smooth,
        'states': states
    }

def plot_larva_integrated_visualization(trx_data, larva_id=None, smooth_window=5, time_window=50):
    """
    Create an integrated visualization of larva behavior with four components:
    1. Left-Top: Larva contour showing body parts and shape
    2. Left-Bottom: Polar plot showing orientation and bend angle vectors
    3. Right-Top: Global trajectory of the larva
    4. Right-Middle: Orientation angle over time 
    5. Right-Bottom: Head angle (upper-lower angle) over time
    
    All visualizations update together with interactive time controls.
    
    Args:
        trx_data (dict): Tracking data dictionary
        larva_id (str, optional): ID of specific larva to analyze, if None, selects a random larva
        smooth_window (int): Window size for angle smoothing (default: 5)
        time_window (int): Number of seconds to show in time plots (default: 50)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from matplotlib.patches import Patch, FancyArrow
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter1d
    from ipywidgets import Play, IntSlider, HBox, jslink
    from IPython.display import display
    
    # Define visualization parameters
    FIGURE_SIZE = (12, 10)  # Smaller figure size
    LINE_WIDTH = 2
    MARKER_SIZE = 8
    SPINE_MARKER_SIZE = 4
    WINDOW_SIZE = 2  # Size of zoom window for contour
    ARROW_WIDTH = 0.02  # Smaller arrow width
    ARROW_HEAD_WIDTH = 0.12
    ARROW_HEAD_LENGTH = 0.15
    ALPHA = 0.7
    TIME_WINDOW = time_window
    
    # Define behavior color scheme
    behavior_colors = {
        1: [0.0, 0.0, 0.0],      # Black (for Run/Crawl)
        2: [1.0, 0.0, 0.0],      # Red (for Cast/Bend)
        3: [0.0, 1.0, 0.0],      # Green (for Stop)
        4: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5: [1.0, 0.5, 0.0],      # Orange (for Backup)
        6: [0.5, 0.0, 0.5],      # Purple (for Roll)
        7: [0.7, 0.7, 0.7]       # Light gray (for Small Actions)
    }
    
    # Behavior labels for legend
    behavior_labels = {
        1: 'Run/Crawl',
        2: 'Cast/Bend',
        3: 'Stop',
        4: 'Hunch',
        5: 'Backup',
        6: 'Roll',
        7: 'Small Actions'
    }
    
    # Vector colors (fixed)
    ORIENTATION_COLOR = 'blue'
    BEND_COLOR = 'red'
    
    # Helper function to calculate orientation angle (relative to negative x-axis)
    def calculate_orientation_angle(vector):
        """
        Calculate angle between vector and negative x-axis in degrees.
        0° = aligned with negative x-axis (downstream), 180° = aligned with positive x-axis (upstream)
        +90° = pointing right (y-axis), -90° = pointing left (negative y-axis)
        """
        # Negative x-axis unit vector
        neg_x_axis = np.array([-1.0, 0.0])
        
        # Calculate angle with negative x-axis
        if np.linalg.norm(vector) == 0:
            return np.nan
            
        # Normalize the vector
        normalized_v = vector / np.linalg.norm(vector)
        
        # Calculate angle with negative x-axis using dot product
        dot_product = np.dot(normalized_v, neg_x_axis)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Determine sign based on y-component
        if normalized_v[1] < 0:
            angle_deg = -angle_deg
            
        return angle_deg
    
    # Helper function to smooth data
    def smooth_data(data, window_size):
        """Apply Gaussian smoothing to data"""
        sigma = window_size / 3.0  # Approximately equivalent to moving average
        return gaussian_filter1d(data, sigma)
    
    # Determine which data structure we're working with
    if 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data
    
    # Select a larva if not specified
    if larva_id is None:
        valid_larvae = []
        for lid, larva_data in data_to_process.items():
            try:
                # Extract larva data if nested
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                
                # Check if larva has required data
                if ('x_spine' in larva_data and 
                    'y_spine' in larva_data and
                    'x_center' in larva_data and
                    'y_center' in larva_data and
                    'x_contour' in larva_data and
                    'y_contour' in larva_data and
                    'angle_upper_lower_smooth_5' in larva_data and
                    'global_state_large_state' in larva_data):
                    valid_larvae.append(lid)
            except:
                continue
        
        if not valid_larvae:
            raise ValueError("No larvae with valid data found")
        
        # Select exactly one random larva
        larva_id = random.choice(valid_larvae)
        print(f"Selected random larva: {larva_id}")
    
    # Get data for the selected larva
    larva_data = data_to_process[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract data arrays
    try:
        # Coordinates for angle calculations
        x_spine = np.array(larva_data['x_spine'])
        y_spine = np.array(larva_data['y_spine'])
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_contour = np.atleast_2d(np.array(larva_data['x_contour']))
        y_contour = np.atleast_2d(np.array(larva_data['y_contour']))
        
        # Additional body part coordinates if available
        try:
            x_neck = np.array(larva_data['x_neck']).flatten()
            y_neck = np.array(larva_data['y_neck']).flatten()
            x_neck_down = np.array(larva_data['x_neck_down']).flatten()
            y_neck_down = np.array(larva_data['y_neck_down']).flatten()
            x_neck_top = np.array(larva_data['x_neck_top']).flatten()
            y_neck_top = np.array(larva_data['y_neck_top']).flatten()
            has_neck_data = True
        except KeyError:
            has_neck_data = False
        
        # Pre-calculated angles and behavioral states
        angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
        time = np.array(larva_data['t']).flatten()
        states = np.array(larva_data['global_state_large_state']).flatten()
        
        # Check for sufficient data
        if len(time) < 10:  # Require at least 10 frames
            raise ValueError(f"Insufficient data for larva {larva_id}")
            
    except KeyError as e:
        raise KeyError(f"Missing required data field for larva {larva_id}: {str(e)}")
    
    # Handle different spine data shapes
    if x_spine.ndim == 1:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
        x_head = x_spine
        y_head = y_spine
    else:  # 2D array with shape (spine_points, frames)
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
        x_head = x_spine[0].flatten()
        y_head = y_spine[0].flatten()
    
    # Calculate tail-to-center vectors and orientation angles
    tail_to_center_vectors = []
    orientation_angles = []
    for i in range(len(x_center)):
        vector = np.array([x_center[i] - x_tail[i], y_center[i] - y_tail[i]])
        tail_to_center_vectors.append(vector)
        orientation_angles.append(calculate_orientation_angle(vector))
    
    # Convert to numpy arrays
    orientation_angles = np.array(orientation_angles)
    
    # Convert upper-lower angle to degrees
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Apply smoothing
    orientation_angles_smooth = smooth_data(orientation_angles, smooth_window)
    angle_upper_lower_deg_smooth = smooth_data(angle_upper_lower_deg, smooth_window)
    
    # Create trajectory colors based on behavioral states
    trajectory_colors = []
    for state in states.flatten():
        try:
            state_val = int(state)
            trajectory_colors.append(behavior_colors.get(state_val, [1, 1, 1]))
        except:
            trajectory_colors.append([1, 1, 1])
    
    # Create figure and axes with the requested layout
    fig = plt.figure(figsize=FIGURE_SIZE)
    
    # Create a grid layout with equal size for contour and polar plots
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.15)
    
    # Create subplots
    ax_contour = fig.add_subplot(gs[0, 0])  # Left-Top: Larva contour
    ax_polar = fig.add_subplot(gs[1, 0], projection='polar')  # Left-Bottom: Polar plot (now same size as contour)
    ax_trajectory = fig.add_subplot(gs[0, 1:])  # Right-Top: Global trajectory (spans 2 columns)
    ax_orientation = fig.add_subplot(gs[1, 1])  # Bottom-Middle: Orientation angle plot
    ax_head_angle = fig.add_subplot(gs[1, 2])  # Bottom-Right: Head angle plot
    
    #----- Initialize contour plot -----
    contour_line, = ax_contour.plot([], [], 'k-', linewidth=LINE_WIDTH)
    contour_fill = ax_contour.fill([], [], color='gray', alpha=ALPHA)[0]
    
    # Initialize spine visualization
    spine_line, = ax_contour.plot([], [], '-', lw=LINE_WIDTH, alpha=ALPHA)
    spine_points = []
    
    # Create placeholder points for each spine point
    if x_spine.ndim > 1:
        num_spine_points = x_spine.shape[0]
    else:
        num_spine_points = 1
    
    for i in range(num_spine_points):
        point, = ax_contour.plot([], [], 'o', ms=SPINE_MARKER_SIZE)
        spine_points.append(point)
    
    # Initialize specific body part points
    head_point, = ax_contour.plot([], [], 'o', ms=MARKER_SIZE, label='Head')
    tail_point, = ax_contour.plot([], [], 's', ms=MARKER_SIZE, label='Tail')
    center_point, = ax_contour.plot([], [], '^', ms=MARKER_SIZE, label='Center')
    
    # Add neck points if available
    if has_neck_data:
        neck_point, = ax_contour.plot([], [], 'D', ms=MARKER_SIZE, label='Neck')
        neck_down_point, = ax_contour.plot([], [], 'X', ms=MARKER_SIZE, label='Neck Down')
        neck_top_point, = ax_contour.plot([], [], '*', ms=MARKER_SIZE, label='Neck Top')
    
    #----- Initialize polar plot -----
    # We'll create arrows in the update function
    orientation_arrow = None
    bend_arrow = None
    
    # Set up polar plot - make it more compact
    ax_polar.set_theta_zero_location('N')  # 0 degrees at the top (North)
    ax_polar.set_theta_direction(-1)  # Clockwise
    ax_polar.set_rlabel_position(45)  # Move radial labels away from plotted line
    ax_polar.set_rticks([0.5, 1.0])  # Less radial ticks
    ax_polar.set_rlim(0, 1.1)  # Set radius limit (slightly smaller)
    
    # Make tick labels smaller and more compact
    ax_polar.set_xticklabels(['90°\n(R)', '45°', '0°\n(D)', '315°', '270°\n(L)', '225°', '180°\n(U)', '135°'], 
                           fontsize=8)
    ax_polar.tick_params(axis='y', labelsize=8)
    
    # Set polar plot title to just "Vector Angles" - no dynamic content
    ax_polar.set_title('Vector Angles', fontsize=10)
    
    #----- Initialize trajectory plot -----
    # Plot the full global trajectory with segment colors based on behavior
    for i in range(len(x_center)-1):
        ax_trajectory.plot(x_center[i:i+2], y_center[i:i+2], 
                          color=trajectory_colors[i], 
                          linewidth=LINE_WIDTH, 
                          alpha=ALPHA)
    # Current position marker
    current_pos, = ax_trajectory.plot([], [], 'o', ms=MARKER_SIZE)
    
    #----- Initialize orientation angle plot -----
    ax_orientation.plot(time, orientation_angles_smooth, 'b-', linewidth=1.5)
    ax_orientation.set_ylabel('Orientation angle (°)', fontsize=9)
    ax_orientation.set_title('Orientation Angle', fontsize=10)
    
    # Add reference lines at 0, ±90, and 180 degrees
    ax_orientation.axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='Downstream')
    ax_orientation.axhline(y=180, color='gray', linestyle='--', alpha=0.3, label='Upstream')
    ax_orientation.axhline(y=-180, color='gray', linestyle='--', alpha=0.3)
    ax_orientation.axhline(y=90, color='gray', linestyle=':', alpha=0.3, label='Right')
    ax_orientation.axhline(y=-90, color='gray', linestyle=':', alpha=0.3, label='Left')
    
    #----- Initialize head angle plot -----
    ax_head_angle.plot(time, angle_upper_lower_deg_smooth, 'r-', linewidth=1.5)
    ax_head_angle.set_xlabel('Time (seconds)', fontsize=9)
    ax_head_angle.set_ylabel('Head angle (°)', fontsize=9)
    ax_head_angle.set_title('Head Bend Angle', fontsize=10)
    
    # Make tick labels smaller for all plots
    for ax in [ax_contour, ax_trajectory, ax_orientation, ax_head_angle]:
        ax.tick_params(axis='both', labelsize=8)
    
    #----- Add behavioral state shading to both angle plots -----
    for state_val in range(1, 8):
        segments = []
        in_segment = False
        segment_start = 0
        states_flat = states.flatten()
        
        for i, s in enumerate(states_flat):
            if s == state_val and not in_segment:
                in_segment = True
                segment_start = i
            elif s != state_val and in_segment:
                in_segment = False
                segments.append((segment_start, i))
                
        # Handle case when segment extends to end of data
        if in_segment:
            segments.append((segment_start, len(states_flat)-1))
            
        # Shade each segment on both angle plots
        for seg_start, seg_end in segments:
            if seg_start < seg_end:  # Only shade non-empty segments
                ax_orientation.axvspan(time[seg_start], time[seg_end], 
                                     color=behavior_colors[state_val], 
                                     alpha=0.3)
                ax_head_angle.axvspan(time[seg_start], time[seg_end], 
                                    color=behavior_colors[state_val], 
                                    alpha=0.3)
    
    #----- Initialize time marker lines -----
    time_marker_orientation, = ax_orientation.plot([0, 0], [ax_orientation.get_ylim()[0], ax_orientation.get_ylim()[1]], 
                                                 'r-', linewidth=2.0)
    time_marker_head, = ax_head_angle.plot([0, 0], [ax_head_angle.get_ylim()[0], ax_head_angle.get_ylim()[1]], 
                                         'r-', linewidth=2.0)
    
    def update(frame):
        nonlocal orientation_arrow, bend_arrow
        current_time = float(time[frame])
        
        #----- Update contour plot -----
        # Extract current frame data for contour
        x_frame = x_contour[:, frame]
        y_frame = y_contour[:, frame]

        # Get state and color
        try:
            state = int(states.flatten()[frame])
            color = behavior_colors.get(state, [1, 1, 1])
        except:
            color = [1, 1, 1]

        # Update contour shape and color
        contour_line.set_data(x_frame, y_frame)
        contour_line.set_color(color)
        contour_fill.set_xy(np.column_stack((x_frame, y_frame)))
        contour_fill.set_facecolor(color)
        
        # Extract spine point coordinates for this frame
        if x_spine.ndim > 1:  # 2D array (multiple spine points)
            spine_x = x_spine[:, frame]
            spine_y = y_spine[:, frame]
        else:  # 1D array (single spine point)
            spine_x = np.array([x_spine[frame]])
            spine_y = np.array([y_spine[frame]])
        
        # Update spine line and points
        spine_line.set_data(spine_x, spine_y)
        spine_line.set_color(color)
        
        # Update individual spine points
        for i, point in enumerate(spine_points):
            if i < len(spine_x):
                point.set_data([spine_x[i]], [spine_y[i]])
                point.set_color(color)
            else:
                point.set_data([], [])  # Hide extra points
        
        # Get body part coordinates
        head_x = spine_x[0] if len(spine_x) > 0 else None
        head_y = spine_y[0] if len(spine_y) > 0 else None
        tail_x = spine_x[-1] if len(spine_x) > 0 else None
        tail_y = spine_y[-1] if len(spine_y) > 0 else None
        center_x = x_center[frame]
        center_y = y_center[frame]
        
        # Update body part points
        head_point.set_data([head_x], [head_y])
        tail_point.set_data([tail_x], [tail_y])
        center_point.set_data([center_x], [center_y])
        
        # Update colors for contour points
        head_point.set_color(color)
        tail_point.set_color(color)
        center_point.set_color(color)
        
        # Update neck-related points if available
        if has_neck_data:
            neck_x = x_neck[frame]
            neck_y = y_neck[frame]
            neck_down_x = x_neck_down[frame]
            neck_down_y = y_neck_down[frame]
            neck_top_x = x_neck_top[frame]
            neck_top_y = y_neck_top[frame]
            
            neck_point.set_data([neck_x], [neck_y])
            neck_down_point.set_data([neck_down_x], [neck_down_y])
            neck_top_point.set_data([neck_top_x], [neck_top_y])
            
            neck_point.set_color(color)
            neck_down_point.set_color(color)
            neck_top_point.set_color(color)
        
        # Update contour zoom window
        ax_contour.set_xlim(center_x - WINDOW_SIZE, center_x + WINDOW_SIZE)
        ax_contour.set_ylim(center_y - WINDOW_SIZE, center_y + WINDOW_SIZE)
        ax_contour.set_title(f"Larva Contour (t={current_time:.2f}s)", fontsize=10)
        
        #----- Update polar plot -----
        # Get current angles
        orientation_angle = orientation_angles[frame]
        bend_angle = angle_upper_lower_deg[frame]
        
        # Convert orientation angle to radians for polar plot
        orientation_rad = np.radians(orientation_angle)
        
        # Remove previous arrows if they exist
        if orientation_arrow:
            orientation_arrow.remove()
        if bend_arrow:
            bend_arrow.remove()
        
        # Create orientation arrow with fixed length of 1
        orientation_arrow = ax_polar.arrow(orientation_rad, 0, 0, 1.0,
                                         alpha=ALPHA, 
                                         width=ARROW_WIDTH,
                                         head_width=ARROW_HEAD_WIDTH, 
                                         head_length=ARROW_HEAD_LENGTH,
                                         color=ORIENTATION_COLOR,
                                         zorder=10)
        
        # Calculate bend angle direction based on sign
        if bend_angle >= 0:
            bend_direction = orientation_rad + np.radians(30)  # 30 degrees clockwise
        else:
            bend_direction = orientation_rad - np.radians(30)  # 30 degrees counterclockwise
        
        # Create bend angle arrow with fixed length of 1
        bend_arrow = ax_polar.arrow(bend_direction, 0, 0, 1.0,
                                  alpha=ALPHA, 
                                  width=ARROW_WIDTH,
                                  head_width=ARROW_HEAD_WIDTH, 
                                  head_length=ARROW_HEAD_LENGTH,
                                  color=BEND_COLOR,
                                  zorder=10)
        
        # No dynamic title on polar plot - removed as requested
        
        #----- Update trajectory plot -----
        current_pos.set_data([center_x], [center_y])
        current_pos.set_color(color)
        
        #----- Update time markers in angle plots -----
        time_marker_orientation.set_data([current_time, current_time], 
                                       [ax_orientation.get_ylim()[0], ax_orientation.get_ylim()[1]])
        time_marker_head.set_data([current_time, current_time], 
                                [ax_head_angle.get_ylim()[0], ax_head_angle.get_ylim()[1]])
        
        #----- Update visible window in angle plots -----
        if time[-1] - time[0] > TIME_WINDOW:
            time_start = max(time[0], current_time - TIME_WINDOW/2)
            time_end = min(time[-1], current_time + TIME_WINDOW/2)
            ax_orientation.set_xlim(time_start, time_end)
            ax_head_angle.set_xlim(time_start, time_end)
        else:
            # If recording is shorter than TIME_WINDOW, show the full range
            ax_orientation.set_xlim(time[0], time[-1])
            ax_head_angle.set_xlim(time[0], time[-1])
        
        fig.canvas.draw_idle()
        return (contour_line, contour_fill, spine_line, head_point, tail_point, center_point,
                orientation_arrow, bend_arrow, current_pos, time_marker_orientation, time_marker_head)
    
    #----- Set up interactive controls -----
    play = Play(value=0, min=0, max=len(time)-1, step=1, interval=50, description="Play")
    slider = IntSlider(min=0, max=len(time)-1, value=0, description="Frame:",
                      continuous_update=True, layout={'width': '800px'})  # Slightly smaller slider
    jslink((play, 'value'), (slider, 'value'))
    
    def on_value_change(change):
        if change['name'] == 'value':
            update(change['new'])
    slider.observe(on_value_change)
    
    #----- Configure axes -----
    # Contour plot
    ax_contour.set_aspect("equal")
    ax_contour.set_xlabel("X position", fontsize=9)
    ax_contour.set_ylabel("Y position", fontsize=9)
    
    # Trajectory plot
    ax_trajectory.set_aspect("equal")
    ax_trajectory.set_xlabel("X position", fontsize=9)
    ax_trajectory.set_ylabel("Y position", fontsize=9)
    ax_trajectory.set_title("Global Trajectory", fontsize=10)
    
    # Orientation angle plot
    ax_orientation.set_ylim(-200, 200)
    
    # Reference label annotations for orientation angle plot - made smaller
    ax_orientation.text(time[0] + 0.01*(time[-1]-time[0]), 0, "Downstream", 
                       verticalalignment='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
    ax_orientation.text(time[0] + 0.01*(time[-1]-time[0]), 180, "Upstream", 
                       verticalalignment='center', fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
    
    # Configure initial view for angle plots
    if len(time) > 0:
        time_start = time[0]
        time_end = min(time[-1], time[0] + TIME_WINDOW)
        ax_orientation.set_xlim(time_start, time_end)
        ax_head_angle.set_xlim(time_start, time_end)
    
    #----- Create legend elements -----
    # Behavior state legend
    behavior_legend_elements = [
        Patch(facecolor=behavior_colors[i], alpha=0.6, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in behavior_colors
    ]
    
    # Reference line legend
    reference_legend = [
        Line2D([0], [0], color='gray', linestyle='-', alpha=0.5, label='Downstream (0°)'),
        Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Upstream (180°)'),
        Line2D([0], [0], color='gray', linestyle=':', alpha=0.5, label='Left/Right (±90°)')
    ]
    
    # Vector legend
    vector_legend = [
        Line2D([0], [0], color=ORIENTATION_COLOR, linewidth=LINE_WIDTH, alpha=ALPHA, label='Orientation'),
        Line2D([0], [0], color=BEND_COLOR, linewidth=LINE_WIDTH, alpha=ALPHA, label='Bend')
    ]
    
    #----- Add legends to plots -----
    # Add behavior legend to bottom of figure with smaller font
    fig.legend(handles=behavior_legend_elements, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 0), 
              ncol=7,  # Put all on one row
              fontsize=7, 
              title='Behaviors', 
              title_fontsize=8)
    
    # Add vector legend to polar plot with smaller font
    ax_polar.legend(handles=vector_legend, loc='lower left', fontsize=7)
    
    # Add reference line legend to orientation plot with smaller font
    ax_orientation.legend(handles=reference_legend, loc='upper right', fontsize=7)
    
    # Display title with smaller font
    fig.suptitle(f'Integrated Visualization - Larva {larva_id}', fontsize=12)
    
    # Display interactive controls
    display(HBox([play, slider]))
    
    # Initialize with first frame
    update(0)
    
    # Tighter layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.2, wspace=0.15)  # Tighter spacing
    plt.show()
    
    return {
        'larva_id': larva_id,
        'time': time,
        'orientation_angles': orientation_angles,
        'orientation_angles_smooth': orientation_angles_smooth,
        'angle_upper_lower_deg': angle_upper_lower_deg,
        'angle_upper_lower_deg_smooth': angle_upper_lower_deg_smooth,
        'states': states
    }


def analyze_cast_orientations_single(trx_data):
    """
    Analyze cast orientations for all larvae in a single experiment,
    separating large casts, small casts, and total casts.
    
    Args:
        trx_data: Dictionary containing larva tracking data
        
    Returns:
        dict: Contains orientation data for large casts, small casts, and total casts
    """
    import numpy as np
    
    # Initialize separate storage for large, small, and total casts
    large_cast_orientations = []
    small_cast_orientations = []
    all_cast_orientations = []  # Combined large and small
    
    # Process each larva
    for larva_id, larva in trx_data.items():
        try:
            # Get position data - handle both nested and flat structures
            if 'data' in larva:
                larva = larva['data']
            
            # Extract and validate arrays
            x_center = np.asarray(larva['x_center']).flatten()
            y_center = np.asarray(larva['y_center']).flatten()
            x_spine = np.asarray(larva['x_spine'])
            y_spine = np.asarray(larva['y_spine'])
            
            # Get global state data - use small_large_state if available
            if 'global_state_small_large_state' in larva:
                states = np.asarray(larva['global_state_small_large_state']).flatten()
                # Define masks for large casts (state = 2.0) and small casts (state = 1.5)
                large_cast_mask = states == 2.0
                small_cast_mask = states == 1.5
                all_cast_mask = large_cast_mask | small_cast_mask
            else:
                # Fall back to regular large_state if small_large_state isn't available
                states = np.asarray(larva['global_state_large_state']).flatten()
                # With just large state, only state 2 is cast
                large_cast_mask = states == 2
                small_cast_mask = np.zeros_like(states, dtype=bool)  # No small casts
                all_cast_mask = large_cast_mask
            
            # Get tail positions (last spine point)
            x_tail = x_spine[-1].flatten() if x_spine.ndim > 1 else x_spine.flatten()
            y_tail = y_spine[-1].flatten() if y_spine.ndim > 1 else y_spine.flatten()
            
            # Calculate tail-to-center vectors
            tail_to_center = np.column_stack([
                x_center - x_tail,
                y_center - y_tail
            ])
            
            # Calculate orientations in degrees
            orientations = np.degrees(np.arctan2(
                tail_to_center[:, 1],
                tail_to_center[:, 0]
            ))
            
            # Add orientations to respective lists
            large_cast_orientations.extend(orientations[large_cast_mask])
            small_cast_orientations.extend(orientations[small_cast_mask])
            all_cast_orientations.extend(orientations[all_cast_mask])
            
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    return {
        'large_cast_orientations': np.array(large_cast_orientations),
        'small_cast_orientations': np.array(small_cast_orientations),
        'all_cast_orientations': np.array(all_cast_orientations),
        'n_larvae': len(trx_data),
        'n_large_casts': len(large_cast_orientations),
        'n_small_casts': len(small_cast_orientations),
        'n_total_casts': len(all_cast_orientations)
    }

def analyze_cast_orientations_all(experiments_data):
    """
    Analyze cast orientations across all experiments,
    separating large casts, small casts, and total casts.
    
    Args:
        experiments_data: Dict containing all experiments data
        
    Returns:
        dict: Contains orientation data and statistics for large, small, and total casts
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Initialize separate storage for large, small, and total casts
    large_cast_orientations = []
    small_cast_orientations = []
    all_cast_orientations = []
    total_larvae = 0
    
    # Handle nested data structure
    if 'data' in experiments_data:
        data_to_process = experiments_data['data']
        total_larvae = experiments_data['metadata']['total_larvae']
    else:
        data_to_process = experiments_data
        total_larvae = len(data_to_process)
    
    # Process the data
    if isinstance(data_to_process, dict):
        results = analyze_cast_orientations_single(data_to_process)
        large_cast_orientations.extend(results['large_cast_orientations'])
        small_cast_orientations.extend(results['small_cast_orientations'])
        all_cast_orientations.extend(results['all_cast_orientations'])
    else:
        for exp_data in data_to_process.values():
            results = analyze_cast_orientations_single(exp_data)
            large_cast_orientations.extend(results['large_cast_orientations'])
            small_cast_orientations.extend(results['small_cast_orientations'])
            all_cast_orientations.extend(results['all_cast_orientations'])
    
    # Convert to numpy arrays
    large_cast_orientations = np.array(large_cast_orientations)
    small_cast_orientations = np.array(small_cast_orientations)
    all_cast_orientations = np.array(all_cast_orientations)
    
    # Create histograms for each type
    bins = np.linspace(-180, 180, 37)  # 36 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate histograms
    hist_large = np.histogram(large_cast_orientations, bins=bins, density=True)[0] if len(large_cast_orientations) > 0 else np.zeros(36)
    hist_small = np.histogram(small_cast_orientations, bins=bins, density=True)[0] if len(small_cast_orientations) > 0 else np.zeros(36)
    hist_all = np.histogram(all_cast_orientations, bins=bins, density=True)[0] if len(all_cast_orientations) > 0 else np.zeros(36)
    
    # Apply smoothing
    smoothed_large = gaussian_filter1d(hist_large, sigma=1)
    smoothed_small = gaussian_filter1d(hist_small, sigma=1)
    smoothed_all = gaussian_filter1d(hist_all, sigma=1)
    
    # Create figure with three subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Large casts
    ax1.plot(bin_centers, hist_large, 'k-', alpha=0.3, linewidth=1)
    ax1.plot(bin_centers, smoothed_large, 'r-', linewidth=2)
    ax1.set_xlabel('Body orientation (°)')
    ax1.set_ylabel('Relative probability')
    ax1.set_xlim(-180, 180)
    ax1.set_title(f'Large Casts (n={len(large_cast_orientations)})')
    
    # Plot 2: Small casts
    ax2.plot(bin_centers, hist_small, 'k-', alpha=0.3, linewidth=1)
    ax2.plot(bin_centers, smoothed_small, 'b-', linewidth=2)
    ax2.set_xlabel('Body orientation (°)')
    ax2.set_ylabel('Relative probability')
    ax2.set_xlim(-180, 180)
    ax2.set_title(f'Small Casts (n={len(small_cast_orientations)})')
    
    # Plot 3: All casts combined
    ax3.plot(bin_centers, hist_all, 'k-', alpha=0.3, linewidth=1)
    ax3.plot(bin_centers, smoothed_all, 'g-', linewidth=2)
    ax3.set_xlabel('Body orientation (°)')
    ax3.set_ylabel('Relative probability')
    ax3.set_xlim(-180, 180)
    ax3.set_title(f'All Casts Combined (n={len(all_cast_orientations)})')
    
    # Add reference lines for upstream/downstream orientation
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, label='Downstream')
        ax.axvline(x=180, color='gray', linestyle='--', alpha=0.3, label='Upstream')
        ax.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    
    # Add super title
    plt.suptitle(f'Cast Orientation Distributions (Total larvae: {total_larvae})', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create comparison plot - all distributions on one plot
    plt.figure(figsize=(6, 4))
    
    if len(large_cast_orientations) > 0:
        plt.plot(bin_centers, smoothed_large, 'r-', linewidth=2, label='Large casts')
    
    if len(small_cast_orientations) > 0:
        plt.plot(bin_centers, smoothed_small, 'b-', linewidth=2, label='Small casts')
    
    if len(all_cast_orientations) > 0:
        plt.plot(bin_centers, smoothed_all, 'g-', linewidth=2, label='All casts')
    
    plt.xlabel('Body orientation (°)')
    plt.ylabel('Relative probability')
    plt.xlim(-180, 180)
    plt.title(f'Cast Orientation Comparison (n={total_larvae} larvae)')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'large_cast_orientations': large_cast_orientations,
        'small_cast_orientations': small_cast_orientations,
        'all_cast_orientations': all_cast_orientations,
        'hist_large': hist_large,
        'hist_small': hist_small,
        'hist_all': hist_all,
        'bin_centers': bin_centers,
        'smoothed_large': smoothed_large,
        'smoothed_small': smoothed_small,
        'smoothed_all': smoothed_all,
        'n_larvae': total_larvae,
        'n_large_casts': len(large_cast_orientations),
        'n_small_casts': len(small_cast_orientations),
        'n_total_casts': len(all_cast_orientations)
    }

def analyze_run_rate_by_orientation(trx_data, larva_id=None, bin_width=10):
    """
    Calculate run rate (probability) as a function of orientation for large runs, small runs, and all runs.
    
    Args:
        trx_data: Dictionary containing tracking data
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae
        bin_width: Width of orientation bins in degrees
        
    Returns:
        dict: Contains run rates and orientation bins for large, small, and all runs
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    
    def get_orientations_and_states(larva_data):
        """Extract orientations and run states for large and small runs."""
        # Calculate orientation
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        # CHECK NEGATIVE AXIS IS ZERO
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
        
        # Get run states
        # Check if small_large_state is available
        if 'global_state_small_large_state' in larva_data:
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            # Define masks for large runs (state = 1.0) and small runs (state = 0.5)
            is_large_run = states == 1.0
            is_small_run = states == 0.5
            is_any_run = is_large_run | is_small_run
        else:
            # Fall back to regular large_state if small_large_state isn't available
            states = np.array(larva_data['global_state_large_state']).flatten()
            # With just large state, only state 1 is run
            is_large_run = states == 1
            is_small_run = np.zeros_like(states, dtype=bool)  # No small runs
            is_any_run = is_large_run
        
        return orientations, is_large_run, is_small_run, is_any_run
    
    # Initialize storage for large, small, and all runs
    all_orientations = []
    large_run_states = []
    small_run_states = []
    all_run_states = []
    
    # Process data
    if larva_id is not None:
        # Single larva analysis
        larva_data = trx_data[larva_id]
        if 'data' in larva_data:
            larva_data = larva_data['data']
        orientations, is_large, is_small, is_any = get_orientations_and_states(larva_data)
        all_orientations.extend(orientations)
        large_run_states.extend(is_large)
        small_run_states.extend(is_small)
        all_run_states.extend(is_any)
        n_larvae = 1
        title = f'Larva {larva_id} - Run Probability'
    else:
        # All larvae analysis
        if 'data' in trx_data:
            data_to_process = trx_data['data']
            n_larvae = trx_data['metadata']['total_larvae']
        else:
            data_to_process = trx_data
            n_larvae = len(data_to_process)
            
        for larva_id, larva_data in data_to_process.items():
            try:
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                orientations, is_large, is_small, is_any = get_orientations_and_states(larva_data)
                all_orientations.extend(orientations)
                large_run_states.extend(is_large)
                small_run_states.extend(is_small)
                all_run_states.extend(is_any)
            except Exception as e:
                print(f"Error processing larva {larva_id}: {str(e)}")
                continue
        title = f'Run Probability (n={n_larvae})'
    
    # Convert to numpy arrays
    all_orientations = np.array(all_orientations)
    large_run_states = np.array(large_run_states)
    small_run_states = np.array(small_run_states)
    all_run_states = np.array(all_run_states)
    
    # Create orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate run rates for each bin and run type
    large_run_rates = []
    small_run_rates = []
    all_run_rates = []
    
    for i in range(len(bins)-1):
        mask = (all_orientations >= bins[i]) & (all_orientations < bins[i+1])
        if np.sum(mask) > 0:
            large_run_rates.append(np.mean(large_run_states[mask]))
            small_run_rates.append(np.mean(small_run_states[mask]))
            all_run_rates.append(np.mean(all_run_states[mask]))
        else:
            large_run_rates.append(0)
            small_run_rates.append(0)
            all_run_rates.append(0)
    
    large_run_rates = np.array(large_run_rates)
    small_run_rates = np.array(small_run_rates)
    all_run_rates = np.array(all_run_rates)
    
    # Apply smoothing
    large_smoothed = gaussian_filter1d(large_run_rates, sigma=1)
    small_smoothed = gaussian_filter1d(small_run_rates, sigma=1)
    all_smoothed = gaussian_filter1d(all_run_rates, sigma=1)
    
    # Create figure with three subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Large runs
    ax1.plot(bin_centers, large_run_rates, 'k-', alpha=0.3, linewidth=1)
    ax1.plot(bin_centers, large_smoothed, 'r-', linewidth=2)
    ax1.set_xlabel('Orientation (°)')
    ax1.set_ylabel('Run probability')
    ax1.set_xlim(-180, 180)
    ax1.set_title('Large Runs')
    
    # Plot 2: Small runs
    ax2.plot(bin_centers, small_run_rates, 'k-', alpha=0.3, linewidth=1)
    ax2.plot(bin_centers, small_smoothed, 'b-', linewidth=2)
    ax2.set_xlabel('Orientation (°)')
    ax2.set_ylabel('Run probability')
    ax2.set_xlim(-180, 180)
    ax2.set_title('Small Runs')
    
    # Plot 3: All runs combined
    ax3.plot(bin_centers, all_run_rates, 'k-', alpha=0.3, linewidth=1)
    ax3.plot(bin_centers, all_smoothed, 'g-', linewidth=2)
    ax3.set_xlabel('Orientation (°)')
    ax3.set_ylabel('Run probability')
    ax3.set_xlim(-180, 180)
    ax3.set_title('All Runs Combined')
    
    # Add reference lines for upstream/downstream orientation
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, label='Downstream')
        ax.axvline(x=180, color='gray', linestyle='--', alpha=0.3, label='Upstream')
        ax.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    
    # Add super title
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create comparison plot - all distributions on one plot
    plt.figure(figsize=(6, 4))
    
    # Count non-zero values to check if we have data
    n_large_nonzero = np.count_nonzero(large_run_rates)
    n_small_nonzero = np.count_nonzero(small_run_rates)
    n_all_nonzero = np.count_nonzero(all_run_rates)
    
    if n_large_nonzero > 0:
        plt.plot(bin_centers, large_smoothed, 'r-', linewidth=2, label='Large runs')
    
    if n_small_nonzero > 0:
        plt.plot(bin_centers, small_smoothed, 'b-', linewidth=2, label='Small runs')
    
    if n_all_nonzero > 0:
        plt.plot(bin_centers, all_smoothed, 'g-', linewidth=2, label='All runs')
    
    plt.xlabel('Orientation (°)')
    plt.ylabel('Run probability')
    plt.xlim(-180, 180)
    plt.title(f'Run Probability Comparison (n={n_larvae} larvae)')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate counts and frequencies for statistical comparison
    large_run_count = np.sum(large_run_states)
    small_run_count = np.sum(small_run_states)
    all_run_count = np.sum(all_run_states)
    
    return {
        'orientations': all_orientations,
        'large_run_states': large_run_states,
        'small_run_states': small_run_states,
        'all_run_states': all_run_states,
        'bin_centers': bin_centers,
        'large_run_rates': large_run_rates,
        'small_run_rates': small_run_rates,
        'all_run_rates': all_run_rates,
        'large_smoothed': large_smoothed,
        'small_smoothed': small_smoothed,
        'all_smoothed': all_smoothed,
        'n_larvae': n_larvae,
        'large_run_count': int(large_run_count),
        'small_run_count': int(small_run_count),
        'all_run_count': int(all_run_count)
    }

def analyze_cast_head_angles_by_orientation(trx_data, larva_id=None, bin_width=10):
    """
    Calculate head-to-center angles during cast events as a function of orientation.
    
    This measures the actual angle between the head and center line during casts,
    not just the frame-to-frame changes.
    
    Args:
        trx_data: Dictionary containing tracking data
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae
        bin_width: Width of orientation bins in degrees
        
    Returns:
        dict: Contains head angle data for large casts, small casts, and all casts
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt

    def get_orientations_angles_and_states(larva_data):
        """Extract orientations, head angles, and cast states."""
        # Calculate body orientation using tail-to-center vector
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        
        # Get head position (first point in spine)
        x_head = np.array(larva_data['x_spine'])[0].flatten()
        y_head = np.array(larva_data['y_spine'])[0].flatten()
        
        # Body orientation vectors (tail to center)
        body_vectors = np.column_stack([x_center - x_tail, y_center - y_tail])
        
        # Head vectors (center to head)
        head_vectors = np.column_stack([x_head - x_center, y_head - y_center])
        
        # Calculate body orientations
        body_orientations = np.degrees(np.arctan2(body_vectors[:, 1], -body_vectors[:, 0]))
        
        # Calculate head angles (relative to body orientation)
        head_angles = np.zeros_like(body_orientations)
        
        for i in range(len(body_orientations)):
            # Get unit vectors
            if np.linalg.norm(body_vectors[i]) > 0 and np.linalg.norm(head_vectors[i]) > 0:
                body_unit = body_vectors[i] / np.linalg.norm(body_vectors[i])
                head_unit = head_vectors[i] / np.linalg.norm(head_vectors[i])
                
                # Calculate dot product and angle
                dot_product = np.clip(np.dot(body_unit, head_unit), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot_product))
                
                # Determine sign (left or right) using cross product
                cross_z = body_unit[0] * head_unit[1] - body_unit[1] * head_unit[0]
                if cross_z < 0:
                    angle = -angle
                    
                head_angles[i] = angle
            else:
                head_angles[i] = np.nan
        
        # Get cast states - handle both large and small casts
        if 'global_state_small_large_state' in larva_data:
            states = np.array(larva_data['global_state_small_large_state']).flatten()
            # Define masks for large casts (state = 2.0) and small casts (state = 1.5)
            is_large_cast = states == 2.0
            is_small_cast = states == 1.5
            is_any_cast = is_large_cast | is_small_cast
        else:
            # Fall back to regular large_state if small_large_state isn't available
            states = np.array(larva_data['global_state_large_state']).flatten()
            # With just large state, only state 2 is cast
            is_large_cast = states == 2
            is_small_cast = np.zeros_like(states, dtype=bool)  # No small casts
            is_any_cast = is_large_cast
            
        return body_orientations, head_angles, is_large_cast, is_small_cast, is_any_cast
    
    # Storage for orientation values and corresponding head angles
    large_cast_angles = []
    small_cast_angles = []
    all_cast_angles = []
    
    large_cast_orientations = []
    small_cast_orientations = []
    all_cast_orientations = []
    
    # Process data either for single larva or all larvae
    if larva_id is not None:
        # Single larva analysis
        larva_data = trx_data[larva_id]
        if 'data' in larva_data:
            larva_data = larva_data['data']
            
        orientations, head_angles, is_large_cast, is_small_cast, is_any_cast = get_orientations_angles_and_states(larva_data)
        
        # Store head angles during casts
        large_cast_angles.extend(head_angles[is_large_cast])
        large_cast_orientations.extend(orientations[is_large_cast])
        
        small_cast_angles.extend(head_angles[is_small_cast])
        small_cast_orientations.extend(orientations[is_small_cast])
        
        all_cast_angles.extend(head_angles[is_any_cast])
        all_cast_orientations.extend(orientations[is_any_cast])
        
        n_larvae = 1
        title = f'Larva {larva_id} - Cast Head Angles'
    else:
        # Multiple larvae analysis
        if 'data' in trx_data:
            data_to_process = trx_data['data']
            n_larvae = trx_data['metadata']['total_larvae']
        else:
            data_to_process = trx_data
            n_larvae = len(data_to_process)
        
        title = f'Cast Head Angles (n={n_larvae})'
        
        for larva_id, larva in data_to_process.items():
            try:
                if 'data' in larva:
                    larva = larva['data']
                
                orientations, head_angles, is_large_cast, is_small_cast, is_any_cast = get_orientations_angles_and_states(larva)
                
                # Store head angles during casts
                large_cast_angles.extend(head_angles[is_large_cast])
                large_cast_orientations.extend(orientations[is_large_cast])
                
                small_cast_angles.extend(head_angles[is_small_cast])
                small_cast_orientations.extend(orientations[is_small_cast])
                
                all_cast_angles.extend(head_angles[is_any_cast])
                all_cast_orientations.extend(orientations[is_any_cast])
            except Exception as e:
                print(f"Error processing larva {larva_id}: {str(e)}")
                continue
    
    # Convert to numpy arrays
    large_cast_angles = np.array(large_cast_angles)
    small_cast_angles = np.array(small_cast_angles)
    all_cast_angles = np.array(all_cast_angles)
    
    large_cast_orientations = np.array(large_cast_orientations)
    small_cast_orientations = np.array(small_cast_orientations)
    all_cast_orientations = np.array(all_cast_orientations)
    
    # Take absolute values of head angles (magnitudes only)
    large_cast_angles_abs = np.abs(large_cast_angles)
    small_cast_angles_abs = np.abs(small_cast_angles)
    all_cast_angles_abs = np.abs(all_cast_angles)
    
    # Create orientation bins from -180 to 180 degrees
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the head angles by body orientation
    large_mean_angles = []
    small_mean_angles = []
    all_mean_angles = []
    
    for i in range(len(bins)-1):
        # For large casts
        if len(large_cast_orientations) > 0:
            mask = (large_cast_orientations >= bins[i]) & (large_cast_orientations < bins[i+1])
            if np.sum(mask) > 0:
                large_mean_angles.append(np.mean(large_cast_angles_abs[mask]))
            else:
                large_mean_angles.append(np.nan)
        else:
            large_mean_angles.append(np.nan)
            
        # For small casts
        if len(small_cast_orientations) > 0:
            mask = (small_cast_orientations >= bins[i]) & (small_cast_orientations < bins[i+1])
            if np.sum(mask) > 0:
                small_mean_angles.append(np.mean(small_cast_angles_abs[mask]))
            else:
                small_mean_angles.append(np.nan)
        else:
            small_mean_angles.append(np.nan)
        
        # For all casts
        if len(all_cast_orientations) > 0:
            mask = (all_cast_orientations >= bins[i]) & (all_cast_orientations < bins[i+1])
            if np.sum(mask) > 0:
                all_mean_angles.append(np.mean(all_cast_angles_abs[mask]))
            else:
                all_mean_angles.append(np.nan)
        else:
            all_mean_angles.append(np.nan)
    
    # Convert to numpy arrays
    large_mean_angles = np.array(large_mean_angles)
    small_mean_angles = np.array(small_mean_angles)
    all_mean_angles = np.array(all_mean_angles)
    
    # Apply smoothing (handling NaNs)
    large_smooth_input = np.nan_to_num(large_mean_angles, nan=0)
    small_smooth_input = np.nan_to_num(small_mean_angles, nan=0)
    all_smooth_input = np.nan_to_num(all_mean_angles, nan=0)
    
    large_smoothed = gaussian_filter1d(large_smooth_input, sigma=1)
    small_smoothed = gaussian_filter1d(small_smooth_input, sigma=1)
    all_smoothed = gaussian_filter1d(all_smooth_input, sigma=1)
    
    # Create figure with three subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Large cast head angles
    ax1.plot(bin_centers, large_smoothed, 'r-', linewidth=2)
    ax1.set_xlabel('Body Orientation (°)')
    ax1.set_ylabel('Mean Head Angle (°)')
    ax1.set_xlim(-180, 180)
    ax1.set_title(f'Large Casts (n={len(large_cast_angles)})')
    
    # Plot 2: Small cast head angles
    ax2.plot(bin_centers, small_smoothed, 'b-', linewidth=2)
    ax2.set_xlabel('Body Orientation (°)')
    ax2.set_ylabel('Mean Head Angle (°)')
    ax2.set_xlim(-180, 180)
    ax2.set_title(f'Small Casts (n={len(small_cast_angles)})')
    
    # Plot 3: All cast head angles
    ax3.plot(bin_centers, all_smoothed, 'g-', linewidth=2)
    ax3.set_xlabel('Body Orientation (°)')
    ax3.set_ylabel('Mean Head Angle (°)')
    ax3.set_xlim(-180, 180)
    ax3.set_title(f'All Casts Combined (n={len(all_cast_angles)})')
    
    # Add super title
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create comparison plot - all curves on one plot
    plt.figure(figsize=(6, 4))
    
    # Check if we have data to plot
    has_large_data = len(large_cast_angles) > 0 and not np.all(np.isnan(large_mean_angles))
    has_small_data = len(small_cast_angles) > 0 and not np.all(np.isnan(small_mean_angles))
    has_all_data = len(all_cast_angles) > 0 and not np.all(np.isnan(all_mean_angles))
    
    if has_large_data:
        plt.plot(bin_centers, large_smoothed, 'r-', linewidth=2, label='Large casts')
    
    if has_small_data:
        plt.plot(bin_centers, small_smoothed, 'b-', linewidth=2, label='Small casts')
    
    if has_all_data:
        plt.plot(bin_centers, all_smoothed, 'g-', linewidth=2, label='All casts')
    
    plt.xlabel('Body Orientation (°)')
    plt.ylabel('Mean Head Angle (°)')
    plt.xlim(-180, 180)
    plt.title(f'Cast Head Angle Comparison (n={n_larvae} larvae)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot head angle distributions during casts
    plt.figure(figsize=(8, 4))
    
    # Create histogram bins
    angle_bins = np.linspace(-90, 90, 37)  # 36 bins covering -90° to 90°
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    # Plot histograms
    if len(large_cast_angles) > 0:
        hist_large = np.histogram(large_cast_angles, bins=angle_bins, density=True)[0]
        smoothed_hist_large = gaussian_filter1d(hist_large, sigma=1)
        plt.plot(angle_bin_centers, smoothed_hist_large, 'r-', linewidth=2, label='Large casts')
    
    if len(small_cast_angles) > 0:
        hist_small = np.histogram(small_cast_angles, bins=angle_bins, density=True)[0]
        smoothed_hist_small = gaussian_filter1d(hist_small, sigma=1)
        plt.plot(angle_bin_centers, smoothed_hist_small, 'b-', linewidth=2, label='Small casts')
    
    if len(all_cast_angles) > 0:
        hist_all = np.histogram(all_cast_angles, bins=angle_bins, density=True)[0]
        smoothed_hist_all = gaussian_filter1d(hist_all, sigma=1)
        plt.plot(angle_bin_centers, smoothed_hist_all, 'g-', linewidth=2, label='All casts')
    
    plt.xlabel('Head Angle (°)')
    plt.ylabel('Probability Density')
    plt.title(f'Distribution of Head Angles During Casts (n={n_larvae} larvae)')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return the computed data and statistics
    return {
        'large_cast_orientations': large_cast_orientations,
        'small_cast_orientations': small_cast_orientations,
        'all_cast_orientations': all_cast_orientations,
        'large_cast_angles': large_cast_angles,
        'small_cast_angles': small_cast_angles,
        'all_cast_angles': all_cast_angles,
        'bin_centers': bin_centers,
        'large_mean_angles': large_mean_angles,
        'small_mean_angles': small_mean_angles,
        'all_mean_angles': all_mean_angles,
        'large_smoothed': large_smoothed,
        'small_smoothed': small_smoothed,
        'all_smoothed': all_smoothed,
        'n_larvae': n_larvae,
        'n_large_casts': len(large_cast_angles),
        'n_small_casts': len(small_cast_angles),
        'n_all_casts': len(all_cast_angles)
    }