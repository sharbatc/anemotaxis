import multiprocessing as mp
import os
from datetime import datetime
import h5py
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
plt.style.use('../anemotaxis.mplstyle')

def get_orientation_array(larva_data):
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
        return compute_orientation_tail_to_neck(
            x_tail[:min_len], y_tail[:min_len], x_neck[:min_len], y_neck[:min_len]
        )
    return None

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
    
def filter_by_behavior_type(data, behavior_type, min_count=1):
    """Filter larvae that have at least min_count instances of a behavior."""
    # Implementation here...
    pass

def filter_by_experiment_date(data, start_date, end_date):
    """Filter data by experiment date range."""
    # Implementation here...
    pass

def filter_larvae_by_excess_stop_time(data, max_stop_percentage=0.5):
    """
    Filter out larvae that spend too much time in the stop state.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        max_stop_percentage: Maximum allowed percentage of time in stop state (0.0-1.0)
        
    Returns:
        dict: Filtered data with same structure as input, excluding larvae with too much stop time
        dict: Statistics about stop time for each larva
    """
    import numpy as np
    import copy
    
    # Make a deep copy to avoid modifying original data
    filtered_data = copy.deepcopy(data)
    
    # Handle data structure
    if 'data' in data:
        extracted_data = filtered_data['data']
        is_multi_exp = True
    else:
        extracted_data = filtered_data
        is_multi_exp = False
    
    # Track larvae to remove and their statistics
    larvae_to_remove = []
    larva_stats = {}
    
    # Analyze each larva
    for larva_id, larva_data in list(extracted_data.items()):
        # Skip if necessary data is not available
        if 'global_state_large_state' not in larva_data or 't' not in larva_data:
            continue
            
        # Get the behavior states and time data
        states = np.array(larva_data['global_state_large_state']).flatten()
        times = np.array(larva_data['t']).flatten()
        
        # Calculate total tracked time
        total_time = times[-1] - times[0]
        
        # Identify stop states (3.0 for large_stop, 2.5 for small_stop)
        stop_states = [3.0, 2.5]
        stop_mask = np.isin(states, stop_states)
        
        # Calculate time spent in each state
        time_in_states = {}
        current_state = None
        state_start_time = None
        
        for i in range(len(states)):
            state = states[i]
            time = times[i]
            
            # State transition or end of sequence
            if state != current_state or i == len(states) - 1:
                # Record previous state duration if there was one
                if current_state is not None and state_start_time is not None:
                    duration = time - state_start_time
                    if current_state not in time_in_states:
                        time_in_states[current_state] = 0
                    time_in_states[current_state] += duration
                
                # Start new state
                current_state = state
                state_start_time = time
        
        # Calculate total time in stop states
        total_stop_time = sum(time_in_states.get(stop_state, 0) for stop_state in stop_states)
        stop_percentage = total_stop_time / total_time if total_time > 0 else 0
        
        # Store statistics
        larva_stats[larva_id] = {
            'total_tracked_time': total_time,
            'time_in_stop': total_stop_time,
            'stop_percentage': stop_percentage
        }
        
        # Flag larvae with excessive stop time
        if stop_percentage > max_stop_percentage:
            larvae_to_remove.append(larva_id)
    
    # Remove flagged larvae
    for larva_id in larvae_to_remove:
        if larva_id in extracted_data:
            del extracted_data[larva_id]
    
    # Update metadata if necessary
    if is_multi_exp and 'metadata' in filtered_data:
        filtered_data['metadata'].update({
            'total_larvae': len(extracted_data),
            'larvae_removed_excess_stop': len(larvae_to_remove),
            'max_stop_percentage': max_stop_percentage
        })
    
    # Print filtering results
    print(f"Excess stop time filtering results (threshold: {max_stop_percentage*100:.0f}%):")
    print(f"  - Removed {len(larvae_to_remove)} larvae with >{int(max_stop_percentage*100)}% time in stop state")
    print(f"  - {len(extracted_data)} larvae remaining")
    
    return filtered_data, larva_stats

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

def plot_behavior_duration_histograms(data, show_plot=True, using_frames = True, title=None, 
                                     bin_width=None, max_value=None, log_scale=False,
                                     save_path=None):
    """Plot histograms of behavior durations.
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        show_plot: Whether to show visualization (default: True)
        title: Optional plot title override (default: None)
        bin_width: Width of histogram bins (default: 0.2 for seconds, 1 for frames)
        max_value: Maximum value to include in histogram (default: None, auto-calculated per behavior)
        log_scale: Whether to use log scale for y-axis (default: False)
        save_path: Path to save the figure (default: None)
        
    Returns:
        dict: Statistics and histogram data for each behavior type, including indices of behaviors above mean and median
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Define behavior mappings for large_small arrays
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
        'run': [0.0, 0.0, 0.0],      # Black
        'cast': [1.0, 0.0, 0.0],     # Red
        'stop': [0.0, 1.0, 0.0],     # Green
        'hunch': [0.0, 0.0, 1.0],    # Blue
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
    
    # Add colors for total behaviors (intermediate shade)
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
            'duration_indices': [],  # Track larva_id and action indices
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'median_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'histogram_data': None,
            'above_mean_indices': [],  # Will store indices of behaviors above mean
            'above_median_indices': []  # Will store indices of behaviors above median
        }
    
    # Initialize total behaviors (combined large and small)
    for base_name in base_behavior_colors:
        behavior_stats[f"{base_name}_total"] = {
            'durations': [],
            'duration_indices': [],  # Track larva_id and action indices
            'n_actions': 0,
            'total_duration': 0,
            'mean_duration': 0,
            'median_duration': 0,
            'std_duration': 0,
            'percent_of_total': 0,
            'histogram_data': None,
            'above_mean_indices': [],  # Will store indices of behaviors above mean
            'above_median_indices': []  # Will store indices of behaviors above median
        }
    
    # Process each larva
    total_actions = {'large': 0, 'small': 0, 'total': 0}
    
    # Determine if we're using frames (n_duration_large_small) or seconds (duration_large_small)
    duration_key = 'n_duration_large_small' if using_frames else 'duration_large_small'
    
    # Set appropriate bin width based on units
    if bin_width is None:
        bin_width = 1 if using_frames else 0.2
    
    # Set appropriate unit label
    unit_label = 'frames' if using_frames else 'seconds'
    unit_abbr = 'fr' if using_frames else 's'
    
    for larva_id, larva_data in extracted_data.items():
        # Skip if necessary data is not available
        if duration_key not in larva_data or 'nb_action_large_small' not in larva_data:
            continue
            
        # Get the large_small arrays
        duration_large_small = larva_data[duration_key]
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
                
                # Get non-NaN indices and values
                valid_indices = ~np.isnan(durations.flatten())
                clean_durations = durations.flatten()[valid_indices]
                
                if n > 0 and len(clean_durations) > 0:
                    # Add to behavior stats
                    behavior_stats[behavior_key]['n_actions'] += n
                    behavior_stats[behavior_key]['durations'].extend(clean_durations)
                    
                    # Store indices information as (larva_id, action_index, duration)
                    for i, duration in enumerate(clean_durations):
                        behavior_stats[behavior_key]['duration_indices'].append((larva_id, i, duration))
                    
                    # Update action counts by size
                    total_actions[size_group] += n
                    total_actions['total'] += n
                    
                    # Also add to total behavior type
                    total_key = f"{base_name}_total"
                    behavior_stats[total_key]['durations'].extend(clean_durations)
                    
                    # Store indices information for totals too
                    for i, duration in enumerate(clean_durations):
                        behavior_stats[total_key]['duration_indices'].append((larva_id, i, duration))
    
    # Calculate statistics for each behavior
    for behavior in behavior_stats:
        stats = behavior_stats[behavior]
        durations = stats['durations']
        duration_indices = stats['duration_indices']
        
        if durations:
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
            mean_duration = float(np.mean(durations))
            median_duration = float(np.median(durations))
            stats.update({
                'total_duration': float(np.sum(durations)),
                'mean_duration': mean_duration,
                'median_duration': median_duration,
                'std_duration': float(np.std(durations)),
                'percent_of_total': 100 * stats['n_actions'] / group_total if group_total > 0 else 0
            })
            
            # Find indices of behaviors above mean
            above_mean_indices = []
            for larva_id, action_idx, duration in duration_indices:
                if duration > mean_duration:
                    above_mean_indices.append((larva_id, action_idx))
            
            stats['above_mean_indices'] = above_mean_indices
            
            # Find indices of behaviors above median
            above_median_indices = []
            for larva_id, action_idx, duration in duration_indices:
                if duration > median_duration:
                    above_median_indices.append((larva_id, action_idx))
            
            stats['above_median_indices'] = above_median_indices
    base_behavior_names = ['run', 'cast', 'stop', 'hunch', 'backup', 'roll']
    if show_plot:
        # Define the base behavior types to plot        
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
            # Set up the figure layout - one row for each behavior type
            n_behaviors = len(active_behavior_types)
            fig, axs = plt.subplots(n_behaviors, 3, figsize=(8, 3 * n_behaviors))
            
            # Handle case with single behavior (reshape axs)
            if n_behaviors == 1:
                axs = np.array([axs])
            
            # Plot histograms for each behavior type
            for i, base_name in enumerate(active_behavior_types):
                large_key = f"large_{base_name}"
                small_key = f"small_{base_name}"
                total_key = f"{base_name}_total"
                
                # Get durations
                large_durations = behavior_stats[large_key]['durations']
                small_durations = behavior_stats[small_key]['durations']
                total_durations = behavior_stats[total_key]['durations']
                
                # Calculate max durations for each behavior variant with 5% buffer
                max_large = max(large_durations) * 1.05 if large_durations else 0
                max_small = max(small_durations) * 1.05 if small_durations else 0
                max_total = max(total_durations) * 1.05 if total_durations else 0
                
                # If a global max_value is provided, use that instead
                if max_value is not None:
                    max_large = max_small = max_total = max_value
                
                # Generate bins for each histogram individually
                large_bins = np.arange(0, max_large + bin_width, bin_width) if large_durations else None
                small_bins = np.arange(0, max_small + bin_width, bin_width) if small_durations else None
                total_bins = np.arange(0, max_total + bin_width, bin_width) if total_durations else None
                
                # Plot large behavior durations
                if large_durations:
                    hist_large, bins_large, _ = axs[i, 0].hist(
                        large_durations, bins=large_bins, alpha=0.7, 
                        color=behavior_colors[large_key],
                        label=f"n={len(large_durations)}")
                    
                    # Store histogram data
                    behavior_stats[large_key]['histogram_data'] = {
                        'counts': hist_large,
                        'bins': bins_large
                    }
                    
                    # Add mean and median lines
                    mean_val = behavior_stats[large_key]['mean_duration']
                    median_val = behavior_stats[large_key]['median_duration']
                    axs[i, 0].axvline(mean_val, color='k', linestyle='--', 
                                     label=f'Mean: {mean_val:.2f}{unit_abbr}')
                    axs[i, 0].axvline(median_val, color='r', linestyle='-', 
                                     label=f'Median: {median_val:.2f}{unit_abbr}')
                    
                    axs[i, 0].set_title(f'Large {base_name.capitalize()} Durations')
                    axs[i, 0].legend(fontsize='small')
                    axs[i, 0].set_xlim(0, max_large)
                else:
                    axs[i, 0].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                  transform=axs[i, 0].transAxes)
                
                # Plot small behavior durations
                if small_durations:
                    hist_small, bins_small, _ = axs[i, 1].hist(
                        small_durations, bins=small_bins, alpha=0.7, 
                        color=behavior_colors[small_key], 
                        label=f"n={len(small_durations)}")
                    
                    # Store histogram data
                    behavior_stats[small_key]['histogram_data'] = {
                        'counts': hist_small,
                        'bins': bins_small
                    }
                    
                    # Add mean and median lines
                    mean_val = behavior_stats[small_key]['mean_duration']
                    median_val = behavior_stats[small_key]['median_duration']
                    axs[i, 1].axvline(mean_val, color='k', linestyle='--', 
                                     label=f'Mean: {mean_val:.2f}{unit_abbr}')
                    axs[i, 1].axvline(median_val, color='r', linestyle='-', 
                                     label=f'Median: {median_val:.2f}{unit_abbr}')
                    
                    axs[i, 1].set_title(f'Small {base_name.capitalize()} Durations')
                    axs[i, 1].legend(fontsize='small')
                    axs[i, 1].set_xlim(0, max_small)
                else:
                    axs[i, 1].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                  transform=axs[i, 1].transAxes)
                
                # Plot total behavior durations
                if total_durations:
                    hist_total, bins_total, _ = axs[i, 2].hist(
                        total_durations, bins=total_bins, alpha=0.7, 
                        color=behavior_colors[total_key],
                        label=f"n={len(total_durations)}")
                    
                    # Store histogram data
                    behavior_stats[total_key]['histogram_data'] = {
                        'counts': hist_total,
                        'bins': bins_total
                    }
                    
                    # Add mean and median lines
                    mean_val = behavior_stats[total_key]['mean_duration']
                    median_val = behavior_stats[total_key]['median_duration']
                    axs[i, 2].axvline(mean_val, color='k', linestyle='--', 
                                     label=f'Mean: {mean_val:.2f}{unit_abbr}')
                    axs[i, 2].axvline(median_val, color='r', linestyle='-', 
                                     label=f'Median: {median_val:.2f}{unit_abbr}')
                    
                    axs[i, 2].set_title(f'All {base_name.capitalize()} Durations')
                    axs[i, 2].legend(fontsize='small')
                    axs[i, 2].set_xlim(0, max_total)
                else:
                    axs[i, 2].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                  transform=axs[i, 2].transAxes)
                
                # Set x and y labels for each row's plots
                for j in range(3):
                    if i == n_behaviors - 1:  # Only add x labels to bottom row
                        axs[i, j].set_xlabel(f'Duration ({unit_label})')
                    axs[i, j].set_ylabel('Count')
                    
                    # Apply log scale if requested
                    if log_scale:
                        axs[i, j].set_yscale('log')
                    
                    # Remove top and right spines
                    axs[i, j].spines['top'].set_visible(False)
                    axs[i, j].spines['right'].set_visible(False)
            
            # Add a super title
            units_info = " in frames" if using_frames else ""
            fig.suptitle(f"Behavior Duration Distributions{units_info} - {title}\n(n = {n_larvae} larvae)", 
                        fontsize=14)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)  # Make room for suptitle
            
            # Save the figure if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
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
            
            print(f"\n{group_name} duration analysis ({unit_label}) for {title}")
            print(f"Number of larvae: {n_larvae}")
            print(f"Total actions: {group_total_actions}\n")
            print(f"{'Behavior':>12} {'Events':>8} {'%Total':>7} {'Mean':>10} {'Median':>10} {'Std':>10} {'Above Mean':>10} {'Above Med':>10}")
            print("-" * 95)
            
            for behavior in behaviors:
                stats = behavior_stats[behavior]
                if stats['durations']:
                    # Get readable name
                    if prefix == '_total':
                        display_name = behavior.replace('_total', '')
                    else:
                        display_name = behavior.replace(prefix, '')
                    
                    above_mean_count = len(stats['above_mean_indices'])
                    above_median_count = len(stats['above_median_indices'])
                    
                    print(f"{display_name:>12}: {stats['n_actions']:8d} {stats['percent_of_total']:6.1f}%"
                          f"{stats['mean_duration']:10.2f} {stats['median_duration']:10.2f} {stats['std_duration']:10.2f}"
                          f"{above_mean_count:10d} {above_median_count:10d}")
    
    return behavior_stats
 
def merge_short_stop_sequences(data, behavior_stats, using_frames=True, min_stop_duration_cast=2.0, min_stop_duration_run=3.0):
    """
    Merge sequences with short stops according to the following rules:
    1. Cast→short_stop→cast: Merge into a single cast
    2. Run→short_stop→run: Merge into a single run
    3. Run→short_stop→cast or cast→short_stop→run: Merge stop with the longer adjacent behavior
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        behavior_stats: Statistics returned from plot_behavior_duration_histograms
        using_frames: Whether durations are in frames (True) or seconds (False)
        min_stop_duration_cast: Minimum duration (in seconds) for stops between casts
        min_stop_duration_run: Minimum duration (in seconds) for stops between runs
        
    Returns:
        dict: Modified data with merged behaviors
        dict: Summary of merging operations
    """
    import numpy as np
    import copy
    
    # Make a deep copy to avoid modifying original data
    merged_data = copy.deepcopy(data)
    
    # Get mean duration for stops (use the total stops stats)
    if 'stop_total' in behavior_stats:
        mean_stop_duration = behavior_stats['stop_total']['mean_duration']
        # Use the larger of mean duration or specified minimum
        threshold_cast = min_stop_duration_cast
        threshold_run = min_stop_duration_run
    else:
        print(f"No stop behavior statistics found. Using defaults of {min_stop_duration_cast}s for casts and {min_stop_duration_run}s for runs.")
        threshold_cast = min_stop_duration_cast
        threshold_run = min_stop_duration_run
    
    # Handle data structure
    if 'data' in data:
        extracted_data = merged_data['data']
    else:
        extracted_data = merged_data
    
    # Track merging statistics
    merge_summary = {
        "cast_stop_cast_count": 0,
        "run_stop_run_count": 0,
        "mixed_count": 0,
        "merged_sequences": [],
        "total_duration_saved": 0
    }
    
    # Process each larva
    for larva_id, larva_data in extracted_data.items():
        # Check if we have the necessary data
        if 'global_state_large_state' not in larva_data or 't' not in larva_data:
            continue
            
        # Get the time and state data
        states = np.array(larva_data['global_state_large_state']).flatten()
        times = np.array(larva_data['t']).flatten()
        
        # Find continuous segments of behavior
        segments = []
        current_state = None
        start_idx = 0
        
        for i, state in enumerate(states):
            if state != current_state:
                # Record the previous segment if there was one
                if current_state is not None and i > start_idx:
                    segments.append({
                        'state': current_state,
                        'start_idx': start_idx,
                        'end_idx': i - 1,
                        'duration': times[i - 1] - times[start_idx]
                    })
                current_state = state
                start_idx = i
        
        # Add the final segment
        if current_state is not None and len(states) > start_idx:
            segments.append({
                'state': current_state,
                'start_idx': start_idx,
                'end_idx': len(states) - 1,
                'duration': times[-1] - times[start_idx]
            })
        
        # Look for patterns with short stops
        i = 0
        new_states = np.copy(states)
        
        while i < len(segments) - 2:
            # Map segment states to behavior names
            state1 = segments[i]['state']
            state2 = segments[i+1]['state']
            state3 = segments[i+2]['state']
            
            # Get durations
            duration1 = segments[i]['duration']
            duration2 = segments[i+1]['duration']  # Stop duration
            duration3 = segments[i+2]['duration']
            
            # Check different patterns
            is_cast1 = state1 in [2, 1.5]  # large_cast or small_cast
            is_run1 = state1 in [1, 0.5]   # large_run or small_run
            is_stop = state2 in [3, 2.5]   # large_stop or small_stop
            is_cast3 = state3 in [2, 1.5]  # large_cast or small_cast
            is_run3 = state3 in [1, 0.5]   # large_run or small_run
            
            # Pattern 1: Cast → short stop → cast
            if is_cast1 and is_stop and is_cast3 and duration2 < threshold_cast:
                # Record and merge as cast-stop-cast
                merge_summary["cast_stop_cast_count"] += 1
                merge_summary["merged_sequences"].append({
                    "type": "cast-stop-cast",
                    "larva_id": larva_id,
                    "behavior1_duration": duration1,
                    "stop_duration": duration2,
                    "behavior3_duration": duration3,
                    "total_duration": duration1 + duration2 + duration3
                })
                merge_summary["total_duration_saved"] += duration2
                
                # Convert the stop segment to a cast
                start_stop = segments[i+1]['start_idx']
                end_stop = segments[i+1]['end_idx']
                new_states[start_stop:end_stop+1] = state1  # Use the state of the first cast
                
                # Skip ahead
                i += 3
                continue
                
            # Pattern 2: Run → short stop → run
            elif is_run1 and is_stop and is_run3 and duration2 < threshold_run:
                # Record and merge as run-stop-run
                merge_summary["run_stop_run_count"] += 1
                merge_summary["merged_sequences"].append({
                    "type": "run-stop-run",
                    "larva_id": larva_id,
                    "behavior1_duration": duration1,
                    "stop_duration": duration2,
                    "behavior3_duration": duration3,
                    "total_duration": duration1 + duration2 + duration3
                })
                merge_summary["total_duration_saved"] += duration2
                
                # Convert the stop segment to a run
                start_stop = segments[i+1]['start_idx']
                end_stop = segments[i+1]['end_idx']
                new_states[start_stop:end_stop+1] = state1  # Use the state of the first run
                
                # Skip ahead
                i += 3
                continue
                
            # Pattern 3: Mixed (run → short stop → cast OR cast → short stop → run)
            elif ((is_run1 and is_stop and is_cast3) or (is_cast1 and is_stop and is_run3)) and duration2 < threshold_cast:
                # Determine which behavior has longer duration
                if duration1 >= duration3:
                    # Use the first behavior (preceding the stop)
                    target_state = state1
                    longer_type = "preceding" + ("_run" if is_run1 else "_cast")
                else:
                    # Use the third behavior (following the stop)
                    target_state = state3
                    longer_type = "following" + ("_run" if is_run3 else "_cast")
                
                # Record and merge as mixed
                merge_summary["mixed_count"] += 1
                merge_summary["merged_sequences"].append({
                    "type": "mixed",
                    "longer": longer_type,
                    "larva_id": larva_id,
                    "behavior1_duration": duration1,
                    "stop_duration": duration2,
                    "behavior3_duration": duration3,
                    "total_duration": duration1 + duration2 + duration3
                })
                merge_summary["total_duration_saved"] += duration2
                
                # Convert the stop segment to the longer behavior
                start_stop = segments[i+1]['start_idx']
                end_stop = segments[i+1]['end_idx']
                new_states[start_stop:end_stop+1] = target_state
                
                # Skip ahead
                i += 3
                continue
            
            # Move to next segment if no merge happened
            i += 1
        
        # Update the modified states back to the larva data
        larva_data['global_state_large_state'] = new_states
    
    # Print summary statistics
    total_merged = merge_summary["cast_stop_cast_count"] + merge_summary["run_stop_run_count"] + merge_summary["mixed_count"]
    print(f"Merged {total_merged} sequences with short stops:")
    print(f"  - {merge_summary['cast_stop_cast_count']} cast-stop-cast sequences")
    print(f"  - {merge_summary['run_stop_run_count']} run-stop-run sequences") 
    print(f"  - {merge_summary['mixed_count']} mixed sequences (run-stop-cast or cast-stop-run)")
    print(f"Total duration saved: {merge_summary['total_duration_saved']:.2f} {'frames' if using_frames else 'seconds'}")
    
    return merged_data, merge_summary

def plot_behavior_merge_differences(original_data, merged_data, larva_id=None, time_window=None):
    """
    Create a comparison visualization that shows where cast-stop-cast sequences
    were merged in the behavior data.
    
    Args:
        original_data: The original unmodified behavior data
        merged_data: The data after merging cast-stop-cast sequences
        larva_id: Optional specific larva ID to visualize (if None, will use first available)
        time_window: Optional [start_time, end_time] to zoom in on a specific section
        
    Returns:
        fig: The matplotlib figure object
    """
    from matplotlib.colors import ListedColormap
    
    # If no larva_id is provided, use the first one
    if larva_id is None:
        larva_id = next(iter(original_data.keys()))
    
    # Get state data for the selected larva from both datasets
    orig_states = np.array(original_data[larva_id]['global_state_large_state']).flatten()
    merged_states = np.array(merged_data[larva_id]['global_state_large_state']).flatten()
    times = np.array(original_data[larva_id]['t']).flatten()
    
    # Create a difference array (where states were changed)
    diff_states = np.zeros_like(orig_states)
    diff_states[(orig_states != merged_states)] = 1  # Mark changed positions
    
    # Apply time window if provided
    if time_window is not None:
        start_idx = np.searchsorted(times, time_window[0])
        end_idx = np.searchsorted(times, time_window[1])
        orig_states = orig_states[start_idx:end_idx]
        merged_states = merged_states[start_idx:end_idx]
        diff_states = diff_states[start_idx:end_idx]
        times = times[start_idx:end_idx]
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Define colormap for the behavior states
    behavior_colors = {
        0: 'white',      # No behavior
        1: [0.0, 0.0, 0.0],  # large_run (Black)
        2: [1.0, 0.0, 0.0],  # large_cast (Red)
        3: [0.0, 1.0, 0.0],  # large_stop (Green)
        4: [0.0, 0.0, 1.0],  # large_hunch (Blue)
        5: [1.0, 0.5, 0.0],  # large_backup (Orange)
        6: [0.5, 0.0, 0.5],  # large_roll (Purple)
        0.5: [0.7, 0.7, 0.7],  # small_run (Light gray)
        1.5: [1.0, 0.7, 0.7],  # small_cast (Light red)
        2.5: [0.7, 1.0, 0.7],  # small_stop (Light green)
        3.5: [0.7, 0.7, 1.0],  # small_hunch (Light blue)
        4.5: [1.0, 0.8, 0.6],  # small_backup (Light orange)
        5.5: [0.8, 0.6, 0.8]   # small_roll (Light purple)
    }
    
    # Create list of colors for the colormap
    unique_states = sorted(list(set(np.concatenate([orig_states, merged_states]))))
    color_list = [behavior_colors[state] if state in behavior_colors else 'gray' for state in unique_states]
    
    # Create behavior colormap
    behavior_cmap = ListedColormap(color_list)
    
    # Plot original state data
    ax1.imshow(orig_states.reshape(1, -1), aspect='auto', cmap=behavior_cmap)
    ax1.set_yticks([])
    ax1.set_title(f"Original Behavior (Larva {larva_id})")
    
    # Plot merged state data
    ax2.imshow(merged_states.reshape(1, -1), aspect='auto', cmap=behavior_cmap)
    ax2.set_yticks([])
    ax2.set_title(f"Merged Behavior (Larva {larva_id})")
    
    # Plot difference (highlighting changed areas)
    diff_cmap = ListedColormap(['white', 'red'])
    ax3.imshow(diff_states.reshape(1, -1), aspect='auto', cmap=diff_cmap)
    ax3.set_yticks([])
    ax3.set_title("Differences (Red = Changed)")
    
    # Add x-axis label
    ax3.set_xlabel("Time (s)")
    
    # Add time ticks
    if len(times) > 20:
        # Add fewer time markers for readability
        step = max(len(times) // 10, 1)
        ax3.set_xticks(np.arange(0, len(times), step))
        ax3.set_xticklabels([f"{times[i]:.1f}" for i in range(0, len(times), step)], rotation=45)
    else:
        ax3.set_xticks(np.arange(len(times)))
        ax3.set_xticklabels([f"{t:.1f}" for t in times], rotation=90)
    
    # Add a legend for the behavior states
    behavior_names = {
        0: 'None',
        1: 'large_run',
        2: 'large_cast',
        3: 'large_stop',
        4: 'large_hunch',
        5: 'large_backup',
        6: 'large_roll',
        0.5: 'small_run',
        1.5: 'small_cast',
        2.5: 'small_stop',
        3.5: 'small_hunch',
        4.5: 'small_backup',
        5.5: 'small_roll'
    }
    
    # Create legend patches
    patches = []
    for state in unique_states:
        if state in behavior_names:
            color = behavior_colors[state]
            patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, 
                                        label=behavior_names[state]))
    
    # Add the legend to the figure
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for the legend
    
    # Print some statistics about the changes
    n_changes = np.sum(diff_states)
    pct_changes = (n_changes / len(diff_states)) * 100
    print(f"Total time points changed: {n_changes} ({pct_changes:.2f}%)")
    
    # Count cast-stop-cast sequences that were merged
    csc_patterns = 0
    for i in range(1, len(orig_states)-1):
        # Check for cast → stop → cast pattern that was changed
        if (orig_states[i] in [3, 2.5] and            # stop in middle
            merged_states[i] in [2, 1.5] and          # now a cast
            orig_states[i-1] in [2, 1.5] and          # cast before
            orig_states[i+1] in [2, 1.5]):            # cast after
            csc_patterns += 1
    
    print(f"Cast-stop-cast sequences identified in this view: {csc_patterns}")
    
    return fig


def plot_global_behavior_matrix(trx_data, show_separate_totals=True, show_plot=True, ax=None):
    """
    Plot global behavior using the global state.
    
    This function visualizes behavioral states across time for all larvae.
    It processes both large_state (1-6) and small_large_state (0.5-5.5) values.
    
    Args:
        trx_data: Dictionary of larva data
        show_separate_totals: If True, show large, small, and total behaviors as separate rows
                             If False, show only a single row per larva
        show_plot: Whether to display the plot immediately
        ax: Optional matplotlib axes to plot on
    
    Returns:
        Behavior matrix or dict of processed data depending on mode
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Create axis if none provided
    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        created_fig = True
    else:
        created_fig = False
    
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

    # Process larvae data
    if isinstance(trx_data, dict) and 'data' in trx_data:
        larvae_data = trx_data['data']
    else:
        larvae_data = trx_data
        
    larva_ids = sorted(larvae_data.keys())
    n_larvae = len(larva_ids)
    
    if n_larvae == 0:
        ax.text(0.5, 0.5, "No larvae data available", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    # Compute time ranges
    tmins = []
    tmaxs = []
    for lid in larva_ids:
        if 't' in larvae_data[lid] and len(larvae_data[lid]['t']) > 0:
            times = np.array(larvae_data[lid]['t']).flatten()
            if len(times) > 0:
                tmins.append(np.min(times))
                tmaxs.append(np.max(times))
    
    if not tmins:
        ax.text(0.5, 0.5, "No time data available", 
                ha='center', va='center', transform=ax.transAxes)
        return {}
    
    t_min = min(tmins)
    t_max = max(tmaxs)
    
    # Decide on plotting approach based on show_separate_totals
    if show_separate_totals:
        # Create full behavior matrix with colors (as in original function)
        resolution = 1000
        behavior_matrix = np.full((n_larvae * 3, resolution, 3), fill_value=1.0)  # white background
        
        # Process each larva
        for larva_idx, lid in enumerate(larva_ids):
            # Get time and state data
            if 't' not in larvae_data[lid] or len(larvae_data[lid]['t']) == 0:
                continue  # Skip larvae without required data
                
            larva_time = np.array(larvae_data[lid]['t']).flatten()
            
            # Use global_state_small_large_state if available, otherwise use global_state_large_state
            if 'global_state_small_large_state' in larvae_data[lid]:
                states = np.array(larvae_data[lid]['global_state_small_large_state']).flatten()
            elif 'global_state_large_state' in larvae_data[lid]:
                states = np.array(larvae_data[lid]['global_state_large_state']).flatten()
            else:
                continue  # Skip if no state data available
            
            # Ensure arrays have same length
            min_len = min(len(larva_time), len(states))
            if min_len == 0:
                continue
                
            larva_time = larva_time[:min_len]
            states = states[:min_len]
            
            # Convert times to indices
            time_indices = np.floor(
                ((larva_time - t_min) / (t_max - t_min) * (resolution - 1))
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
            
            # Assign the arrays to the behavior matrix
            row_large = larva_idx * 3
            row_small = larva_idx * 3 + 1
            row_total = larva_idx * 3 + 2
            
            behavior_matrix[row_large] = large_behaviors
            behavior_matrix[row_small] = small_behaviors
            behavior_matrix[row_total] = total_behaviors

        # Plot the behavior matrix
        ax.imshow(behavior_matrix, aspect='auto', interpolation='nearest', alpha=0.8,
                extent=[t_min, t_max, behavior_matrix.shape[0], 0])
        
        # Create y-tick labels
        ytick_positions = []
        ytick_labels = []
        
        for i, lid in enumerate(larva_ids):
            base_pos = i * 3
            # Position ticks in the middle of each row
            ytick_positions.extend([base_pos + 0.5, base_pos + 1.5, base_pos + 2.5])
            ytick_labels.extend([f"{lid} (large)", f"{lid} (small)", f"{lid} (total)"])
        
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize='small')
        
        # Add horizontal lines to separate larvae
        for i in range(1, n_larvae):
            y_pos = i * 3
            ax.axhline(y=y_pos, color='black', linestyle='-', linewidth=0.5)
        
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
        
        ax.legend(handles=legend_elements, loc='center left', 
                bbox_to_anchor=(1, 0.5), title='Behavioral States')
        
    else:
        # Use the scatter plot approach for single row per larva, plotting ALL behaviors
        ytick_positions = []
        ytick_labels = []
        
        for i, larva_id in enumerate(larva_ids):
            if 'global_state_large_state' not in larvae_data[larva_id] or 't' not in larvae_data[larva_id]:
                continue  # Skip larvae without required data
            
            times = np.array(larvae_data[larva_id]['t']).flatten()
            states = np.array(larvae_data[larva_id]['global_state_large_state']).flatten()
            
            # Check if times are sorted
            if not np.all(np.diff(times) >= 0) and len(times) > 1:
                # Sort times and states
                sorted_indices = np.argsort(times)
                times = times[sorted_indices]
                states = states[sorted_indices]
            
            # Ensure both arrays have same length
            min_len = min(len(times), len(states))
            if min_len == 0:
                continue
                
            times = times[:min_len]
            states = states[:min_len]
            
            # Draw thin line across total time range
            ax.plot([times[0], times[-1]], [i, i], 'k-', linewidth=0.5, alpha=0.2)
            
            # Draw colored segments for each behavior state with appropriate colors
            # Plot ALL behaviors, not just run, cast, and stop
            for state_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
                if state_val in state_mapping:
                    info = state_mapping[state_val]
                    mask = states == state_val
                    if np.any(mask):
                        ax.scatter(times[mask], [i] * sum(mask), 
                                 color=info['color'], s=5, marker='s', alpha=0.8)
            
            ytick_positions.append(i)
            ytick_labels.append(str(larva_id))
        
        # Set axis properties
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        
        # Add legend with all behavioral states
        legend_elements = []
        for state_val in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            if state_val in state_mapping:
                info = state_mapping[state_val]
                legend_elements.append(
                    Line2D([0], [0], marker='s', color='w', 
                          markerfacecolor=info['color'], markersize=8, 
                          label=f"{info['base']}")
                )
        
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Set common axis properties
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Larva ID')
    # Title removed as requested
    ax.set_xlim(t_min, t_max)
    
    # Only show if we created our own figure and show_plot is True
    if created_fig and show_plot:
        plt.tight_layout()
        plt.show()
    
    # Return the processed data
    result_data = {}
    for larva_id in larva_ids:
        if 'global_state_large_state' in larvae_data[larva_id] and 't' in larvae_data[larva_id]:
            result_data[larva_id] = {
                'times': np.array(larvae_data[larva_id]['t']).flatten(),
                'states': np.array(larvae_data[larva_id]['global_state_large_state']).flatten()
            }
    
    return result_data

def _create_behavior_matrix(trx_data, show_separate_totals=True):
    """
    Create a behavior matrix for visualization without displaying it.
    
    Args:
        trx_data: Dictionary with larva tracking data
        show_separate_totals: If True, show large, small, and total behaviors as separate rows
        
    Returns:
        numpy.ndarray: The behavior matrix
        tuple: The extent (left, right, bottom, top) for imshow
    """
    import numpy as np
    
    # Define the base behavior names
    base_behavior_names = {
        1.0: 'run', 
        2.0: 'cast', 
        3.0: 'stop'
    }
    
    # Get larvae IDs
    if isinstance(trx_data, dict) and 'data' in trx_data:
        # Handle nested data structure (likely from multi-experiment data)
        larvae_data = trx_data['data']
    else:
        # Direct data structure
        larvae_data = trx_data
        
    larva_ids = sorted(larvae_data.keys())
    n_larvae = len(larva_ids)
    
    if n_larvae == 0:
        return np.zeros((1, 1)), (0, 1, 0, 1)
    
    # Compute time range - with better error handling
    tmins = []
    tmaxs = []
    for lid in larva_ids:
        if 't' in larvae_data[lid] and len(larvae_data[lid]['t']) > 0:
            # Convert to numpy array if it's not already and flatten
            times = np.array(larvae_data[lid]['t']).flatten()
            if len(times) > 0:
                tmins.append(np.min(times))
                tmaxs.append(np.max(times))
    
    if not tmins or not tmaxs:
        return np.zeros((1, 1)), (0, 1, 0, 1)
        
    t_min = min(tmins)
    t_max = max(tmaxs)
    
    # Create a time grid (100 points is reasonable for visualization)
    n_time_points = 100
    time_grid = np.linspace(t_min, t_max, n_time_points)
    
    # Initialize behavior matrix
    rows_per_larva = 3 if show_separate_totals else 1
    behavior_matrix = np.zeros((n_larvae * rows_per_larva, n_time_points))
    
    # Fill in the behavior matrix
    for i, larva_id in enumerate(larva_ids):
        if 'global_state_large_state' not in larvae_data[larva_id] or 't' not in larvae_data[larva_id]:
            continue  # Skip larvae without required data
        
        try:
            # Convert both times and behaviors to flat numpy arrays
            times = np.array(larvae_data[larva_id]['t']).flatten()
            behaviors = np.array(larvae_data[larva_id]['global_state_large_state']).flatten()
            
            # Check if times are sorted (required for np.interp)
            if not np.all(np.diff(times) >= 0):
                # Sort times and behaviors if times are not monotonically increasing
                sorted_indices = np.argsort(times)
                times = times[sorted_indices]
                behaviors = behaviors[sorted_indices]
            
            # Ensure both arrays have the same length
            min_len = min(len(times), len(behaviors))
            if min_len == 0:
                continue  # Skip if either array is empty
            
            times = times[:min_len]
            behaviors = behaviors[:min_len]
            
            # Interpolate behaviors to the time grid
            interp_behaviors = np.interp(time_grid, times, behaviors)
            
            if show_separate_totals:
                # Row 0: Large behaviors (run=1.0, cast=2.0, stop=3.0)
                # Row 1: Small behaviors (run=1.5, cast=2.5, stop=3.5) - not used here
                # Row 2: All behaviors combined
                row_idx = i * 3
                behavior_matrix[row_idx] = interp_behaviors
                behavior_matrix[row_idx + 2] = interp_behaviors  # Combined
            else:
                # Just one row per larva
                behavior_matrix[i] = interp_behaviors
                
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            # Skip this larva and continue with others
            continue
    
    # Return the matrix and the extent for imshow
    extent = (t_min, t_max, n_larvae * rows_per_larva, 0)
    return behavior_matrix, extent

def plot_behavior_matrices_for_all_dates(base_path, show_separate_totals=False, min_duration=300,
                                         max_stop_percentage=0.4, min_stop_duration_cast=3.0,
                                         min_stop_duration_run=3.0, fig_size=(15, 10), max_dates=None,
                                         save_path=None):
    """
    Process all trx.mat files in date subfolders, filter, merge stops, and create a grid of behavior matrices.
    
    Args:
        base_path: Base directory containing date subfolders with trx.mat files
        show_separate_totals: If True, show large, small, and total behaviors as separate rows
        min_duration: Minimum duration (in seconds) for filtering larvae
        max_stop_percentage: Maximum percentage of time a larva can spend in stop state (0.0-1.0)
        min_stop_duration_cast: Minimum duration for stops between casts to be merged
        min_stop_duration_run: Minimum duration for stops between runs to be merged
        fig_size: Size of the figure (width, height)
        max_dates: Maximum number of dates to process (None for all)
        save_path: Path to save the figure (None for no saving)
        
    Returns:
        dict: Dictionary mapping date names to their merged datasets
        fig: The matplotlib figure
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import src.data_loader as data_loader
    import src.processor as processor
    import traceback
    
    # Define behavior state color scheme
    state_colors = {
        0.0: [1.0, 1.0, 1.0],      # White/Background
        1.0: [0.0, 0.0, 0.0],      # Black (for Run/Crawl)
        2.0: [1.0, 0.0, 0.0],      # Red (for Cast/Bend)
        3.0: [0.0, 1.0, 0.0],      # Green (for Stop)
        4.0: [0.0, 0.0, 1.0],      # Blue (for Hunch)
        5.0: [1.0, 0.5, 0.0],      # Orange (for Backup)
        6.0: [0.5, 0.0, 0.5],      # Purple (for Roll)
        0.5: [0.7, 0.7, 0.7],      # Light gray (for small Run)
        1.5: [1.0, 0.7, 0.7],      # Light red (for small Cast)
        2.5: [0.7, 1.0, 0.7],      # Light green (for small Stop)
        3.5: [0.7, 0.7, 1.0],      # Light blue (for small Hunch)
        4.5: [1.0, 0.8, 0.6],      # Light orange (for small Backup)
        5.5: [0.8, 0.6, 0.8]       # Light purple (for small Roll)
    }
    
    # Store merged data for all dates
    all_merged_data = {}
    
    # Find all trx.mat files in the base path
    all_trx_paths = []
    for root, dirs, files in os.walk(base_path):
        if 'trx.mat' in files:
            # Extract date name from folder
            date_name = os.path.basename(root)
            trx_path = os.path.join(root, "trx.mat")
            all_trx_paths.append((date_name, trx_path))
    
    # Limit to max_dates if specified
    if max_dates is not None:
        all_trx_paths = all_trx_paths[:max_dates]
    
    # Calculate grid dimensions based on number of files
    n_dates = len(all_trx_paths)
    if n_dates == 0:
        print("No trx.mat files found!")
        return {}, None
    
    # Calculate a reasonable grid layout
    n_cols = min(3, n_dates)  # Maximum 3 columns
    n_rows = (n_dates + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Process each date and create a subplot
    for i, (date_name, trx_path) in enumerate(all_trx_paths):
        print(f"\nProcessing {date_name}")
        row = i // n_cols
        col = i % n_cols
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        try:
            # Process the data using data_loader functions
            date_str, extracted_data, metadata = data_loader.process_single_file(trx_path)
            
            # Step 1: Filter by duration
            filtered_data = processor.filter_larvae_by_duration(extracted_data, min_total_duration=min_duration)
            n_after_duration = len(filtered_data)
            print(f"  - After duration filtering: {n_after_duration} larvae")
            
            # Step 2: Filter by excess stop time
            filtered_data, stop_time_stats = processor.filter_larvae_by_excess_stop_time(
                filtered_data, max_stop_percentage=max_stop_percentage
            )
            n_after_stop = len(filtered_data)
            print(f"  - After stop percentage filtering: {n_after_stop} larvae")
            
            # Step 3: Calculate behavior statistics for merging
            behavior_stats = processor.plot_behavior_duration_histograms(
                filtered_data, show_plot=False, using_frames=False
            )
            
            # Step 4: Merge short stop sequences
            merged_data, merge_summary = processor.merge_short_stop_sequences(
                filtered_data, behavior_stats,
                min_stop_duration_cast=min_stop_duration_cast,
                min_stop_duration_run=min_stop_duration_run
            )
            
            print(f"  - Merged {merge_summary['cast_stop_cast_count']} cast-stop-cast sequences")
            print(f"  - Merged {merge_summary['run_stop_run_count']} run-stop-run sequences")
            
            # Store the merged data for this date
            all_merged_data[date_name] = merged_data
            
            # Get number of larvae after all processing
            n_larvae = len(merged_data)
            
            if n_larvae > 0:
                try:
                    # Create behavior matrix using the merged data and our fixed helper
                    behavior_matrix, extent = _create_behavior_matrix(merged_data, show_separate_totals)
                    
                    # Create a proper colormap for behavior states
                    # First identify all unique states in the matrix
                    unique_states = np.unique(behavior_matrix)
                    
                    # Create color list and bounds for behavior states
                    color_list = []
                    bounds = []
                    
                    for state in sorted(unique_states):
                        bounds.append(state - 0.1)  # Lower bound
                        if state in state_colors:
                            color_list.append(state_colors[state])
                        else:
                            # Use white for unknown states
                            color_list.append([1, 1, 1])
                    
                    # Add upper bound for last state
                    bounds.append(sorted(unique_states)[-1] + 0.1)
                    
                    # Create colormap and norm
                    cmap = ListedColormap(color_list)
                    norm = BoundaryNorm(bounds, cmap.N)
                    
                    # Show the matrix with our custom colormap
                    ax.imshow(behavior_matrix, aspect='auto', interpolation='nearest', 
                             alpha=0.8, extent=extent, cmap=cmap, norm=norm)
                    
                    # Set title and labels
                    ax.set_title(f"{date_name} (n={n_larvae})", fontsize=10)
                    
                    # Only add x and y labels on the outer edges of the grid
                    if row == n_rows - 1:  # Bottom row
                        ax.set_xlabel('Time (s)', fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    
                    if col == 0:  # Leftmost column
                        ax.set_ylabel('Larvae', fontsize=8)
                    else:
                        ax.set_yticklabels([])
                    
                    # Minimize ticks for cleaner appearance
                    ax.tick_params(axis='both', which='major', labelsize=6)
                except Exception as inner_e:
                    print(f"Error creating behavior matrix for {date_name}: {str(inner_e)}")
                    traceback.print_exc()  # Print the full traceback
                    ax.text(0.5, 0.5, f"Error creating\nbehavior matrix", 
                           ha='center', va='center', transform=ax.transAxes, color='red')
            else:
                ax.text(0.5, 0.5, f"No data for {date_name}\nafter filtering", 
                       ha='center', va='center', transform=ax.transAxes)
        
        except Exception as e:
            print(f"Error processing {trx_path}: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
            ax.text(0.5, 0.5, f"Error processing\n{date_name}", 
                   ha='center', va='center', transform=ax.transAxes, color='red')
    
    # If there's enough data for a decent plot, add a color legend
    if n_dates > 0:
        # Create color legend patches for the most common behavior states
        from matplotlib.patches import Patch
        
        behavior_labels = {
            1.0: 'Run/Crawl',
            2.0: 'Cast/Bend',
            3.0: 'Stop',
            4.0: 'Hunch',
            5.0: 'Backup',
            6.0: 'Roll',
            0.5: 'Small Run',
            1.5: 'Small Cast',
            2.5: 'Small Stop'
        }
        
        legend_elements = []
        for state, label in behavior_labels.items():
            if state in state_colors:
                legend_elements.append(
                    Patch(facecolor=state_colors[state], alpha=0.8, 
                         edgecolor='none', label=label)
                )
        
        # Add legend if we have elements
        if legend_elements:
            fig.legend(handles=legend_elements, loc='lower center', 
                     ncol=min(5, len(legend_elements)), 
                     fontsize=8, bbox_to_anchor=(0.5, 0.01))
    
    fig.suptitle(f"Behavior Matrices - {os.path.basename(base_path)}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.10)  # Make room for suptitle and legend
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return all_merged_data, fig
def plot_global_behavior_matrix_above_mean(trx_data):
    """
    Plot global behavior matrix showing only behaviors with durations above the mean.
    Calculates behavior duration means directly from the data.
    
    Args:
        trx_data: Dictionary containing larva tracking data
        
    Returns:
        numpy.ndarray: The resulting behavior matrix
    """
    import copy
    import numpy as np
    
    # Create a deep copy to avoid modifying original data
    filtered_data = copy.deepcopy(trx_data)
    
    # Get sorted larva IDs
    larva_ids = sorted(filtered_data.keys())
    
    # Define state to behavior mapping
    state_to_behavior = {
        1: 'large_run',
        2: 'large_cast',
        3: 'large_stop',
        4: 'large_hunch',
        5: 'large_backup',
        6: 'large_roll',
        0.5: 'small_run',
        1.5: 'small_cast',
        2.5: 'small_stop',
        3.5: 'small_hunch',
        4.5: 'small_backup',
        5.5: 'small_roll'
    }
    
    # First pass: Calculate durations for each behavior type
    behavior_durations = {behavior: [] for behavior in state_to_behavior.values()}
    
    for larva_id in larva_ids:
        larva_data = filtered_data[larva_id]
        
        # Check if we have the necessary data
        if 'global_state_large_state' not in larva_data or 't' not in larva_data:
            continue
            
        # Get the time and state data
        states = np.array(larva_data['global_state_large_state']).flatten()
        times = np.array(larva_data['t']).flatten()
        
        # Find continuous segments of behavior
        segments = []
        current_state = None
        start_idx = 0
        
        for i, state in enumerate(states):
            if state != current_state:
                # Record the previous segment if there was one
                if current_state is not None and i > start_idx:
                    segments.append({
                        'state': current_state,
                        'start_idx': start_idx,
                        'end_idx': i - 1,
                        'duration': times[i - 1] - times[start_idx]  # This is in seconds
                    })
                current_state = state
                start_idx = i
        
        # Add the final segment
        if current_state is not None and len(states) > start_idx:
            segments.append({
                'state': current_state,
                'start_idx': start_idx,
                'end_idx': len(states) - 1,
                'duration': times[-1] - times[start_idx]  # This is in seconds
            })
        
        # Collect durations for each behavior type
        for segment in segments:
            state = segment['state']
            if state in state_to_behavior:
                behavior_key = state_to_behavior[state]
                behavior_durations[behavior_key].append(segment['duration'])
    
    # Calculate mean durations for each behavior type
    mean_durations = {}
    for behavior, durations in behavior_durations.items():
        if durations:  # Only calculate if we have data
            mean_durations[behavior] = np.nanmedian(durations)
            print(f"Mean duration for {behavior}: {mean_durations[behavior]:.2f}s")
        else:
            mean_durations[behavior] = 0
    
    # Second pass: Filter behaviors that are below the mean
    for larva_id in larva_ids:
        larva_data = filtered_data[larva_id]
        
        # Check if we have the necessary data
        if 'global_state_large_state' not in larva_data or 't' not in larva_data:
            continue
            
        # Get the time and state data
        states = np.array(larva_data['global_state_large_state']).flatten()
        times = np.array(larva_data['t']).flatten()
        
        # Create a mask for behaviors to keep
        keep_mask = np.zeros_like(states, dtype=bool)
        
        # Find continuous segments of behavior again
        segments = []
        current_state = None
        start_idx = 0
        
        for i, state in enumerate(states):
            if state != current_state:
                # Record the previous segment if there was one
                if current_state is not None and i > start_idx:
                    segments.append({
                        'state': current_state,
                        'start_idx': start_idx,
                        'end_idx': i - 1,
                        'duration': times[i - 1] - times[start_idx]
                    })
                current_state = state
                start_idx = i
        
        # Add the final segment
        if current_state is not None and len(states) > start_idx:
            segments.append({
                'state': current_state,
                'start_idx': start_idx,
                'end_idx': len(states) - 1,
                'duration': times[-1] - times[start_idx]
            })
        
        # Check each segment against the mean duration
        for segment in segments:
            state = segment['state']
            if state in state_to_behavior:
                behavior_key = state_to_behavior[state]
                
                # Skip if we don't have a mean for this behavior
                if behavior_key not in mean_durations or mean_durations[behavior_key] == 0:
                    continue
                
                # Keep this segment if duration is above mean
                if segment['duration'] > mean_durations[behavior_key]:
                    start = segment['start_idx']
                    end = segment['end_idx']
                    keep_mask[start:end+1] = True
        
        # Set states we don't want to keep to 0 (which will be white in the plot)
        filtered_states = np.copy(states)
        filtered_states[~keep_mask] = 0
        
        # Update the filtered data
        filtered_data[larva_id]['global_state_large_state'] = filtered_states
    
    # Use the standard plotting function with our filtered data
    # Set show_separate_totals=False to show large and small behaviors on same row
    behavior_matrix = plot_global_behavior_matrix(filtered_data, show_separate_totals=False)
    
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
def compute_orientation_tail_to_neck(x_tail, y_tail, x_neck, y_neck):
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
def analyze_turn_rate_by_orientation_true(
    trx_data, bin_width=10, show_plot=True, ax=None, sigma=2, min_turn_amplitude=10
):
    """
    Analyze turn rate (turns/sec) vs orientation using tail-to-neck orientation.
    A turn is defined as a cast (state==2 or state==1.5) that results in an orientation change >= min_turn_amplitude (deg).
    The orientation at which the turn occurs is the *initial* orientation at cast onset.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    def get_orientation_array(larva_data):
        required_keys = ['x_tail', 'y_tail', 'x_neck', 'y_neck']
        if all(k in larva_data for k in required_keys):
            x_tail = np.array(larva_data['x_tail']).flatten()
            y_tail = np.array(larva_data['y_tail']).flatten()
            x_neck = np.array(larva_data['x_neck']).flatten()
            y_neck = np.array(larva_data['y_neck']).flatten()
            min_len = min(len(x_tail), len(y_tail), len(x_neck), len(y_neck))
            return compute_orientation_tail_to_neck(
                x_tail[:min_len], y_tail[:min_len], x_neck[:min_len], y_neck[:min_len]
            )
        return None

    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        created_fig = False

    if isinstance(trx_data, dict) and 'data' in trx_data:
        data_to_process = trx_data['data']
    else:
        data_to_process = trx_data

    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    turn_counts = np.zeros_like(bin_centers, dtype=float)
    total_time = np.zeros_like(bin_centers, dtype=float)

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_large_state' not in larva_data or 't' not in larva_data:
                continue
            states = np.array(larva_data['global_state_large_state']).flatten()
            t = np.array(larva_data['t']).flatten()
            orientations = get_orientation_array(larva_data)
            if orientations is None:
                continue
            min_len = min(len(states), len(t), len(orientations))
            states = states[:min_len]
            t = t[:min_len]
            orientations = orientations[:min_len]

            # Find all cast (state==2 or state==1.5) segments
            i = 0
            while i < len(states):
                if states[i] == 2 or states[i] == 1.5:
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
                            # Bin by initial orientation
                            for j in range(len(bin_centers)):
                                if bins[j] <= orient_start < bins[j+1]:
                                    turn_counts[j] += 1
                                    break
                else:
                    i += 1

            # Bin total time spent in each orientation (all frames)
            for j in range(len(bin_centers)):
                bin_mask = (orientations >= bins[j]) & (orientations < bins[j+1])
                t_bin = t[bin_mask]
                if len(t_bin) > 1:
                    total_time[j] += np.sum(np.diff(t_bin))
                elif len(t_bin) == 1 and len(t) > 1:
                    dt = np.median(np.diff(t))
                    total_time[j] += dt

        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    # Compute turn rate per second
    turn_rate_per_sec = np.zeros_like(bin_centers, dtype=float)
    for j in range(len(bin_centers)):
        if total_time[j] > 0:
            turn_rate_per_sec[j] = turn_counts[j] / total_time[j]

    smoothed_rates = gaussian_filter1d(turn_rate_per_sec, sigma=sigma)

    if ax is not None:
        ax.plot(bin_centers, smoothed_rates, color='purple', linewidth=2)
        ax.set_xlabel('Orientation (°)')
        ax.set_ylabel('Turn Rate (turns/sec)')
        ax.set_xlim(-180, 180)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=180, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=-180, color='black', linestyle='--', alpha=0.5)
        if created_fig and show_plot:
            plt.tight_layout()
            plt.show()

    return {
        'bin_centers': bin_centers,
        'turn_counts': turn_counts,
        'total_time': total_time,
        'turn_rate_per_sec': turn_rate_per_sec,
        'smoothed_rates': smoothed_rates
    }

def analyze_run_orientations_all(experiments_data, bin_width=10, show_plot=True, ax=None, sigma=2):
    """Analyze run orientations using tail-to-neck orientation definition."""
    if ax is None and show_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        created_fig = False

    all_run_orientations = []

    if isinstance(experiments_data, dict) and 'data' in experiments_data:
        data_to_process = experiments_data['data']
    else:
        data_to_process = experiments_data

    for larva_id, larva_data in data_to_process.items():
        try:
            if 'global_state_large_state' not in larva_data:
                continue
            states = np.array(larva_data['global_state_large_state']).flatten()
            runs = np.logical_or(states == 1, states == 0.5)
            if runs.sum() == 0:
                continue
            orientations = get_orientation_array(larva_data)
            if orientations is None:
                continue
            min_len = min(len(orientations), len(runs))
            orientations = orientations[:min_len]
            runs = runs[:min_len]
            all_run_orientations.extend(orientations[runs])
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")

    if not all_run_orientations:
        ax.text(0.5, 0.5, "No run orientation data available", ha='center', va='center', transform=ax.transAxes)
        return {'orientations': [], 'hist': np.array([]), 'bin_centers': np.array([])}

    bins = np.arange(-180, 181, bin_width)
    hist, bin_edges = np.histogram(all_run_orientations, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

    ax.plot(bin_centers, smoothed_hist, color='black', linewidth=2)
    ax.set_xlabel('Orientation (°)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(-180, 180)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=180, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-180, color='black', linestyle='--', alpha=0.5)

    if created_fig and show_plot:
        plt.tight_layout()
        plt.show()

    return {'orientations': all_run_orientations, 'hist': hist, 'bin_centers': bin_centers, 'smoothed_hist': smoothed_hist}




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

def analyze_turn_rate_by_orientation_new(trx_data, larva_id=None, bin_width=10, 
                                    smooth_window=5, jump_threshold=15, 
                                    peak_threshold=5, peak_prominence=3):
    """
    Calculate turn rate as a function of orientation for large turns, small turns, and all turns,
    using peak detection for more accurate cast identification.
    
    Args:
        trx_data: Dictionary containing tracking data
        larva_id: Optional specific larva to analyze. If None, analyzes all larvae
        bin_width: Width of orientation bins in degrees
        smooth_window: Window size for smoothing angle data
        jump_threshold: Threshold for detecting orientation jumps in degrees/frame
        peak_threshold: Minimum height for a bend angle peak to be considered a cast
        peak_prominence: Minimum prominence for peak detection
        
    Returns:
        dict: Contains turn rates and orientation bins for large, small, and all turns
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    
    def get_orientations_and_states(larva_data):
        """Extract orientations, bend angles, and turn states for large and small turns."""
        # Calculate orientation
        x_center = np.array(larva_data['x_neck']).flatten()
        y_center = np.array(larva_data['y_neck']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        # Calculate orientations in degrees
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
        
        # Get upper-lower bend angle
        if 'angle_upper_lower_smooth_5' in larva_data:
            angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
            angle_upper_lower_deg = np.degrees(angle_upper_lower)
        else:
            # If bend angle data is missing, use zeros (this will rely on state labels only)
            angle_upper_lower_deg = np.zeros_like(orientations)
        
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
        
        return orientations, angle_upper_lower_deg, is_large_turn, is_small_turn, is_any_turn
    
    def detect_turn_peaks(orientations, bend_angles, is_turn, smooth_window, peak_threshold, peak_prominence):
        """Detect actual peaks in bend angles during turns."""
        # Apply smoothing to bend angles
        bend_angles_smooth = gaussian_filter1d(bend_angles, sigma=smooth_window/3.0)
        
        # Calculate angle derivative to find slope changes
        bend_angle_diff = np.diff(bend_angles_smooth)
        # Add zero at the beginning to match length
        bend_angle_diff = np.insert(bend_angle_diff, 0, 0)
        
        # Find peaks in absolute bend angles during turn segments
        peak_mask = np.zeros_like(is_turn, dtype=bool)
        
        # Find continuous segments of turns
        turn_segments = []
        in_turn = False
        turn_start = 0
        
        for i in range(len(is_turn)):
            if is_turn[i] and not in_turn:
                # Start of a new turn
                in_turn = True
                turn_start = i
            elif not is_turn[i] and in_turn:
                # End of a turn
                in_turn = False
                if i - turn_start >= 3:  # Require at least 3 frames
                    turn_segments.append((turn_start, i))
        
        # Handle case when still in turn at end of data
        if in_turn and len(is_turn) - turn_start >= 3:
            turn_segments.append((turn_start, len(is_turn)))
        
        # Process each turn segment to find peaks
        for start, end in turn_segments:
            segment_angles = bend_angles_smooth[start:end]
            
            # Find peaks in absolute bend angles
            abs_angles = np.abs(segment_angles)
            peak_indices, _ = find_peaks(
                abs_angles, 
                height=peak_threshold,
                prominence=peak_prominence,
                distance=3
            )
            
            # Mark the peaks in the original array
            for idx in peak_indices:
                if start + idx < len(peak_mask):
                    peak_mask[start + idx] = True
        
        # Return the orientations at peak times and the peak mask
        return orientations[peak_mask], peak_mask
    
    # Initialize storage for state-based and peak-based analyses
    all_orientations = []
    large_turn_states = []
    small_turn_states = []
    all_turn_states = []
    
    large_peak_orientations = []
    small_peak_orientations = []
    all_peak_orientations = []
    
    # Process data
    if larva_id is not None:
        # Single larva analysis
        larva_data = trx_data[larva_id]
        if 'data' in larva_data:
            larva_data = larva_data['data']
            
        orientations, bend_angles, is_large, is_small, is_any = get_orientations_and_states(larva_data)
        
        # Store state-based data
        all_orientations.extend(orientations)
        large_turn_states.extend(is_large)
        small_turn_states.extend(is_small)
        all_turn_states.extend(is_any)
        
        # Detect peaks for each turn type
        large_peaks, large_peak_mask = detect_turn_peaks(
            orientations, bend_angles, is_large, smooth_window, peak_threshold, peak_prominence)
        small_peaks, small_peak_mask = detect_turn_peaks(
            orientations, bend_angles, is_small, smooth_window, peak_threshold, peak_prominence)
        all_peaks, all_peak_mask = detect_turn_peaks(
            orientations, bend_angles, is_any, smooth_window, peak_threshold, peak_prominence)
        
        # Store peak-based orientations
        large_peak_orientations.extend(large_peaks)
        small_peak_orientations.extend(small_peaks)
        all_peak_orientations.extend(all_peaks)
        
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
            
        for larva_id, larva_data in data_to_process.items():
            try:
                if 'data' in larva_data:
                    larva_data = larva_data['data']
                    
                orientations, bend_angles, is_large, is_small, is_any = get_orientations_and_states(larva_data)
                
                # Store state-based data
                all_orientations.extend(orientations)
                large_turn_states.extend(is_large)
                small_turn_states.extend(is_small)
                all_turn_states.extend(is_any)
                
                # Detect peaks for each turn type
                large_peaks, large_peak_mask = detect_turn_peaks(
                    orientations, bend_angles, is_large, smooth_window, peak_threshold, peak_prominence)
                small_peaks, small_peak_mask = detect_turn_peaks(
                    orientations, bend_angles, is_small, smooth_window, peak_threshold, peak_prominence)
                all_peaks, all_peak_mask = detect_turn_peaks(
                    orientations, bend_angles, is_any, smooth_window, peak_threshold, peak_prominence)
                
                # Store peak-based orientations
                large_peak_orientations.extend(large_peaks)
                small_peak_orientations.extend(small_peaks)
                all_peak_orientations.extend(all_peaks)
            except Exception as e:
                print(f"Error processing larva {larva_id}: {str(e)}")
                continue
                
        title = f'Cast Probability (n={n_larvae})'
    
    # Convert to numpy arrays
    all_orientations = np.array(all_orientations)
    large_turn_states = np.array(large_turn_states)
    small_turn_states = np.array(small_turn_states)
    all_turn_states = np.array(all_turn_states)
    
    large_peak_orientations = np.array(large_peak_orientations)
    small_peak_orientations = np.array(small_peak_orientations)
    all_peak_orientations = np.array(all_peak_orientations)
    
    # Create orientation bins
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate turn rates for each bin and turn type (state-based)
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
    
    # Calculate peak-based distributions
    large_peak_distribution = np.zeros(len(bin_centers))
    small_peak_distribution = np.zeros(len(bin_centers))
    all_peak_distribution = np.zeros(len(bin_centers))
    
    # Count peaks in each bin
    for i in range(len(bins)-1):
        # For large casts
        if len(large_peak_orientations) > 0:
            mask = (large_peak_orientations >= bins[i]) & (large_peak_orientations < bins[i+1])
            large_peak_distribution[i] = np.sum(mask)
        
        # For small casts
        if len(small_peak_orientations) > 0:
            mask = (small_peak_orientations >= bins[i]) & (small_peak_orientations < bins[i+1])
            small_peak_distribution[i] = np.sum(mask)
        
        # For all casts
        if len(all_peak_orientations) > 0:
            mask = (all_peak_orientations >= bins[i]) & (all_peak_orientations < bins[i+1])
            all_peak_distribution[i] = np.sum(mask)
    
    # Normalize distributions if we have peaks
    if np.sum(large_peak_distribution) > 0:
        large_peak_distribution = large_peak_distribution / np.sum(large_peak_distribution)
    if np.sum(small_peak_distribution) > 0:
        small_peak_distribution = small_peak_distribution / np.sum(small_peak_distribution)
    if np.sum(all_peak_distribution) > 0:
        all_peak_distribution = all_peak_distribution / np.sum(all_peak_distribution)
    
    # Apply smoothing to both state-based rates and peak-based distributions
    large_turn_rates = np.array(large_turn_rates)
    small_turn_rates = np.array(small_turn_rates)
    all_turn_rates = np.array(all_turn_rates)
    
    state_large_smoothed = gaussian_filter1d(large_turn_rates, sigma=1)
    state_small_smoothed = gaussian_filter1d(small_turn_rates, sigma=1)
    state_all_smoothed = gaussian_filter1d(all_turn_rates, sigma=1)
    
    peak_large_smoothed = gaussian_filter1d(large_peak_distribution, sigma=1)
    peak_small_smoothed = gaussian_filter1d(small_peak_distribution, sigma=1)
    peak_all_smoothed = gaussian_filter1d(all_peak_distribution, sigma=1)
    
    # Create figure with two rows of subplots: state-based and peak-based
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Large turn peaks
    axes[0].plot(bin_centers, large_peak_distribution, 'k-', alpha=0.3, linewidth=1)
    axes[0].plot(bin_centers, peak_large_smoothed, 'r-', linewidth=2)
    axes[0].set_xlabel('Orientation (°)')
    axes[0].set_ylabel('Relative frequency')
    axes[0].set_xlim(-180, 180)
    axes[0].set_title(f'Large Casts (n={len(large_peak_orientations)})')
    
    # Plot 2: Small turn peaks
    axes[1].plot(bin_centers, small_peak_distribution, 'k-', alpha=0.3, linewidth=1)
    axes[1].plot(bin_centers, peak_small_smoothed, 'b-', linewidth=2)
    axes[1].set_xlabel('Orientation (°)')
    axes[1].set_ylabel('Relative frequency')
    axes[1].set_xlim(-180, 180)
    axes[1].set_title(f'Small Casts (n={len(small_peak_orientations)})')
    
    # Plot 3: All turn peaks
    axes[2].plot(bin_centers, all_peak_distribution, 'k-', alpha=0.3, linewidth=1)
    axes[2].plot(bin_centers, peak_all_smoothed, 'g-', linewidth=2)
    axes[2].set_xlabel('Orientation (°)')
    axes[2].set_ylabel('Relative frequency')
    axes[2].set_xlim(-180, 180)
    axes[2].set_title(f'All Casts (n={len(all_peak_orientations)})')
    
    # Add reference lines for upstream/downstream orientation
    for ax in axes:
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, label='Downstream')
        ax.axvline(x=180, color='gray', linestyle='--', alpha=0.3, label='Upstream')
        ax.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    
    # Add super title
    plt.suptitle(f'Peak-Based Cast Probability (n={n_larvae} larvae)', fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the suptitle
    plt.show()
    
    # Create combined comparison plot
    plt.figure(figsize=(6, 4))
    
    if len(large_peak_orientations) > 0:
        plt.plot(bin_centers, peak_large_smoothed, 'r-', linewidth=2, label='Large casts')
    
    if len(small_peak_orientations) > 0:
        plt.plot(bin_centers, peak_small_smoothed, 'b-', linewidth=2, label='Small casts')
    
    if len(all_peak_orientations) > 0:
        plt.plot(bin_centers, peak_all_smoothed, 'g-', linewidth=2, label='All casts')
    
    plt.xlabel('Orientation (°)')
    plt.ylabel('Relative frequency')
    plt.xlim(-180, 180)
    plt.title(f'Peak-Based Cast Distribution (n={n_larvae} larvae)')
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    # Calculate counts and frequencies for statistical comparison
    large_turn_count = np.sum(large_turn_states)
    small_turn_count = np.sum(small_turn_states)
    all_turn_count = np.sum(all_turn_states)
    
    return {
        # State-based analysis
        'orientations': all_orientations,
        'large_turn_states': large_turn_states,
        'small_turn_states': small_turn_states,
        'all_turn_states': all_turn_states,
        'bin_centers': bin_centers,
        'large_turn_rates': large_turn_rates,
        'small_turn_rates': small_turn_rates,
        'all_turn_rates': all_turn_rates,
        'state_large_smoothed': state_large_smoothed,
        'state_small_smoothed': state_small_smoothed,
        'state_all_smoothed': state_all_smoothed,
        
        # Peak-based analysis
        'large_peak_orientations': large_peak_orientations,
        'small_peak_orientations': small_peak_orientations,
        'all_peak_orientations': all_peak_orientations,
        'large_peak_distribution': large_peak_distribution,
        'small_peak_distribution': small_peak_distribution,
        'all_peak_distribution': all_peak_distribution,
        'peak_large_smoothed': peak_large_smoothed,
        'peak_small_smoothed': peak_small_smoothed,
        'peak_all_smoothed': peak_all_smoothed,
        
        # Metadata
        'n_larvae': n_larvae,
        'large_turn_count': int(large_turn_count),
        'small_turn_count': int(small_turn_count),
        'all_turn_count': int(all_turn_count),
        'n_large_peaks': len(large_peak_orientations),
        'n_small_peaks': len(small_peak_orientations),
        'n_all_peaks': len(all_peak_orientations)
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
            orientations = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
            
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


def analyze_perpendicular_cast_directions(trx_data, angle_width=5, min_frame=3, basepath=None):
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
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.lines import Line2D
    
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
            
            # Track all casts separately instead of combining during the loop
            all_casts_data = []
            
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
                    if len(cast_sequence) < min_frame:  # Need at least min_frame frames
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
                    
                    # Store this cast for the "all" category
                    all_casts_data.append({
                        'type': 'large',
                        'direction': cast_direction,
                        'init_angle': init_orientation,
                        'max_angle': max_cast_angle
                    })
                    
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
                        if len(cast_sequence) < min_frame:  # Need at least min_frame frames
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
                        
                        # Store this cast for the "all" category
                        all_casts_data.append({
                            'type': 'small',
                            'direction': cast_direction,
                            'init_angle': init_orientation,
                            'max_angle': max_cast_angle
                        })
                        
                    except (IndexError, ValueError):
                        continue
            
            # Now process the "all" category after both large and small casts have been analyzed
            for cast_data in all_casts_data:
                direction = cast_data['direction']
                total_counts['all'][direction] += 1
                larva_counts['all'][direction] += 1
            
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
    if basepath is not None:
        # Create filename with angle width and timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(basepath, f"perpendicular_cast_boxplot_angle{angle_width}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved boxplot to: {filepath}")
    
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
        #     # Save the figure if basepath is provided
        # if basepath is not None:
        #     # Create filename with angle width and timestamp
        #     from datetime import datetime
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filepath = os.path.join(basepath, f"perpendicular_cast_boxplot_angle{angle_width}_{timestamp}.svg")
        #     plt.savefig(filepath, format='svg', bbox_inches='tight')
        #     print(f"Saved boxplot to: {filepath}")
    
    
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


def analyze_perpendicular_cast_directions_new(trx_data, angle_width=5, min_frame=3,basepath=None):
    """
    Analyze whether casts are biased toward upstream or downstream when larvae are perpendicular to flow.
    Perpendicular is defined as orientation around ±90° within angle_width window.
    
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
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.lines import Line2D
    
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
                # Fall back to just large state if small_large_state isn't available
                cast_states = np.array(larva_data.get('cast', larva_data.get('global_state_large_state', []))).flatten()
                large_cast_mask = cast_states == 2  # Only large casts available
                small_cast_mask = np.zeros_like(large_cast_mask, dtype=bool)  # No small casts
                any_cast_mask = large_cast_mask
                
            if len(cast_states) == 0:
                continue
                
            # Find start frames for different cast types
            large_cast_starts = np.where((large_cast_mask[1:]) & (~large_cast_mask[:-1]))[0] + 1
            small_cast_starts = np.where((small_cast_mask[1:]) & (~small_cast_mask[:-1]))[0] + 1
            
            # If no cast starts were found, try looking for any frame in a cast state
            if len(large_cast_starts) == 0:
                large_cast_starts = np.where(large_cast_mask)[0]
            if len(small_cast_starts) == 0 and has_small_large_state:
                small_cast_starts = np.where(small_cast_mask)[0]
                
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
            
            # Store all casts from this larva to process together
            all_casts_data = []
            
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
                        
                    # Get maximum cast angle within a short window
                    end = min(start + 6, len(cast_angles))  # Look at first 6 frames of cast
                    if end <= start or end >= len(cast_angles):
                        continue
                        
                    cast_sequence = cast_angles[start:end]
                    if len(cast_sequence) < min_frame:  # Need at least min_frame frames
                        continue
                    
                    # Find frame with maximum deviation
                    angle_diffs = np.abs(cast_sequence - init_orientation)
                    max_deviation_idx = np.argmax(angle_diffs)
                    max_cast_angle = cast_sequence[max_deviation_idx]
                    
                    # Determine if cast is upstream or downstream
                    cast_direction = determine_cast_direction(init_orientation, max_cast_angle)
                    
                    # Store this cast data
                    all_casts_data.append({
                        'type': 'large',
                        'direction': cast_direction,
                        'init_angle': init_orientation,
                        'max_angle': max_cast_angle
                    })
                    
                except (IndexError, ValueError) as e:
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
                        if len(cast_sequence) < min_frame:  # Need at least min_frame frames
                            continue
                        
                        # Find frame with maximum deviation
                        angle_diffs = np.abs(cast_sequence - init_orientation)
                        max_deviation_idx = np.argmax(angle_diffs)
                        max_cast_angle = cast_sequence[max_deviation_idx]
                        
                        # Determine if cast is upstream or downstream
                        cast_direction = determine_cast_direction(init_orientation, max_cast_angle)
                        
                        # Store this cast data
                        all_casts_data.append({
                            'type': 'small',
                            'direction': cast_direction,
                            'init_angle': init_orientation,
                            'max_angle': max_cast_angle
                        })
                        
                    except (IndexError, ValueError) as e:
                        continue
            
            # Process all the casts we've collected
            for cast_data in all_casts_data:
                cast_type = cast_data['type']
                direction = cast_data['direction']
                
                # Update counts for this specific cast type
                total_counts[cast_type][direction] += 1
                larva_counts[cast_type][direction] += 1
                
                # Also update the 'all' category
                total_counts['all'][direction] += 1
                larva_counts['all'][direction] += 1
            
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
        if p_val is not None:
            # Get y positions
            y_max = max(max(data[0]) if len(data[0]) > 0 else 0, 
                        max(data[1]) if len(data[1]) > 0 else 0) + 0.05
            
            # Plot the line
            x1, x2 = 0, 1
            ax.plot([x1, x1, x2, x2], [y_max, y_max + 0.03, y_max + 0.03, y_max], lw=1.5, c='black')
            
            # Add stars based on significance level
            sig_str = ""
            if p_val < 0.05:
                sig_str = "*" 
                if p_val < 0.01:
                    sig_str = "**"
                    if p_val < 0.001:
                        sig_str = "***"
            
            ax.text((x1 + x2) * 0.5, y_max + 0.04, sig_str, ha='center', va='bottom', color='black', fontsize=14)
            
            # Add p-value with * indicator of significance
            sig_indicator = "*" if p_val < 0.05 else "n.s."
            ax.text((x1 + x2) * 0.5, y_max + 0.07, f'p = {p_val:.4f} ({sig_indicator})', ha='center', va='bottom', fontsize=8)
        
        # Add reference line for 0.5 probability (chance level)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        
        # Add raw counts as text on plot
        ax.text(0, 0.05, f'n = {total_counts[cast_type]["upstream"]}', ha='center', fontsize=8)
        ax.text(1, 0.05, f'n = {total_counts[cast_type]["downstream"]}', ha='center', fontsize=8)
        
        # Format plot
        if i == 0:
            ax.set_ylabel('Probability', fontsize=10)
        
        # Increase y-axis limit to make room for p-values and significance bars
        ax.set_ylim(0, 1.2)  # Increased from 1.1 to 1.2
        
        # Add type label as title
        type_labels = {
            'large': 'Large Casts',
            'small': 'Small Casts',
            'all': 'All Casts'
        }
        
        ax.set_title(f'{type_labels[cast_type]}\n(n={len(larva_probabilities[cast_type]["upstream"])} larvae)', 
                   fontsize=10)
        
        # Calculate and display median instead of mean
        if len(data[0]) > 0:
            upstream_median = np.median(data[0])
            upstream_std = np.std(data[0])
            ax.text(0, upstream_median + 0.03, f"Median: {upstream_median:.2f}", 
                  ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(0, upstream_median - 0.12, f"Std: {upstream_std:.2f}", 
                  ha='center', va='bottom', fontsize=8)
        
        if len(data[1]) > 0:
            downstream_median = np.median(data[1])
            downstream_std = np.std(data[1])
            ax.text(1, downstream_median + 0.03, f"Median: {downstream_median:.2f}", 
                  ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(1, downstream_median - 0.12, f"Std: {downstream_std:.2f}", 
                  ha='center', va='bottom', fontsize=8)
    
    # Add super title
    fig.suptitle(f'Cast Direction When Perpendicular to Flow (±{angle_width}°)', 
              fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.3)  # Make room for the suptitle
    plt.show()

    if basepath is not None:
        # Create filename with angle width and timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(basepath, f"perpendicular_cast_boxplot_angle{angle_width}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved boxplot to: {filepath}")
    
    # 2. Second figure: per-larva bar plots for cast types
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

        
    
    # Print summary of statistical results
    print("\nSummary of Cast Direction Analysis:")
    print(f"Analyzed {larvae_processed} larvae with cast events during perpendicular orientation (±{angle_width}°)")
    
    for cast_type in ['all', 'large', 'small']:
        if cast_type in plot_types:
            print(f"\n{cast_type.capitalize()} Casts:")
            n_up = total_counts[cast_type]['upstream']
            n_down = total_counts[cast_type]['downstream']
            total = n_up + n_down
            
            if total > 0:
                up_pct = (n_up / total) * 100
                down_pct = (n_down / total) * 100
                
                print(f"  Total: {total} casts from {len(larva_probabilities[cast_type]['upstream'])} larvae")
                print(f"  Upstream: {n_up} ({up_pct:.1f}%), Downstream: {n_down} ({down_pct:.1f}%)")
                
                # Add mean and standard deviation information
                up_probs = np.array(larva_probabilities[cast_type]['upstream'])
                down_probs = np.array(larva_probabilities[cast_type]['downstream'])
                
                if len(up_probs) > 0:
                    up_mean = np.mean(up_probs)
                    up_std = np.std(up_probs)
                    down_mean = np.mean(down_probs)
                    down_std = np.std(down_probs)
                    
                    print(f"  Upstream mean: {up_mean:.3f} ± {up_std:.3f} (std)")
                    print(f"  Downstream mean: {down_mean:.3f} ± {down_std:.3f} (std)")
                    
                    # Also print median values for comparison
                    up_median = np.median(up_probs)
                    down_median = np.median(down_probs)
                    print(f"  Upstream median: {up_median:.3f}")
                    print(f"  Downstream median: {down_median:.3f}")
                
                # Chi-square test result
                chi2_result = stats_results[cast_type]['chi2_result']
                if chi2_result is not None:
                    chi2, p_val = chi2_result
                    sig_str = "significant" if p_val < 0.05 else "not significant"
                    print(f"  Chi-Square Test: χ² = {chi2:.2f}, p = {p_val:.4f} ({sig_str})")
                
                # T-test result
                ttest_result = stats_results[cast_type]['ttest_result']
                if ttest_result is not None:
                    t_stat, p_val = ttest_result
                    sig_str = "significant" if p_val < 0.05 else "not significant"
                    print(f"  Paired T-Test: t = {t_stat:.2f}, p = {p_val:.4f} ({sig_str})")
    
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

def plot_filtered_angles(trx_data, larva_id=None, smooth_window=5, jump_threshold=15):
    """
    Plot orientation and upper-lower angles with intelligent filtering and behavior state highlighting.
    Classifies body bends as upstream or downstream based on orientation.
    
    Parameters:
    -----------
    trx_data : dict
        The tracking data dictionary
    larva_id : str or int, optional
        ID of specific larva to analyze, if None, selects a random larva
    smooth_window : int
        Window size for smoothing (default=5)
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame (default=15)
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from matplotlib.patches import Patch
    
    # Define behavior color scheme
    behavior_colors = {
        1: [0.0, 0.0, 0.0],  # Run/Crawl
        2: [1.0, 0.0, 0.0],  # Cast/Bend
        3: [0.0, 1.0, 0.0],  # Stop
        4: [0.0, 0.0, 1.0],  # Hunch
        5: [1.0, 0.5, 0.0],  # Backup
        6: [0.5, 0.0, 0.5],  # Roll
        7: [0.7, 0.7, 0.7]   # Small Actions
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
    
    # Select larva if not specified
    if larva_id is None:
        larva_ids = list(trx_data.keys())
        larva_id = np.random.choice(larva_ids)
    
    # Extract larva data
    larva_data = trx_data[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract required data fields
    time = np.array(larva_data['t']).flatten()
    states = np.array(larva_data['global_state_large_state']).flatten()
    
    # Get orientation angles (calculate or extract)
    if 'orientation_angle' in larva_data:
        orientation_angles = np.array(larva_data['orientation_angle']).flatten()
    else:
        # Calculate orientation from tail to center (negative x-axis is 0 degrees)
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten() if np.array(larva_data['x_spine']).ndim > 1 else np.array(larva_data['x_spine']).flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten() if np.array(larva_data['y_spine']).ndim > 1 else np.array(larva_data['y_spine']).flatten()
        
        dx = x_center - x_tail
        dy = y_center - y_tail
        orientation_angles = np.degrees(np.arctan2(dy, -dx))  # -dx because 0° is negative x-axis
    
    # Get upper-lower bend angle
    angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Ensure all arrays have the same length
    min_length = min(len(time), len(orientation_angles), len(angle_upper_lower_deg), len(states))
    time = time[:min_length]
    orientation_angles = orientation_angles[:min_length]
    angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
    states = states[:min_length]
    
    # ==== FILTERING AND PROCESSING ====
    
    # 1. Orientation angle jump detection and correction
    orientation_raw = orientation_angles.copy()
    
    # Calculate derivative to detect jumps
    orientation_diff = np.abs(np.diff(orientation_angles))
    # Add zero at the beginning to match length
    orientation_diff = np.insert(orientation_diff, 0, 0)
    
    # Find jumps bigger than threshold
    jumps = orientation_diff > jump_threshold
    
    # Create masked array for plotting
    orientation_masked = np.ma.array(orientation_angles, mask=jumps)
    
    # Apply Gaussian smoothing to the filtered data
    # First interpolate masked values for smoothing
    orientation_interp = orientation_masked.filled(np.nan)
    mask = np.isnan(orientation_interp)
    
    # Only interpolate if we have valid data
    if np.sum(~mask) > 1:
        indices = np.arange(len(orientation_interp))
        valid_indices = indices[~mask]
        valid_values = orientation_interp[~mask]
        orientation_interp[mask] = np.interp(indices[mask], valid_indices, valid_values)
    
    # Apply smoothing
    orientation_smooth = gaussian_filter1d(orientation_interp, smooth_window/3.0)
    
    # 2. Intelligent peak detection for bend angle
    # Calculate slope changes
    bend_angle_diff = np.diff(angle_upper_lower_deg)
    # Add zero at the beginning to match length
    bend_angle_diff = np.insert(bend_angle_diff, 0, 0)
    
    # Smooth the bend angle
    bend_angle_smooth = gaussian_filter1d(angle_upper_lower_deg, smooth_window/3.0)
    
    # Find peaks with intelligent filtering:
    # - Ignore peaks that start from flat regions (slope near zero)
    # - Require minimum height and prominence
    pos_peaks, _ = find_peaks(
        bend_angle_smooth, 
        height=5,        
        prominence=3,        
        distance=5
    )
    
    neg_peaks, _ = find_peaks(
        -bend_angle_smooth, 
        height=5,        
        prominence=3,        
        distance=5
    )
    
    # Combine positive and negative peaks
    all_peaks = np.union1d(pos_peaks, neg_peaks)
    
    # Filter out peaks that start from flat regions
    slope_threshold = 0.5  # Define what's considered "flat"
    filtered_peaks = []
    for peak in all_peaks:
        if peak > 0 and abs(bend_angle_diff[peak-1]) > slope_threshold:
            filtered_peaks.append(peak)
    
    # Classify peaks as upstream or downstream based on orientation
    upstream_peaks = []
    downstream_peaks = []
    
    for peak in filtered_peaks:
        # Get orientation at time of peak (after smoothing)
        orientation = orientation_smooth[peak]
        bend_angle = bend_angle_smooth[peak]
        
        # Normalize orientation to -180 to 180 range
        while orientation > 180:
            orientation -= 360
        while orientation <= -180:
            orientation += 360
            
        # Classify based on orientation and bend direction
        if (orientation > 0 and orientation < 180):  # Right side (positive orientation)
            if bend_angle < 0:  # Negative bend is upstream
                upstream_peaks.append(peak)
            else:  # Positive bend is downstream
                downstream_peaks.append(peak)
        else:  # Left side (negative orientation)
            if bend_angle > 0:  # Positive bend is upstream
                upstream_peaks.append(peak)
            else:  # Negative bend is downstream
                downstream_peaks.append(peak)
    
    # ==== PLOTTING ====
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot orientation angle
    ax1.plot(time, orientation_raw, 'gray', alpha=0.3, label='Raw')
    ax1.plot(time, orientation_smooth, 'b-', linewidth=1.5, label='Smoothed')
    ax1.scatter(time[jumps], orientation_raw[jumps], color='r', s=20, alpha=0.5, label='Detected jumps')
    ax1.set_ylabel('Orientation angle (°)')
    ax1.set_title('Larva Orientation Angle (with jump detection)')
    ax1.legend(loc='upper right')
    
    # Add reference lines for orientation
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(y=180, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=-180, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.3)
    ax1.axhline(y=-90, color='gray', linestyle=':', alpha=0.3)
    
    # Plot bend angle (upper-lower)
    ax2.plot(time, angle_upper_lower_deg, 'gray', alpha=0.3, label='Raw')
    ax2.plot(time, bend_angle_smooth, 'r-', linewidth=1.5, label='Smoothed')
    
    # Plot classified peaks
    ax2.scatter(time[upstream_peaks], bend_angle_smooth[upstream_peaks], 
               color='blue', s=80, marker='^', label='Upstream Peaks')
    ax2.scatter(time[downstream_peaks], bend_angle_smooth[downstream_peaks], 
               color='green', s=80, marker='v', label='Downstream Peaks')
    
    ax2.set_ylabel('Upper-lower angle (°)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Body Bend Angle (with upstream/downstream detection)')
    ax2.legend(loc='upper right')
    
    # Add behavioral state highlighting to both plots
    for i in range(1, 8):
        if i in behavior_colors:
            mask = states == i
            if np.any(mask):
                ax1.fill_between(time, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                                where=mask, color=behavior_colors[i], alpha=0.2)
                ax2.fill_between(time, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                                where=mask, color=behavior_colors[i], alpha=0.2)
    
    # Add explanatory annotations
    ax1.text(time[0], 150, "0° = neg. x-axis\n±180° = pos. x-axis\n+90° = right\n-90° = left", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add upstream/downstream explanatory text
    ax2.text(time[0], ax2.get_ylim()[1]*0.8, 
             "Right side (+90°): -ve bend = upstream, +ve bend = downstream\n" +
             "Left side (-90°): +ve bend = upstream, -ve bend = downstream", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add behavior legend
    legend_elements = [
        Patch(facecolor=behavior_colors[i], alpha=0.3, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in behavior_colors
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              title='Behaviors', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0))
    
    # Format plot
    plt.tight_layout()
    plt.suptitle(f'Filtered Angle Analysis for Larva {larva_id}', fontsize=14, y=1.02)
    plt.subplots_adjust(bottom=0.15)  # Make room for the behavior legend
    
    plt.show()
    
    # Return results for further analysis
    return {
        'larva_id': larva_id,
        'time': time,
        'orientation_raw': orientation_raw,
        'orientation_smooth': orientation_smooth,
        'orientation_jumps': np.where(jumps)[0],
        'bend_angle_raw': angle_upper_lower_deg,
        'bend_angle_smooth': bend_angle_smooth,
        'all_peaks': filtered_peaks,
        'upstream_peaks': upstream_peaks,
        'downstream_peaks': downstream_peaks,
        'states': states
    }

def plot_filtered_angles_enhanced(trx_data, larva_id=None, smooth_window=5, jump_threshold=15,
                                 sweep_start_threshold=20.0, sweep_end_threshold=10.0,
                                 velocity_run_threshold=0.5):
    """
    Plot orientation and upper-lower angles with intelligent filtering and head sweep detection.
    
    Parameters:
    -----------
    trx_data : dict
        The tracking data dictionary
    larva_id : str or int, optional
        ID of specific larva to analyze, if None, selects a random larva
    smooth_window : int
        Window size for smoothing (default=5)
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame (default=15)
    sweep_start_threshold : float
        Threshold angle in degrees to identify the start of a head sweep (default=20.0)
    sweep_end_threshold : float
        Threshold angle in degrees to identify the end of a head sweep (default=10.0)
    velocity_run_threshold : float
        Threshold for emotion velocity norm to identify runs (default=0.5)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import random
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Define custom behavior color scheme
    custom_behavior_colors = {
        'run': [0.0, 0.5, 0.0],      # Green for runs
        'turn': [0.7, 0.7, 0.0],      # Yellow for turns
        'accepted_sweep': [0.0, 0.8, 0.3],  # Bright green for accepted head sweeps
        'rejected_sweep': [0.8, 0.0, 0.0]   # Red for rejected head sweeps
    }
    
    # Select larva if not specified
    if larva_id is None:
        larva_id = random.choice(list(trx_data.keys()))
        print(f"Selected random larva: {larva_id}")
    
    # Extract larva data
    larva_data = trx_data[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract required data fields
    time = np.array(larva_data['t']).flatten()
    states = np.array(larva_data['global_state_large_state']).flatten()
    
    # Extract coordinates for orientation calculation
    x_center = np.array(larva_data['x_center']).flatten()
    y_center = np.array(larva_data['y_center']).flatten()
    
    # Extract emotion velocity norm for run detection
    if 'motion_velocity_norm_smooth_5' in larva_data:
        velocity = np.array(larva_data['tail_velocity_norm_smooth_5']).flatten()
    else:
        # If emotion velocity isn't available, calculate velocity from center positions
        print("Warning: motion_velocity_norm_smooth_5 not found, calculating from positions")
        velocity = np.zeros_like(time)
        for i in range(1, len(time)):
            # Distance moved by center
            dx = x_center[i] - x_center[i-1]
            dy = y_center[i] - y_center[i-1]
            # Velocity magnitude
            dt = time[i] - time[i-1]
            if dt > 0:
                velocity[i] = np.sqrt(dx*dx + dy*dy) / dt
            else:
                velocity[i] = 0
        
        # Apply smoothing
        velocity = gaussian_filter1d(velocity, 5/3.0)
    
    # Handle different spine data shapes
    x_spine = np.array(larva_data['x_spine'])
    y_spine = np.array(larva_data['y_spine'])
    
    if x_spine.ndim > 1:  # 2D array
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
        x_neck = x_spine[0].flatten()  # First point is neck
        y_neck = y_spine[0].flatten()
    else:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
        x_neck = x_spine  # Can't differentiate neck in 1D case
        y_neck = y_spine
    
    # Calculate orientation angles using tail-to-neck vector
    orientation_angles = []
    for i in range(len(x_neck)):
        vector = np.array([x_neck[i] - x_tail[i], y_neck[i] - y_tail[i]])
        if np.linalg.norm(vector) == 0:
            orientation_angles.append(np.nan)
        else:
            angle_deg = np.degrees(np.arctan2(vector[1], -vector[0]))
            orientation_angles.append(angle_deg)
    
    orientation_angles = np.array(orientation_angles)
    
    # Get upper-lower bend angle
    angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Ensure all arrays have the same length
    min_length = min(len(time), len(orientation_angles), len(angle_upper_lower_deg), len(states), len(velocity))
    time = time[:min_length]
    orientation_angles = orientation_angles[:min_length]
    angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
    states = states[:min_length]
    velocity = velocity[:min_length]
    
    # ==== FILTERING AND PROCESSING ====
    
    # 1. Orientation angle jump detection and correction
    orientation_raw = orientation_angles.copy()
    
    # Calculate derivative to detect jumps
    orientation_diff = np.abs(np.diff(orientation_angles))
    # Add zero at the beginning to match length
    orientation_diff = np.insert(orientation_diff, 0, 0)
    
    # Find jumps bigger than threshold
    jumps = orientation_diff > jump_threshold
    
    # Create masked array for plotting
    orientation_masked = np.ma.array(orientation_angles, mask=jumps)
    
    # Apply Gaussian smoothing to the filtered data
    # First interpolate masked values for smoothing
    orientation_interp = orientation_masked.filled(np.nan)
    mask = np.isnan(orientation_interp)
    
    # Only interpolate if we have valid data
    if np.sum(~mask) > 1:
        orientation_interp[mask] = np.interp(
            np.flatnonzero(mask), 
            np.flatnonzero(~mask), 
            orientation_interp[~mask]
        )
    
    # Apply smoothing
    orientation_smooth = gaussian_filter1d(orientation_interp, smooth_window/3.0)
    
    # Apply smoothing to bend angle
    bend_angle_smooth = gaussian_filter1d(angle_upper_lower_deg, smooth_window/3.0)
    
    # 2. Determine run vs turn segments based on emotion velocity
    # A run is when emotion velocity exceeds the threshold
    is_run = velocity > velocity_run_threshold
    
    # 3. Find run and turn segments
    run_segments = []
    turn_segments = []
    
    # Define minimum run and turn length (to avoid brief fluctuations)
    min_run_length = 3  # frames
    min_turn_length = 3  # frames
    
    # Process runs with minimum duration
    in_run = False
    run_start = 0
    for i in range(min_length):
        if is_run[i] and not in_run:
            # Potential start of run
            run_start = i
            in_run = True
        elif not is_run[i] and in_run:
            # End of run
            if i - run_start >= min_run_length:
                run_segments.append((run_start, i))
            in_run = False
    
    # Handle case when still in run at end of data
    if in_run and min_length - run_start >= min_run_length:
        run_segments.append((run_start, min_length-1))
    
    # Calculate turn segments as gaps between runs
    if len(run_segments) > 0:
        # If first run doesn't start at beginning, add initial turn
        if run_segments[0][0] > 0:
            turn_segments.append((0, run_segments[0][0]))
            
        # Add turns between runs
        for i in range(len(run_segments)-1):
            # Only add turn if it's long enough
            turn_start = run_segments[i][1]
            turn_end = run_segments[i+1][0]
            if turn_end - turn_start >= min_turn_length:
                turn_segments.append((turn_start, turn_end))
            
        # If last run doesn't end at end, add final turn
        if run_segments[-1][1] < min_length-1:
            turn_segments.append((run_segments[-1][1], min_length-1))
    else:
        # If no runs at all, entire sequence is a turn
        turn_segments.append((0, min_length-1))
    
    # 4. Find head sweeps during turns using the customizable thresholds
    head_sweep_segments = []
    
    # Check each turn segment for head sweeps
    for turn_start, turn_end in turn_segments:
        in_head_sweep = False
        head_sweep_start = turn_start
        head_sweep_sign = 0  # Direction of bend: +1 for positive, -1 for negative
        
        for i in range(turn_start, turn_end+1):
            bend_angle = bend_angle_smooth[i]
            abs_bend = abs(bend_angle)
            
            if not in_head_sweep and abs_bend > sweep_start_threshold:
                # Start a new head sweep
                in_head_sweep = True
                head_sweep_start = i
                head_sweep_sign = 1 if bend_angle > 0 else -1
            
            elif in_head_sweep:
                # Check if head sweep should end
                if (abs_bend < sweep_end_threshold or 
                    (bend_angle * head_sweep_sign < 0) or  # Sign changed
                    (i == turn_end)):  # Turn ended
                    
                    # Head sweep ended, save segment
                    head_sweep_segments.append((head_sweep_start, i, turn_start, turn_end))
                    in_head_sweep = False
    
    # 5. Classify head sweeps as accepted or rejected
    accepted_sweeps = []
    rejected_sweeps = []
    
    for sweep_start, sweep_end, turn_start, turn_end in head_sweep_segments:
        # A head sweep is accepted if it begins a new run
        # This means the sweep is the last one in the turn
        is_last_in_turn = True
        
        # Look if there are other sweeps in the same turn that start after this one
        for other_start, other_end, other_turn_start, other_turn_end in head_sweep_segments:
            if (other_turn_start == turn_start and other_start > sweep_start):
                is_last_in_turn = False
                break
        
        # Check if the turn is immediately followed by a run
        followed_by_run = False
        for run_start, run_end in run_segments:
            if run_start == turn_end:
                followed_by_run = True
                break
        
        # Accepted if last sweep in turn and turn is followed by a run
        if is_last_in_turn and followed_by_run:
            accepted_sweeps.append((sweep_start, sweep_end))
        else:
            rejected_sweeps.append((sweep_start, sweep_end))
    
    # ==== PLOTTING ====
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot orientation angle
    ax1.plot(time, orientation_raw, 'gray', alpha=0.3, label='Raw')
    ax1.plot(time, orientation_smooth, 'b-', linewidth=1.5, label='Smoothed')
    ax1.scatter(time[jumps], orientation_raw[jumps], color='r', s=20, alpha=0.5, label='Detected jumps')
    ax1.set_ylabel('Orientation angle (°)')
    ax1.set_title('Larva Orientation Angle (Tail-to-Neck)')
    ax1.legend(loc='upper right')
    
    # Add reference lines for orientation
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(y=180, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=-180, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.3)
    ax1.axhline(y=-90, color='gray', linestyle=':', alpha=0.3)
    
    # Plot bend angle with original behavior and emotion velocity
    ax2.plot(time, angle_upper_lower_deg, 'gray', alpha=0.3, label='Raw bend')
    ax2.plot(time, bend_angle_smooth, 'r-', linewidth=1.5, label='Smoothed bend')
    
    # Plot emotion velocity on second y-axis
    ax2b = ax2.twinx()
    ax2b.plot(time, velocity, 'b-', linewidth=1.0, alpha=0.6, label='Emotion velocity')
    ax2b.axhline(y=velocity_run_threshold, color='b', linestyle='--', alpha=0.5, 
                label=f'Run threshold: {velocity_run_threshold}')
    ax2b.set_ylabel('Emotion velocity', color='b')
    ax2b.tick_params(axis='y', labelcolor='b')
    
    # Add legend for both lines
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_ylabel('Body bend angle (°)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Body Bend Angle and Emotion Velocity')
    
    # Plot bend angle with custom behavior classification for third plot
    ax3.plot(time, bend_angle_smooth, 'k-', linewidth=1.0)
    ax3.set_ylabel('Body bend angle (°)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Body Bend Angle with Custom Run/Turn Classification')
    
    # Add threshold reference lines to third plot
    ax3.axhline(y=sweep_start_threshold, color='blue', linestyle='-', alpha=0.5)
    ax3.axhline(y=-sweep_start_threshold, color='blue', linestyle='-', alpha=0.5)
    ax3.axhline(y=sweep_end_threshold, color='cyan', linestyle='--', alpha=0.5)
    ax3.axhline(y=-sweep_end_threshold, color='cyan', linestyle='--', alpha=0.5)
    
    # Highlight run segments
    for run_start, run_end in run_segments:
        ax3.axvspan(time[run_start], time[run_end], 
                   color=custom_behavior_colors['run'], alpha=0.3)
    
    # Highlight turn segments
    for turn_start, turn_end in turn_segments:
        ax3.axvspan(time[turn_start], time[turn_end], 
                   color=custom_behavior_colors['turn'], alpha=0.3)
    
    # Highlight accepted head sweeps with higher alpha to make them stand out
    for sweep_start, sweep_end in accepted_sweeps:
        ax3.axvspan(time[sweep_start], time[sweep_end],
                   color=custom_behavior_colors['accepted_sweep'], alpha=0.7)
    
    # Highlight rejected head sweeps with higher alpha to make them stand out
    for sweep_start, sweep_end in rejected_sweeps:
        ax3.axvspan(time[sweep_start], time[sweep_end],
                   color=custom_behavior_colors['rejected_sweep'], alpha=0.7)
    
    # Add explanatory annotation
    ax1.text(time[0], 150, "0° = neg. x-axis\n±180° = pos. x-axis\n+90° = right\n-90° = left", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add custom behavior legend
    custom_legend_elements = [
        Patch(facecolor=custom_behavior_colors['run'], alpha=0.3, 
              edgecolor='none', label='Run (emotion velocity > threshold)'),
        Patch(facecolor=custom_behavior_colors['turn'], alpha=0.3, 
              edgecolor='none', label='Turn (between runs)'),
        Patch(facecolor=custom_behavior_colors['accepted_sweep'], alpha=0.7, 
              edgecolor='none', label='Accepted head sweep'),
        Patch(facecolor=custom_behavior_colors['rejected_sweep'], alpha=0.7, 
              edgecolor='none', label='Rejected head sweep'),
        Line2D([0], [0], color='blue', linestyle='-', alpha=0.5, 
              label=f'Sweep start threshold (±{sweep_start_threshold}°)'),
        Line2D([0], [0], color='cyan', linestyle='--', alpha=0.5, 
              label=f'Sweep end threshold (±{sweep_end_threshold}°)'),
        Line2D([0], [0], color='b', linestyle='--', alpha=0.5, 
              label=f'Run velocity threshold ({velocity_run_threshold})')
    ]
    
    # Add threshold legend
    fig.legend(handles=custom_legend_elements, loc='lower center', 
              ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0))
    
    # Format plot
    plt.tight_layout()
    plt.suptitle(f'Custom Run/Turn Analysis for Larva {larva_id}', fontsize=14, y=1.02)
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    
    plt.show()
    
    # Return results for further analysis
    return {
        'larva_id': larva_id,
        'time': time,
        'orientation_smooth': orientation_smooth,
        'bend_angle_smooth': bend_angle_smooth,
        'emotion_velocity': velocity,
        'run_segments': run_segments,
        'turn_segments': turn_segments,
        'accepted_sweeps': accepted_sweeps,
        'rejected_sweeps': rejected_sweeps
    }

def plot_combined_behavioral_analysis(trx_data, larva_id=None, smooth_window=5, jump_threshold=15,
                                  sweep_start_threshold=20.0, sweep_end_threshold=10.0,
                                  velocity_run_threshold=0.5):
    """
    Plot orientation and upper-lower angles with both behavioral state highlighting 
    and accepted/rejected head sweep detection.
    
    Parameters:
    -----------
    trx_data : dict
        The tracking data dictionary
    larva_id : str or int, optional
        ID of specific larva to analyze, if None, selects a random larva
    smooth_window : int
        Window size for smoothing (default=5)
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame (default=15)
    sweep_start_threshold : float
        Threshold angle in degrees to identify the start of a head sweep (default=20.0)
    sweep_end_threshold : float
        Threshold angle in degrees to identify the end of a head sweep (default=10.0)
    velocity_run_threshold : float
        Threshold for motion velocity norm to identify runs (default=0.5)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    import random
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
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
    
    # Define custom head sweep color scheme
    sweep_colors = {
        'run': [0.0, 0.5, 0.0],           # Green for runs
        'turn': [0.7, 0.7, 0.0],          # Yellow for turns
        'accepted_sweep': [0.0, 0.8, 0.3], # Bright green for accepted head sweeps
        'rejected_sweep': [0.8, 0.0, 0.0]  # Red for rejected head sweeps
    }
    
    # Select larva if not specified
    if larva_id is None:
        larva_id = random.choice(list(trx_data.keys()))
        print(f"Selected random larva: {larva_id}")
    
    # Extract larva data
    larva_data = trx_data[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract required data fields
    time = np.array(larva_data['t']).flatten()
    states = np.array(larva_data['global_state_large_state']).flatten()
    
    # Extract coordinates for orientation calculation
    x_center = np.array(larva_data['x_center']).flatten()
    y_center = np.array(larva_data['y_center']).flatten()
    
    # Extract motion velocity norm for run detection
    if 'motion_velocity_norm_smooth_5' in larva_data:
        velocity = np.array(larva_data['motion_velocity_norm_smooth_5']).flatten()
    elif 'tail_velocity_norm_smooth_5' in larva_data:
        velocity = np.array(larva_data['tail_velocity_norm_smooth_5']).flatten()
    else:
        # If motion velocity isn't available, calculate velocity from center positions
        print("Warning: motion_velocity_norm_smooth_5 not found, calculating from positions")
        velocity = np.zeros_like(time)
        for i in range(1, len(time)):
            # Distance moved by center
            dx = x_center[i] - x_center[i-1]
            dy = y_center[i] - y_center[i-1]
            # Velocity magnitude
            dt = time[i] - time[i-1]
            if dt > 0:
                velocity[i] = np.sqrt(dx*dx + dy*dy) / dt
            else:
                velocity[i] = 0
        
        # Apply smoothing
        velocity = gaussian_filter1d(velocity, 5/3.0)
    
    # Handle different spine data shapes
    x_spine = np.array(larva_data['x_spine'])
    y_spine = np.array(larva_data['y_spine'])
    
    if x_spine.ndim > 1:  # 2D array
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
        x_neck = x_spine[0].flatten()  # First point is neck
        y_neck = y_spine[0].flatten()
    else:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
        x_neck = x_spine  # Can't differentiate neck in 1D case
        y_neck = y_spine
    
    # Calculate orientation angles
    orientation_angles = []
    for i in range(len(x_center)):
        vector = np.array([x_center[i] - x_tail[i], y_center[i] - y_tail[i]])
        if np.linalg.norm(vector) == 0:
            orientation_angles.append(np.nan)
        else:
            angle_deg = np.degrees(np.arctan2(vector[1], -vector[0]))
            orientation_angles.append(angle_deg)
    
    orientation_angles = np.array(orientation_angles)
    
    # Get upper-lower bend angle
    angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Ensure all arrays have the same length
    min_length = min(len(time), len(orientation_angles), len(angle_upper_lower_deg), len(states), len(velocity))
    time = time[:min_length]
    orientation_angles = orientation_angles[:min_length]
    angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
    states = states[:min_length]
    velocity = velocity[:min_length]
    
    # ==== FILTERING AND PROCESSING ====
    
    # 1. Orientation angle jump detection and correction
    orientation_raw = orientation_angles.copy()
    
    # Calculate derivative to detect jumps
    orientation_diff = np.abs(np.diff(orientation_angles))
    # Add zero at the beginning to match length
    orientation_diff = np.insert(orientation_diff, 0, 0)
    
    # Find jumps bigger than threshold
    jumps = orientation_diff > jump_threshold
    
    # Create masked array for plotting
    orientation_masked = np.ma.array(orientation_angles, mask=jumps)
    
    # Apply Gaussian smoothing to the filtered data
    # First interpolate masked values for smoothing
    orientation_interp = orientation_masked.filled(np.nan)
    mask = np.isnan(orientation_interp)
    
    # Only interpolate if we have valid data
    if np.sum(~mask) > 1:
        orientation_interp[mask] = np.interp(
            np.flatnonzero(mask), 
            np.flatnonzero(~mask), 
            orientation_interp[~mask]
        )
    
    # Apply smoothing
    orientation_smooth = gaussian_filter1d(orientation_interp, smooth_window/3.0)
    
    # 2. Intelligent peak detection for bend angle
    # Calculate slope changes
    bend_angle_diff = np.diff(angle_upper_lower_deg)
    # Add zero at the beginning to match length
    bend_angle_diff = np.insert(bend_angle_diff, 0, 0)
    
    # Smooth the bend angle
    bend_angle_smooth = gaussian_filter1d(angle_upper_lower_deg, smooth_window/3.0)
    
    # Find peaks with intelligent filtering
    bend_peaks, _ = find_peaks(
        np.abs(bend_angle_smooth), 
        height=5,       # Minimum peak height
        prominence=3,   # Minimum prominence
        distance=5      # Minimum distance between peaks
    )
    
    # Filter out peaks that start from flat regions
    slope_threshold = 0.5  # Define what's considered "flat"
    filtered_peaks = []
    for peak in bend_peaks:
        # Check at least 3 points before the peak for slope
        if peak > 3:
            # Calculate average slope before the peak
            pre_peak_slopes = np.abs(bend_angle_diff[peak-3:peak])
            avg_pre_slope = np.mean(pre_peak_slopes)
            if avg_pre_slope > slope_threshold:
                filtered_peaks.append(peak)
    
    # 3. Determine run vs turn segments based on motion velocity
    # A run is when motion velocity exceeds the threshold
    is_run = velocity > velocity_run_threshold
    
    # 4. Find run and turn segments
    run_segments = []
    turn_segments = []
    
    # Define minimum run and turn length (to avoid brief fluctuations)
    min_run_length = 3  # frames
    min_turn_length = 3  # frames
    
    # Process runs with minimum duration
    in_run = False
    run_start = 0
    for i in range(min_length):
        if is_run[i] and not in_run:
            # Potential start of run
            run_start = i
            in_run = True
        elif not is_run[i] and in_run:
            # End of run
            if i - run_start >= min_run_length:
                run_segments.append((run_start, i))
            in_run = False
    
    # Handle case when still in run at end of data
    if in_run and min_length - run_start >= min_run_length:
        run_segments.append((run_start, min_length-1))
    
    # Calculate turn segments as gaps between runs
    if len(run_segments) > 0:
        # If first run doesn't start at beginning, add initial turn
        if run_segments[0][0] > 0:
            turn_segments.append((0, run_segments[0][0]))
            
        # Add turns between runs
        for i in range(len(run_segments)-1):
            # Only add turn if it's long enough
            turn_start = run_segments[i][1]
            turn_end = run_segments[i+1][0]
            if turn_end - turn_start >= min_turn_length:
                turn_segments.append((turn_start, turn_end))
            
        # If last run doesn't end at end, add final turn
        if run_segments[-1][1] < min_length-1:
            turn_segments.append((run_segments[-1][1], min_length-1))
    else:
        # If no runs at all, entire sequence is a turn
        turn_segments.append((0, min_length-1))
    
    # 5. Find head sweeps during turns using the customizable thresholds
    head_sweep_segments = []
    
    # Check each turn segment for head sweeps
    for turn_start, turn_end in turn_segments:
        in_head_sweep = False
        head_sweep_start = turn_start
        head_sweep_sign = 0  # Direction of bend: +1 for positive, -1 for negative
        
        for i in range(turn_start, turn_end+1):
            bend_angle = bend_angle_smooth[i]
            abs_bend = abs(bend_angle)
            
            if not in_head_sweep and abs_bend > sweep_start_threshold:
                # Start a new head sweep
                in_head_sweep = True
                head_sweep_start = i
                head_sweep_sign = 1 if bend_angle > 0 else -1
            
            elif in_head_sweep:
                # Check if head sweep should end
                if (abs_bend < sweep_end_threshold or 
                    (bend_angle * head_sweep_sign < 0) or  # Sign changed
                    (i == turn_end)):  # Turn ended
                    
                    # Head sweep ended, save segment
                    head_sweep_segments.append((head_sweep_start, i, turn_start, turn_end))
                    in_head_sweep = False
    
    # 6. Classify head sweeps as accepted or rejected
    accepted_sweeps = []
    rejected_sweeps = []
    
    for sweep_start, sweep_end, turn_start, turn_end in head_sweep_segments:
        # A head sweep is accepted if it begins a new run
        # This means the sweep is the last one in the turn
        is_last_in_turn = True
        
        # Look if there are other sweeps in the same turn that start after this one
        for other_start, other_end, other_turn_start, other_turn_end in head_sweep_segments:
            if (other_turn_start == turn_start and other_start > sweep_start):
                is_last_in_turn = False
                break
        
        # Check if the turn is immediately followed by a run
        followed_by_run = False
        for run_start, run_end in run_segments:
            if run_start == turn_end:
                followed_by_run = True
                break
        
        # Accepted if last sweep in turn and turn is followed by a run
        if is_last_in_turn and followed_by_run:
            accepted_sweeps.append((sweep_start, sweep_end))
        else:
            rejected_sweeps.append((sweep_start, sweep_end))
    
    # ==== PLOTTING ====
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Orientation angle with jumps
    ax1.plot(time, orientation_raw, 'gray', alpha=0.3, label='Raw')
    ax1.plot(time, orientation_smooth, 'b-', linewidth=1.5, label='Smoothed')
    ax1.scatter(time[jumps], orientation_raw[jumps], color='r', s=20, alpha=0.5, label='Detected jumps')
    ax1.set_ylabel('Orientation angle (°)')
    ax1.set_title('Larva Orientation Angle (Tail-to-Center)')
    ax1.legend(loc='upper right')
    
    # Add reference lines for orientation
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Downstream
    ax1.axhline(y=180, color='gray', linestyle='--', alpha=0.3)  # Upstream
    ax1.axhline(y=-180, color='gray', linestyle='--', alpha=0.3)  # Upstream
    ax1.axhline(y=90, color='gray', linestyle=':', alpha=0.3)  # Right
    ax1.axhline(y=-90, color='gray', linestyle=':', alpha=0.3)  # Left
    
    # Add explanatory annotation
    ax1.text(time[0], 150, "0° = neg. x-axis\n±180° = pos. x-axis\n+90° = right\n-90° = left", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot 2: Bend angle with behavioral state highlighting
    ax2.plot(time, angle_upper_lower_deg, 'gray', alpha=0.3, label='Raw')
    ax2.plot(time, bend_angle_smooth, 'r-', linewidth=1.5, label='Smoothed')
    
    # Plot velocity on second y-axis
    ax2b = ax2.twinx()
    ax2b.plot(time, velocity, 'b-', linewidth=1.0, alpha=0.6, label='Motion velocity')
    ax2b.axhline(y=velocity_run_threshold, color='b', linestyle='--', alpha=0.5, 
                label=f'Run threshold: {velocity_run_threshold}')
    ax2b.set_ylabel('Motion velocity', color='b')
    ax2b.tick_params(axis='y', labelcolor='b')
    
    # Add legend for both lines
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax2.set_ylabel('Body bend angle (°)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Body Bend Angle with Behavioral States')
    
    # Add behavioral state highlighting to the second plot
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
        
        # Highlight each segment on the middle plot
        for start, end in behavior_segments:
            if end > start:  # Check for valid segment
                color = behavior_colors[i]
                alpha = 0.3
                ax2.axvspan(time[start], time[end], color=color, alpha=alpha)
    
    # Plot 3: Bend angle with head sweep classification
    ax3.plot(time, bend_angle_smooth, 'k-', linewidth=1.0)
    ax3.set_ylabel('Body bend angle (°)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Body Bend Angle with Head Sweep Classification')
    
    # Add threshold reference lines to third plot
    ax3.axhline(y=sweep_start_threshold, color='blue', linestyle='-', alpha=0.5)
    ax3.axhline(y=-sweep_start_threshold, color='blue', linestyle='-', alpha=0.5)
    ax3.axhline(y=sweep_end_threshold, color='cyan', linestyle='--', alpha=0.5)
    ax3.axhline(y=-sweep_end_threshold, color='cyan', linestyle='--', alpha=0.5)
    
    # Highlight run and turn segments
    for run_start, run_end in run_segments:
        ax3.axvspan(time[run_start], time[run_end], 
                   color=sweep_colors['run'], alpha=0.3)
    
    for turn_start, turn_end in turn_segments:
        ax3.axvspan(time[turn_start], time[turn_end], 
                   color=sweep_colors['turn'], alpha=0.3)
    
    # Highlight accepted head sweeps (on top of the run/turn highlighting)
    for sweep_start, sweep_end in accepted_sweeps:
        ax3.axvspan(time[sweep_start], time[sweep_end],
                   color=sweep_colors['accepted_sweep'], alpha=0.7)
    
    # Highlight rejected head sweeps (also on top of the run/turn highlighting)
    for sweep_start, sweep_end in rejected_sweeps:
        ax3.axvspan(time[sweep_start], time[sweep_end],
                   color=sweep_colors['rejected_sweep'], alpha=0.7)
    
    # Create legends for both sets of colors
    # Legend for behavioral states
    behavior_legend_elements = [
        Patch(facecolor=behavior_colors[i], alpha=0.3, 
              edgecolor='none', label=behavior_labels[i])
        for i in range(1, 8) if i in behavior_colors
    ]
    
    # Legend for head sweep classification
    sweep_legend_elements = [
        Patch(facecolor=sweep_colors['run'], alpha=0.3, 
              edgecolor='none', label='Run (velocity > threshold)'),
        Patch(facecolor=sweep_colors['turn'], alpha=0.3, 
              edgecolor='none', label='Turn (between runs)'),
        Patch(facecolor=sweep_colors['accepted_sweep'], alpha=0.7, 
              edgecolor='none', label='Accepted head sweep'),
        Patch(facecolor=sweep_colors['rejected_sweep'], alpha=0.7, 
              edgecolor='none', label='Rejected head sweep'),
        Line2D([0], [0], color='blue', linestyle='-', alpha=0.5, 
              label=f'Sweep start threshold (±{sweep_start_threshold}°)'),
        Line2D([0], [0], color='cyan', linestyle='--', alpha=0.5, 
              label=f'Sweep end threshold (±{sweep_end_threshold}°)')
    ]
    
    # Add the legends in a two-row layout at the bottom
    fig.legend(handles=behavior_legend_elements, loc='lower center', 
              title='Behavioral States', ncol=4, fontsize=9, 
              bbox_to_anchor=(0.5, 0.01))
    
    # Add the sweep classification legend below the behavior legend
    fig.legend(handles=sweep_legend_elements, loc='lower center', 
              title='Head Sweep Classification', ncol=3, fontsize=9, 
              bbox_to_anchor=(0.5, 0.10))
    
    # Format plot
    plt.tight_layout()
    plt.suptitle(f'Combined Behavioral Analysis for Larva {larva_id}', fontsize=14, y=1.02)
    plt.subplots_adjust(bottom=0.22)  # Make room for the two legends
    
    plt.show()
    
    # Return results for further analysis
    return {
        'larva_id': larva_id,
        'time': time,
        'orientation_smooth': orientation_smooth,
        'bend_angle_smooth': bend_angle_smooth,
        'velocity': velocity,
        'run_segments': run_segments,
        'turn_segments': turn_segments,
        'accepted_sweeps': accepted_sweeps,
        'rejected_sweeps': rejected_sweeps,
        'states': states,
        'bend_peaks': filtered_peaks
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
    FIGURE_SIZE = (12, 6)
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
        if np.linalg.norm(vector) == 0:
            return np.nan
            
        # Calculate angle with negative x-axis using arctan2
        # Note: arctan2(y, x) gives angle from positive x-axis, we want from negative x-axis
        # So we use arctan2(y, -x) or equivalently arctan2(-y, x) + 180°
        angle_deg = np.degrees(np.arctan2(vector[1], -vector[0]))
        
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
    ax1.set_theta_zero_location('E')  # 0 degrees at the right (East)
    ax1.set_theta_direction(1)  # Antilockwise
    ax1.set_rlabel_position(45)  # Move radial labels away from plotted line
    ax1.set_rticks([])  # Less radial ticks
    ax1.set_rlim(0, 1.2)  # Set radius limit
    
    # Add cardinal direction labels with upstream/downstream
    ax1.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Right)', '135°', '±180°\n(Upstream)', '-135','-90°\n(Left)', '-45°'])
    
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
        # For polar plots, 0 radians is to the right (east), and angles increase counterclockwise
        orientation_rad = np.radians(orientation_angle)  # Adjust for polar plot convention
        bend_rad = np.radians(bend_angle)  # Same adjustment for bend angle
        
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
        
        
        # Create bend angle arrow with fixed length of 1
        bend_arrow = ax1.arrow(bend_rad, 0, 0, 1.0,
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
                     fontsize=10, pad=35)
        
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
    ax_polar.set_theta_zero_location('E')  # 0 degrees at the top (North)
    ax_polar.set_theta_direction(1)  # Anticlockwise
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
                -tail_to_center[:, 0]
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

def analyze_cast_orientations_all_new(experiments_data, smooth_window=5, jump_threshold=15):
    """
    Analyze cast orientations across all experiments using sophisticated peak detection,
    separating large casts, small casts, and total casts.
    
    Args:
        experiments_data: Dict containing all experiments data
        smooth_window: Window size for smoothing (default=5)
        jump_threshold: Threshold for detecting orientation jumps in degrees/frame (default=15)
        
    Returns:
        dict: Contains orientation data and statistics for large, small, and total casts
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    
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
        # Single experiment case
        large_cast_o, small_cast_o, all_cast_o = extract_cast_orientations(data_to_process, smooth_window, jump_threshold)
        large_cast_orientations.extend(large_cast_o)
        small_cast_orientations.extend(small_cast_o)
        all_cast_orientations.extend(all_cast_o)
    else:
        # Multiple experiments case
        for exp_data in data_to_process.values():
            large_cast_o, small_cast_o, all_cast_o = extract_cast_orientations(exp_data, smooth_window, jump_threshold)
            large_cast_orientations.extend(large_cast_o)
            small_cast_orientations.extend(small_cast_o)
            all_cast_orientations.extend(all_cast_o)
    
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

def extract_cast_orientations(data, smooth_window=5, jump_threshold=15):
    """
    Extract cast orientations using sophisticated peak detection techniques.
    
    Args:
        data: Dictionary containing larva tracking data
        smooth_window: Window size for smoothing (default=5)
        jump_threshold: Threshold for detecting orientation jumps (default=15)
        
    Returns:
        tuple: (large_cast_orientations, small_cast_orientations, all_cast_orientations)
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    
    # Initialize orientation storage
    large_cast_orientations = []
    small_cast_orientations = []
    all_cast_orientations = []
    
    # Process each larva
    for larva_id, larva_data in data.items():
        # Extract nested data if needed
        if 'data' in larva_data:
            larva_data = larva_data['data']
            
        # Extract required data
        try:
            # Extract basic time series data
            time = np.array(larva_data['t']).flatten()
            states = np.array(larva_data['global_state_large_state']).flatten()
            
            # Check for small_large_state or fall back to large_state
            has_small_large_state = 'global_state_small_large_state' in larva_data
            
            if has_small_large_state:
                # Extract both small and large cast states
                small_large_states = np.array(larva_data['global_state_small_large_state']).flatten()
                large_cast_mask = small_large_states == 2.0  # Large casts = 2.0
                small_cast_mask = small_large_states == 1.5  # Small casts = 1.5
            else:
                # Fall back to just large state if small_large_state isn't available
                large_cast_mask = states == 2  # Only large casts available
                small_cast_mask = np.zeros_like(states, dtype=bool)  # No small casts
            
            any_cast_mask = large_cast_mask | small_cast_mask
            
            # Get orientation angles
            x_center = np.array(larva_data['x_center']).flatten()
            y_center = np.array(larva_data['y_center']).flatten()
            x_spine = np.array(larva_data['x_spine'])
            y_spine = np.array(larva_data['y_spine'])
            
            if x_spine.ndim > 1:
                x_tail = x_spine[-1].flatten()
                y_tail = y_spine[-1].flatten()
            else:
                x_tail = x_spine
                y_tail = y_spine
            
            # Calculate orientation vectors
            dx = x_center - x_tail
            dy = y_center - y_tail
            orientation_angles = np.degrees(np.arctan2(dy, -dx))  # -dx because 0° is negative x-axis
            
            # Get upper-lower bend angle
            angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
            angle_upper_lower_deg = np.degrees(angle_upper_lower)
            
            # Ensure all arrays have the same length
            min_length = min(len(time), len(orientation_angles), len(angle_upper_lower_deg), len(states))
            time = time[:min_length]
            orientation_angles = orientation_angles[:min_length]
            angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
            states = states[:min_length]
            large_cast_mask = large_cast_mask[:min_length]
            small_cast_mask = small_cast_mask[:min_length]
            any_cast_mask = any_cast_mask[:min_length]
            
            # ==== FILTERING AND PROCESSING ====
            
            # 1. Orientation angle jump detection and correction
            orientation_raw = orientation_angles.copy()
            
            # Calculate derivative to detect jumps
            orientation_diff = np.abs(np.diff(orientation_angles))
            # Add zero at the beginning to match length
            orientation_diff = np.insert(orientation_diff, 0, 0)
            
            # Find jumps bigger than threshold
            jumps = orientation_diff > jump_threshold
            
            # Create masked array for plotting
            orientation_masked = np.ma.array(orientation_angles, mask=jumps)
            
            # Apply Gaussian smoothing to the filtered data
            # First interpolate masked values for smoothing
            orientation_interp = orientation_masked.filled(np.nan)
            mask = np.isnan(orientation_interp)
            
            # Only interpolate if we have valid data
            if np.sum(~mask) > 1:
                indices = np.arange(len(orientation_interp))
                valid_indices = indices[~mask]
                valid_values = orientation_interp[~mask]
                orientation_interp[mask] = np.interp(indices[mask], valid_indices, valid_values)
            
            # Apply smoothing
            orientation_smooth = gaussian_filter1d(orientation_interp, smooth_window/3.0)
            
            # 2. Intelligent peak detection for bend angle
            # Calculate slope changes
            bend_angle_diff = np.diff(angle_upper_lower_deg)
            # Add zero at the beginning to match length
            bend_angle_diff = np.insert(bend_angle_diff, 0, 0)
            
            # Smooth the bend angle
            bend_angle_smooth = gaussian_filter1d(angle_upper_lower_deg, smooth_window/3.0)
            
            # Find peaks with intelligent filtering
            pos_peaks, _ = find_peaks(
                bend_angle_smooth, 
                height=5,
                prominence=3,
                distance=5
            )
            
            neg_peaks, _ = find_peaks(
                -bend_angle_smooth, 
                height=5,
                prominence=3,
                distance=5
            )
            
            # Combine positive and negative peaks
            all_peaks = np.union1d(pos_peaks, neg_peaks)
            
            # Filter out peaks that start from flat regions
            slope_threshold = 0.5  # Define what's considered "flat"
            filtered_peaks = []
            for peak in all_peaks:
                if peak > 0 and abs(bend_angle_diff[peak-1]) > slope_threshold:
                    filtered_peaks.append(peak)
            
            # Collect orientations at valid cast peaks
            for peak in filtered_peaks:
                if large_cast_mask[peak]:
                    large_cast_orientations.append(orientation_smooth[peak])
                    all_cast_orientations.append(orientation_smooth[peak])
                elif small_cast_mask[peak]:
                    small_cast_orientations.append(orientation_smooth[peak])
                    all_cast_orientations.append(orientation_smooth[peak])
                elif any_cast_mask[peak]:  # Fallback for any cast
                    all_cast_orientations.append(orientation_smooth[peak])
            
        except Exception as e:
            print(f"Error processing larva {larva_id}: {str(e)}")
            continue
    
    return large_cast_orientations, small_cast_orientations, all_cast_orientations


def detect_cast_peaks(trx_data, smooth_window=5, jump_threshold=15, peak_threshold=10.0, acceptance_threshold=30.0):
    """
    Detect peaks in cast events for all larvae in the tracking data.
    
    Args:
        trx_data: Dictionary containing tracking data
        smooth_window: Window size for smoothing the angle data
        jump_threshold: Threshold for detecting orientation jumps in degrees/frame
        peak_threshold: Minimum height for a peak to be considered significant
        acceptance_threshold: Maximum difference (degrees) between peak orientation and final orientation
                             for a peak to be considered "accepted"
        
    Returns:
        dict: Contains detected peaks with their properties
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    
    # Storage for detected peaks
    all_peaks = {
        'orientations': [],      # Body orientation at peak time
        'amplitudes': [],        # Peak amplitude (bend angle)
        'directions': [],        # Direction of cast (positive/negative)
        'larva_ids': [],         # ID of larva for each peak
        'cast_types': [],        # Whether it was a large or small cast
        'is_first_peak': [],     # Flag for first peak in a segment
        'is_last_peak': [],      # Flag for last peak in a segment
        'is_accepted': [],       # Flag if orientation changed to match peak
        'frame_indices': []      # Store frame indices for reference
    }
    
    # Process each larva
    for larva_id, larva_data in trx_data.items():
        if 'data' in larva_data:
            larva_data = larva_data['data']
            
        # Skip larvae with insufficient data
        if not all(key in larva_data for key in ['global_state_large_state', 'angle_upper_lower_smooth_5']):
            continue
            
        # Extract behavior states and angles
        states = np.array(larva_data['global_state_large_state']).flatten()
        angles = np.degrees(np.array(larva_data['angle_upper_lower_smooth_5']).flatten())
        
        # Get orientation angles
        x_center = np.array(larva_data['x_center']).flatten()
        y_center = np.array(larva_data['y_center']).flatten()
        x_tail = np.array(larva_data['x_spine'])[-1].flatten()
        y_tail = np.array(larva_data['y_spine'])[-1].flatten()
        
        tail_to_center = np.column_stack([x_center - x_tail, y_center - y_tail])
        orientations = np.degrees(np.arctan2(tail_to_center[:, 1], -tail_to_center[:, 0]))
        
        # Ensure all arrays have the same length
        min_length = min(len(states), len(angles), len(orientations))
        states = states[:min_length]
        angles = angles[:min_length]
        orientations = orientations[:min_length]
        
        # Apply smoothing to angle data
        angles_smooth = gaussian_filter1d(angles, sigma=smooth_window/3.0)
        
        # Identify segments with cast behavior (state = 2 for large casts, 1.5 for small casts)
        large_cast_mask = (states == 2.0)
        small_cast_mask = (states == 1.5)
        
        # Process cast segments to find peaks
        for cast_type, cast_mask in [('large', large_cast_mask), ('small', small_cast_mask)]:
            # Find continuous segments of casting behavior
            cast_segments = []
            in_cast = False
            cast_start = 0
            
            for i in range(len(cast_mask)):
                if cast_mask[i] and not in_cast:
                    # Start of a new cast
                    in_cast = True
                    cast_start = i
                elif not cast_mask[i] and in_cast:
                    # End of a cast
                    in_cast = False
                    if i - cast_start >= 3:  # Require at least 3 frames for a valid cast
                        cast_segments.append((cast_start, i))
            
            # Handle case when still in cast at end of data
            if in_cast and len(cast_mask) - cast_start >= 3:
                cast_segments.append((cast_start, len(cast_mask)))
            
            # Process each cast segment
            for start, end in cast_segments:
                segment_angles = angles_smooth[start:end]
                segment_orientations = orientations[start:end]
                
                # Find peaks in absolute angle values
                peak_indices, peak_props = find_peaks(
                    np.abs(segment_angles), 
                    height=peak_threshold,
                    prominence=3.0,
                    distance=3
                )
                
                # Skip if no peaks found
                if len(peak_indices) == 0:
                    continue
                
                # Get final orientation after the cast (if available)
                # Check the orientation a few frames after the cast ends
                post_cast_frames = 5  # Look ahead this many frames
                final_orientation = None
                if end + post_cast_frames < len(orientations):
                    # Use the average orientation from several frames after the cast
                    final_orientation = np.mean(orientations[end:end+post_cast_frames])
                elif end < len(orientations):
                    # If at the edge of data, use what's available
                    final_orientation = np.mean(orientations[end:])
                elif end - 1 < len(orientations):
                    # If at the very edge, use the last frame
                    final_orientation = orientations[end - 1]
                
                # Process detected peaks
                for i, idx in enumerate(peak_indices):
                    if start + idx < len(orientations):
                        global_idx = start + idx  # Global frame index
                        peak_orientation = segment_orientations[idx]
                        peak_amplitude = segment_angles[idx]
                        peak_direction = 1 if peak_amplitude > 0 else -1  # +1 for rightward, -1 for leftward
                        
                        # Mark first and last peaks
                        is_first = (i == 0)
                        is_last = (i == len(peak_indices) - 1)
                        
                        # Determine if the peak is "accepted"
                        # A peak is accepted if the final body orientation is close to the peak orientation
                        is_accepted = False
                        if final_orientation is not None:
                            # Calculate circular difference between orientations (handles angle wrap-around)
                            diff = abs((peak_orientation - final_orientation + 180) % 360 - 180)
                            is_accepted = diff <= acceptance_threshold
                        
                        # Store all peak data
                        all_peaks['orientations'].append(peak_orientation)
                        all_peaks['amplitudes'].append(peak_amplitude)
                        all_peaks['directions'].append(peak_direction)
                        all_peaks['larva_ids'].append(larva_id)
                        all_peaks['cast_types'].append(cast_type)
                        all_peaks['is_first_peak'].append(is_first)
                        all_peaks['is_last_peak'].append(is_last)
                        all_peaks['is_accepted'].append(is_accepted)
                        all_peaks['frame_indices'].append(global_idx)
    
    # Convert lists to numpy arrays
    for key in all_peaks:
        all_peaks[key] = np.array(all_peaks[key])
    
    all_peaks['n_peaks'] = len(all_peaks['orientations'])
    
    # Print summary statistics
    n_first = np.sum(all_peaks['is_first_peak'])
    n_last = np.sum(all_peaks['is_last_peak'])
    n_accepted = np.sum(all_peaks['is_accepted'])
    n_total = all_peaks['n_peaks']
    
    print(f"Detected {n_total} cast peaks across {len(np.unique(all_peaks['larva_ids']))} larvae")
    print(f"First peaks: {n_first} ({n_first/n_total*100:.1f}%), Last peaks: {n_last} ({n_last/n_total*100:.1f}%)")
    print(f"Accepted peaks: {n_accepted} ({n_accepted/n_total*100:.1f}%)")
    
    return all_peaks


def analyze_cast_peaks_by_orientation(peaks_data, bin_width=10):
    """
    Analyze how cast peak characteristics vary with body orientation.
    
    Args:
        peaks_data: Dictionary containing detected cast peaks
        bin_width: Width of orientation bins in degrees
        
    Returns:
        dict: Contains binned statistics for cast peaks
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    
    # Create orientation bins from -180 to 180 degrees
    bins = np.arange(-180, 181, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Separate large and small cast peaks
    large_mask = (peaks_data['cast_types'] == 'large')
    small_mask = (peaks_data['cast_types'] == 'small')
    
    # Filter data by cast type
    large_orientations = peaks_data['orientations'][large_mask]
    large_amplitudes = peaks_data['amplitudes'][large_mask]
    large_directions = peaks_data['directions'][large_mask]
    
    small_orientations = peaks_data['orientations'][small_mask]
    small_amplitudes = peaks_data['amplitudes'][small_mask]
    small_directions = peaks_data['directions'][small_mask]
    
    # All casts combined
    all_orientations = peaks_data['orientations']
    all_amplitudes = peaks_data['amplitudes']
    all_directions = peaks_data['directions']
    
    # Initialize arrays for binned statistics
    large_mean_amplitudes = np.zeros(len(bin_centers))
    large_direction_bias = np.zeros(len(bin_centers))
    large_count = np.zeros(len(bin_centers))
    
    small_mean_amplitudes = np.zeros(len(bin_centers))
    small_direction_bias = np.zeros(len(bin_centers))
    small_count = np.zeros(len(bin_centers))
    
    all_mean_amplitudes = np.zeros(len(bin_centers))
    all_direction_bias = np.zeros(len(bin_centers))
    all_count = np.zeros(len(bin_centers))
    
    # Calculate binned statistics
    for i in range(len(bin_centers)):
        # Define bin range
        bin_min = bins[i]
        bin_max = bins[i+1]
        
        # Filter peaks by orientation
        large_bin_mask = (large_orientations >= bin_min) & (large_orientations < bin_max)
        small_bin_mask = (small_orientations >= bin_min) & (small_orientations < bin_max)
        all_bin_mask = (all_orientations >= bin_min) & (all_orientations < bin_max)
        
        # Calculate statistics for large casts
        if np.sum(large_bin_mask) > 0:
            large_mean_amplitudes[i] = np.mean(np.abs(large_amplitudes[large_bin_mask]))
            large_direction_bias[i] = np.mean(large_directions[large_bin_mask])
            large_count[i] = np.sum(large_bin_mask)
        
        # Calculate statistics for small casts
        if np.sum(small_bin_mask) > 0:
            small_mean_amplitudes[i] = np.mean(np.abs(small_amplitudes[small_bin_mask]))
            small_direction_bias[i] = np.mean(small_directions[small_bin_mask])
            small_count[i] = np.sum(small_bin_mask)
        
        # Calculate statistics for all casts
        if np.sum(all_bin_mask) > 0:
            all_mean_amplitudes[i] = np.mean(np.abs(all_amplitudes[all_bin_mask]))
            all_direction_bias[i] = np.mean(all_directions[all_bin_mask])
            all_count[i] = np.sum(all_bin_mask)
    
    # Apply smoothing
    large_amplitude_smooth = gaussian_filter1d(large_mean_amplitudes, sigma=1)
    large_direction_smooth = gaussian_filter1d(large_direction_bias, sigma=1)
    
    small_amplitude_smooth = gaussian_filter1d(small_mean_amplitudes, sigma=1)
    small_direction_smooth = gaussian_filter1d(small_direction_bias, sigma=1)
    
    all_amplitude_smooth = gaussian_filter1d(all_mean_amplitudes, sigma=1)
    all_direction_smooth = gaussian_filter1d(all_direction_bias, sigma=1)
    
    # Create visualization - Amplitude by orientation
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bin_centers, large_amplitude_smooth, 'r-', label='Large casts')
    plt.plot(bin_centers, small_amplitude_smooth, 'b-', label='Small casts')
    plt.plot(bin_centers, all_amplitude_smooth, 'g-', label='All casts')
    plt.xlabel('Body Orientation (°)')
    plt.ylabel('Mean Peak Amplitude (°)')
    plt.title('Cast Peak Amplitude by Orientation')
    plt.xlim(-180, 180)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    
    # Create visualization - Direction bias by orientation
    plt.subplot(1, 2, 2)
    plt.plot(bin_centers, large_direction_smooth, 'r-', label='Large casts')
    plt.plot(bin_centers, small_direction_smooth, 'b-', label='Small casts')
    plt.plot(bin_centers, all_direction_smooth, 'g-', label='All casts')
    plt.xlabel('Body Orientation (°)')
    plt.ylabel('Direction Bias\n(+1: Right, -1: Left)')
    plt.title('Cast Direction Bias by Orientation')
    plt.xlim(-180, 180)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'bin_centers': bin_centers,
        'large_mean_amplitudes': large_mean_amplitudes,
        'large_direction_bias': large_direction_bias,
        'large_count': large_count,
        'large_amplitude_smooth': large_amplitude_smooth,
        'large_direction_smooth': large_direction_smooth,
        'small_mean_amplitudes': small_mean_amplitudes,
        'small_direction_bias': small_direction_bias,
        'small_count': small_count,
        'small_amplitude_smooth': small_amplitude_smooth,
        'small_direction_smooth': small_direction_smooth,
        'all_mean_amplitudes': all_mean_amplitudes,
        'all_direction_bias': all_direction_bias,
        'all_count': all_count,
        'all_amplitude_smooth': all_amplitude_smooth,
        'all_direction_smooth': all_direction_smooth,
        'n_large_peaks': np.sum(large_mask),
        'n_small_peaks': np.sum(small_mask),
        'n_total_peaks': len(all_orientations)
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


def plot_combined_run_orientation_analysis(data, bin_width=10, save_path=None):
    """
    Plot both run orientation distribution and run probability on the same polar plot.
    
    Args:
        data: The filtered data containing larva tracking information
        bin_width: Width of the orientation bins in degrees
        save_path: Path to save the figure (None for no saving)
    
    Returns:
        dict: Results from both analyses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import src.processor as processor
    
    # Run both analyses
    orientation_results = processor.analyze_run_orientations_all(data)
    rate_results = processor.analyze_run_rate_by_orientation(data, bin_width=bin_width)
    
    # Extract data for plotting
    orientations = np.array(orientation_results['bin_centers'])
    orientation_dist = np.array(orientation_results['smoothed_all'])
    
    rates_orientations = np.array(rate_results['bin_centers'])
    run_rates = np.array(rate_results['all_smoothed'])
    
    # Convert to radians for polar plot
    orientations_rad = np.deg2rad(orientations)
    rates_orientations_rad = np.deg2rad(rates_orientations)
    
    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
    
    # Plot orientation distribution (where larvae face when running)
    line1 = ax.plot(orientations_rad, orientation_dist, 
             color='blue', linewidth=2, label='Orientation Distribution')
    ax.fill_between(orientations_rad, 0, orientation_dist, alpha=0.2, color='blue')
    
    # Create a second y-axis for run probability
    ax2 = ax.twinx()
    
    # Plot run probability (how likely to run at each orientation)
    line2 = ax2.plot(rates_orientations_rad, run_rates, 
             color='red', linewidth=2, linestyle='--', label='Run Probability')
    
    # Set up the primary axis
    ax.set_theta_zero_location('N')  # 0 degrees at the top
    ax.set_theta_direction(-1)       # clockwise
    ax.set_rlabel_position(45)       # Move radial labels
    
    # Add labels and titles
    plt.title('Run Behavior Analysis by Orientation', y=1.08, fontsize=14)
    
    # Add a custom legend with both lines
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    
    # Add explanatory text
    plt.figtext(0.5, 0.01, 
                'Blue: Distribution of orientations during runs (P(orientation | running))\n'
                'Red dashed: Probability of running at each orientation (P(running | orientation))', 
                ha='center', fontsize=10)
    
    # Set axis limits (normalize for better visualization)
    ax.set_ylim(0, max(orientation_dist) * 1.1)
    ax2.set_ylim(0, max(run_rates) * 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add orientation markers
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Crosswind (left))', '135°', 
                      '180°\n(Upstream)', '225°', '270°\n(Crosswind (right))', '315°'])
    
    # Add subtle reference ring
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta, np.ones_like(theta) * max(orientation_dist)/2, 
            linestyle=':', color='gray', alpha=0.5, linewidth=0.5)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'orientation_results': orientation_results,
        'rate_results': rate_results
    }

def plot_polar_run_types_comparison(data, bin_width=10, save_path=None):
    """
    Plot polar comparisons of all run types (large, small, all) for both
    orientation distribution and run probability.
    
    Args:
        data: The filtered data containing larva tracking information
        bin_width: Width of the orientation bins in degrees
        save_path: Path to save the figure (None for no saving)
    
    Returns:
        dict: Results from both analyses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.lines import Line2D
    
    # Run both analyses
    orientation_results = analyze_run_orientations_all(data)
    rate_results = analyze_run_rate_by_orientation(data, bin_width=bin_width)
    
    # Define colors for different run types
    colors = {
        'large': 'red',
        'small': 'blue',
        'all': 'green'
    }
    
    # Create a figure with two polar subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                  subplot_kw={'projection': 'polar'})
    
    # Initialize maximum values for scaling
    max_dist = 0
    max_rate = 0
    
    # Plot each run type on both plots
    for run_type in ['large', 'small', 'all']:
        color = colors[run_type]
        
        # Extract orientation distribution data
        orientations = np.array(orientation_results['bin_centers'])
        orientation_dist = np.array(orientation_results[f'smoothed_{run_type}'])
        max_dist = max(max_dist, np.max(orientation_dist))
        
        # Extract run probability data
        rates_orientations = np.array(rate_results['bin_centers'])
        run_rates = np.array(rate_results[f'{run_type}_smoothed'])
        max_rate = max(max_rate, np.max(run_rates))
        
        # Convert to radians for polar plot
        orientations_rad = np.deg2rad(orientations)
        rates_orientations_rad = np.deg2rad(rates_orientations)
        
        # Plot on first subplot (orientation distribution)
        ax1.plot(orientations_rad, orientation_dist, color=color, linewidth=2, 
                label=f'{run_type.capitalize()} runs')
        
        # Plot on second subplot (run probability)
        ax2.plot(rates_orientations_rad, run_rates, color=color, linewidth=2,
                label=f'{run_type.capitalize()} runs')
    
    # After plotting all lines, add filled areas with lower alpha to avoid overlap
    for run_type in ['large', 'small', 'all']:
        color = colors[run_type]
        alpha = 0.15  # Very light fill
        
        # Fill for orientation distribution
        orientations = np.array(orientation_results['bin_centers'])
        orientation_dist = np.array(orientation_results[f'smoothed_{run_type}'])
        orientations_rad = np.deg2rad(orientations)
        ax1.fill_between(orientations_rad, 0, orientation_dist, alpha=alpha, color=color)
        
        # Fill for run probability
        rates_orientations = np.array(rate_results['bin_centers'])
        run_rates = np.array(rate_results[f'{run_type}_smoothed'])
        rates_orientations_rad = np.deg2rad(rates_orientations)
        ax2.fill_between(rates_orientations_rad, 0, run_rates, alpha=alpha, color=color)
    
    # Configure both polar axes
    for i, ax in enumerate([ax1, ax2]):
        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)       # Clockwise direction
        ax.set_rlabel_position(45)       # Position radius labels
        
        # Add orientation labels
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Crosswind (left))', '135°', 
                      '180°\n(Upstream)', '225°', '270°\n(Crosswind (right))', '315°'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add titles
        if i == 0:
            ax.set_title('Run Orientation Distribution\n(Where larvae face during runs)', y=1.08)
        else:
            ax.set_title('Run Probability by Orientation\n(How likely to run at each angle)', y=1.08)
    
    # Add a single legend for both subplots
    legend_elements = [
        Line2D([0], [0], color=colors['large'], linewidth=2, label='Large runs'),
        Line2D([0], [0], color=colors['small'], linewidth=2, label='Small runs'),
        Line2D([0], [0], color=colors['all'], linewidth=2, label='All runs')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    # Add overall title
    plt.suptitle(f'Run Analysis Comparison (n={orientation_results["n_larvae"]} larvae)', 
               fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the legend
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'orientation_results': orientation_results,
        'rate_results': rate_results
    }

def plot_polar_run_comparison(data, run_type='all', bin_width=10, save_path=None):
    """
    Plot polar representations of run orientation distribution and run probability
    for a specific run type (large, small, or all).
    
    Args:
        data: The filtered data containing larva tracking information
        run_type: Type of runs to analyze ('large', 'small', or 'all')
        bin_width: Width of the orientation bins in degrees
        save_path: Path to save the figure (None for no saving)
    
    Returns:
        dict: Results from both analyses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Validate run_type
    if run_type not in ['large', 'small', 'all']:
        raise ValueError("run_type must be 'large', 'small', or 'all'")
    
    # Run both analyses
    orientation_results = analyze_run_orientations_all(data)
    rate_results = analyze_run_rate_by_orientation(data, bin_width=bin_width)
    
    # Extract data for the specified run type
    orientations = np.array(orientation_results['bin_centers'])
    orientation_dist = np.array(orientation_results[f'smoothed_{run_type}'])
    
    rates_orientations = np.array(rate_results['bin_centers'])
    run_rates = np.array(rate_results[f'{run_type}_smoothed'])
    
    # Convert to radians for polar plot
    orientations_rad = np.deg2rad(orientations)
    rates_orientations_rad = np.deg2rad(rates_orientations)
    
    # Define colors based on run type
    colors = {
        'large': 'red',
        'small': 'blue',
        'all': 'green'
    }
    color = colors[run_type]
    
    # Create a figure with two polar subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   subplot_kw={'projection': 'polar'})
    
    # Plot 1: Run orientation distribution
    ax1.plot(orientations_rad, orientation_dist, color=color, linewidth=2)
    ax1.fill_between(orientations_rad, 0, orientation_dist, alpha=0.3, color=color)
    ax1.set_title(f'{run_type.capitalize()} Run Orientation Distribution\n(Where larvae face during runs)', 
                 y=1.08)
    
    # Plot 2: Run probability by orientation
    ax2.plot(rates_orientations_rad, run_rates, color=color, linewidth=2)
    ax2.fill_between(rates_orientations_rad, 0, run_rates, alpha=0.3, color=color)
    ax2.set_title(f'{run_type.capitalize()} Run Probability by Orientation\n(How likely to run at each angle)', 
                 y=1.08)
    
    # Configure both polar axes
    for ax in [ax1, ax2]:
        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)       # Clockwise direction
        ax.set_rlabel_position(45)       # Position radius labels
        
        # Add orientation labels
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Crosswind (left))', '135°', 
                      '180°\n(Upstream)', '225°', '270°\n(Crosswind (right))', '315°'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Normalize the radial axis for better visualization
        ax.set_ylim(0, None)  # Auto-scale the upper limit
        
        # Add subtle reference rings
        max_val = ax.get_ylim()[1]
        theta = np.linspace(0, 2*np.pi, 100)
        for r in np.linspace(0, max_val, 4)[1:]:
            ax.plot(theta, np.ones_like(theta) * r, 
                   linestyle=':', color='gray', alpha=0.3, linewidth=0.5)
    
    # Add overall title
    n_runs = orientation_results[f'n_{run_type}_runs'] if run_type != 'all' else orientation_results['n_total_runs']
    plt.suptitle(f'{run_type.capitalize()} Run Analysis (n={orientation_results["n_larvae"]} larvae, {n_runs} runs)', 
               fontsize=14)
    
    # Add explanation text at the bottom
    plt.figtext(0.5, 0.01, 
               'Left: Distribution of orientations during runs (P(orientation | running))\n'
               'Right: Probability of running at each orientation (P(running | orientation))', 
               ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'orientation_results': orientation_results,
        'rate_results': rate_results
    }

def plot_polar_run_analysis(data, bin_width=10, save_path=None):
    """
    Plot polar representations of run orientation distribution and run probability.
    
    Args:
        data: The filtered data containing larva tracking information
        bin_width: Width of the orientation bins in degrees
        save_path: Path to save the figure (None for no saving)
    
    Returns:
        dict: Results from both analyses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    
    # Run both analyses
    orientation_results = analyze_run_orientations_all(data)
    rate_results = analyze_run_rate_by_orientation(data, bin_width=bin_width)
    
    # Extract data for plotting
    orientations = np.array(orientation_results['bin_centers'])
    orientation_dist = np.array(orientation_results['smoothed_all'])
    
    rates_orientations = np.array(rate_results['bin_centers'])
    run_rates = np.array(rate_results['all_smoothed'])
    
    # Convert to radians for polar plot
    orientations_rad = np.deg2rad(orientations)
    rates_orientations_rad = np.deg2rad(rates_orientations)
    
    # Create a figure with two polar subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   subplot_kw={'projection': 'polar'})
    
    # Plot 1: Run orientation distribution
    ax1.plot(orientations_rad, orientation_dist, color='blue', linewidth=2)
    ax1.fill_between(orientations_rad, 0, orientation_dist, alpha=0.3, color='blue')
    ax1.set_title('Run Orientation Distribution\n(Where larvae face when running)', 
                 y=1.08)
    
    # Plot 2: Run probability by orientation
    ax2.plot(rates_orientations_rad, run_rates, color='red', linewidth=2)
    ax2.fill_between(rates_orientations_rad, 0, run_rates, alpha=0.3, color='red')
    ax2.set_title('Run Probability by Orientation\n(How likely to run at each angle)', 
                 y=1.08)
    
    # Configure both polar axes
    for ax in [ax1, ax2]:
        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)       # Clockwise direction
        ax.set_rlabel_position(45)       # Position radius labels
        
        # Add orientation labels
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Crosswind (left))', '135°', 
                      '180°\n(Upstream)', '225°', '270°\n(Crosswind (right))', '315°'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Run Behavior Analysis (n={orientation_results["n_larvae"]} larvae)', 
               fontsize=14)
    
    # Add explanation text at the bottom
    plt.figtext(0.5, 0.01, 
               'Left: Distribution of orientations during runs (P(orientation | running))\n'
               'Right: Probability of running at each orientation (P(running | orientation))', 
               ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'orientation_results': orientation_results,
        'rate_results': rate_results
    }

def plot_polar_run_histograms(data, bin_width=10, save_path=None):
    """
    Plot polar histograms of run orientation distribution and run probability.
    
    Args:
        data: The filtered data containing larva tracking information
        bin_width: Width of the orientation bins in degrees
        save_path: Path to save the figure (None for no saving)
    
    Returns:
        dict: Results from both analyses
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import src.processor as processor
    # Run both analyses
    orientation_results = processor.analyze_run_orientations_all(data)
    rate_results = processor.analyze_run_rate_by_orientation(data, bin_width=bin_width)
    
    # Extract data for plotting
    orientations = np.array(orientation_results['bin_centers'])
    orientation_dist = np.array(orientation_results['smoothed_all'])
    
    rates_orientations = np.array(rate_results['bin_centers'])
    run_rates = np.array(rate_results['all_smoothed'])
    
    # Convert to radians for polar plot
    orientations_rad = np.deg2rad(orientations)
    bin_width_rad = np.deg2rad(bin_width)
    
    # Create a figure with two polar subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   subplot_kw={'projection': 'polar'})
    
    # Plot 1: Run orientation distribution as polar histogram
    bars1 = ax1.bar(orientations_rad, orientation_dist, width=bin_width_rad, bottom=0.0, 
                   color='royalblue', alpha=0.7, edgecolor='blue', linewidth=0.5)
    ax1.set_title('Run Orientation Distribution\n(Where larvae face during runs)', 
                 y=1.08)
    
    # Plot 2: Run probability by orientation as polar histogram
    bars2 = ax2.bar(np.deg2rad(rates_orientations), run_rates, width=bin_width_rad, bottom=0.0,
                   color='firebrick', alpha=0.7, edgecolor='darkred', linewidth=0.5)
    ax2.set_title('Run Probability by Orientation\n(How likely to run at each angle)', 
                 y=1.08)
    
    # Configure both polar axes
    for ax in [ax1, ax2]:
        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)       # Clockwise direction
        ax.set_rlabel_position(45)       # Position radius labels
        
        # Add orientation labels
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Crosswind (left))', '135°', 
                      '180°\n(Upstream)', '225°', '270°\n(Crosswind (right))', '315°'])
        
        # Add grid
        ax.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Run Behavior Analysis (n={orientation_results["n_larvae"]} larvae)', 
               fontsize=14)
    
    # Add explanation text at the bottom
    plt.figtext(0.5, 0.01, 
               'Left: Distribution of orientations during runs (P(orientation | running))\n'
               'Right: Probability of running at each orientation (P(running | orientation))', 
               ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'orientation_results': orientation_results,
        'rate_results': rate_results
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



def compare_genotype_distributions_by_date():
    """
    Compare orientation distributions for runs and casts across different genotypes,
    analyzing each experiment date separately. Automatically detects available dates.
    """
    # Define genotype paths to analyze
    genotype_paths = {
        "SS01948": "/Users/sharbat/Projects/anemotaxis/data/GMR_SS01948@UAS_TNT_2_0003/p_5gradient2_2s1x600s0s#n#n#n",
        "SS01757": "/Users/sharbat/Projects/anemotaxis/data/GMR_SS01757@UAS_TNT_2_0003/p_5gradient2_2s1x600s0s#n#n#n",
        "control": "/Users/sharbat/Projects/anemotaxis/data/FCF_attP2-40@UAS_TNT_2_0003/p_5gradient2_2s1x600s0s#n#n#n",
    }
    
    # Automatically discover experiment dates for each genotype
    experiment_dates = {}
    for genotype, base_path in genotype_paths.items():
        # Find all subdirectories with trx.mat files
        date_dirs = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file == "trx.mat":
                    # Extract date directory name (immediate parent of trx.mat)
                    date_dir = os.path.basename(root)
                    date_dirs.append(date_dir)
        
        # Store unique, sorted date directories
        experiment_dates[genotype] = sorted(list(set(date_dirs)))
        print(f"Found {len(experiment_dates[genotype])} experiment dates for {genotype}")
    
    # Storage for results - structure: genotype -> date -> results
    genotype_date_results = {}
    
    # Define color maps for each genotype and line styles for dates
    genotype_colors = {
        "SS01948": "blue",
        "SS01757": "red",
        "control": "green"
    }
    
    # Define line styles based on the number of dates we have
    max_dates = max([len(dates) for dates in experiment_dates.values()])
    line_styles = [
        ("-", 1.5),    # Solid thick
        ("--", 1.5),   # Dashed thick
        (":", 1.8),    # Dotted thick
        ("-.", 1.5),   # Dash-dot thick
        ("-", 1.0),    # Solid thin
        ("--", 1.0),   # Dashed thin
        (":", 1.2),    # Dotted thin
        ("-.", 1.0)    # Dash-dot thin
    ]
    
    # Ensure we have enough line styles
    while len(line_styles) < max_dates:
        line_styles.extend(line_styles[:4])
    
    # Create date_styles mapping
    date_styles = {}
    for genotype, dates in experiment_dates.items():
        for i, date in enumerate(dates):
            date_styles[date] = line_styles[i]
    
    # Process each genotype
    for genotype, base_path in genotype_paths.items():
        print(f"Processing genotype: {genotype}")
        genotype_date_results[genotype] = {}
        
        # Process each experiment date for this genotype
        for date in experiment_dates[genotype]:
            try:
                trx_path = os.path.join(base_path, date, "trx.mat")
                print(f"  Loading {trx_path}")
                
                # Check if file exists
                if not os.path.exists(trx_path):
                    print(f"  File not found: {trx_path}")
                    continue
                
                # Load and process data using the correct function names
                processed_data = process_single_file(trx_path)
                
                # Apply filters
                filtered_data = filter_larvae_by_duration(processed_data, 
                                                             min_total_duration=300)
                
                if not filtered_data or len(filtered_data["larvae"]) == 0:
                    print(f"  No larvae passed filters for {date}")
                    continue
                
                num_larvae = len(filtered_data["larvae"])
                print(f"  {num_larvae} larvae passed filters for {date}")
                
                # Analyze run orientations
                run_results = analyze_run_orientations_all(filtered_data)
                
                # Analyze cast orientations
                cast_results = analyze_cast_orientations_all(filtered_data)
                
                # Store the results for this date
                genotype_date_results[genotype][date] = {
                    "run_results": run_results,
                    "cast_results": cast_results,
                    "n_larvae": num_larvae
                }
                
                print(f"  Completed analysis for {genotype}, {date} with {num_larvae} larvae")
                
            except Exception as e:
                print(f"  Error processing {genotype}, {date}: {str(e)}")
                continue
    
    # Check if we have enough data to compare
    if not any(genotype_date_results.values()):
        print("No data available to compare")
        return genotype_date_results
    
    # Combine the dates within each genotype for a genotype summary plot
    genotype_combined_results = {}
    for genotype, date_results in genotype_date_results.items():
        if not date_results:  # Skip empty results
            continue
            
        # Initialize combined arrays
        large_cast_orientations = []
        small_cast_orientations = []
        all_cast_orientations = []
        large_run_orientations = []
        small_run_orientations = []
        all_run_orientations = []
        total_larvae = 0
        
        # Combine data from all dates for this genotype
        for date, results in date_results.items():
            # Add cast data
            if 'cast_results' in results:
                large_cast_orientations.extend(results['cast_results'].get('large_cast_orientations', []))
                small_cast_orientations.extend(results['cast_results'].get('small_cast_orientations', []))
                all_cast_orientations.extend(results['cast_results'].get('all_cast_orientations', []))
            
            # Add run data
            if 'run_results' in results:
                large_run_orientations.extend(results['run_results'].get('large_run_orientations', []))
                small_run_orientations.extend(results['run_results'].get('small_run_orientations', []))
                all_run_orientations.extend(results['run_results'].get('all_run_orientations', []))
            
            # Track larvae count
            total_larvae += results.get('n_larvae', 0)
        
        # Calculate combined histograms and smoothed data
        bins = np.linspace(-180, 180, 37)  # 36 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Casts
        if len(all_cast_orientations) > 0:
            cast_hist = np.histogram(all_cast_orientations, bins=bins, density=True)[0]
            cast_smoothed = gaussian_filter1d(cast_hist, sigma=1)
        else:
            cast_hist = np.zeros(36)
            cast_smoothed = np.zeros(36)
        
        # Runs
        if len(all_run_orientations) > 0:
            run_hist = np.histogram(all_run_orientations, bins=bins, density=True)[0]
            run_smoothed = gaussian_filter1d(run_hist, sigma=1)
        else:
            run_hist = np.zeros(36)
            run_smoothed = np.zeros(36)
        
        # Store combined results
        genotype_combined_results[genotype] = {
            'cast_hist': cast_hist,
            'cast_smoothed': cast_smoothed,
            'run_hist': run_hist,
            'run_smoothed': run_smoothed,
            'bin_centers': bin_centers,
            'total_larvae': total_larvae,
            'n_casts': len(all_cast_orientations),
            'n_runs': len(all_run_orientations)
        }
    
    # Skip plots if no combined results
    if not genotype_combined_results:
        print("No combined results available to plot")
        return genotype_date_results
    
    # Create genotype comparison plots
    # 1. Run Orientations
    plt.figure(figsize=(8, 6))
    for genotype, results in genotype_combined_results.items():
        if results['n_runs'] > 0:
            plt.plot(results['bin_centers'], results['run_smoothed'], 
                     color=genotype_colors.get(genotype, 'black'), linewidth=2.5,
                     label=f"{genotype} (n={results['total_larvae']} larvae, {results['n_runs']} runs)")
    
    plt.xlabel('Body Orientation (°)', fontsize=12)
    plt.ylabel('Relative Probability', fontsize=12)
    plt.xlim(-180, 180)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.title('Run Orientation Distribution Comparison by Genotype', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('run_orientation_genotype_comparison.png', dpi=300)
    plt.show()
    
    # 2. Cast Orientations
    plt.figure(figsize=(8, 6))
    for genotype, results in genotype_combined_results.items():
        if results['n_casts'] > 0:
            plt.plot(results['bin_centers'], results['cast_smoothed'], 
                     color=genotype_colors.get(genotype, 'black'), linewidth=2.5,
                     label=f"{genotype} (n={results['total_larvae']} larvae, {results['n_casts']} casts)")
    
    plt.xlabel('Body Orientation (°)', fontsize=12)
    plt.ylabel('Relative Probability', fontsize=12)
    plt.xlim(-180, 180)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    plt.title('Cast Orientation Distribution Comparison by Genotype', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('cast_orientation_genotype_comparison.png', dpi=300)
    plt.show()
    
    # Create plots comparing run orientations with all dates
    plt.figure(figsize=(12, 8))
    
    # Custom legend elements
    genotype_legend_elements = [Line2D([0], [0], color=color, lw=2, label=genotype) 
                               for genotype, color in genotype_colors.items() 
                               if genotype in genotype_date_results and genotype_date_results[genotype]]
    
    # Store bin centers for x-axis
    bin_centers = None
    
    # Plot run orientation distributions for each genotype and date
    for genotype, date_results in genotype_date_results.items():
        for date, results in date_results.items():
            if 'run_results' not in results:
                continue
                
            run_data = results["run_results"]
            
            # Get line style for this date
            linestyle, linewidth = date_styles.get(date, ('-', 1.0))
            
            # Store bin centers from first result for x-axis
            if bin_centers is None and 'bin_centers' in run_data:
                bin_centers = run_data["bin_centers"]
            
            if 'smoothed_all' in run_data and len(run_data['smoothed_all']) > 0:
                plt.plot(run_data["bin_centers"], run_data["smoothed_all"], 
                        color=genotype_colors.get(genotype, 'black'),
                        linestyle=linestyle, linewidth=linewidth,
                        label=f"{genotype}, {date} (n={results['n_larvae']})")
    
    plt.xlabel('Body Orientation (°)', fontsize=12)
    plt.ylabel('Relative Probability', fontsize=12)
    plt.xlim(-180, 180)
    plt.title('Run Orientation Distribution Comparison by Genotype and Date', fontsize=14)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    
    # Create a legend for genotypes
    if genotype_legend_elements:
        plt.legend(handles=genotype_legend_elements, loc='upper left', 
                  title="Genotype", bbox_to_anchor=(1.02, 1))
        
        # Add date legend as text
        date_text = []
        for i, (date, style) in enumerate(date_styles.items()):
            date_text.append(f"{date}: {style[0]}")
        
        plt.figtext(0.98, 0.6, "\n".join(date_text), va="top", ha="right", 
                   bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('run_orientation_by_date_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create plots comparing cast orientations
    plt.figure(figsize=(12, 8))
    
    # Plot cast orientation distributions for each genotype and date
    for genotype, date_results in genotype_date_results.items():
        for date, results in date_results.items():
            if 'cast_results' not in results:
                continue
                
            cast_data = results["cast_results"]
            
            # Get line style for this date
            linestyle, linewidth = date_styles.get(date, ('-', 1.0))
            
            if 'smoothed_all' in cast_data and len(cast_data['smoothed_all']) > 0:
                plt.plot(cast_data["bin_centers"], cast_data["smoothed_all"], 
                        color=genotype_colors.get(genotype, 'black'),
                        linestyle=linestyle, linewidth=linewidth,
                        label=f"{genotype}, {date} (n={results['n_larvae']})")
    
    plt.xlabel('Body Orientation (°)', fontsize=12)
    plt.ylabel('Relative Probability', fontsize=12)
    plt.xlim(-180, 180)
    plt.title('Cast Orientation Distribution Comparison by Genotype and Date', fontsize=14)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
    
    # Create a legend for genotypes
    if genotype_legend_elements:
        plt.legend(handles=genotype_legend_elements, loc='upper left', 
                  title="Genotype", bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig('cast_orientation_by_date_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find all unique dates across genotypes to create date-specific comparisons
    all_dates = set()
    for dates_dict in experiment_dates.values():
        all_dates.update(dates_dict)
    all_dates = sorted(list(all_dates))
    
    # Run comparisons for each date
    for date in all_dates:
        genotypes_with_data = []
        for genotype in genotype_paths.keys():
            if (genotype in genotype_date_results and 
                date in genotype_date_results[genotype] and
                'run_results' in genotype_date_results[genotype][date] and
                'cast_results' in genotype_date_results[genotype][date]):
                genotypes_with_data.append(genotype)
        
        if len(genotypes_with_data) < 2:
            print(f"Not enough genotypes with data for date {date}")
            continue
        
        # Create figure with two subplots (run and cast)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot run orientations for this date
        ax1.set_title(f'Run Orientations - {date}', fontsize=14)
        for genotype in genotypes_with_data:
            results = genotype_date_results[genotype][date]
            run_data = results["run_results"]
            
            if 'smoothed_all' in run_data and len(run_data['smoothed_all']) > 0:
                ax1.plot(run_data["bin_centers"], run_data["smoothed_all"], 
                       color=genotype_colors.get(genotype, 'black'), linewidth=2,
                       label=f"{genotype} (n={results['n_larvae']})")
        
        ax1.set_xlabel('Body Orientation (°)', fontsize=12)
        ax1.set_ylabel('Relative Probability', fontsize=12)
        ax1.set_xlim(-180, 180)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax1.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
        ax1.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
        ax1.legend()
        
        # Plot cast orientations for this date
        ax2.set_title(f'Cast Orientations - {date}', fontsize=14)
        for genotype in genotypes_with_data:
            results = genotype_date_results[genotype][date]
            cast_data = results["cast_results"]
            
            if 'smoothed_all' in cast_data and len(cast_data['smoothed_all']) > 0:
                ax2.plot(cast_data["bin_centers"], cast_data["smoothed_all"], 
                       color=genotype_colors.get(genotype, 'black'), linewidth=2,
                       label=f"{genotype} (n={results['n_larvae']})")
        
        ax2.set_xlabel('Body Orientation (°)', fontsize=12)
        ax2.set_ylabel('Relative Probability', fontsize=12)
        ax2.set_xlim(-180, 180)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
        ax2.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'orientation_comparison_{date}.png', dpi=300)
        plt.show()
    
    # Statistical comparisons by date
    print("\nStatistical Comparisons by Date:")
    
    for date in all_dates:
        genotypes_with_data = []
        for genotype in genotype_paths.keys():
            if (genotype in genotype_date_results and 
                date in genotype_date_results[genotype] and
                'run_results' in genotype_date_results[genotype][date] and
                'cast_results' in genotype_date_results[genotype][date]):
                genotypes_with_data.append(genotype)
        
        if len(genotypes_with_data) < 2:
            continue
        
        print(f"\nComparisons for date: {date}")
        
        # Run comparisons
        print("  Run Orientation Distribution:")
        for i in range(len(genotypes_with_data)):
            for j in range(i+1, len(genotypes_with_data)):
                g1 = genotypes_with_data[i]
                g2 = genotypes_with_data[j]
                
                # Get the raw orientation data
                try:
                    runs_g1 = genotype_date_results[g1][date]["run_results"]["all_run_orientations"]
                    runs_g2 = genotype_date_results[g2][date]["run_results"]["all_run_orientations"]
                    
                    # Ensure both arrays have elements before running KS test
                    if len(runs_g1) > 0 and len(runs_g2) > 0:
                        # Kolmogorov-Smirnov test
                        ks_stat, p_value = ks_2samp(runs_g1, runs_g2)
                        
                        print(f"    {g1} vs {g2}: KS statistic = {ks_stat:.4f}, p-value = {p_value:.6f}")
                        if p_value < 0.05:
                            print(f"      Significant difference detected (p < 0.05)")
                    else:
                        print(f"    {g1} vs {g2}: Insufficient data for statistical comparison")
                except (KeyError, ValueError) as e:
                    print(f"    {g1} vs {g2}: Error in data comparison - {str(e)}")
        
        # Cast comparisons
        print("  Cast Orientation Distribution:")
        for i in range(len(genotypes_with_data)):
            for j in range(i+1, len(genotypes_with_data)):
                g1 = genotypes_with_data[i]
                g2 = genotypes_with_data[j]
                
                # Get the raw orientation data
                try:
                    casts_g1 = genotype_date_results[g1][date]["cast_results"]["all_cast_orientations"]
                    casts_g2 = genotype_date_results[g2][date]["cast_results"]["all_cast_orientations"]
                    
                    # Ensure both arrays have elements before running KS test
                    if len(casts_g1) > 0 and len(casts_g2) > 0:
                        # Kolmogorov-Smirnov test
                        ks_stat, p_value = ks_2samp(casts_g1, casts_g2)
                        
                        print(f"    {g1} vs {g2}: KS statistic = {ks_stat:.4f}, p-value = {p_value:.6f}")
                        if p_value < 0.05:
                            print(f"      Significant difference detected (p < 0.05)")
                    else:
                        print(f"    {g1} vs {g2}: Insufficient data for statistical comparison")
                except (KeyError, ValueError) as e:
                    print(f"    {g1} vs {g2}: Error in data comparison - {str(e)}")
    
    # Statistical comparisons for combined genotype data
    print("\nStatistical Comparisons for Combined Genotype Data:")
    
    # Compare run distributions
    print("\nRun Orientation Distribution Comparisons (all dates combined):")
    genotypes = list(genotype_combined_results.keys())
    for i in range(len(genotypes)):
        for j in range(i+1, len(genotypes)):
            g1 = genotypes[i]
            g2 = genotypes[j]
            
            # Get all run orientations from all dates for each genotype
            runs_g1 = []
            runs_g2 = []
            
            for date, results in genotype_date_results.get(g1, {}).items():
                if 'run_results' in results and 'all_run_orientations' in results['run_results']:
                    runs_g1.extend(results["run_results"]["all_run_orientations"])
                
            for date, results in genotype_date_results.get(g2, {}).items():
                if 'run_results' in results and 'all_run_orientations' in results['run_results']:
                    runs_g2.extend(results["run_results"]["all_run_orientations"])
            
            # Kolmogorov-Smirnov test
            if len(runs_g1) > 0 and len(runs_g2) > 0:
                ks_stat, p_value = ks_2samp(runs_g1, runs_g2)
                
                print(f"  {g1} vs {g2}: KS statistic = {ks_stat:.4f}, p-value = {p_value:.6f}")
                print(f"    {g1}: {len(runs_g1)} runs, {g2}: {len(runs_g2)} runs")
                if p_value < 0.05:
                    print(f"    Significant difference detected (p < 0.05)")
            else:
                print(f"  {g1} vs {g2}: Insufficient data for statistical comparison")
    
    # Compare cast distributions
    print("\nCast Orientation Distribution Comparisons (all dates combined):")
    for i in range(len(genotypes)):
        for j in range(i+1, len(genotypes)):
            g1 = genotypes[i]
            g2 = genotypes[j]
            
            # Get all cast orientations from all dates for each genotype
            casts_g1 = []
            casts_g2 = []
            
            for date, results in genotype_date_results.get(g1, {}).items():
                if 'cast_results' in results and 'all_cast_orientations' in results['cast_results']:
                    casts_g1.extend(results["cast_results"]["all_cast_orientations"])
                
            for date, results in genotype_date_results.get(g2, {}).items():
                if 'cast_results' in results and 'all_cast_orientations' in results['cast_results']:
                    casts_g2.extend(results["cast_results"]["all_cast_orientations"])
            
            # Kolmogorov-Smirnov test
            if len(casts_g1) > 0 and len(casts_g2) > 0:
                ks_stat, p_value = ks_2samp(casts_g1, casts_g2)
                
                print(f"  {g1} vs {g2}: KS statistic = {ks_stat:.4f}, p-value = {p_value:.6f}")
                print(f"    {g1}: {len(casts_g1)} casts, {g2}: {len(casts_g2)} casts")
                if p_value < 0.05:
                    print(f"    Significant difference detected (p < 0.05)")
            else:
                print(f"  {g1} vs {g2}: Insufficient data for statistical comparison")
    
    return genotype_date_results


def compare_cast_head_angles(dataset1, dataset2, labels=None, bin_width=10, basepath=None):
    """
    Compare head angle distributions during casts between two datasets and test for statistical differences.
    
    Args:
        dataset1 (dict): First tracking data dictionary
        dataset2 (dict): Second tracking data dictionary to compare with the first
        labels (tuple): Optional tuple of (label1, label2) for the datasets
        bin_width (int): Width of orientation bins in degrees
        basepath (str): Optional base path to save output SVG files
        
    Returns:
        dict: Contains comparison statistics and test results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    import os
    from datetime import datetime
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Dataset 1', 'Dataset 2')
    
    # Analyze both datasets individually
    print(f"Analyzing {labels[0]}...")
    results1 = analyze_cast_head_angles_by_orientation(dataset1, bin_width=bin_width)
    
    print(f"Analyzing {labels[1]}...")
    results2 = analyze_cast_head_angles_by_orientation(dataset2, bin_width=bin_width)
    
    # Extract key data for comparison
    large_angles1 = results1['large_cast_angles']
    small_angles1 = results1['small_cast_angles']
    all_angles1 = results1['all_cast_angles']
    
    large_angles2 = results2['large_cast_angles']
    small_angles2 = results2['small_cast_angles']
    all_angles2 = results2['all_cast_angles']
    
    # Store statistical test results
    stat_results = {}
    
    # Function to perform statistical tests
    def perform_tests(data1, data2, name):
        if len(data1) > 0 and len(data2) > 0:
            # Kolmogorov-Smirnov test (non-parametric, distribution-free)
            ks_stat, ks_pval = stats.ks_2samp(data1, data2)
            
            # Mann-Whitney U test (non-parametric, compares medians)
            u_stat, u_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            # t-test (parametric, compares means)
            t_stat, t_pval = stats.ttest_ind(data1, data2, equal_var=False)
            
            return {
                'ks_test': {'statistic': ks_stat, 'pvalue': ks_pval},
                'mannwhitney_test': {'statistic': u_stat, 'pvalue': u_pval},
                't_test': {'statistic': t_stat, 'pvalue': t_pval},
                'n_samples1': len(data1),
                'n_samples2': len(data2),
                'mean1': np.mean(data1),
                'mean2': np.mean(data2),
                'median1': np.median(data1),
                'median2': np.median(data2),
                'std1': np.std(data1),
                'std2': np.std(data2)
            }
        else:
            return None
    
    # Perform tests for all cast types
    stat_results['large_casts'] = perform_tests(large_angles1, large_angles2, 'large casts')
    stat_results['small_casts'] = perform_tests(small_angles1, small_angles2, 'small casts')
    stat_results['all_casts'] = perform_tests(all_angles1, all_angles2, 'all casts')
    
    # Plot comparison of distributions
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create histogram bins
    angle_bins = np.linspace(-90, 90, 37)  # 36 bins covering -90° to 90°
    angle_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    # Function to plot histograms
    def plot_hist_comparison(ax, data1, data2, title, color1='blue', color2='red'):
        if len(data1) > 0 and len(data2) > 0:
            # Calculate histograms
            hist1, _ = np.histogram(data1, bins=angle_bins, density=True)
            hist2, _ = np.histogram(data2, bins=angle_bins, density=True)
            
            # Smooth histograms
            smoothed1 = gaussian_filter1d(hist1, sigma=1)
            smoothed2 = gaussian_filter1d(hist2, sigma=1)
            
            # Plot histograms
            ax.plot(angle_bin_centers, smoothed1, color=color1, linewidth=2, 
                   label=f"{labels[0]} (n={len(data1)})")
            ax.plot(angle_bin_centers, smoothed2, color=color2, linewidth=2, 
                   label=f"{labels[1]} (n={len(data2)})")
            
            # Add statistical test results if available
            test_results = stat_results.get(title.lower().replace(' ', '_'))
            if test_results:
                # Format p-values
                ks_p = test_results['ks_test']['pvalue']
                mw_p = test_results['mannwhitney_test']['pvalue']
                t_p = test_results['t_test']['pvalue']
                
                # Add significance asterisks
                ks_sig = '***' if ks_p < 0.001 else ('**' if ks_p < 0.01 else ('*' if ks_p < 0.05 else 'n.s.'))
                mw_sig = '***' if mw_p < 0.001 else ('**' if mw_p < 0.01 else ('*' if mw_p < 0.05 else 'n.s.'))
                t_sig = '***' if t_p < 0.001 else ('**' if t_p < 0.01 else ('*' if t_p < 0.05 else 'n.s.'))
                
                # Add text for statistical test results
                text_y = ax.get_ylim()[1] * 0.9
                ax.text(0.05, text_y, f"KS: p={ks_p:.4f} {ks_sig}", fontsize=9, ha='left')
                ax.text(0.05, text_y * 0.9, f"MW: p={mw_p:.4f} {mw_sig}", fontsize=9, ha='left')
                ax.text(0.05, text_y * 0.8, f"t-test: p={t_p:.4f} {t_sig}", fontsize=9, ha='left')
                
                # Add means with standard error
                mean1 = test_results['mean1']
                mean2 = test_results['mean2']
                se1 = test_results['std1'] / np.sqrt(test_results['n_samples1'])
                se2 = test_results['std2'] / np.sqrt(test_results['n_samples2'])
                
                ax.text(0.05, text_y * 0.7, f"{labels[0]}: {mean1:.2f}±{se1:.2f}°", fontsize=9, ha='left', color=color1)
                ax.text(0.05, text_y * 0.6, f"{labels[1]}: {mean2:.2f}±{se2:.2f}°", fontsize=9, ha='left', color=color2)
        
        ax.set_xlabel('Head Angle (°)')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.legend()
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Plot the three comparisons
    plot_hist_comparison(axs[0], large_angles1, large_angles2, 'Large Casts', 'blue', 'red')
    plot_hist_comparison(axs[1], small_angles1, small_angles2, 'Small Casts', 'purple', 'orange')
    plot_hist_comparison(axs[2], all_angles1, all_angles2, 'All Casts', 'green', 'brown')
    
    plt.suptitle(f'Comparison of Cast Head Angles: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
        label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
        filepath = os.path.join(basepath, f"head_angle_comparison_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved comparison plot to: {filepath}")
    
    plt.show()
    
    # Create box plots for a clearer comparison of distributions
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Function to create box plots
    def create_boxplot(ax, data1, data2, title):
        if len(data1) > 0 and len(data2) > 0:
            # Create box plot
            bp = ax.boxplot([data1, data2], labels=[labels[0], labels[1]], patch_artist=True,
                          notch=True, showfliers=False)
            
            # Customize colors
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            # Add individual data points with jitter
            for i, data in enumerate([data1, data2]):
                pos = i + 1
                jitter = np.random.normal(0, 0.04, size=len(data))
                ax.scatter(np.full_like(data, pos) + jitter, data, alpha=0.3, s=10, 
                          color='blue' if i == 0 else 'red')
            
            # Add statistical test results if available
            test_results = stat_results.get(title.lower().replace(' ', '_'))
            if test_results:
                # Get p-value from Mann-Whitney test (most appropriate for boxplots)
                p_val = test_results['mannwhitney_test']['pvalue']
                
                # Add significance bar
                if p_val < 0.05:
                    y_max = max(np.percentile(data1, 95), np.percentile(data2, 95)) + 10
                    x1, x2 = 1, 2
                    ax.plot([x1, x1, x2, x2], [y_max, y_max+5, y_max+5, y_max], lw=1.5, c='black')
                    
                    # Add stars based on significance level
                    stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
                    ax.text((x1 + x2) * 0.5, y_max + 6, stars, ha='center', va='bottom', fontsize=14)
                
                # Add p-value text
                sig_indicator = "*" if p_val < 0.05 else "n.s."
                ax.text((1 + 2) * 0.5, ax.get_ylim()[1] * 0.9, f'p = {p_val:.4f} ({sig_indicator})', 
                       ha='center', va='bottom', fontsize=10)
                
                # Add medians text
                median1 = test_results['median1']
                median2 = test_results['median2']
                ax.text(1, median1 + 5, f"Median: {median1:.2f}°", ha='center', fontsize=9)
                ax.text(2, median2 + 5, f"Median: {median2:.2f}°", ha='center', fontsize=9)
        
        ax.set_ylabel('Head Angle (°)')
        ax.set_title(title)
    
    # Create the boxplots
    create_boxplot(axs[0], large_angles1, large_angles2, 'Large Casts')
    create_boxplot(axs[1], small_angles1, small_angles2, 'Small Casts')
    create_boxplot(axs[2], all_angles1, all_angles2, 'All Casts')
    
    plt.suptitle(f'Head Angle Distributions: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
        label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
        filepath = os.path.join(basepath, f"head_angle_boxplot_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved boxplot to: {filepath}")
    
    plt.show()
    
    # NEW SECTION: Compare head angle as a function of body orientation
    # Get the data for orientation-dependent head angles
    bin_centers1 = results1['bin_centers']
    bin_centers2 = results2['bin_centers']
    
    large_mean_angles1 = results1['large_mean_angles']
    large_mean_angles2 = results2['large_mean_angles']
    
    small_mean_angles1 = results1['small_mean_angles']
    small_mean_angles2 = results2['small_mean_angles']
    
    all_mean_angles1 = results1['all_mean_angles']
    all_mean_angles2 = results2['all_mean_angles']
    
    large_smoothed1 = results1['large_smoothed']
    large_smoothed2 = results2['large_smoothed']
    
    small_smoothed1 = results1['small_smoothed']
    small_smoothed2 = results2['small_smoothed']
    
    all_smoothed1 = results1['all_smoothed']
    all_smoothed2 = results2['all_smoothed']
    
    # Create figure for orientation-dependent analysis
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # First row: Overlay plots for each cast type
    def plot_orientation_comparison(ax, centers, data1, data2, title, color1='blue', color2='red'):
        ax.plot(centers, data1, color=color1, linewidth=2, label=labels[0])
        ax.plot(centers, data2, color=color2, linewidth=2, label=labels[1])
        ax.set_xlabel('Body Orientation (°)')
        ax.set_ylabel('Mean Head Angle (°)')
        ax.set_xlim(-180, 180)
        ax.set_title(title)
        ax.legend()
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=90, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=-90, color='gray', linestyle=':', alpha=0.3)
    
    # Plot comparisons in the first row
    plot_orientation_comparison(axs[0, 0], bin_centers1, large_smoothed1, large_smoothed2, 'Large Casts', 'blue', 'red')
    plot_orientation_comparison(axs[0, 1], bin_centers1, small_smoothed1, small_smoothed2, 'Small Casts', 'purple', 'orange')
    plot_orientation_comparison(axs[0, 2], bin_centers1, all_smoothed1, all_smoothed2, 'All Casts', 'green', 'brown')
    
    # Second row: Difference plots for each cast type
    def plot_orientation_difference(ax, centers, data1, data2, title, color='purple'):
        # Calculate difference (data2 - data1)
        diff = data2 - data1
        
        # Find regions where both curves have valid data
        valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
        
        # Plot difference
        ax.plot(centers[valid_mask], diff[valid_mask], color=color, linewidth=2)
        ax.fill_between(centers[valid_mask], 0, diff[valid_mask], where=(diff[valid_mask] > 0), 
                       color=color, alpha=0.3, interpolate=True)
        ax.fill_between(centers[valid_mask], diff[valid_mask], 0, where=(diff[valid_mask] < 0), 
                       color=color, alpha=0.3, interpolate=True)
        
        ax.set_xlabel('Body Orientation (°)')
        ax.set_ylabel('Difference in Head Angle (°)')
        ax.set_xlim(-180, 180)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'Difference in {title} ({labels[1]} - {labels[0]})')
        
        # Add reference lines at key orientations
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=90, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=-90, color='gray', linestyle=':', alpha=0.3)
        
        # Add mean difference as text
        mean_diff = np.nanmean(diff)
        ax.text(0.05, 0.95, f'Mean difference: {mean_diff:.2f}°', 
               transform=ax.transAxes, fontsize=10, va='top', ha='left')
    
    # Plot differences in the second row
    plot_orientation_difference(axs[1, 0], bin_centers1, large_smoothed1, large_smoothed2, 'Large Casts', 'darkred')
    plot_orientation_difference(axs[1, 1], bin_centers1, small_smoothed1, small_smoothed2, 'Small Casts', 'darkblue')
    plot_orientation_difference(axs[1, 2], bin_centers1, all_smoothed1, all_smoothed2, 'All Casts', 'darkgreen')
    
    plt.suptitle(f'Comparison of Head Angles by Body Orientation: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3)
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(basepath, f"head_angle_by_orientation_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved orientation-dependent analysis to: {filepath}")
    
    plt.show()
    
    # Print summary of comparison results
    print("\nSummary of Cast Head Angle Comparison:")
    print(f"Comparing {labels[0]} vs {labels[1]}")
    
    for cast_type in ['large_casts', 'small_casts', 'all_casts']:
        results = stat_results.get(cast_type)
        if results:
            print(f"\n{cast_type.replace('_', ' ').title()}:")
            n1 = results['n_samples1']
            n2 = results['n_samples2']
            mean1 = results['mean1']
            mean2 = results['mean2']
            median1 = results['median1']
            median2 = results['median2']
            std1 = results['std1']
            std2 = results['std2']
            
            print(f"  {labels[0]}: n={n1}, mean={mean1:.2f}±{std1:.2f}°, median={median1:.2f}°")
            print(f"  {labels[1]}: n={n2}, mean={mean2:.2f}±{std2:.2f}°, median={median2:.2f}°")
            print(f"  Mean difference: {mean2-mean1:.2f}°")
            print(f"  Median difference: {median2-median1:.2f}°")
            
            # Print statistical test results
            ks_p = results['ks_test']['pvalue']
            mw_p = results['mannwhitney_test']['pvalue']
            t_p = results['t_test']['pvalue']
            
            print(f"  Kolmogorov-Smirnov test: p={ks_p:.6f} {'(significant)' if ks_p < 0.05 else '(not significant)'}")
            print(f"  Mann-Whitney U test: p={mw_p:.6f} {'(significant)' if mw_p < 0.05 else '(not significant)'}")
            print(f"  t-test: p={t_p:.6f} {'(significant)' if t_p < 0.05 else '(not significant)'}")
    
    # Also analyze specific orientation regions - FIX HERE
    print("\nAnalysis of Head Angles at Key Orientations:")
    
    # Define key orientation regions
    regions = [
        ('Upwind', -30, 30),
        ('Downwind', [-180, -150, 150, 180]),
        ('Perpendicular', [-120, -60, 60, 120])
    ]
    
    for region_info in regions:
        region_name = region_info[0]
        limits = region_info[1:]
        
        print(f"\n{region_name} Orientation Region:")
        
        # Function to filter angles by orientation - FIX HERE
        def filter_by_orientation(angles, orientations, limits):
            # Handle different region formats
            if isinstance(limits, list) and len(limits) == 4:
                # Multiple ranges: e.g., [-180, -150, 150, 180] means -180 to -150 OR 150 to 180
                mask1 = (orientations >= limits[0]) & (orientations <= limits[1])
                mask2 = (orientations >= limits[2]) & (orientations <= limits[3])
                return angles[mask1 | mask2]
            elif len(limits) == 2 and isinstance(limits[0], (int, float)) and isinstance(limits[1], (int, float)):
                # Simple range: e.g., -30 to 30
                lower, upper = limits
                return angles[(orientations >= lower) & (orientations <= upper)]
            elif isinstance(limits[0], (int, float)):
                # Single value - probably the upwind case which has 2 values in the tuple
                lower, upper = limits[0], region_info[2]
                return angles[(orientations >= lower) & (orientations <= upper)]
            else:
                print(f"Warning: Unhandled limits format: {limits}")
                return np.array([])
        
        # Get angles in each region for both datasets
        for cast_type, name in [('large', 'Large Casts'), ('small', 'Small Casts'), ('all', 'All Casts')]:
            angles1 = results1[f'{cast_type}_cast_angles']
            orientations1 = results1[f'{cast_type}_cast_orientations']
            
            angles2 = results2[f'{cast_type}_cast_angles']
            orientations2 = results2[f'{cast_type}_cast_orientations']
            
            # Filter angles by orientation region
            filtered_angles1 = filter_by_orientation(angles1, orientations1, limits)
            filtered_angles2 = filter_by_orientation(angles2, orientations2, limits)
            
            if len(filtered_angles1) > 0 and len(filtered_angles2) > 0:
                # Calculate statistics
                mean1 = np.mean(filtered_angles1)
                mean2 = np.mean(filtered_angles2)
                std1 = np.std(filtered_angles1)
                std2 = np.std(filtered_angles2)
                
                # Run statistical tests
                _, t_pval = stats.ttest_ind(filtered_angles1, filtered_angles2, equal_var=False)
                _, mw_pval = stats.mannwhitneyu(filtered_angles1, filtered_angles2, alternative='two-sided')
                
                print(f"  {name}:")
                print(f"    {labels[0]}: n={len(filtered_angles1)}, mean={mean1:.2f}±{std1:.2f}°")
                print(f"    {labels[1]}: n={len(filtered_angles2)}, mean={mean2:.2f}±{std2:.2f}°")
                print(f"    Difference: {mean2-mean1:.2f}° (t-test p={t_pval:.4f}, MW test p={mw_pval:.4f})")
    
    # Return results for further analysis
    return {
        'results1': results1,
        'results2': results2,
        'statistical_tests': stat_results,
        'labels': labels
    }


def compare_run_orientations(dataset1, dataset2, labels=None, bin_width=12, basepath=None):
    """
    Compare run orientation distributions between two datasets and test for statistical differences.
    
    Args:
        dataset1 (dict): First tracking data dictionary
        dataset2 (dict): Second tracking data dictionary to compare with the first
        labels (tuple): Optional tuple of (label1, label2) for the datasets
        bin_width (int): Width of orientation bins in degrees
        basepath (str): Optional base path to save output SVG files
        
    Returns:
        dict: Contains comparison statistics and test results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    import os
    from datetime import datetime
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Dataset 1', 'Dataset 2')
    
    # Analyze both datasets individually
    print(f"Analyzing {labels[0]}...")
    results1 = analyze_run_orientations_all(dataset1)
    
    print(f"Analyzing {labels[1]}...")
    results2 = analyze_run_orientations_all(dataset2)
    
    # Extract key data for comparison
    large_orientations1 = results1['large_run_orientations']
    small_orientations1 = results1['small_run_orientations']
    all_orientations1 = results1['all_run_orientations']
    
    large_orientations2 = results2['large_run_orientations']
    small_orientations2 = results2['small_run_orientations']
    all_orientations2 = results2['all_run_orientations']
    
    # Store statistical test results
    stat_results = {}
    
    # Function to perform statistical tests
    def perform_tests(data1, data2, name):
        if len(data1) > 0 and len(data2) > 0:
            # Kolmogorov-Smirnov test (non-parametric, distribution-free)
            ks_stat, ks_pval = stats.ks_2samp(data1, data2)
            
            # Watson-Williams test for circular data (parametric)
            # First convert to radians for circular statistics
            data1_rad = np.radians(data1)
            data2_rad = np.radians(data2)
            
            # For Watson U² test, we need to combine, sort, and find ranks
            try:
                import pycircstat
                watson_u2, watson_pval = pycircstat.watson_williams_test(data1_rad, data2_rad)
            except:
                # Fallback if pycircstat not available
                watson_u2, watson_pval = None, None
            
            # Mean resultant vector length (measure of concentration for circular data)
            r1 = np.abs(np.mean(np.exp(1j * data1_rad)))
            r2 = np.abs(np.mean(np.exp(1j * data2_rad)))
            
            # Calculate circular means
            mean1 = np.angle(np.mean(np.exp(1j * data1_rad)), deg=True)
            mean2 = np.angle(np.mean(np.exp(1j * data2_rad)), deg=True)
            
            return {
                'ks_test': {'statistic': ks_stat, 'pvalue': ks_pval},
                'watson_williams_test': {'statistic': watson_u2, 'pvalue': watson_pval},
                'n_samples1': len(data1),
                'n_samples2': len(data2),
                'circular_mean1': mean1,
                'circular_mean2': mean2,
                'resultant_length1': r1,
                'resultant_length2': r2
            }
        else:
            return None
    
    # Perform tests for all run types
    stat_results['large_runs'] = perform_tests(large_orientations1, large_orientations2, 'large runs')
    stat_results['small_runs'] = perform_tests(small_orientations1, small_orientations2, 'small runs')
    stat_results['all_runs'] = perform_tests(all_orientations1, all_orientations2, 'all runs')
    
    # Create figure for comparing histograms
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create histogram bins
    bins = np.linspace(-180, 180, int(360/bin_width) + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Function to plot histogram comparison
    def plot_hist_comparison(ax, data1, data2, title, color1='blue', color2='red'):
        if len(data1) > 0 and len(data2) > 0:
            # Calculate histograms
            hist1, _ = np.histogram(data1, bins=bins, density=True)
            hist2, _ = np.histogram(data2, bins=bins, density=True)
            
            # Smooth histograms
            smoothed1 = gaussian_filter1d(hist1, sigma=1)
            smoothed2 = gaussian_filter1d(hist2, sigma=1)
            
            # Plot histograms
            ax.plot(bin_centers, smoothed1, color=color1, linewidth=2, 
                   label=f"{labels[0]} (n={len(data1)})")
            ax.plot(bin_centers, smoothed2, color=color2, linewidth=2, 
                   label=f"{labels[1]} (n={len(data2)})")
            
            # Add statistical test results if available
            test_results = stat_results.get(title.lower().replace(' ', '_'))
            if test_results:
                # Format p-values
                ks_p = test_results['ks_test']['pvalue']
                ww_p = test_results['watson_williams_test']['pvalue'] if test_results['watson_williams_test'] else None
                
                # Add significance asterisks
                ks_sig = '***' if ks_p < 0.001 else ('**' if ks_p < 0.01 else ('*' if ks_p < 0.05 else 'n.s.'))
                ww_sig = '***' if ww_p and ww_p < 0.001 else ('**' if ww_p and ww_p < 0.01 else ('*' if ww_p and ww_p < 0.05 else 'n.s.'))
                
                # Add text for statistical test results
                y_pos = ax.get_ylim()[1] * 0.95
                text_props = {'fontsize': 9, 'ha': 'left', 'transform': ax.transAxes}
                
                stat_text = f"KS test: p={ks_p:.4f} ({ks_sig})"
                if ww_p:
                    stat_text += f"\nW-W test: p={ww_p:.4f} ({ww_sig})"
                
                ax.text(0.05, 0.95, stat_text, va='top', **text_props)
                
                # Add circular means with directional information
                mean1 = test_results['circular_mean1']
                mean2 = test_results['circular_mean2']
                r1 = test_results['resultant_length1']
                r2 = test_results['resultant_length2']
                
                ax.text(0.05, 0.05, 
                      f"{labels[0]}: {mean1:.1f}°, r={r1:.2f}\n{labels[1]}: {mean2:.1f}°, r={r2:.2f}", 
                      va='bottom', **text_props)
                
                # Add markers for mean directions
                arrow_length = 0.9 * max(np.max(smoothed1), np.max(smoothed2))
                
                # FIX: Changed arrow parameter names to match matplotlib's expected parameters
                arrow_props = {'width': 0.008, 'headwidth': 0.02, 'headlength': 0.02, 'fc': 'black', 'ec': 'black', 'alpha': 0.7}
                
                # Plot arrows at mean directions
                if not np.isnan(mean1):
                    ax.annotate('', xy=(mean1, arrow_length), xytext=(mean1, 0),
                               arrowprops=dict(**arrow_props, color=color1))
                
                if not np.isnan(mean2):
                    ax.annotate('', xy=(mean2, arrow_length), xytext=(mean2, 0),
                               arrowprops=dict(**arrow_props, color=color2))
        
        ax.set_xlabel('Orientation (°)')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(-180, 180)
        ax.set_title(title)
        ax.legend(loc='upper right')
        
        # Add reference lines at 0, 90, -90 degrees
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=90, color='black', linestyle=':', alpha=0.3)
        ax.axvline(x=-90, color='black', linestyle=':', alpha=0.3)
    
    # Plot the three comparisons
    plot_hist_comparison(axs[0], large_orientations1, large_orientations2, 'Large Runs', 'darkred', 'salmon')
    plot_hist_comparison(axs[1], small_orientations1, small_orientations2, 'Small Runs', 'darkblue', 'skyblue')
    plot_hist_comparison(axs[2], all_orientations1, all_orientations2, 'All Runs', 'darkgreen', 'lightgreen')
    
    plt.suptitle(f'Comparison of Run Orientations: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
        label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
        filepath = os.path.join(basepath, f"run_orientation_comparison_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved comparison plot to: {filepath}")
    
    plt.show()
    
    # Create circular plots for orientation data
    fig = plt.figure(figsize=(15, 5))
    
    # Function to create polar histogram plot
    def plot_circular_hist(ax, data, bins=36, color='blue', alpha=0.5, label=None):
        if len(data) == 0:
            return
            
        # Convert to radians for circular histogram
        data_rad = np.radians(data)
        
        # Create bins in radians
        bins_rad = np.linspace(-np.pi, np.pi, bins+1)
        
        # Calculate histogram
        hist, _ = np.histogram(data_rad, bins=bins_rad, density=True)
        
        # Smooth histogram
        hist = gaussian_filter1d(hist, sigma=1)
        
        # Plot histogram as filled area
        bin_width = 2 * np.pi / bins
        bin_centers = bins_rad[:-1] + bin_width/2
        
        # Scale histogram values 
        max_radius = 1.0
        hist_scaled = hist / np.max(hist) * max_radius if np.max(hist) > 0 else hist
        
        # Plot as filled line
        ax.plot(bin_centers, hist_scaled, color=color, linewidth=2, label=label)
        ax.fill_between(bin_centers, 0, hist_scaled, color=color, alpha=alpha)
        
        # Add mean direction as arrow
        mean_angle = np.angle(np.mean(np.exp(1j * data_rad)))
        mean_r = np.abs(np.mean(np.exp(1j * data_rad)))
        
        # FIX: Use native arrow function for polar plots instead of annotate
        ax.arrow(0, 0, mean_angle, mean_r*0.8, 
                alpha=0.8, width=0.02, head_width=0.1, head_length=0.1,
                fc=color, ec='black')
    
    # Create three polar subplots
    ax1 = fig.add_subplot(131, projection='polar')
    ax2 = fig.add_subplot(132, projection='polar')
    ax3 = fig.add_subplot(133, projection='polar')
    
    # Plot polar histograms
    # Large runs
    if len(large_orientations1) > 0 or len(large_orientations2) > 0:
        plot_circular_hist(ax1, large_orientations1, color='darkred', alpha=0.3, label=labels[0])
        plot_circular_hist(ax1, large_orientations2, color='salmon', alpha=0.3, label=labels[1])
        ax1.set_title('Large Runs')
    
    # Small runs
    if len(small_orientations1) > 0 or len(small_orientations2) > 0:
        plot_circular_hist(ax2, small_orientations1, color='darkblue', alpha=0.3, label=labels[0])
        plot_circular_hist(ax2, small_orientations2, color='skyblue', alpha=0.3, label=labels[1])
        ax2.set_title('Small Runs')
    
    # All runs
    if len(all_orientations1) > 0 or len(all_orientations2) > 0:
        plot_circular_hist(ax3, all_orientations1, color='darkgreen', alpha=0.3, label=labels[0])
        plot_circular_hist(ax3, all_orientations2, color='lightgreen', alpha=0.3, label=labels[1])
        ax3.set_title('All Runs')
    
    # Configure polar axes
    for ax in [ax1, ax2, ax3]:
        # Set ticks and labels
        ax.set_theta_zero_location('N')  # 0 at the top
        ax.set_theta_direction(-1)      # clockwise
        ax.set_thetagrids([0, 90, 180, 270], ['0°', '90°', '±180°', '-90°'])
        ax.set_rlabel_position(45)      # Move radial labels
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.suptitle(f'Circular Comparison of Run Orientations: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(basepath, f"run_orientation_circular_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved circular plot to: {filepath}")
    
    plt.show()
    
    # Print summary of comparison results
    print("\nSummary of Run Orientation Comparison:")
    print(f"Comparing {labels[0]} vs {labels[1]}")
    
    for run_type in ['large_runs', 'small_runs', 'all_runs']:
        results = stat_results.get(run_type)
        if results:
            print(f"\n{run_type.replace('_', ' ').title()}:")
            n1 = results['n_samples1']
            n2 = results['n_samples2']
            mean1 = results['circular_mean1']
            mean2 = results['circular_mean2']
            r1 = results['resultant_length1']
            r2 = results['resultant_length2']
            
            print(f"  {labels[0]}: n={n1}, mean direction={mean1:.1f}°, concentration={r1:.2f}")
            print(f"  {labels[1]}: n={n2}, mean direction={mean2:.1f}°, concentration={r2:.2f}")
            print(f"  Direction difference: {(mean2-mean1)%360:.1f}°")
            
            # Print statistical test results
            ks_p = results['ks_test']['pvalue']
            ww_p = results['watson_williams_test']['pvalue'] if results['watson_williams_test'] else None
            
            print(f"  Kolmogorov-Smirnov test: p={ks_p:.6f} {'(significant)' if ks_p < 0.05 else '(not significant)'}")
            if ww_p:
                print(f"  Watson-Williams test: p={ww_p:.6f} {'(significant)' if ww_p < 0.05 else '(not significant)'}")
    
    # Return results for further analysis
    return {
        'results1': results1,
        'results2': results2,
        'statistical_tests': stat_results,
        'labels': labels
    }


def compare_cast_orientations(dataset1, dataset2, labels=None, bin_width=10, basepath=None):
    """
    Compare cast orientation distributions between two datasets and test for statistical differences.
    
    Args:
        dataset1 (dict): First tracking data dictionary
        dataset2 (dict): Second tracking data dictionary to compare with the first
        labels (tuple): Optional tuple of (label1, label2) for the datasets
        bin_width (int): Width of orientation bins in degrees
        basepath (str): Optional base path to save output SVG files
        
    Returns:
        dict: Contains comparison statistics and test results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    import os
    from datetime import datetime
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Dataset 1', 'Dataset 2')
    
    # Analyze both datasets individually
    print(f"Analyzing {labels[0]}...")
    results1 = analyze_cast_orientations_all(dataset1)
    
    print(f"Analyzing {labels[1]}...")
    results2 = analyze_cast_orientations_all(dataset2)
    
    # Extract key data for comparison
    large_orientations1 = results1['large_cast_orientations']
    small_orientations1 = results1['small_cast_orientations']
    all_orientations1 = results1['all_cast_orientations']
    
    large_orientations2 = results2['large_cast_orientations']
    small_orientations2 = results2['small_cast_orientations']
    all_orientations2 = results2['all_cast_orientations']
    
    # Store statistical test results
    stat_results = {}
    
    # Function to perform statistical tests
    def perform_tests(data1, data2, name):
        if len(data1) > 0 and len(data2) > 0:
            # Kolmogorov-Smirnov test (non-parametric, distribution-free)
            ks_stat, ks_pval = stats.ks_2samp(data1, data2)
            
            # For circular data, we need specialized tests
            # First convert to radians for circular statistics
            data1_rad = np.radians(data1)
            data2_rad = np.radians(data2)
            
            # Try to use pycircstat if available, otherwise fallback
            try:
                import pycircstat
                # Watson-Williams test (parametric, for circular data)
                ww_stat, ww_pval = pycircstat.watson_williams_test(data1_rad, data2_rad)
            except:
                # Fallback if pycircstat not available
                ww_stat, ww_pval = None, None
            
            # Calculate circular means and concentration parameters
            mean1 = np.degrees(np.angle(np.mean(np.exp(1j * data1_rad))))
            mean2 = np.degrees(np.angle(np.mean(np.exp(1j * data2_rad))))
            r1 = np.abs(np.mean(np.exp(1j * data1_rad)))  # resultant vector length (concentration)
            r2 = np.abs(np.mean(np.exp(1j * data2_rad)))
            
            return {
                'ks_test': {'statistic': ks_stat, 'pvalue': ks_pval},
                'watson_williams_test': {'statistic': ww_stat, 'pvalue': ww_pval},
                'n_samples1': len(data1),
                'n_samples2': len(data2),
                'circular_mean1': mean1,
                'circular_mean2': mean2,
                'resultant_length1': r1,
                'resultant_length2': r2
            }
        else:
            return None
    
    # Perform tests for all cast types
    stat_results['large_casts'] = perform_tests(large_orientations1, large_orientations2, 'large casts')
    stat_results['small_casts'] = perform_tests(small_orientations1, small_orientations2, 'small casts')
    stat_results['all_casts'] = perform_tests(all_orientations1, all_orientations2, 'all casts')
    
    # Create figure with subplots for distribution comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create histogram bins
    bins = np.linspace(-180, 180, int(360/bin_width) + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Function to plot histogram comparison
    def plot_hist_comparison(ax, data1, data2, title, color1='blue', color2='red'):
        if len(data1) > 0 and len(data2) > 0:
            # Calculate histograms
            hist1, _ = np.histogram(data1, bins=bins, density=True)
            hist2, _ = np.histogram(data2, bins=bins, density=True)
            
            # Smooth histograms
            smoothed1 = gaussian_filter1d(hist1, sigma=1)
            smoothed2 = gaussian_filter1d(hist2, sigma=1)
            
            # Plot histograms with raw data points
            ax.plot(bin_centers, hist1, 'k-', alpha=0.2, linewidth=0.5)
            ax.plot(bin_centers, hist2, 'k-', alpha=0.2, linewidth=0.5)
            
            # Plot smoothed curves
            ax.plot(bin_centers, smoothed1, color=color1, linewidth=2, 
                   label=f"{labels[0]} (n={len(data1)})")
            ax.plot(bin_centers, smoothed2, color=color2, linewidth=2, 
                   label=f"{labels[1]} (n={len(data2)})")
            
            # Add statistical test results if available
            test_results = stat_results.get(title.lower().replace(' ', '_'))
            if test_results:
                # Format p-values
                ks_p = test_results['ks_test']['pvalue']
                ww_p = test_results['watson_williams_test']['pvalue'] if test_results['watson_williams_test'] else None
                
                # Add significance asterisks
                ks_sig = '***' if ks_p < 0.001 else ('**' if ks_p < 0.01 else ('*' if ks_p < 0.05 else 'n.s.'))
                ww_sig = '***' if ww_p and ww_p < 0.001 else ('**' if ww_p and ww_p < 0.01 else ('*' if ww_p and ww_p < 0.05 else 'n.s.'))
                
                # Add text for statistical test results
                y_pos = 0.95
                text_props = {'fontsize': 9, 'ha': 'left', 'transform': ax.transAxes}
                
                stat_text = f"KS test: p={ks_p:.4f} ({ks_sig})"
                if ww_p:
                    stat_text += f"\nW-W test: p={ww_p:.4f} ({ww_sig})"
                
                ax.text(0.05, y_pos, stat_text, va='top', **text_props)
                
                # Add circular means with directional information
                mean1 = test_results['circular_mean1']
                mean2 = test_results['circular_mean2']
                r1 = test_results['resultant_length1']
                r2 = test_results['resultant_length2']
                
                mean_text = (f"{labels[0]}: {mean1:.1f}°, r={r1:.2f}\n"
                            f"{labels[1]}: {mean2:.1f}°, r={r2:.2f}")
                
                ax.text(0.05, 0.05, mean_text, va='bottom', **text_props)
                
                # Add arrows at mean directions
                arrow_height = 0.9 * max(np.max(smoothed1), np.max(smoothed2))
                
                # Add arrows at mean directions if they're valid
                if not np.isnan(mean1):
                    ax.annotate('', xy=(mean1, arrow_height), xytext=(mean1, 0),
                               arrowprops=dict(width=0.008, headwidth=0.02, headlength=0.02, 
                                             fc=color1, ec=color1, alpha=0.7))
                
                if not np.isnan(mean2):
                    ax.annotate('', xy=(mean2, arrow_height), xytext=(mean2, 0),
                               arrowprops=dict(width=0.008, headwidth=0.02, headlength=0.02, 
                                             fc=color2, ec=color2, alpha=0.7))
                
                # Add significance bar if p-value is significant
                if ks_p < 0.05 or (ww_p and ww_p < 0.05):
                    y_max = ax.get_ylim()[1] * 0.9
                    ax.plot([-180, 180], [y_max, y_max], 'k-', linewidth=1.5)
                    ax.text(0, y_max * 1.05, "Significant difference", 
                          ha='center', va='bottom', fontsize=10)
        
        # Add reference lines and format
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=180, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=-180, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Cast Orientation (°)')
        ax.set_ylabel('Probability Density')
        ax.set_xlim(-180, 180)
        ax.set_title(title)
        ax.legend(loc='upper right')
    
    # Plot the three comparisons
    plot_hist_comparison(axs[0], large_orientations1, large_orientations2, 'Large Casts', 'darkred', 'salmon')
    plot_hist_comparison(axs[1], small_orientations1, small_orientations2, 'Small Casts', 'darkblue', 'skyblue')
    plot_hist_comparison(axs[2], all_orientations1, all_orientations2, 'All Casts', 'darkgreen', 'lightgreen')
    
    plt.suptitle(f'Comparison of Cast Orientations: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
        label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
        filepath = os.path.join(basepath, f"cast_orientation_comparison_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved comparison plot to: {filepath}")
    
    plt.show()
    
    # Create circular plots for better visualization of cast orientations
    fig = plt.figure(figsize=(15, 5))
    
    # Function to create polar histogram plot
    def plot_circular_hist(ax, data, bins=36, color='blue', alpha=0.5, label=None):
        if len(data) == 0:
            return
            
        # Convert to radians for circular histogram
        data_rad = np.radians(data)
        
        # Create bins in radians
        bins_rad = np.linspace(-np.pi, np.pi, bins+1)
        
        # Calculate histogram
        hist, _ = np.histogram(data_rad, bins=bins_rad, density=True)
        
        # Smooth histogram
        hist = gaussian_filter1d(hist, sigma=1)
        
        # Plot histogram as filled area
        bin_width = 2 * np.pi / bins
        bin_centers = bins_rad[:-1] + bin_width/2
        
        # Scale histogram values 
        max_radius = 1.0
        hist_scaled = hist / np.max(hist) * max_radius if np.max(hist) > 0 else hist
        
        # Plot as filled line
        ax.plot(bin_centers, hist_scaled, color=color, linewidth=2, label=label)
        ax.fill_between(bin_centers, 0, hist_scaled, color=color, alpha=alpha)
        
        # Add mean direction as arrow
        mean_angle = np.angle(np.mean(np.exp(1j * data_rad)))
        mean_r = np.abs(np.mean(np.exp(1j * data_rad)))
        
        # Add arrow showing mean direction
        ax.arrow(0, 0, mean_angle, mean_r*0.8, 
                alpha=0.8, width=0.02, head_width=0.1, head_length=0.1,
                fc=color, ec='black')
    
    # Create three polar subplots
    ax1 = fig.add_subplot(131, projection='polar')
    ax2 = fig.add_subplot(132, projection='polar')
    ax3 = fig.add_subplot(133, projection='polar')
    
    # Plot polar histograms
    # Large casts
    if len(large_orientations1) > 0 or len(large_orientations2) > 0:
        plot_circular_hist(ax1, large_orientations1, color='darkred', alpha=0.3, label=labels[0])
        plot_circular_hist(ax1, large_orientations2, color='salmon', alpha=0.3, label=labels[1])
        ax1.set_title('Large Casts')
    
    # Small casts
    if len(small_orientations1) > 0 or len(small_orientations2) > 0:
        plot_circular_hist(ax2, small_orientations1, color='darkblue', alpha=0.3, label=labels[0])
        plot_circular_hist(ax2, small_orientations2, color='skyblue', alpha=0.3, label=labels[1])
        ax2.set_title('Small Casts')
    
    # All casts
    if len(all_orientations1) > 0 or len(all_orientations2) > 0:
        plot_circular_hist(ax3, all_orientations1, color='darkgreen', alpha=0.3, label=labels[0])
        plot_circular_hist(ax3, all_orientations2, color='lightgreen', alpha=0.3, label=labels[1])
        ax3.set_title('All Casts')
    
    # Configure polar axes
    for ax in [ax1, ax2, ax3]:
        # Set ticks and labels
        ax.set_theta_zero_location('N')  # 0 at the top
        ax.set_theta_direction(-1)      # clockwise
        ax.set_thetagrids([0, 90, 180, 270], ['0°', '90°', '±180°', '-90°'])
        ax.set_rlabel_position(45)      # Move radial labels
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.suptitle(f'Circular Comparison of Cast Orientations: {labels[0]} vs {labels[1]}', fontsize=14)
    plt.tight_layout()
    
    # Save figure if basepath is provided
    if basepath is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(basepath, f"cast_orientation_circular_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved circular plot to: {filepath}")
    
    plt.show()
    
    # Print summary of comparison results
    print("\nSummary of Cast Orientation Comparison:")
    print(f"Comparing {labels[0]} vs {labels[1]}")
    
    for cast_type in ['large_casts', 'small_casts', 'all_casts']:
        results = stat_results.get(cast_type)
        if results:
            print(f"\n{cast_type.replace('_', ' ').title()}:")
            n1 = results['n_samples1']
            n2 = results['n_samples2']
            mean1 = results['circular_mean1']
            mean2 = results['circular_mean2']
            r1 = results['resultant_length1']
            r2 = results['resultant_length2']
            
            print(f"  {labels[0]}: n={n1}, mean direction={mean1:.1f}°, concentration={r1:.2f}")
            print(f"  {labels[1]}: n={n2}, mean direction={mean2:.1f}°, concentration={r2:.2f}")
            print(f"  Direction difference: {abs(mean2-mean1)%360:.1f}°")
            
            # Print statistical test results
            ks_p = results['ks_test']['pvalue']
            ww_p = results['watson_williams_test']['pvalue'] if results['watson_williams_test'] else None
            
            print(f"  Kolmogorov-Smirnov test: p={ks_p:.6f} {'(significant)' if ks_p < 0.05 else '(not significant)'}")
            if ww_p:
                print(f"  Watson-Williams test: p={ww_p:.6f} {'(significant)' if ww_p < 0.05 else '(not significant)'}")
    
    # Return results for further analysis
    return {
        'results1': results1,
        'results2': results2,
        'statistical_tests': stat_results,
        'labels': labels
    }



def compare_cast_directions_perpendicular(genotype1_data, genotype2_data, labels=None, angle_width=5, min_frame=3, basepath=None):
    """
    Compare upstream and downstream cast distributions between two genotypes when larvae are perpendicular to flow.
    
    Args:
        genotype1_data (dict): Tracking data dictionary for first genotype
        genotype2_data (dict): Tracking data dictionary for second genotype
        labels (tuple): Optional tuple of (label1, label2) for the genotypes
        angle_width (int): Width of perpendicular orientation sector in degrees
        min_frame (int): Minimum number of frames to consider for a cast
        basepath (str): Base path for saving output files
        
    Returns:
        dict: Contains comparison statistics and test results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import os
    from datetime import datetime
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Genotype 1', 'Genotype 2')
    
    # Create safe filenames from labels
    label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
    label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
    
    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure basepath exists
    if basepath is not None:
        os.makedirs(basepath, exist_ok=True)
    
    # Analyze both genotypes separately
    print(f"Analyzing {labels[0]}...")
    genotype1_results = analyze_perpendicular_cast_directions_new(
        genotype1_data, angle_width=angle_width, min_frame=min_frame, basepath=None)
    
    print(f"Analyzing {labels[1]}...")
    genotype2_results = analyze_perpendicular_cast_directions_new(
        genotype2_data, angle_width=angle_width, min_frame=min_frame, basepath=None)
    
    # Store comparison results
    comparison_results = {
        'angle_width': angle_width,
        'labels': labels,
        'statistical_tests': {},
        'counts': {}
    }
    
    # Cast types to analyze
    cast_types = ['all', 'large', 'small']
    
    # Colors for each genotype
    genotype_colors = {
        labels[0]: {
            'upstream': '#1f77b4',   # blue
            'downstream': '#9ecae1'  # light blue
        },
        labels[1]: {
            'upstream': '#d62728',   # red
            'downstream': '#ff9896'  # light red
        }
    }
    
    # Create figure for comparison
    n_cast_types = sum(1 for ct in cast_types 
                       if len(genotype1_results['larva_probabilities'][ct]['upstream']) > 0 
                       and len(genotype2_results['larva_probabilities'][ct]['upstream']) > 0)
    
    if n_cast_types == 0:
        print("No valid cast types found with sufficient data in both genotypes.")
        return
    
    fig, axes = plt.subplots(1, n_cast_types, figsize=(4.5 * n_cast_types, 6))
    
    # Handle the case of a single subplot
    if n_cast_types == 1:
        axes = [axes]
    
    # Variable to store statistical results for text file
    stat_results_txt = []
    stat_results_txt.append(f"Cast Direction Comparison: {labels[0]} vs {labels[1]}")
    stat_results_txt.append(f"Analysis timestamp: {timestamp}")
    stat_results_txt.append(f"Perpendicular angle width: ±{angle_width}°")
    stat_results_txt.append("\n")
    
    # Process each cast type
    ax_idx = 0  # Counter for valid subplots
    for cast_type in cast_types:
        # Skip if not enough data in either genotype
        if (len(genotype1_results['larva_probabilities'][cast_type]['upstream']) == 0 or
            len(genotype2_results['larva_probabilities'][cast_type]['upstream']) == 0):
            continue
        
        ax = axes[ax_idx]
        ax_idx += 1
        
        # Get the per-larva probabilities
        g1_upstream = np.array(genotype1_results['larva_probabilities'][cast_type]['upstream'])
        g1_downstream = np.array(genotype1_results['larva_probabilities'][cast_type]['downstream'])
        g2_upstream = np.array(genotype2_results['larva_probabilities'][cast_type]['upstream'])
        g2_downstream = np.array(genotype2_results['larva_probabilities'][cast_type]['downstream'])
        
        # Get raw counts
        g1_up_count = genotype1_results['total_counts'][cast_type]['upstream']
        g1_down_count = genotype1_results['total_counts'][cast_type]['downstream']
        g2_up_count = genotype2_results['total_counts'][cast_type]['upstream']
        g2_down_count = genotype2_results['total_counts'][cast_type]['downstream']
        
        comparison_results['counts'][cast_type] = {
            labels[0]: {'upstream': g1_up_count, 'downstream': g1_down_count},
            labels[1]: {'upstream': g2_up_count, 'downstream': g2_down_count}
        }
        
        # Calculate statistics
        g1_up_mean = np.mean(g1_upstream)
        g1_up_std = np.std(g1_upstream)
        g1_up_sem = g1_up_std / np.sqrt(len(g1_upstream))
        g1_up_median = np.median(g1_upstream)
        
        g1_down_mean = np.mean(g1_downstream)
        g1_down_std = np.std(g1_downstream)
        g1_down_sem = g1_down_std / np.sqrt(len(g1_downstream))
        g1_down_median = np.median(g1_downstream)
        
        g2_up_mean = np.mean(g2_upstream)
        g2_up_std = np.std(g2_upstream)
        g2_up_sem = g2_up_std / np.sqrt(len(g2_upstream))
        g2_up_median = np.median(g2_upstream)
        
        g2_down_mean = np.mean(g2_downstream)
        g2_down_std = np.std(g2_downstream)
        g2_down_sem = g2_down_std / np.sqrt(len(g2_downstream))
        g2_down_median = np.median(g2_downstream)
        
        # Statistical tests
        # 1. Compare upstream probabilities between genotypes
        up_ttest = stats.ttest_ind(g1_upstream, g2_upstream, equal_var=False)
        up_mw = stats.mannwhitneyu(g1_upstream, g2_upstream)
        
        # 2. Compare downstream probabilities between genotypes
        down_ttest = stats.ttest_ind(g1_downstream, g2_downstream, equal_var=False)
        down_mw = stats.mannwhitneyu(g1_downstream, g2_downstream)
        
        # 3. Compare within each genotype (upstream vs. downstream)
        g1_within_ttest = stats.ttest_rel(g1_upstream, g1_downstream)
        g2_within_ttest = stats.ttest_rel(g2_upstream, g2_downstream)
        
        # Store results
        comparison_results['statistical_tests'][cast_type] = {
            'upstream_comparison': {
                'ttest': {'statistic': up_ttest.statistic, 'pvalue': up_ttest.pvalue},
                'mannwhitney': {'statistic': up_mw.statistic, 'pvalue': up_mw.pvalue}
            },
            'downstream_comparison': {
                'ttest': {'statistic': down_ttest.statistic, 'pvalue': down_ttest.pvalue},
                'mannwhitney': {'statistic': down_mw.statistic, 'pvalue': down_mw.pvalue}
            },
            'within_genotype': {
                labels[0]: {'statistic': g1_within_ttest.statistic, 'pvalue': g1_within_ttest.pvalue},
                labels[1]: {'statistic': g2_within_ttest.statistic, 'pvalue': g2_within_ttest.pvalue}
            },
            'means': {
                labels[0]: {'upstream': g1_up_mean, 'downstream': g1_down_mean},
                labels[1]: {'upstream': g2_up_mean, 'downstream': g2_down_mean}
            },
            'medians': {
                labels[0]: {'upstream': g1_up_median, 'downstream': g1_down_median},
                labels[1]: {'upstream': g2_up_median, 'downstream': g2_down_median}
            },
            'std': {
                labels[0]: {'upstream': g1_up_std, 'downstream': g1_down_std},
                labels[1]: {'upstream': g2_up_std, 'downstream': g2_down_std}
            },
            'sem': {
                labels[0]: {'upstream': g1_up_sem, 'downstream': g1_down_sem},
                labels[1]: {'upstream': g2_up_sem, 'downstream': g2_down_sem}
            },
            'n_larvae': {
                labels[0]: len(g1_upstream),
                labels[1]: len(g2_upstream)
            }
        }
        
        # Add cast type results to text output
        stat_results_txt.append(f"--- {cast_type.upper()} CASTS ---")
        stat_results_txt.append(f"Sample sizes:")
        stat_results_txt.append(f"  {labels[0]}: {len(g1_upstream)} larvae, {g1_up_count + g1_down_count} casts ({g1_up_count} upstream, {g1_down_count} downstream)")
        stat_results_txt.append(f"  {labels[1]}: {len(g2_upstream)} larvae, {g2_up_count + g2_down_count} casts ({g2_up_count} upstream, {g2_down_count} downstream)")
        stat_results_txt.append("\nUpstream probability:")
        stat_results_txt.append(f"  {labels[0]}: mean={g1_up_mean:.3f} ± {g1_up_sem:.3f} (SEM), median={g1_up_median:.3f}")
        stat_results_txt.append(f"  {labels[1]}: mean={g2_up_mean:.3f} ± {g2_up_sem:.3f} (SEM), median={g2_up_median:.3f}")
        stat_results_txt.append(f"  Difference: {g2_up_mean - g1_up_mean:.3f}")
        stat_results_txt.append(f"  t-test: t={up_ttest.statistic:.3f}, p={up_ttest.pvalue:.6f} ({'significant' if up_ttest.pvalue < 0.05 else 'not significant'})")
        stat_results_txt.append(f"  Mann-Whitney: U={up_mw.statistic:.1f}, p={up_mw.pvalue:.6f} ({'significant' if up_mw.pvalue < 0.05 else 'not significant'})")
        
        stat_results_txt.append("\nDownstream probability:")
        stat_results_txt.append(f"  {labels[0]}: mean={g1_down_mean:.3f} ± {g1_down_sem:.3f} (SEM), median={g1_down_median:.3f}")
        stat_results_txt.append(f"  {labels[1]}: mean={g2_down_mean:.3f} ± {g2_down_sem:.3f} (SEM), median={g2_down_median:.3f}")
        stat_results_txt.append(f"  Difference: {g2_down_mean - g1_down_mean:.3f}")
        stat_results_txt.append(f"  t-test: t={down_ttest.statistic:.3f}, p={down_ttest.pvalue:.6f} ({'significant' if down_ttest.pvalue < 0.05 else 'not significant'})")
        stat_results_txt.append(f"  Mann-Whitney: U={down_mw.statistic:.1f}, p={down_mw.pvalue:.6f} ({'significant' if down_mw.pvalue < 0.05 else 'not significant'})")
        
        stat_results_txt.append("\nWithin-genotype comparison (upstream vs downstream):")
        stat_results_txt.append(f"  {labels[0]}: t={g1_within_ttest.statistic:.3f}, p={g1_within_ttest.pvalue:.6f} ({'significant' if g1_within_ttest.pvalue < 0.05 else 'not significant'})")
        stat_results_txt.append(f"  {labels[1]}: t={g2_within_ttest.statistic:.3f}, p={g2_within_ttest.pvalue:.6f} ({'significant' if g2_within_ttest.pvalue < 0.05 else 'not significant'})")
        stat_results_txt.append("\n")
        
        # Create grouped bar chart
        bar_width = 0.35
        x = np.arange(2)  # Two directions: upstream and downstream
        
        # Create bars
        ax.bar(x - bar_width/2, [g1_up_mean, g1_down_mean], bar_width, 
               yerr=[g1_up_sem, g1_down_sem], capsize=5,
               color=[genotype_colors[labels[0]]['upstream'], genotype_colors[labels[0]]['downstream']], 
               label=labels[0], alpha=0.8)
        
        ax.bar(x + bar_width/2, [g2_up_mean, g2_down_mean], bar_width,
               yerr=[g2_up_sem, g2_down_sem], capsize=5,
               color=[genotype_colors[labels[1]]['upstream'], genotype_colors[labels[1]]['downstream']], 
               label=labels[1], alpha=0.8)
        
        # Add individual data points
        jitter_amount = 0.05
        # Upstream - genotype 1
        jitter1 = np.random.normal(0, jitter_amount, size=len(g1_upstream))
        ax.scatter(np.full_like(g1_upstream, x[0] - bar_width/2) + jitter1, g1_upstream, 
                   color=genotype_colors[labels[0]]['upstream'], alpha=0.4, s=15)
        
        # Downstream - genotype 1
        jitter2 = np.random.normal(0, jitter_amount, size=len(g1_downstream))
        ax.scatter(np.full_like(g1_downstream, x[1] - bar_width/2) + jitter2, g1_downstream, 
                   color=genotype_colors[labels[0]]['downstream'], alpha=0.4, s=15)
        
        # Upstream - genotype 2
        jitter3 = np.random.normal(0, jitter_amount, size=len(g2_upstream))
        ax.scatter(np.full_like(g2_upstream, x[0] + bar_width/2) + jitter3, g2_upstream, 
                   color=genotype_colors[labels[1]]['upstream'], alpha=0.4, s=15)
        
        # Downstream - genotype 2
        jitter4 = np.random.normal(0, jitter_amount, size=len(g2_downstream))
        ax.scatter(np.full_like(g2_downstream, x[1] + bar_width/2) + jitter4, g2_downstream, 
                   color=genotype_colors[labels[1]]['downstream'], alpha=0.4, s=15)
        
        # Add significance bars and asterisks for between-genotype comparison
        y_level = 1.1  # Starting height for significance bars
        bar_height = 0.05
        
        # Function to add significance bars and text
        def add_significance(x1, x2, y, p_value, ax, text_offset=0.02):
            if p_value >= 0.05:
                sig_text = "n.s."
            elif p_value < 0.001:
                sig_text = "***"
            elif p_value < 0.01:
                sig_text = "**"
            else:
                sig_text = "*"
            
            # Draw the bar
            ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y], 'k-', linewidth=1.5)
            
            # Add the text
            ax.text((x1 + x2) / 2, y + bar_height + text_offset, sig_text, 
                    ha='center', va='bottom', fontsize=12)
            
            # Return the new y level
            return y + bar_height + 0.08
        
        # Upstream comparison (between genotypes)
        if up_ttest.pvalue < 0.1:  # Show even marginally significant results
            y_level = add_significance(x[0] - bar_width/2, x[0] + bar_width/2, y_level, up_ttest.pvalue, ax)
        
        # Downstream comparison (between genotypes)
        if down_ttest.pvalue < 0.1:
            y_level = add_significance(x[1] - bar_width/2, x[1] + bar_width/2, y_level, down_ttest.pvalue, ax)
        
        # Within-genotype comparison (if both are significant, stack the bars)
        within_g1_y = 0.96
        within_g2_y = 0.96
        
        if g1_within_ttest.pvalue < 0.1:
            within_g1_y = add_significance(x[0] - bar_width/2, x[1] - bar_width/2, within_g1_y, g1_within_ttest.pvalue, ax)
        
        if g2_within_ttest.pvalue < 0.1:
            within_g2_y = add_significance(x[0] + bar_width/2, x[1] + bar_width/2, within_g2_y, g2_within_ttest.pvalue, ax)
        
        # Add sample sizes to the bars
        def add_counts(x, y, count, ax, va='bottom'):
            ax.text(x, y, f"n={count}", ha='center', va=va, fontsize=8, color='black')
        
        # Add counts text
        add_counts(x[0] - bar_width/2, g1_up_mean + g1_up_sem + 0.03, g1_up_count, ax)
        add_counts(x[1] - bar_width/2, g1_down_mean + g1_down_sem + 0.03, g1_down_count, ax)
        add_counts(x[0] + bar_width/2, g2_up_mean + g2_up_sem + 0.03, g2_up_count, ax)
        add_counts(x[1] + bar_width/2, g2_down_mean + g2_down_sem + 0.03, g2_down_count, ax)
        
        # Add legend, title, and labels
        ax.set_ylabel('Cast Probability')
        ax.set_xticks(x)
        ax.set_xticklabels(['Upstream', 'Downstream'])
        ax.set_ylim(0, min(1.5, y_level + 0.1))  # Adjust based on significance bars
        
        type_labels = {
            'large': 'Large Casts',
            'small': 'Small Casts',
            'all': 'All Casts'
        }
        
        # Add count of larvae
        ax.set_title(f"{type_labels.get(cast_type, cast_type.capitalize())}\n{labels[0]}: n={len(g1_upstream)} larvae\n{labels[1]}: n={len(g2_upstream)} larvae")
        
        # Reference line at 0.5 (chance level)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Add legend
        if ax_idx == 1:  # Only for the first subplot
            ax.legend(title="Genotype", loc='upper right')
    
    # Add overall title
    fig.suptitle(f'Comparison of Cast Directions When Perpendicular to Flow (±{angle_width}°)', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure if basepath is provided
    if basepath is not None:
        filepath = os.path.join(basepath, f"cast_direction_comparison_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved comparison plot to: {filepath}")
    
    plt.show()
    
    # Save statistical results to text file
    if basepath is not None:
        txt_filepath = os.path.join(basepath, f"cast_direction_stats_{label1_safe}_vs_{label2_safe}_{timestamp}.txt")
        with open(txt_filepath, 'w') as f:
            f.write('\n'.join(stat_results_txt))
        print(f"Saved statistical results to: {txt_filepath}")
    
    # Return results for further analysis
    return comparison_results


def compare_cast_directions_new(genotype1_data, genotype2_data, labels=None, angle_width=5, min_frame=3, basepath=None):
    """
    Compare upstream and downstream cast distributions between two genotypes when larvae are perpendicular to flow.
    Also tests if mean head angle as a function of body orientation differs statistically between groups.
    
    Args:
        genotype1_data (dict): Tracking data dictionary for first genotype
        genotype2_data (dict): Tracking data dictionary for second genotype
        labels (tuple): Optional tuple of (label1, label2) for the genotypes
        angle_width (int): Width of perpendicular orientation sector in degrees
        min_frame (int): Minimum number of frames to consider for a cast
        basepath (str): Base path for saving output files
        
    Returns:
        dict: Contains comparison statistics and test results
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import os
    from datetime import datetime
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Genotype 1', 'Genotype 2')
    
    # Create safe filenames from labels
    label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
    label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
    
    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure basepath exists
    if basepath is not None:
        os.makedirs(basepath, exist_ok=True)
    
    # Analyze both genotypes separately
    print(f"Analyzing {labels[0]}...")
    genotype1_results = analyze_perpendicular_cast_directions_new(
        genotype1_data, angle_width=angle_width, min_frame=min_frame, basepath=None)
    
    print(f"Analyzing {labels[1]}...")
    genotype2_results = analyze_perpendicular_cast_directions_new(
        genotype2_data, angle_width=angle_width, min_frame=min_frame, basepath=None)
    
    # Get head angle vs body orientation data for both genotypes
    # This would typically come from analyze_cast_head_angles_by_orientation or similar function
    g1_head_angle_results = analyze_cast_head_angles_by_orientation(genotype1_data)
    g2_head_angle_results = analyze_cast_head_angles_by_orientation(genotype2_data)
    
    # Statistical comparison of head angle vs body orientation curves
    orientation_stats = {}
    
    # Get the orientation bins and head angles for both genotypes
    bin_centers = g1_head_angle_results['bin_centers']
    
    # For each cast type (large, small, all)
    for cast_type in ['large', 'small', 'all']:
        # Get the smoothed curves for each genotype
        g1_curve = g1_head_angle_results[f'{cast_type}_smoothed']
        g2_curve = g2_head_angle_results[f'{cast_type}_smoothed']
        
        # Get raw data points (head angles and orientations) for permutation test
        g1_head_angles = g1_head_angle_results[f'{cast_type}_cast_angles']
        g1_orientations = g1_head_angle_results[f'{cast_type}_cast_orientations']
        
        g2_head_angles = g2_head_angle_results[f'{cast_type}_cast_angles']
        g2_orientations = g2_head_angle_results[f'{cast_type}_cast_orientations']
        
        # Calculate the mean head angle for each genotype
        g1_mean_angle = np.nanmean(g1_head_angles)
        g2_mean_angle = np.nanmean(g2_head_angles)
        
        # Statistical test for overall head angle difference (t-test and Mann-Whitney)
        t_stat, t_p = stats.ttest_ind(g1_head_angles, g2_head_angles, equal_var=False)
        u_stat, mw_p = stats.mannwhitneyu(g1_head_angles, g2_head_angles, alternative='two-sided')
        
        # Calculate differences between curves where both have valid data
        valid_mask = ~np.isnan(g1_curve) & ~np.isnan(g2_curve)
        diff_curve = g2_curve[valid_mask] - g1_curve[valid_mask]
        valid_bins = bin_centers[valid_mask]
        
        # Calculate mean absolute difference and mean squared difference
        mean_abs_diff = np.mean(np.abs(diff_curve))
        mean_sqr_diff = np.mean(diff_curve**2)
        
        # Permutation test for curve differences
        # This tests if the two curves differ significantly
        # by randomly reassigning data points between genotypes
        n_permutations = 1000
        permutation_diffs = np.zeros(n_permutations)
        
        # Combine all data points
        all_angles = np.concatenate([g1_head_angles, g2_head_angles])
        all_orientations = np.concatenate([g1_orientations, g2_orientations])
        n1 = len(g1_head_angles)
        n2 = len(g2_head_angles)
        
        for i in range(n_permutations):
            # Randomly shuffle the data
            perm_idx = np.random.permutation(len(all_angles))
            perm_angles = all_angles[perm_idx]
            perm_orientations = all_orientations[perm_idx]
            
            # Split into two groups of original sizes
            perm_g1_angles = perm_angles[:n1]
            perm_g1_orientations = perm_orientations[:n1]
            perm_g2_angles = perm_angles[n1:]
            perm_g2_orientations = perm_orientations[n1:]
            
            # Compute histogram for each group
            hist1, _ = np.histogram(perm_g1_orientations, bins=36, range=(-180, 180), weights=perm_g1_angles)
            count1, _ = np.histogram(perm_g1_orientations, bins=36, range=(-180, 180))
            
            hist2, _ = np.histogram(perm_g2_orientations, bins=36, range=(-180, 180), weights=perm_g2_angles)
            count2, _ = np.histogram(perm_g2_orientations, bins=36, range=(-180, 180))
            
            # Calculate mean angle in each bin
            with np.errstate(divide='ignore', invalid='ignore'):
                perm_mean1 = hist1 / count1
                perm_mean2 = hist2 / count2
            
            # Calculate difference and mean absolute difference
            perm_valid_mask = ~np.isnan(perm_mean1) & ~np.isnan(perm_mean2)
            if np.sum(perm_valid_mask) > 0:
                perm_diff = perm_mean2[perm_valid_mask] - perm_mean1[perm_valid_mask]
                permutation_diffs[i] = np.mean(np.abs(perm_diff))
            else:
                permutation_diffs[i] = 0
        
        # Calculate p-value from permutation test
        perm_p = np.mean(permutation_diffs >= mean_abs_diff)
        
        # Store results
        orientation_stats[cast_type] = {
            'overall_head_angle': {
                'g1_mean': g1_mean_angle,
                'g2_mean': g2_mean_angle,
                'difference': g2_mean_angle - g1_mean_angle,
                't_test': {'statistic': t_stat, 'pvalue': t_p},
                'mannwhitney': {'statistic': u_stat, 'pvalue': mw_p}
            },
            'orientation_curves': {
                'mean_abs_diff': mean_abs_diff,
                'mean_sqr_diff': mean_sqr_diff,
                'permutation_test_pvalue': perm_p
            }
        }
    
    # Create figure to visualize the statistical comparison of orientation curves
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, cast_type in enumerate(['large', 'small', 'all']):
        ax = axs[i]
        
        # Get the data
        bin_centers = g1_head_angle_results['bin_centers']
        g1_curve = g1_head_angle_results[f'{cast_type}_smoothed']
        g2_curve = g2_head_angle_results[f'{cast_type}_smoothed']
        
        # Get the error bars if available
        g1_sem = g1_head_angle_results.get(f'{cast_type}_sem', np.zeros_like(g1_curve))
        g2_sem = g2_head_angle_results.get(f'{cast_type}_sem', np.zeros_like(g2_curve))
        
        # Plot the curves with error regions
        ax.plot(bin_centers, g1_curve, color='blue', linewidth=2, label=labels[0])
        ax.fill_between(bin_centers, g1_curve - g1_sem, g1_curve + g1_sem, color='blue', alpha=0.2)
        
        ax.plot(bin_centers, g2_curve, color='red', linewidth=2, label=labels[1])
        ax.fill_between(bin_centers, g2_curve - g2_sem, g2_curve + g2_sem, color='red', alpha=0.2)
        
        # Add reference lines
        # ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        # ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=90, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(x=-90, color='gray', linestyle=':', alpha=0.3)
        
        # Add statistical test results
        stats_text = (
            f"Mean Abs Diff: {orientation_stats[cast_type]['orientation_curves']['mean_abs_diff']:.2f}°\n"
            f"Permutation p: {orientation_stats[cast_type]['orientation_curves']['permutation_test_pvalue']:.4f}"
        )
        
        # Highlight result with a star if significant
        is_significant = orientation_stats[cast_type]['orientation_curves']['permutation_test_pvalue'] < 0.05
        if is_significant:
            stats_text += " *"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))
        
        # Add title
        title = f"{cast_type.capitalize()} Casts"
        ax.set_title(title)
        
        # Add labels
        ax.set_xlabel('Body Orientation (°)')
        ax.set_ylabel('Mean Head Angle (°)')
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Set limits
        ax.set_xlim(-180, 180)
    
    plt.tight_layout()
    plt.suptitle(f'Statistical Comparison of Head Angle vs Body Orientation Curves\n{labels[0]} vs {labels[1]}', 
                 fontsize=14, y=1.05)
    
    # Save figure if basepath is provided
    if basepath is not None:
        filepath = os.path.join(basepath, f"head_angle_curves_comparison_{label1_safe}_vs_{label2_safe}_{timestamp}.svg")
        plt.savefig(filepath, format='svg', bbox_inches='tight')
        print(f"Saved head angle curves comparison to: {filepath}")
    
    plt.show()
    
    # Save statistical results to text file
    if basepath is not None:
        # Prepare the results text
        results_text = [
            f"Statistical Comparison of Head Angle vs Body Orientation: {labels[0]} vs {labels[1]}",
            f"Analysis timestamp: {timestamp}",
            "\n"
        ]
        
        for cast_type in ['large', 'small', 'all']:
            stats_data = orientation_stats[cast_type]
            
            results_text.append(f"--- {cast_type.upper()} CASTS ---")
            
            # Overall head angle comparison
            overall = stats_data['overall_head_angle']
            results_text.append("Overall Head Angle:")
            results_text.append(f"  {labels[0]} mean: {overall['g1_mean']:.2f}°")
            results_text.append(f"  {labels[1]} mean: {overall['g2_mean']:.2f}°")
            results_text.append(f"  Difference: {overall['difference']:.2f}°")
            results_text.append(f"  t-test: t={overall['t_test']['statistic']:.3f}, p={overall['t_test']['pvalue']:.6f} ({'significant' if overall['t_test']['pvalue'] < 0.05 else 'not significant'})")
            results_text.append(f"  Mann-Whitney: U={overall['mannwhitney']['statistic']:.1f}, p={overall['mannwhitney']['pvalue']:.6f} ({'significant' if overall['mannwhitney']['pvalue'] < 0.05 else 'not significant'})")
            
            # Curve comparison
            curve = stats_data['orientation_curves']
            results_text.append("\nOrientation-dependent Curve Comparison:")
            results_text.append(f"  Mean absolute difference: {curve['mean_abs_diff']:.2f}°")
            results_text.append(f"  Mean squared difference: {curve['mean_sqr_diff']:.2f}°²")
            results_text.append(f"  Permutation test p-value: {curve['permutation_test_pvalue']:.6f} ({'significant' if curve['permutation_test_pvalue'] < 0.05 else 'not significant'})")
            results_text.append("\n")
        
        # Write to file
        txt_filepath = os.path.join(basepath, f"head_angle_curves_stats_{label1_safe}_vs_{label2_safe}_{timestamp}.txt")
        with open(txt_filepath, 'w') as f:
            f.write('\n'.join(results_text))
        print(f"Saved head angle curves statistics to: {txt_filepath}")
    
    # Return combined results
    return {
        'cast_direction_comparison': {
            'genotype1': genotype1_results,
            'genotype2': genotype2_results
        },
        'orientation_curve_stats': orientation_stats
    }


def plot_angle_timeseries_with_polar(trx_data, larva_id=None, smooth_window=5, jump_threshold=15):
    """
    Create a dual-view visualization of orientation angles:
    1. Linear time series (top) - shows angle changes with unwrapping to avoid jumps
    2. Animated polar representation (bottom) - shows angle on a circle with time indication
    
    Parameters:
    -----------
    trx_data : dict
        The tracking data dictionary
    larva_id : str or int, optional
        ID of specific larva to analyze, if None, selects a random larva
    smooth_window : int
        Window size for smoothing
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame
    """
    from scipy.ndimage import gaussian_filter1d
    from ipywidgets import interact, IntSlider
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    # Select larva if not specified
    if larva_id is None:
        larva_id = random.choice(list(trx_data.keys()))
        print(f"Selected random larva: {larva_id}")
    
    # Extract larva data
    larva_data = trx_data[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract required data fields
    time = np.array(larva_data['t']).flatten()
    states = np.array(larva_data['global_state_large_state']).flatten()
    
    # Get orientation angles
    x_center = np.array(larva_data['x_center']).flatten()
    y_center = np.array(larva_data['y_center']).flatten()
    x_spine = np.array(larva_data['x_spine'])
    y_spine = np.array(larva_data['y_spine'])
    
    if x_spine.ndim > 1:  # 2D array
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
    else:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
    
    # Calculate orientation angles
    orientation_angles = []
    for i in range(len(x_center)):
        vector = np.array([x_center[i] - x_tail[i], y_center[i] - y_tail[i]])
        if np.linalg.norm(vector) == 0:
            orientation_angles.append(np.nan)
        else:
            angle_deg = np.degrees(np.arctan2(vector[1], -vector[0]))
            orientation_angles.append(angle_deg)
    
    orientation_angles = np.array(orientation_angles)
    
    # Smooth the angles
    orientation_smooth = gaussian_filter1d(orientation_angles, smooth_window/3.0)
    
    # Unwrap angles to remove discontinuities at -180/180 boundary
    orientation_rad = np.radians(orientation_smooth)
    unwrapped_rad = np.unwrap(orientation_rad)
    unwrapped_deg = np.degrees(unwrapped_rad)
    
    # Create the figure with two subplots - FIXED: Create polar subplot correctly
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 2])
    
    # Regular cartesian subplot for time series
    ax1 = fig.add_subplot(gs[0])
    
    # Polar subplot for the angle visualization
    ax2 = fig.add_subplot(gs[1], projection='polar')
    
    # Set up the time series plot
    ax1.plot(time, orientation_smooth, 'k-', alpha=0.5, label='Original')
    ax1.plot(time, unwrapped_deg, 'b-', linewidth=1.5, label='Unwrapped')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Orientation (°)')
    ax1.set_title(f'Orientation Time Series - Larva {larva_id}')
    ax1.legend(loc='upper right')
    
    # Add reference lines
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.axhline(y=180, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=-180, color='gray', linestyle='--', alpha=0.3)
    
    # Set up the polar plot
    ax2.set_theta_zero_location('N')  # 0 at top
    ax2.set_theta_direction(-1)  # clockwise
    ax2.set_rticks([])  # No radial ticks
    ax2.set_rlim(0, 1.2)
    ax2.set_title('Orientation on Polar Plot')
    
    # Add cardinal direction labels
    directions = ['Upstream (0°)', 'Right (90°)', 'Downstream (180°)', 'Left (270°)']
    angles = np.radians([0, 90, 180, 270])
    for direction, angle in zip(directions, angles):
        ax2.text(angle, 1.1, direction, ha='center', va='center', fontsize=9)
    
    # Time marker for time series
    time_line = ax1.axvline(x=time[0], color='r', linewidth=1.5)
    
    # Current angle arrow for polar plot
    arrow = ax2.annotate('', xy=(orientation_rad[0], 1), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='r', lw=2))
    
    # Add past trajectory with color gradient
    n_history = 100  # Number of past points to show
    points = []
    for i in range(min(n_history, len(orientation_rad))):
        alpha = (i + 1) / n_history  # Fade out older points
        point, = ax2.plot([orientation_rad[i]], [1], 'o', ms=3, 
                         color='blue', alpha=alpha * 0.5)
        points.append(point)
    
    # Interactive slider to move through time
    def update(frame):
        # Update time marker
        time_line.set_xdata([time[frame], time[frame]])
        
        # Update arrow direction
        angle = orientation_rad[frame]
        arrow.xy = (angle, 1)
        
        # Update trajectory
        for i, point in enumerate(points):
            idx = max(0, frame - n_history + i + 1)
            if idx < frame:
                point.set_data([orientation_rad[idx]], [1])
                alpha = (i + 1) / n_history
                point.set_alpha(alpha * 0.5)
            else:
                point.set_data([], [])
        
        plt.draw()
    
    # Create slider widget
    slider = IntSlider(min=0, max=len(time)-1, step=1, value=0,
                     description='Time Frame:',
                     layout={'width': '100%'})
    
    # Connect the slider to the update function
    interact(update, frame=slider)
    
    plt.tight_layout()
    return {
        'time': time,
        'orientation_smooth': orientation_smooth,
        'unwrapped_deg': unwrapped_deg,
        'fig': fig
    }


def plot_polar_angles_scatter(trx_data, larva_id=None, smooth_window=5, jump_threshold=15, color_by='time', cmap='viridis'):
    """
    Create two polar scatter plots with randomized radii:
    1. Orientation angle scatter plot
    2. Body bend angle scatter plot
    
    Points are colored by time or behavior state using a colormap.
    
    Parameters:
    -----------
    trx_data : dict
        The tracking data dictionary
    larva_id : str or int, optional
        ID of specific larva to analyze, if None, selects a random larva
    smooth_window : int
        Window size for smoothing
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame
    color_by : str
        'time' to color by time progression, 'behavior' to color by behavioral state
    cmap : str or matplotlib.colors.Colormap
        Colormap to use for coloring points (when color_by='time')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
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
    
    # Helper function to calculate orientation angle (relative to negative x-axis)
    def calculate_orientation_angle(vector):
        """
        Calculate angle between vector and negative x-axis in degrees.
        0° = aligned with negative x-axis (downstream), 180° = aligned with positive x-axis (upstream)
        +90° = pointing right (y-axis), -90° = pointing left (negative y-axis)
        """
        if np.linalg.norm(vector) == 0:
            return np.nan
            
        # Calculate angle with negative x-axis using arctan2
        angle_deg = np.degrees(np.arctan2(vector[1], -vector[0]))
        
        return angle_deg
    
    # Helper function to smooth data
    def smooth_data(data, window_size):
        """Apply Gaussian smoothing to data"""
        sigma = window_size / 3.0  # Approximately equivalent to moving average
        return gaussian_filter1d(data, sigma)
    
    # Select larva if not specified
    if larva_id is None:
        larva_id = random.choice(list(trx_data.keys()))
        print(f"Selected random larva: {larva_id}")
    
    # Extract larva data
    larva_data = trx_data[larva_id]
    if 'data' in larva_data:
        larva_data = larva_data['data']
    
    # Extract required data fields
    time = np.array(larva_data['t']).flatten()
    states = np.array(larva_data['global_state_large_state']).flatten()
    
    # Extract coordinates for angle calculations
    x_spine = np.array(larva_data['x_spine'])
    y_spine = np.array(larva_data['y_spine'])
    x_center = np.array(larva_data['x_center']).flatten()
    y_center = np.array(larva_data['y_center']).flatten()
    
    # Handle different spine data shapes
    if x_spine.ndim == 1:  # 1D array
        x_tail = x_spine
        y_tail = y_spine
    else:  # 2D array with shape (spine_points, frames)
        x_tail = x_spine[-1].flatten()
        y_tail = y_spine[-1].flatten()
    
    # Calculate tail-to-center vectors and orientation angles
    tail_to_center_vectors = []
    orientation_angles = []
    for i in range(len(x_center)):
        vector = np.array([x_center[i] - x_tail[i], y_center[i] - y_tail[i]])
        tail_to_center_vectors.append(vector)
        orientation_angles.append(calculate_orientation_angle(vector))
    
    # Get upper-lower bend angle
    angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
    angle_upper_lower_deg = np.degrees(angle_upper_lower)
    
    # Convert to numpy arrays
    orientation_angles = np.array(orientation_angles)
    
    # Apply smoothing
    orientation_angles_smooth = smooth_data(orientation_angles, smooth_window)
    angle_upper_lower_deg_smooth = smooth_data(angle_upper_lower_deg, smooth_window)
    
    # Ensure all arrays have the same length
    min_length = min(len(time), len(orientation_angles_smooth), len(angle_upper_lower_deg_smooth), len(states))
    time = time[:min_length]
    orientation_angles_smooth = orientation_angles_smooth[:min_length]
    angle_upper_lower_deg_smooth = angle_upper_lower_deg_smooth[:min_length]
    states = states[:min_length]
    
    # Generate random radii between 0.5 and 1.0
    np.random.seed(42)  # For reproducibility
    random_radii = 0.5 + 0.5 * np.random.random(min_length)
    
    # Create figure with two polar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': 'polar'})
    
    # Prepare colors based on time or behavior
    if color_by == 'behavior':
        # Use behavior states for coloring
        colors = np.array([behavior_colors.get(int(state), [0.5, 0.5, 0.5]) for state in states])
        color_values = states
    else:  # default to time
        # Use time for coloring
        cmap_obj = plt.get_cmap(cmap)
        norm_time = (time - np.min(time)) / (np.max(time) - np.min(time)) if np.max(time) > np.min(time) else np.zeros_like(time)
        colors = cmap_obj(norm_time)
        color_values = time
    
    # Convert angles to radians for polar plots
    orientation_rad = np.radians(orientation_angles_smooth)
    bend_rad = np.radians(angle_upper_lower_deg_smooth)
    
    # 1. Plot orientation angles on polar plot with random radii
    if color_by == 'behavior':
        scatter1 = ax1.scatter(orientation_rad, random_radii, 
                              c=color_values, cmap=plt.cm.colors.ListedColormap(list(behavior_colors.values())),
                              s=3, alpha=0.5)
    else:
        scatter1 = ax1.scatter(orientation_rad, random_radii, 
                              c=color_values, cmap=cmap, s=3, alpha=0.5)
    
    # Configure orientation plot
    ax1.set_title('Orientation Angle', fontsize=12)
    ax1.set_theta_zero_location('E')  # 0 at right (East)
    ax1.set_theta_direction(1)       # counterclockwise
    
    # Set radius ticks off
    ax1.set_rticks([])
    ax1.set_rlim(0, 1.2)
    
    # Add cardinal direction labels with upstream/downstream
    ax1.set_xticklabels(['0°\n(Downstream)', '45°', '90°\n(Right)', '135°', 
                         '±180°\n(Upstream)', '-135°', '-90°\n(Left)', '-45°'])
    
    # 2. Plot bend angle on polar plot with random radii
    # Use the same random radii for consistency (or generate new ones if preferred)
    if color_by == 'behavior':
        scatter2 = ax2.scatter(bend_rad, random_radii, 
                              c=color_values, cmap=plt.cm.colors.ListedColormap(list(behavior_colors.values())),
                              s=3, alpha=0.5)
    else:
        scatter2 = ax2.scatter(bend_rad, random_radii, 
                              c=color_values, cmap=cmap, s=3, alpha=0.5)
    
    # Configure bend angle plot
    ax2.set_title('Body Bend Angle', fontsize=12)
    ax2.set_theta_zero_location('E')  # 0 at right (East)
    ax2.set_theta_direction(1)       # counterclockwise
    
    # Set radius ticks off
    ax2.set_rticks([])
    ax2.set_rlim(0, 1.2)
    
    # Add labels for the bend angle plot
    ax2.set_xticklabels(['0°\n(No Bend)', '45°\n(Right)', '90°\n(Right)', '135°\n(Right)', 
                        '±180°', '-135°\n(Left)', '-90°\n(Left)', '-45°\n(Left)'])
    
    # Add colorbar or legend
    if color_by == 'behavior':
        # Create legend for behavior states
        unique_states = np.unique(states).astype(int)
        behavior_legend_elements = [
            Patch(facecolor=behavior_colors.get(state, [0.5, 0.5, 0.5]), 
                 alpha=0.8, label=behavior_labels.get(state, f'State {state}'))
            for state in unique_states
        ]
        fig.legend(handles=behavior_legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, 0), ncol=min(len(behavior_legend_elements), 4), fontsize=9)
    else:
        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(time)
        cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', pad=0.1)
        cbar.set_label('Time (seconds)')
    
    # Add vector legend
    vector_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, 
              label='Orientation Vector', alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, 
              label='Bend Angle Vector', alpha=0.8)
    ]
    
    # Add vector legend between the plots
    fig.legend(handles=vector_legend, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=9)
    
    plt.suptitle(f'Polar Angle Distributions for Larva {larva_id}', fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2 if color_by == 'behavior' else 0.15)
    
    return {
        'larva_id': larva_id,
        'time': time,
        'orientation_angles_smooth': orientation_angles_smooth,
        'angle_upper_lower_deg_smooth': angle_upper_lower_deg_smooth,
        'states': states,
        'fig': fig
    }



def compare_cast_directions_peaks(genotype1_data, genotype2_data, labels=None, angle_width=5, 
                                 smooth_window=5, jump_threshold=15, basepath=None):
    """
    Compare upstream and downstream cast distributions between two genotypes when larvae are perpendicular to flow.
    Compares all peaks, first peaks, and last peaks separately to identify behavioral differences.
    
    Parameters:
    -----------
    genotype1_data : dict
        Tracking data dictionary for first genotype
    genotype2_data : dict
        Tracking data dictionary for second genotype
    labels : tuple, optional
        Tuple of (label1, label2) for the genotypes
    angle_width : int
        Width of perpendicular orientation sector in degrees
    smooth_window : int
        Window size for smoothing (default=5)
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame (default=15)
    basepath : str, optional
        Base path for saving output files
        
    Returns:
    --------
    dict: Contains comparison statistics and test results
    """
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from datetime import datetime
    from matplotlib.patches import Patch
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Genotype 1', 'Genotype 2')
    
    # Create safe filenames from labels
    label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
    label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
    
    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure basepath exists
    if basepath is not None:
        import os
        os.makedirs(basepath, exist_ok=True)
    
    # Define perpendicular angle ranges (both left and right sides)
    # With wind blowing towards negative x, perpendicular is ±90°
    left_perp_range = (-90 - angle_width, -90 + angle_width)
    right_perp_range = (90 - angle_width, 90 + angle_width)
    
    def is_perpendicular(angle):
        """Check if an angle is within the perpendicular ranges"""
        return ((left_perp_range[0] <= angle <= left_perp_range[1]) or 
                (right_perp_range[0] <= angle <= right_perp_range[1]))
    
    def analyze_peaks_in_genotype(genotype_data):
        """Analyze cast peaks within a genotype using sophisticated filtering techniques"""
        # Store per-larva counts for each peak type (all, first, last)
        peak_types = ['all', 'first', 'last']
        
        per_larva_counts = {}
        total_counts = {}
        larva_probabilities = {}
        
        for ptype in peak_types:
            per_larva_counts[ptype] = {
                'large': {'upstream': [], 'downstream': [], 'larva_ids': [], 'total': []},
                'small': {'upstream': [], 'downstream': [], 'larva_ids': [], 'total': []}
            }
            total_counts[ptype] = {
                'large': {'upstream': 0, 'downstream': 0},
                'small': {'upstream': 0, 'downstream': 0}
            }
            larva_probabilities[ptype] = {
                'large': {'upstream': [], 'downstream': []},
                'small': {'upstream': [], 'downstream': []}
            }
        
        # Determine which data structure we're working with
        if 'data' in genotype_data:
            data_to_process = genotype_data['data']
            n_larvae = genotype_data['metadata']['total_larvae']
        else:
            data_to_process = genotype_data
            n_larvae = len(data_to_process)
        
        # Process each larva
        larvae_processed = 0
        for larva_id, larva_data in data_to_process.items():
            # Extract nested data if needed
            if 'data' in larva_data:
                larva_data = larva_data['data']
                
            # Extract required data
            try:
                # Extract basic time series data
                time = np.array(larva_data['t']).flatten()
                states = np.array(larva_data['global_state_large_state']).flatten()
                
                # Check for small_large_state or fall back to large_state
                has_small_large_state = 'global_state_small_large_state' in larva_data
                
                if has_small_large_state:
                    # Extract both small and large cast states
                    small_large_states = np.array(larva_data['global_state_small_large_state']).flatten()
                    large_cast_mask = small_large_states == 2.0  # Large casts = 2.0
                    small_cast_mask = small_large_states == 1.5  # Small casts = 1.5
                else:
                    # Fall back to just large state if small_large_state isn't available
                    large_cast_mask = states == 2  # Only large casts available
                    small_cast_mask = np.zeros_like(states, dtype=bool)  # No small casts
                
                any_cast_mask = large_cast_mask | small_cast_mask
                
                # Get orientation angles
                # Calculate orientation from tail to center (negative x-axis is 0 degrees)
                x_center = np.array(larva_data['x_center']).flatten()
                y_center = np.array(larva_data['y_center']).flatten()
                x_spine = np.array(larva_data['x_spine'])
                y_spine = np.array(larva_data['y_spine'])
                
                if x_spine.ndim > 1:
                    x_tail = x_spine[-1].flatten()
                    y_tail = y_spine[-1].flatten()
                else:
                    x_tail = x_spine
                    y_tail = y_spine
                
                # Calculate orientation vectors
                dx = x_center - x_tail
                dy = y_center - y_tail
                orientation_angles = np.degrees(np.arctan2(dy, -dx))  # -dx because 0° is negative x-axis
                
                # Get upper-lower bend angle
                angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
                angle_upper_lower_deg = np.degrees(angle_upper_lower)
                
                # Ensure all arrays have the same length
                min_length = min(len(time), len(orientation_angles), len(angle_upper_lower_deg), len(states))
                time = time[:min_length]
                orientation_angles = orientation_angles[:min_length]
                angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
                states = states[:min_length]
                large_cast_mask = large_cast_mask[:min_length]
                small_cast_mask = small_cast_mask[:min_length]
                any_cast_mask = any_cast_mask[:min_length]
                
                # ==== FILTERING AND PROCESSING ====
                
                # 1. Orientation angle jump detection and correction
                orientation_raw = orientation_angles.copy()
                
                # Calculate derivative to detect jumps
                orientation_diff = np.abs(np.diff(orientation_angles))
                # Add zero at the beginning to match length
                orientation_diff = np.insert(orientation_diff, 0, 0)
                
                # Find jumps bigger than threshold
                jumps = orientation_diff > jump_threshold
                
                # Create masked array for plotting
                orientation_masked = np.ma.array(orientation_angles, mask=jumps)
                
                # Apply Gaussian smoothing to the filtered data
                # First interpolate masked values for smoothing
                orientation_interp = orientation_masked.filled(np.nan)
                mask = np.isnan(orientation_interp)
                
                # Only interpolate if we have valid data
                if np.sum(~mask) > 1:
                    indices = np.arange(len(orientation_interp))
                    valid_indices = indices[~mask]
                    valid_values = orientation_interp[~mask]
                    orientation_interp[mask] = np.interp(indices[mask], valid_indices, valid_values)
                
                # Apply smoothing
                orientation_smooth = gaussian_filter1d(orientation_interp, smooth_window/3.0)
                
                # 2. Intelligent peak detection for bend angle
                # Smooth the bend angle
                bend_angle_smooth = gaussian_filter1d(angle_upper_lower_deg, smooth_window/3.0)
                
                # 3. Find continuous cast segments
                cast_segments = {
                    'large': [],
                    'small': []
                }
                
                # Find large cast segments
                in_segment = False
                start_idx = 0
                for i, is_cast in enumerate(large_cast_mask):
                    if is_cast and not in_segment:
                        # Start of new segment
                        in_segment = True
                        start_idx = i
                    elif not is_cast and in_segment:
                        # End of segment
                        in_segment = False
                        if i - start_idx >= 3:  # Require at least 3 frames
                            cast_segments['large'].append((start_idx, i))
                
                # Handle case when still in segment at end of data
                if in_segment and len(large_cast_mask) - start_idx >= 3:
                    cast_segments['large'].append((start_idx, len(large_cast_mask)))
                
                # Find small cast segments
                in_segment = False
                start_idx = 0
                for i, is_cast in enumerate(small_cast_mask):
                    if is_cast and not in_segment:
                        # Start of new segment
                        in_segment = True
                        start_idx = i
                    elif not is_cast and in_segment:
                        # End of segment
                        in_segment = False
                        if i - start_idx >= 3:  # Require at least 3 frames
                            cast_segments['small'].append((start_idx, i))
                
                # Handle case when still in segment at end of data
                if in_segment and len(small_cast_mask) - start_idx >= 3:
                    cast_segments['small'].append((start_idx, len(small_cast_mask)))
                
                # 4. Find peaks within each cast segment
                all_peaks = {
                    'large': [],
                    'small': []
                }
                
                first_peaks = {
                    'large': [],
                    'small': []
                }
                
                last_peaks = {
                    'large': [],
                    'small': []
                }
                
                # Process each cast type
                for cast_type in ['large', 'small']:
                    for start, end in cast_segments[cast_type]:
                        segment_angles = bend_angle_smooth[start:end]
                        
                        # Find peaks in absolute bend angles
                        abs_angles = np.abs(segment_angles)
                        
                        # Find positive and negative peaks
                        pos_peaks, _ = find_peaks(segment_angles, height=5, prominence=3, distance=3)
                        neg_peaks, _ = find_peaks(-segment_angles, height=5, prominence=3, distance=3)
                        
                        # Combine and sort peaks by position
                        segment_peaks = sorted(list(pos_peaks) + list(neg_peaks))
                        
                        if len(segment_peaks) > 0:
                            # Convert to global indices
                            segment_peaks = [start + idx for idx in segment_peaks]
                            
                            # Store all peaks
                            all_peaks[cast_type].extend(segment_peaks)
                            
                            # Store first and last peak
                            first_peaks[cast_type].append(segment_peaks[0])
                            last_peaks[cast_type].append(segment_peaks[-1])
                
                # 5. Classify peaks as upstream or downstream
                # Storage for cast counts by peak type
                larva_cast_counts = {
                    'all': {'large': {'upstream': 0, 'downstream': 0}, 'small': {'upstream': 0, 'downstream': 0}},
                    'first': {'large': {'upstream': 0, 'downstream': 0}, 'small': {'upstream': 0, 'downstream': 0}},
                    'last': {'large': {'upstream': 0, 'downstream': 0}, 'small': {'upstream': 0, 'downstream': 0}}
                }
                
                # Process all peaks
                for cast_type in ['large', 'small']:
                    # Process all peaks
                    for peak_idx in all_peaks[cast_type]:
                        if is_perpendicular(orientation_smooth[peak_idx]):
                            # Get orientation and bend angle at peak
                            orientation = orientation_smooth[peak_idx]
                            bend_angle = bend_angle_smooth[peak_idx]
                            
                            # Normalize orientation to -180 to 180 range
                            while orientation > 180:
                                orientation -= 360
                            while orientation <= -180:
                                orientation += 360
                            
                            # Classify as upstream or downstream based on orientation and bend direction
                            is_upstream = False
                            if (orientation > 0 and orientation < 180):  # Right side (positive orientation)
                                if bend_angle < 0:  # Negative bend is upstream
                                    is_upstream = True
                                else:  # Positive bend is downstream
                                    is_upstream = False
                            else:  # Left side (negative orientation)
                                if bend_angle > 0:  # Positive bend is upstream
                                    is_upstream = True
                                else:  # Negative bend is downstream
                                    is_upstream = False
                            
                            # Update cast counts for all peaks
                            cast_direction = 'upstream' if is_upstream else 'downstream'
                            larva_cast_counts['all'][cast_type][cast_direction] += 1
                    
                    # Process first peaks
                    for peak_idx in first_peaks[cast_type]:
                        if is_perpendicular(orientation_smooth[peak_idx]):
                            # Get orientation and bend angle at peak
                            orientation = orientation_smooth[peak_idx]
                            bend_angle = bend_angle_smooth[peak_idx]
                            
                            # Normalize orientation to -180 to 180 range
                            while orientation > 180:
                                orientation -= 360
                            while orientation <= -180:
                                orientation += 360
                            
                            # Classify as upstream or downstream
                            is_upstream = False
                            if (orientation > 0 and orientation < 180):  # Right side
                                if bend_angle < 0:  # Negative bend is upstream
                                    is_upstream = True
                                else:  # Positive bend is downstream
                                    is_upstream = False
                            else:  # Left side
                                if bend_angle > 0:  # Positive bend is upstream
                                    is_upstream = True
                                else:  # Negative bend is downstream
                                    is_upstream = False
                            
                            # Update cast counts for first peaks
                            cast_direction = 'upstream' if is_upstream else 'downstream'
                            larva_cast_counts['first'][cast_type][cast_direction] += 1
                    
                    # Process last peaks
                    for peak_idx in last_peaks[cast_type]:
                        if is_perpendicular(orientation_smooth[peak_idx]):
                            # Get orientation and bend angle at peak
                            orientation = orientation_smooth[peak_idx]
                            bend_angle = bend_angle_smooth[peak_idx]
                            
                            # Normalize orientation to -180 to 180 range
                            while orientation > 180:
                                orientation -= 360
                            while orientation <= -180:
                                orientation += 360
                            
                            # Classify as upstream or downstream
                            is_upstream = False
                            if (orientation > 0 and orientation < 180):  # Right side
                                if bend_angle < 0:  # Negative bend is upstream
                                    is_upstream = True
                                else:  # Positive bend is downstream
                                    is_upstream = False
                            else:  # Left side
                                if bend_angle > 0:  # Positive bend is upstream
                                    is_upstream = True
                                else:  # Negative bend is downstream
                                    is_upstream = False
                            
                            # Update cast counts for last peaks
                            cast_direction = 'upstream' if is_upstream else 'downstream'
                            larva_cast_counts['last'][cast_type][cast_direction] += 1
                
                # 6. Store counts for analysis if we have enough data
                has_casts = False
                for peak_type in peak_types:
                    for cast_type in ['large', 'small']:
                        upstream = larva_cast_counts[peak_type][cast_type]['upstream']
                        downstream = larva_cast_counts[peak_type][cast_type]['downstream']
                        total = upstream + downstream
                        
                        if total >= 3:  # Require at least 3 casts for reliable probability
                            has_casts = True
                            
                            # Add to total counts
                            total_counts[peak_type][cast_type]['upstream'] += upstream
                            total_counts[peak_type][cast_type]['downstream'] += downstream
                            
                            # Add to per-larva counts
                            per_larva_counts[peak_type][cast_type]['upstream'].append(upstream)
                            per_larva_counts[peak_type][cast_type]['downstream'].append(downstream)
                            per_larva_counts[peak_type][cast_type]['total'].append(total)
                            per_larva_counts[peak_type][cast_type]['larva_ids'].append(str(larva_id))
                            
                            # Calculate and store probability
                            upstream_prob = upstream / total
                            downstream_prob = downstream / total
                            larva_probabilities[peak_type][cast_type]['upstream'].append(upstream_prob)
                            larva_probabilities[peak_type][cast_type]['downstream'].append(downstream_prob)
                
                if has_casts:
                    larvae_processed += 1
                
            except Exception as e:
                print(f"Error processing larva {larva_id}: {e}")
        
        # Calculate overall probabilities for each cast type and peak type
        probabilities = {}
        for peak_type in peak_types:
            probabilities[peak_type] = {}
            for cast_type in ['large', 'small']:
                upstream = total_counts[peak_type][cast_type]['upstream']
                downstream = total_counts[peak_type][cast_type]['downstream']
                total = upstream + downstream
                
                if total > 0:
                    probabilities[peak_type][cast_type] = {
                        'upstream': upstream / total,
                        'downstream': downstream / total
                    }
                else:
                    probabilities[peak_type][cast_type] = {
                        'upstream': 0,
                        'downstream': 0
                    }
        
        # Calculate mean probabilities from per-larva data
        mean_probabilities = {}
        sem_probabilities = {}  # Standard error of mean
        
        for peak_type in peak_types:
            mean_probabilities[peak_type] = {}
            sem_probabilities[peak_type] = {}
            
            for cast_type in ['large', 'small']:
                up_probs = np.array(larva_probabilities[peak_type][cast_type]['upstream'])
                down_probs = np.array(larva_probabilities[peak_type][cast_type]['downstream'])
                
                if len(up_probs) > 0:
                    mean_probabilities[peak_type][cast_type] = {
                        'upstream': np.mean(up_probs),
                        'downstream': np.mean(down_probs)
                    }
                    sem_probabilities[peak_type][cast_type] = {
                        'upstream': stats.sem(up_probs) if len(up_probs) > 1 else 0,
                        'downstream': stats.sem(down_probs) if len(down_probs) > 1 else 0
                    }
                else:
                    mean_probabilities[peak_type][cast_type] = {'upstream': 0, 'downstream': 0}
                    sem_probabilities[peak_type][cast_type] = {'upstream': 0, 'downstream': 0}
        
        # Statistical tests for each cast type and peak type
        stats_results = {}
        for peak_type in peak_types:
            stats_results[peak_type] = {}
            
            for cast_type in ['large', 'small']:
                upstream_probs = np.array(larva_probabilities[peak_type][cast_type]['upstream'])
                downstream_probs = np.array(larva_probabilities[peak_type][cast_type]['downstream'])
                
                if len(upstream_probs) > 0:
                    # One-sample t-test against chance level (0.5)
                    tstat, pval = stats.ttest_1samp(upstream_probs, 0.5)
                    
                    # Add chi-square test on raw counts
                    observed = np.array([total_counts[peak_type][cast_type]['upstream'], 
                                        total_counts[peak_type][cast_type]['downstream']])
                    expected = np.sum(observed) / 2  # Expected equal distribution
                    chi2, p_chi2 = stats.chisquare(observed)
                    
                    stats_results[peak_type][cast_type] = {
                        'ttest': {'tstat': tstat, 'pval': pval},
                        'chisquare': {'chi2': chi2, 'pval': p_chi2},
                        'n_larvae': len(upstream_probs),
                        'n_upstream': total_counts[peak_type][cast_type]['upstream'],
                        'n_downstream': total_counts[peak_type][cast_type]['downstream']
                    }
                else:
                    stats_results[peak_type][cast_type] = {
                        'ttest': {'tstat': float('nan'), 'pval': float('nan')},
                        'chisquare': {'chi2': float('nan'), 'pval': float('nan')},
                        'n_larvae': 0,
                        'n_upstream': 0,
                        'n_downstream': 0
                    }
        
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
    
    # Analyze both genotypes separately
    print(f"Analyzing {labels[0]}...")
    genotype1_results = analyze_peaks_in_genotype(genotype1_data)
    
    print(f"Analyzing {labels[1]}...")
    genotype2_results = analyze_peaks_in_genotype(genotype2_data)
    
    # Statistical comparison between genotypes
    comparison_results = {
        'angle_width': angle_width,
        'labels': labels,
        'statistical_tests': {},
        'counts': {}
    }
    
    # Cast types to analyze
    cast_types = ['large', 'small']
    peak_types = ['all', 'first', 'last']
    
    # Colors for each genotype
    genotype_colors = {
        labels[0]: {
            'upstream': '#1f77b4',  # Blue
            'downstream': '#9ecae1'  # Light blue
        },
        labels[1]: {
            'upstream': '#d62728',   # Red
            'downstream': '#ff9896'  # Light red
        }
    }
    
    # Count valid cast types (those with data in both genotypes) for each peak type
    valid_data = {}
    for peak_type in peak_types:
        valid_data[peak_type] = []
        for cast_type in cast_types:
            if (len(genotype1_results['larva_probabilities'][peak_type][cast_type]['upstream']) > 0 and
                len(genotype2_results['larva_probabilities'][peak_type][cast_type]['upstream']) > 0):
                valid_data[peak_type].append(cast_type)
    
    # Create figure for each peak type
    print(f"Creating comparison plots...")
    
    # Variable to store statistical results for text file
    stat_results_txt = []
    stat_results_txt.append(f"Cast Direction Comparison: {labels[0]} vs {labels[1]}")
    stat_results_txt.append(f"Analysis timestamp: {timestamp}")
    stat_results_txt.append(f"Perpendicular angle width: ±{angle_width}°")
    stat_results_txt.append(f"Using improved peak detection with jump_threshold={jump_threshold} and smooth_window={smooth_window}")
    stat_results_txt.append("\n")
    
    # Create plots for each peak type
    for peak_type in peak_types:
        valid_cast_types = valid_data[peak_type]
        n_cast_types = len(valid_cast_types)
        
        if n_cast_types == 0:
            print(f"No {peak_type} peak data found in both genotypes. Skipping plot.")
            continue
        
        # Create figure
        fig, axes = plt.subplots(1, n_cast_types, figsize=(4.5 * n_cast_types, 6))
        
        # Handle the case of a single subplot
        if n_cast_types == 1:
            axes = [axes]
        
        # Process each cast type
        for ax_idx, cast_type in enumerate(valid_cast_types):
            # Get data for current cast type
            g1_upstream = genotype1_results['larva_probabilities'][peak_type][cast_type]['upstream']
            g1_downstream = genotype1_results['larva_probabilities'][peak_type][cast_type]['downstream']
            g2_upstream = genotype2_results['larva_probabilities'][peak_type][cast_type]['upstream']
            g2_downstream = genotype2_results['larva_probabilities'][peak_type][cast_type]['downstream']
            
            # Statistical tests for difference between genotypes
            # Two-sample t-test for upstream probabilities
            tstat, pval = stats.ttest_ind(g1_upstream, g2_upstream, equal_var=False)
            comparison_results['statistical_tests'][f'{peak_type}_{cast_type}_upstream'] = {
                'tstat': tstat,
                'pval': pval,
                'n1': len(g1_upstream),
                'n2': len(g2_upstream)
            }
            
            # Add to text results
            stat_results_txt.append(f"{peak_type.capitalize()} peaks - {cast_type.capitalize()} casts - Upstream probability comparison:")
            stat_results_txt.append(f"  {labels[0]}: {np.mean(g1_upstream):.4f} ± {stats.sem(g1_upstream):.4f} (n={len(g1_upstream)})")
            stat_results_txt.append(f"  {labels[1]}: {np.mean(g2_upstream):.4f} ± {stats.sem(g2_upstream):.4f} (n={len(g2_upstream)})")
            stat_results_txt.append(f"  Two-sample t-test: t={tstat:.4f}, p={pval:.4f}")
            
            # Add raw count statistics for this cast type
            g1_up_count = genotype1_results['total_counts'][peak_type][cast_type]['upstream']
            g1_down_count = genotype1_results['total_counts'][peak_type][cast_type]['downstream']
            g2_up_count = genotype2_results['total_counts'][peak_type][cast_type]['upstream']
            g2_down_count = genotype2_results['total_counts'][peak_type][cast_type]['downstream']
            
            stat_results_txt.append(f"  Raw counts:")
            stat_results_txt.append(f"    {labels[0]}: {g1_up_count} upstream, {g1_down_count} downstream")
            stat_results_txt.append(f"    {labels[1]}: {g2_up_count} upstream, {g2_down_count} downstream")
            
            # One-sample t-test against chance (0.5) for each genotype
            for label, data in [(labels[0], g1_upstream), (labels[1], g2_upstream)]:
                tstat, pval = stats.ttest_1samp(data, 0.5)
                stat_results_txt.append(f"  {label} vs chance (0.5): t={tstat:.4f}, p={pval:.4f}")
            
            stat_results_txt.append("")
            
            # Create the plot for this cast type
            ax = axes[ax_idx]
            
            # Data for box plot
            data = [g1_upstream, g1_downstream, g2_upstream, g2_downstream]
            positions = [1, 2, 3.5, 4.5]
            labels_box = [f'{labels[0]}\nUpstream', f'{labels[0]}\nDownstream', 
                         f'{labels[1]}\nUpstream', f'{labels[1]}\nDownstream']
            
            # Create box plot
            bp = ax.boxplot(data, positions=positions, notch=True, patch_artist=True, 
                           widths=0.6, showfliers=False)
            
            # Color boxes by genotype and direction
            colors = [genotype_colors[labels[0]]['upstream'], 
                     genotype_colors[labels[0]]['downstream'],
                     genotype_colors[labels[1]]['upstream'], 
                     genotype_colors[labels[1]]['downstream']]
            
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=colors[i], alpha=0.6)
                bp['medians'][i].set(color='black', linewidth=1.5)
            
            # Add individual data points with jitter
            for i, (pos, d) in enumerate(zip(positions, data)):
                jitter = 0.1 * np.random.randn(len(d))
                ax.scatter(pos + jitter, d, s=25, alpha=0.6, color=colors[i], 
                          edgecolor='k', linewidth=0.5)
            
            # Add annotations: mean ± SEM
            for i, (pos, d) in enumerate(zip(positions, data)):
                if len(d) > 0:
                    mean = np.mean(d)
                    sem = stats.sem(d) if len(d) > 1 else 0
                    ax.text(pos, 1.05, f"{mean:.2f}±{sem:.2f}", ha='center', va='bottom',
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.8, pad=0.1))
            
            # Add raw count annotations for each population
            ax.text(1, -0.05, f"n={g1_up_count}", ha='center', va='top', fontsize=8)
            ax.text(2, -0.05, f"n={g1_down_count}", ha='center', va='top', fontsize=8)
            ax.text(3.5, -0.05, f"n={g2_up_count}", ha='center', va='top', fontsize=8)
            ax.text(4.5, -0.05, f"n={g2_down_count}", ha='center', va='top', fontsize=8)
            
            # Add p-value annotation between genotypes
            # Comparison between genotypes (upstream probabilities)
            pval = comparison_results['statistical_tests'][f'{peak_type}_{cast_type}_upstream']['pval']
            if pval < 0.001:
                ptext = "p<0.001 ***"
            elif pval < 0.01:
                ptext = "p<0.01 **"
            elif pval < 0.05:
                ptext = f"p={pval:.3f} *"
            else:
                ptext = f"p={pval:.3f}"
                
            # Add horizontal line and p-value between genotype upstream probabilities
            ax.plot([1, 3.5], [1.15, 1.15], 'k-', lw=1)
            ax.text(2.25, 1.17, ptext, ha='center', va='bottom', fontsize=10)
            
            # Add significance markers for comparison to chance (0.5)
            # For first genotype
            pval_g1 = genotype1_results['stats_results'][peak_type][cast_type]['ttest']['pval']
            if pval_g1 < 0.05:
                stars_g1 = "*" * sum([pval_g1 < p for p in [0.05, 0.01, 0.001]])
                ax.text(1, 0.45, stars_g1, ha='center', va='top', fontsize=14)
                
            # For second genotype
            pval_g2 = genotype2_results['stats_results'][peak_type][cast_type]['ttest']['pval']
            if pval_g2 < 0.05:
                stars_g2 = "*" * sum([pval_g2 < p for p in [0.05, 0.01, 0.001]])
                ax.text(3.5, 0.45, stars_g2, ha='center', va='top', fontsize=14)
            
            # Add a dashed line at 0.5 (chance level)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
            
            # Add significance markers for within-genotype comparison (upstream vs downstream)
            # For first genotype (compare positions 1 and 2)
            tstat_within_g1, pval_within_g1 = stats.ttest_rel(g1_upstream, g1_downstream)
            if pval_within_g1 < 0.001:
                ptext_g1 = "p<0.001 ***"
            elif pval_within_g1 < 0.01:
                ptext_g1 = "p<0.01 **"
            elif pval_within_g1 < 0.05:
                ptext_g1 = f"p={pval_within_g1:.3f} *"
            else:
                ptext_g1 = f"p={pval_within_g1:.3f}"
                
            # Add horizontal line and p-value between upstream and downstream for first genotype
            line_height_g1 = 1.08
            ax.plot([1, 2], [line_height_g1, line_height_g1], 'k-', lw=1)
            ax.text(1.5, line_height_g1 + 0.02, ptext_g1, ha='center', va='bottom', fontsize=9)

            # For second genotype (compare positions 3.5 and 4.5)
            tstat_within_g2, pval_within_g2 = stats.ttest_rel(g2_upstream, g2_downstream)
            if pval_within_g2 < 0.001:
                ptext_g2 = "p<0.001 ***"
            elif pval_within_g2 < 0.01:
                ptext_g2 = "p<0.01 **"
            elif pval_within_g2 < 0.05:
                ptext_g2 = f"p={pval_within_g2:.3f} *"
            else:
                ptext_g2 = f"p={pval_within_g2:.3f}"
                
            # Add horizontal line and p-value between upstream and downstream for second genotype
            line_height_g2 = 1.08
            ax.plot([3.5, 4.5], [line_height_g2, line_height_g2], 'k-', lw=1)
            ax.text(4.0, line_height_g2 + 0.02, ptext_g2, ha='center', va='bottom', fontsize=9)

            # Update stat_results_txt with within-genotype comparisons
            stat_results_txt.append(f"  Within-genotype comparisons (upstream vs downstream):")
            stat_results_txt.append(f"    {labels[0]}: t={tstat_within_g1:.4f}, p={pval_within_g1:.4f} {'(significant)' if pval_within_g1 < 0.05 else '(not significant)'}")
            stat_results_txt.append(f"    {labels[1]}: t={tstat_within_g2:.4f}, p={pval_within_g2:.4f} {'(significant)' if pval_within_g2 < 0.05 else '(not significant)'}")
            
            # Configure the plot
            ax.set_ylim(0, 1.25)  # Higher upper limit to accommodate annotations
            ax.set_ylabel('Probability of Cast Direction')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_box, rotation=45, ha='right')
            ax.set_title(f'{cast_type.capitalize()} Casts')
        
        # Add overall title for this peak type
        peak_title = {
            'all': 'All Peaks in Cast', 
            'first': 'First Peak in Cast', 
            'last': 'Last Peak in Cast'
        }
        fig.suptitle(f'Comparison of Cast Directions: {peak_title[peak_type]} (±{angle_width}°)', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.20)  # More space at bottom for labels
        
        # Save figure if basepath is provided
        if basepath is not None:
            figname = f"{basepath}/cast_direction_{peak_type}_peaks_{label1_safe}_vs_{label2_safe}_{timestamp}.png"
            plt.savefig(figname, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {figname}")
            
            # Also save as SVG for publication-quality figure
            svgname = f"{basepath}/cast_direction_{peak_type}_peaks_{label1_safe}_vs_{label2_safe}_{timestamp}.svg"
            plt.savefig(svgname, format='svg', bbox_inches='tight')
            print(f"Saved SVG to: {svgname}")
        
        plt.show()
    
    # Save statistical results to text file
    if basepath is not None:
        txtname = f"{basepath}/cast_direction_stats_{label1_safe}_vs_{label2_safe}_{timestamp}.txt"
        with open(txtname, 'w') as f:
            f.write('\n'.join(stat_results_txt))
        print(f"Saved statistics to: {txtname}")
    
    # Return combined results
    return {
        'cast_direction_comparison': {
            'genotype1': genotype1_results,
            'genotype2': genotype2_results
        },
        'statistical_tests': comparison_results['statistical_tests'],
        'angle_width': angle_width,
        'smooth_window': smooth_window,
        'jump_threshold': jump_threshold
    }

def compare_cast_amplitudes_by_direction_peaks(genotype1_data, genotype2_data, labels=None, angle_width=5, 
                                              smooth_window=5, jump_threshold=15, basepath=None):
    """
    Compare bend amplitudes between upstream and downstream casts for two genotypes when larvae are perpendicular to flow.
    Analyzes all peaks, first peaks, and last peaks separately to identify differences at different phases of casts.
    
    Parameters:
    -----------
    genotype1_data : dict
        Tracking data dictionary for first genotype
    genotype2_data : dict
        Tracking data dictionary for second genotype
    labels : tuple, optional
        Tuple of (label1, label2) for the genotypes
    angle_width : int
        Width of perpendicular orientation sector in degrees
    smooth_window : int
        Window size for smoothing (default=5)
    jump_threshold : float
        Threshold for detecting orientation jumps in degrees/frame (default=15)
    basepath : str, optional
        Base path for saving output files
        
    Returns:
    --------
    dict: Contains comparison statistics and test results for different peak types
    """
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from datetime import datetime
    from matplotlib.patches import Patch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Set default labels if not provided
    if labels is None:
        labels = ('Genotype 1', 'Genotype 2')
    
    # Create safe filenames from labels
    label1_safe = ''.join(c if c.isalnum() else '_' for c in labels[0])
    label2_safe = ''.join(c if c.isalnum() else '_' for c in labels[1])
    
    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Make sure basepath exists
    if basepath is not None:
        import os
        os.makedirs(basepath, exist_ok=True)
    
    # Define perpendicular angle ranges (both left and right sides)
    # With wind blowing towards negative x, perpendicular is ±90°
    left_perp_range = (-90 - angle_width, -90 + angle_width)
    right_perp_range = (90 - angle_width, 90 + angle_width)
    
    def is_perpendicular(angle):
        """Check if an angle is within the perpendicular ranges"""
        return ((left_perp_range[0] <= angle <= left_perp_range[1]) or 
                (right_perp_range[0] <= angle <= right_perp_range[1]))
    
    def analyze_peaks_in_genotype(genotype_data):
        """Analyze cast peaks and store bend amplitudes within a genotype"""
        # Define peak types
        peak_types = ['all', 'first', 'last']
        
        # Store bend amplitudes for each cast type, direction, and peak type
        cast_amplitudes = {}
        per_larva_amplitudes = {}
        total_counts = {}
        
        for peak_type in peak_types:
            cast_amplitudes[peak_type] = {
                'small': {'upstream': [], 'downstream': [], 'larva_ids': []},
                'large': {'upstream': [], 'downstream': [], 'larva_ids': []},
                'all': {'upstream': [], 'downstream': [], 'larva_ids': []}
            }
            
            per_larva_amplitudes[peak_type] = {
                'small': {'upstream': [], 'downstream': [], 'larva_ids': []},
                'large': {'upstream': [], 'downstream': [], 'larva_ids': []},
                'all': {'upstream': [], 'downstream': [], 'larva_ids': []}
            }
            
            total_counts[peak_type] = {
                'small': {'upstream': 0, 'downstream': 0},
                'large': {'upstream': 0, 'downstream': 0},
                'all': {'upstream': 0, 'downstream': 0}
            }
        
        # Determine which data structure we're working with
        if 'data' in genotype_data:
            data_to_process = genotype_data['data']
            n_larvae = genotype_data['metadata']['total_larvae']
        else:
            data_to_process = genotype_data
            n_larvae = len(data_to_process)
        
        # Process each larva
        larvae_processed = 0
        for larva_id, larva_data in data_to_process.items():
            # Extract nested data if needed
            if 'data' in larva_data:
                larva_data = larva_data['data']
                
            # Extract required data
            try:
                # Extract basic time series data
                time = np.array(larva_data['t']).flatten()
                states = np.array(larva_data['global_state_large_state']).flatten()
                
                # Check for small_large_state or fall back to large_state
                has_small_large_state = 'global_state_small_large_state' in larva_data
                
                if has_small_large_state:
                    # Extract both small and large cast states
                    small_large_states = np.array(larva_data['global_state_small_large_state']).flatten()
                    large_cast_mask = small_large_states == 2.0  # Large casts = 2.0
                    small_cast_mask = small_large_states == 1.5  # Small casts = 1.5
                else:
                    # Fall back to just large state if small_large_state isn't available
                    large_cast_mask = states == 2  # Only large casts available
                    small_cast_mask = np.zeros_like(states, dtype=bool)  # No small casts
                
                any_cast_mask = large_cast_mask | small_cast_mask
                
                # Get orientation angles
                x_center = np.array(larva_data['x_center']).flatten()
                y_center = np.array(larva_data['y_center']).flatten()
                x_spine = np.array(larva_data['x_spine'])
                y_spine = np.array(larva_data['y_spine'])
                
                if x_spine.ndim > 1:
                    x_tail = x_spine[-1].flatten()
                    y_tail = y_spine[-1].flatten()
                else:
                    x_tail = x_spine
                    y_tail = y_spine
                
                # Calculate orientation vectors
                dx = x_center - x_tail
                dy = y_center - y_tail
                orientation_angles = np.degrees(np.arctan2(dy, -dx))  # -dx because 0° is negative x-axis
                
                # Get upper-lower bend angle
                angle_upper_lower = np.array(larva_data['angle_upper_lower_smooth_5']).flatten()
                angle_upper_lower_deg = np.degrees(angle_upper_lower)
                
                # Ensure all arrays have the same length
                min_length = min(len(time), len(orientation_angles), len(angle_upper_lower_deg), len(states))
                time = time[:min_length]
                orientation_angles = orientation_angles[:min_length]
                angle_upper_lower_deg = angle_upper_lower_deg[:min_length]
                states = states[:min_length]
                large_cast_mask = large_cast_mask[:min_length]
                small_cast_mask = small_cast_mask[:min_length]
                any_cast_mask = any_cast_mask[:min_length]
                
                # ==== FILTERING AND PROCESSING ====
                
                # 1. Orientation angle jump detection and correction
                orientation_raw = orientation_angles.copy()
                
                # Calculate derivative to detect jumps
                orientation_diff = np.abs(np.diff(orientation_angles))
                # Add zero at the beginning to match length
                orientation_diff = np.insert(orientation_diff, 0, 0)
                
                # Find jumps bigger than threshold
                jumps = orientation_diff > jump_threshold
                
                # Create masked array for plotting
                orientation_masked = np.ma.array(orientation_angles, mask=jumps)
                
                # Apply Gaussian smoothing to the filtered data
                # First interpolate masked values for smoothing
                orientation_interp = orientation_masked.filled(np.nan)
                mask = np.isnan(orientation_interp)
                
                # Only interpolate if we have valid data
                if np.sum(~mask) > 1:
                    indices = np.arange(len(orientation_interp))
                    valid_indices = indices[~mask]
                    valid_values = orientation_interp[~mask]
                    orientation_interp[mask] = np.interp(indices[mask], valid_indices, valid_values)
                
                # Apply smoothing
                orientation_smooth = gaussian_filter1d(orientation_interp, smooth_window/3.0)
                
                # 2. Smooth the bend angle
                bend_angle_smooth = gaussian_filter1d(angle_upper_lower_deg, smooth_window/3.0)
                
                # 3. Find continuous cast segments
                cast_segments = {
                    'large': [],
                    'small': []
                }
                
                # Find large cast segments
                in_segment = False
                start_idx = 0
                for i, is_cast in enumerate(large_cast_mask):
                    if is_cast and not in_segment:
                        # Start of new segment
                        in_segment = True
                        start_idx = i
                    elif not is_cast and in_segment:
                        # End of segment
                        in_segment = False
                        if i - start_idx >= 3:  # Require minimum length
                            cast_segments['large'].append((start_idx, i))
                
                # Handle case when still in segment at end of data
                if in_segment and len(large_cast_mask) - start_idx >= 3:
                    cast_segments['large'].append((start_idx, len(large_cast_mask)))
                
                # Find small cast segments
                in_segment = False
                start_idx = 0
                for i, is_cast in enumerate(small_cast_mask):
                    if is_cast and not in_segment:
                        # Start of new segment
                        in_segment = True
                        start_idx = i
                    elif not is_cast and in_segment:
                        # End of segment
                        in_segment = False
                        if i - start_idx >= 3:  # Require minimum length
                            cast_segments['small'].append((start_idx, i))
                
                # Handle case when still in segment at end of data
                if in_segment and len(small_cast_mask) - start_idx >= 3:
                    cast_segments['small'].append((start_idx, len(small_cast_mask)))
                
                # 4. Find peaks within each cast segment
                all_peaks = {
                    'large': [],
                    'small': []
                }
                
                first_peaks = {
                    'large': [],
                    'small': []
                }
                
                last_peaks = {
                    'large': [],
                    'small': []
                }
                
                # Process each cast type
                for cast_type in ['large', 'small']:
                    for start, end in cast_segments[cast_type]:
                        segment_angles = bend_angle_smooth[start:end]
                        
                        # Find peaks in absolute bend angles
                        abs_angles = np.abs(segment_angles)
                        
                        # Find positive and negative peaks
                        pos_peaks, _ = find_peaks(segment_angles, height=5, prominence=3, distance=3)
                        neg_peaks, _ = find_peaks(-segment_angles, height=5, prominence=3, distance=3)
                        
                        # Combine and sort peaks by position
                        segment_peaks = sorted(list(pos_peaks) + list(neg_peaks))
                        
                        if len(segment_peaks) > 0:
                            # Add all peaks to global frame reference
                            global_peaks = [start + idx for idx in segment_peaks]
                            all_peaks[cast_type].extend(global_peaks)
                            
                            # Add first peak
                            first_peaks[cast_type].append(start + segment_peaks[0])
                            
                            # Add last peak
                            last_peaks[cast_type].append(start + segment_peaks[-1])
                
                # 5. Create storage for larva's amplitudes by cast type, peak type, and direction
                larva_amplitudes = {}
                for peak_type in peak_types:
                    larva_amplitudes[peak_type] = {
                        'large': {'upstream': [], 'downstream': []},
                        'small': {'upstream': [], 'downstream': []},
                        'all': {'upstream': [], 'downstream': []}
                    }
                
                # 6. Process peaks to classify as upstream or downstream and store amplitude
                # Process all peaks
                for cast_type in ['large', 'small']:
                    for peak_idx in all_peaks[cast_type]:
                        if is_perpendicular(orientation_smooth[peak_idx]):
                            # Get orientation and bend angle at peak
                            orientation = orientation_smooth[peak_idx]
                            bend_angle = bend_angle_smooth[peak_idx]
                            
                            # Normalize orientation to -180 to 180 range
                            while orientation > 180:
                                orientation -= 360
                            while orientation <= -180:
                                orientation += 360
                            
                            # Classify as upstream or downstream based on orientation and bend direction
                            is_upstream = False
                            if (orientation > 0 and orientation < 180):  # Right side
                                if bend_angle < 0:  # Negative bend is upstream
                                    is_upstream = True
                                else:  # Positive bend is downstream
                                    is_upstream = False
                            else:  # Left side
                                if bend_angle > 0:  # Positive bend is upstream
                                    is_upstream = True
                                else:  # Negative bend is downstream
                                    is_upstream = False
                            
                            # Store the absolute amplitude of the bend
                            cast_direction = 'upstream' if is_upstream else 'downstream'
                            amplitude = abs(bend_angle)  # Absolute value of bend angle
                            
                            # Store in all peaks category
                            larva_amplitudes['all'][cast_type][cast_direction].append(amplitude)
                            
                            # Also store in all casts category
                            larva_amplitudes['all']['all'][cast_direction].append(amplitude)
                
                # Process first peaks
                for cast_type in ['large', 'small']:
                    for peak_idx in first_peaks[cast_type]:
                        if is_perpendicular(orientation_smooth[peak_idx]):
                            # Get orientation and bend angle at peak
                            orientation = orientation_smooth[peak_idx]
                            bend_angle = bend_angle_smooth[peak_idx]
                            
                            # Normalize orientation to -180 to 180 range
                            while orientation > 180:
                                orientation -= 360
                            while orientation <= -180:
                                orientation += 360
                            
                            # Classify as upstream or downstream
                            is_upstream = False
                            if (orientation > 0 and orientation < 180):  # Right side
                                if bend_angle < 0:  # Negative bend is upstream
                                    is_upstream = True
                                else:  # Positive bend is downstream
                                    is_upstream = False
                            else:  # Left side
                                if bend_angle > 0:  # Positive bend is upstream
                                    is_upstream = True
                                else:  # Negative bend is downstream
                                    is_upstream = False
                            
                            # Store the absolute amplitude of the bend
                            cast_direction = 'upstream' if is_upstream else 'downstream'
                            amplitude = abs(bend_angle)  # Absolute value of bend angle
                            
                            # Store in first peaks category
                            larva_amplitudes['first'][cast_type][cast_direction].append(amplitude)
                            
                            # Also store in all casts category
                            larva_amplitudes['first']['all'][cast_direction].append(amplitude)
                
                # Process last peaks
                for cast_type in ['large', 'small']:
                    for peak_idx in last_peaks[cast_type]:
                        if is_perpendicular(orientation_smooth[peak_idx]):
                            # Get orientation and bend angle at peak
                            orientation = orientation_smooth[peak_idx]
                            bend_angle = bend_angle_smooth[peak_idx]
                            
                            # Normalize orientation to -180 to 180 range
                            while orientation > 180:
                                orientation -= 360
                            while orientation <= -180:
                                orientation += 360
                            
                            # Classify as upstream or downstream
                            is_upstream = False
                            if (orientation > 0 and orientation < 180):  # Right side
                                if bend_angle < 0:  # Negative bend is upstream
                                    is_upstream = True
                                else:  # Positive bend is downstream
                                    is_upstream = False
                            else:  # Left side
                                if bend_angle > 0:  # Positive bend is upstream
                                    is_upstream = True
                                else:  # Negative bend is downstream
                                    is_upstream = False
                            
                            # Store the absolute amplitude of the bend
                            cast_direction = 'upstream' if is_upstream else 'downstream'
                            amplitude = abs(bend_angle)  # Absolute value of bend angle
                            
                            # Store in last peaks category
                            larva_amplitudes['last'][cast_type][cast_direction].append(amplitude)
                            
                            # Also store in all casts category
                            larva_amplitudes['last']['all'][cast_direction].append(amplitude)
                
                # 7. Only include larvae with sufficient cast data and update overall statistics
                has_casts = False
                
                for peak_type in peak_types:
                    for cast_type in ['large', 'small', 'all']:
                        upstream_amps = larva_amplitudes[peak_type][cast_type]['upstream']
                        downstream_amps = larva_amplitudes[peak_type][cast_type]['downstream']
                        
                        # Require at least 3 casts in each direction for reliable analysis
                        if len(upstream_amps) >= 3 and len(downstream_amps) >= 3:
                            has_casts = True
                            
                            # Add to total counts
                            total_counts[peak_type][cast_type]['upstream'] += len(upstream_amps)
                            total_counts[peak_type][cast_type]['downstream'] += len(downstream_amps)
                            
                            # Add to overall amplitude lists
                            cast_amplitudes[peak_type][cast_type]['upstream'].extend(upstream_amps)
                            cast_amplitudes[peak_type][cast_type]['downstream'].extend(downstream_amps)
                            
                            # For tracking which larva contributed to each amplitude
                            cast_amplitudes[peak_type][cast_type]['larva_ids'].extend([str(larva_id)] * (len(upstream_amps) + len(downstream_amps)))
                            
                            # Store per-larva amplitude lists for paired analysis
                            per_larva_amplitudes[peak_type][cast_type]['upstream'].append(upstream_amps)
                            per_larva_amplitudes[peak_type][cast_type]['downstream'].append(downstream_amps)
                            per_larva_amplitudes[peak_type][cast_type]['larva_ids'].append(str(larva_id))
                
                if has_casts:
                    larvae_processed += 1
                
            except Exception as e:
                print(f"Error processing larva {larva_id}: {e}")
        
        # Calculate summary statistics for bend amplitudes
        amplitude_stats = {}
        
        for peak_type in peak_types:
            amplitude_stats[peak_type] = {}
            
            for cast_type in ['small', 'large', 'all']:
                upstream_amps = np.array(cast_amplitudes[peak_type][cast_type]['upstream'])
                downstream_amps = np.array(cast_amplitudes[peak_type][cast_type]['downstream'])
                
                if len(upstream_amps) > 0 and len(downstream_amps) > 0:
                    # Calculate means and SEMs
                    amplitude_stats[peak_type][cast_type] = {
                        'upstream': {
                            'mean': np.mean(upstream_amps),
                            'sem': stats.sem(upstream_amps) if len(upstream_amps) > 1 else 0,
                            'n': len(upstream_amps)
                        },
                        'downstream': {
                            'mean': np.mean(downstream_amps),
                            'sem': stats.sem(downstream_amps) if len(downstream_amps) > 1 else 0,
                            'n': len(downstream_amps)
                        }
                    }
                    
                    # Perform statistical tests
                    # Independent t-test (unpaired) for all amplitudes
                    tstat, pval = stats.ttest_ind(upstream_amps, downstream_amps, equal_var=False)
                    amplitude_stats[peak_type][cast_type]['ttest_ind'] = {
                        'tstat': tstat,
                        'pval': pval
                    }
                    
                    # Calculate mean amplitudes per larva for paired tests
                    larvae_mean_upstream = []
                    larvae_mean_downstream = []
                    
                    for ups, downs in zip(per_larva_amplitudes[peak_type][cast_type]['upstream'], 
                                         per_larva_amplitudes[peak_type][cast_type]['downstream']):
                        if len(ups) > 0 and len(downs) > 0:
                            larvae_mean_upstream.append(np.mean(ups))
                            larvae_mean_downstream.append(np.mean(downs))
                    
                    # Paired t-test if we have per-larva data
                    if len(larvae_mean_upstream) > 1:
                        tstat_paired, pval_paired = stats.ttest_rel(larvae_mean_upstream, larvae_mean_downstream)
                        amplitude_stats[peak_type][cast_type]['ttest_paired'] = {
                            'tstat': tstat_paired,
                            'pval': pval_paired,
                            'n_larvae': len(larvae_mean_upstream)
                        }
                else:
                    amplitude_stats[peak_type][cast_type] = None
        
        return {
            'cast_amplitudes': cast_amplitudes,
            'per_larva_amplitudes': per_larva_amplitudes,
            'amplitude_stats': amplitude_stats,
            'total_counts': total_counts,
            'larvae_processed': larvae_processed,
            'total_larvae': n_larvae
        }
    
    # Analyze both genotypes separately
    print(f"Analyzing {labels[0]}...")
    genotype1_results = analyze_peaks_in_genotype(genotype1_data)
    
    print(f"Analyzing {labels[1]}...")
    genotype2_results = analyze_peaks_in_genotype(genotype2_data)
    
    # Store comparison results
    comparison_results = {
        'angle_width': angle_width,
        'labels': labels,
        'statistical_tests': {}
    }
    
    # Define peak types and cast types to analyze
    peak_types = ['all', 'first', 'last']
    cast_types = ['all', 'large', 'small']
    
    # Colors for each condition
    condition_colors = {
        'upstream': {
            labels[0]: '#1f77b4',  # Blue
            labels[1]: '#d62728'   # Red
        },
        'downstream': {
            labels[0]: '#9ecae1',  # Light blue
            labels[1]: '#ff9896'   # Light red
        }
    }
    
    # Variable to store statistical results for text file
    stat_results_txt = []
    stat_results_txt.append(f"Cast Amplitude Comparison: {labels[0]} vs {labels[1]}")
    stat_results_txt.append(f"Analysis timestamp: {timestamp}")
    stat_results_txt.append(f"Perpendicular angle width: ±{angle_width}°")
    stat_results_txt.append(f"Using improved peak detection with jump_threshold={jump_threshold} and smooth_window={smooth_window}")
    stat_results_txt.append("\n")
    
    # Process each peak type separately (create a figure for each peak type)
    for peak_type in peak_types:
        # Count valid cast types for this peak type (those with data in both genotypes)
        valid_cast_types = []
        for cast_type in cast_types:
            g1_stats = genotype1_results['amplitude_stats'].get(peak_type, {}).get(cast_type)
            g2_stats = genotype2_results['amplitude_stats'].get(peak_type, {}).get(cast_type)
            
            if g1_stats and g2_stats:
                valid_cast_types.append(cast_type)
        
        n_cast_types = len(valid_cast_types)
        
        if n_cast_types == 0:
            print(f"No valid cast data found for {peak_type} peaks in both genotypes. Skipping plot.")
            continue
        
        # Create figure for comparison
        fig, axes = plt.subplots(1, n_cast_types, figsize=(5 * n_cast_types, 6))
        
        # Handle the case of a single subplot
        if n_cast_types == 1:
            axes = [axes]
        
        # Process each cast type
        for ax_idx, cast_type in enumerate(valid_cast_types):
            ax = axes[ax_idx]
            
            # Extract amplitude data
            g1_upstream = np.array(genotype1_results['cast_amplitudes'][peak_type][cast_type]['upstream'])
            g1_downstream = np.array(genotype1_results['cast_amplitudes'][peak_type][cast_type]['downstream'])
            g2_upstream = np.array(genotype2_results['cast_amplitudes'][peak_type][cast_type]['upstream'])
            g2_downstream = np.array(genotype2_results['cast_amplitudes'][peak_type][cast_type]['downstream'])
            
            # Data for box plot
            data = [g1_upstream, g1_downstream, g2_upstream, g2_downstream]
            positions = [1, 2, 4, 5]
            labels_box = [f'{labels[0]}\nUpstream', f'{labels[0]}\nDownstream', 
                         f'{labels[1]}\nUpstream', f'{labels[1]}\nDownstream']
            
            # Create box plot
            bp = ax.boxplot(data, positions=positions, notch=True, patch_artist=True, 
                           widths=0.6, showfliers=False)
            
            # Color boxes by condition
            colors = [condition_colors['upstream'][labels[0]], 
                     condition_colors['downstream'][labels[0]],
                     condition_colors['upstream'][labels[1]], 
                     condition_colors['downstream'][labels[1]]]
            
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor=colors[i], alpha=0.6)
                bp['medians'][i].set(color='black', linewidth=1.5)
            
            # Add individual data points with jitter
            for i, (pos, d) in enumerate(zip(positions, data)):
                # Limit to max 100 points for clarity
                if len(d) > 100:
                    # Stratified sampling to ensure representation of the distribution
                    indices = np.linspace(0, len(d)-1, 100, dtype=int)
                    d_sampled = d[indices]
                else:
                    d_sampled = d
                    
                jitter = 0.1 * np.random.randn(len(d_sampled))
                ax.scatter(pos + jitter, d_sampled, s=25, alpha=0.5, color=colors[i], 
                          edgecolor='k', linewidth=0.5)
            
            # Add annotations: mean ± SEM
            for i, (pos, d) in enumerate(zip(positions, data)):
                mean = np.mean(d)
                sem = stats.sem(d) if len(d) > 1 else 0
                ax.text(pos, np.max(d) + 5, f"{mean:.1f}±{sem:.1f}°", ha='center', 
                       fontsize=9, bbox=dict(facecolor='white', alpha=0.7, pad=0.1))
            
            # Add count annotations
            ax.text(1, -5, f"n={len(g1_upstream)}", ha='center', va='top', fontsize=8)
            ax.text(2, -5, f"n={len(g1_downstream)}", ha='center', va='top', fontsize=8)
            ax.text(4, -5, f"n={len(g2_upstream)}", ha='center', va='top', fontsize=8)
            ax.text(5, -5, f"n={len(g2_downstream)}", ha='center', va='top', fontsize=8)
            
            # Add statistical tests for upstream vs downstream within genotypes
            # For first genotype
            g1_stats = genotype1_results['amplitude_stats'][peak_type][cast_type]
            g1_pval = g1_stats['ttest_ind']['pval']
            
            if g1_pval < 0.001:
                g1_ptext = "p<0.001 ***"
            elif g1_pval < 0.01:
                g1_ptext = "p<0.01 **"
            elif g1_pval < 0.05:
                g1_ptext = f"p={g1_pval:.3f} *"
            else:
                g1_ptext = f"p={g1_pval:.3f}"
                
            # Add line and p-value for first genotype
            g1_maxheight = max(np.max(g1_upstream), np.max(g1_downstream)) + 10
            ax.plot([1, 2], [g1_maxheight, g1_maxheight], 'k-', lw=1)
            ax.text(1.5, g1_maxheight + 2, g1_ptext, ha='center', va='bottom', fontsize=9)
            
            # For second genotype
            g2_stats = genotype2_results['amplitude_stats'][peak_type][cast_type]
            g2_pval = g2_stats['ttest_ind']['pval']
            
            if g2_pval < 0.001:
                g2_ptext = "p<0.001 ***"
            elif g2_pval < 0.01:
                g2_ptext = "p<0.01 **"
            elif g2_pval < 0.05:
                g2_ptext = f"p={g2_pval:.3f} *"
            else:
                g2_ptext = f"p={g2_pval:.3f}"
                
            # Add line and p-value for second genotype
            g2_maxheight = max(np.max(g2_upstream), np.max(g2_downstream)) + 10
            ax.plot([4, 5], [g2_maxheight, g2_maxheight], 'k-', lw=1)
            ax.text(4.5, g2_maxheight + 2, g2_ptext, ha='center', va='bottom', fontsize=9)
            
            # Add comparison between genotypes for upstream and downstream
            # Upstream comparison (g1_upstream vs g2_upstream)
            tstat_up, pval_up = stats.ttest_ind(g1_upstream, g2_upstream, equal_var=False)
            comparison_results['statistical_tests'][f'{peak_type}_{cast_type}_upstream'] = {
                'tstat': tstat_up, 
                'pval': pval_up
            }
            
            if pval_up < 0.001:
                up_ptext = "p<0.001 ***"
            elif pval_up < 0.01:
                up_ptext = "p<0.01 **"
            elif pval_up < 0.05:
                up_ptext = f"p={pval_up:.3f} *"
            else:
                up_ptext = f"p={pval_up:.3f}"
                
            # Add line for upstream comparison
            up_maxheight = max(np.max(g1_upstream), np.max(g2_upstream)) + 15
            ax.plot([1, 4], [up_maxheight, up_maxheight], 'k-', lw=1)
            ax.text(2.5, up_maxheight + 2, up_ptext, ha='center', va='bottom', fontsize=9)
            
            # Downstream comparison (g1_downstream vs g2_downstream)
            tstat_down, pval_down = stats.ttest_ind(g1_downstream, g2_downstream, equal_var=False)
            comparison_results['statistical_tests'][f'{peak_type}_{cast_type}_downstream'] = {
                'tstat': tstat_down, 
                'pval': pval_down
            }
            
            if pval_down < 0.001:
                down_ptext = "p<0.001 ***"
            elif pval_down < 0.01:
                down_ptext = "p<0.01 **"
            elif pval_down < 0.05:
                down_ptext = f"p={pval_down:.3f} *"
            else:
                down_ptext = f"p={pval_down:.3f}"
                
            # Add line for downstream comparison
            down_maxheight = max(np.max(g1_downstream), np.max(g2_downstream)) + 20
            ax.plot([2, 5], [down_maxheight, down_maxheight], 'k-', lw=1, linestyle='--')
            ax.text(3.5, down_maxheight + 2, down_ptext, ha='center', va='bottom', fontsize=9)
            
            # Configure the plot
            max_y = max(
                np.max(g1_upstream), np.max(g1_downstream),
                np.max(g2_upstream), np.max(g2_downstream)
            ) + 30  # More room for annotations
            
            ax.set_ylim(0, max_y)
            ax.set_ylabel('Bend Amplitude (degrees)')
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_box, rotation=45, ha='right')
            ax.set_title(f'{cast_type.capitalize()} Casts')
            
            # Add statistical test results to text output
            stat_results_txt.append(f"\n{peak_type.capitalize()} Peaks - {cast_type.capitalize()} Casts - Bend Amplitude Comparison:")
            
            # Within-genotype comparisons (upstream vs downstream)
            stat_results_txt.append(f"  Within-genotype comparisons (upstream vs downstream):")
            stat_results_txt.append(f"    {labels[0]} upstream: {g1_stats['upstream']['mean']:.2f}° ± {g1_stats['upstream']['sem']:.2f}° (n={g1_stats['upstream']['n']})")
            stat_results_txt.append(f"    {labels[0]} downstream: {g1_stats['downstream']['mean']:.2f}° ± {g1_stats['downstream']['sem']:.2f}° (n={g1_stats['downstream']['n']})")
            stat_results_txt.append(f"    {labels[0]} comparison: t={g1_stats['ttest_ind']['tstat']:.3f}, p={g1_stats['ttest_ind']['pval']:.4f} {'(significant)' if g1_stats['ttest_ind']['pval'] < 0.05 else ''}")
            
            stat_results_txt.append(f"    {labels[1]} upstream: {g2_stats['upstream']['mean']:.2f}° ± {g2_stats['upstream']['sem']:.2f}° (n={g2_stats['upstream']['n']})")
            stat_results_txt.append(f"    {labels[1]} downstream: {g2_stats['downstream']['mean']:.2f}° ± {g2_stats['downstream']['sem']:.2f}° (n={g2_stats['downstream']['n']})")
            stat_results_txt.append(f"    {labels[1]} comparison: t={g2_stats['ttest_ind']['tstat']:.3f}, p={g2_stats['ttest_ind']['pval']:.4f} {'(significant)' if g2_stats['ttest_ind']['pval'] < 0.05 else ''}")
            
            # Between-genotype comparisons
            stat_results_txt.append(f"  Between-genotype comparisons:")
            stat_results_txt.append(f"    Upstream ({labels[0]} vs {labels[1]}): t={tstat_up:.3f}, p={pval_up:.4f} {'(significant)' if pval_up < 0.05 else ''}")
            stat_results_txt.append(f"    Downstream ({labels[0]} vs {labels[1]}): t={tstat_down:.3f}, p={pval_down:.4f} {'(significant)' if pval_down < 0.05 else ''}")
        
        # Define peak type titles
        peak_titles = {
            'all': 'All Peaks in Cast', 
            'first': 'First Peak in Cast', 
            'last': 'Last Peak in Cast'
        }
        
        # Add overall title for this peak type
        fig.suptitle(f'Comparison of Cast Bend Amplitudes: {peak_titles[peak_type]} (±{angle_width}°)', fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, bottom=0.20)  # More space at bottom for labels
        
        # Save figure if basepath is provided
        if basepath is not None:
            figname = f"{basepath}/cast_amplitude_{peak_type}_peaks_{label1_safe}_vs_{label2_safe}_{timestamp}.png"
            plt.savefig(figname, dpi=300, bbox_inches='tight')
            print(f"Saved figure to: {figname}")
            
            # Also save as SVG for publication-quality figure
            svgname = f"{basepath}/cast_amplitude_{peak_type}_peaks_{label1_safe}_vs_{label2_safe}_{timestamp}.svg"
            plt.savefig(svgname, format='svg', bbox_inches='tight')
            print(f"Saved SVG to: {svgname}")
        
        plt.show()
    
    # Save statistical results to text file
    if basepath is not None:
        txtname = f"{basepath}/cast_amplitude_stats_{label1_safe}_vs_{label2_safe}_{timestamp}.txt"
        with open(txtname, 'w') as f:
            f.write('\n'.join(stat_results_txt))
        print(f"Saved statistics to: {txtname}")
    
    # Return results
    return {
        'genotype1': genotype1_results,
        'genotype2': genotype2_results,
        'statistical_tests': comparison_results['statistical_tests'],
        'analysis_parameters': {
            'angle_width': angle_width,
            'smooth_window': smooth_window,
            'jump_threshold': jump_threshold
        }
    }