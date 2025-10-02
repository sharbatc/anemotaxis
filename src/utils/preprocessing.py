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

    original_count = len(extracted_data)

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
    
    removed_count = original_count - len(filtered_data)
    
    # Print filtering results
    print(f"Duration filtering results (threshold: {min_total_duration:.1f}s):")
    print(f"  - Removed {removed_count} larvae with <{min_total_duration:.1f}s total duration")
    print(f"  - {len(filtered_data)} larvae remaining")
    
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
    
    return filtered_data

def merge_short_stop_sequences(data, min_stop_duration_cast=2.0, min_stop_duration_run=3.0):
    """
    Merge sequences with short stops according to the following rules:
    1. Cast→short_stop→cast: Merge into a single cast
    2. Run→short_stop→run: Merge into a single run
    3. Run→short_stop→cast or cast→short_stop→run: Merge stop with the longer adjacent behavior
    
    Args:
        data: Either single experiment data (dict) or all experiments data (dict with 'data' key)
        min_stop_duration_cast: Minimum duration (in seconds) for stops between casts
        min_stop_duration_run: Minimum duration (in seconds) for stops between runs
        
    Returns:
        dict: Modified data with merged behaviors
    """
    import numpy as np
    import copy
    
    def calculate_segments(states, times):
        """Helper function to calculate segments from current state array."""
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
                        'start_time': times[start_idx],
                        'end_time': times[i - 1],
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
                'start_time': times[start_idx],
                'end_time': times[-1],
                'duration': times[-1] - times[start_idx]
            })
        
        return segments
    
    # Make a deep copy to avoid modifying original data
    merged_data = copy.deepcopy(data)
    
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
        if 'global_state_small_large_state' not in larva_data or 't' not in larva_data:
            continue
            
        # Get the time and state data
        states = np.array(larva_data['global_state_small_large_state']).flatten()
        times = np.array(larva_data['t']).flatten()
        
        # Ensure same length and sort if needed
        min_len = min(len(times), len(states))
        times, states = times[:min_len], states[:min_len]
        
        if not np.all(np.diff(times) >= 0):
            sorted_idx = np.argsort(times)
            times, states = times[sorted_idx], states[sorted_idx]
        
        # Create working copy of states that we'll modify
        new_states = np.copy(states)
        
        # Keep merging until no more merges are possible
        merged_something = True
        iteration_count = 0
        max_iterations = 100  # Safety limit to prevent infinite loops
        
        while merged_something and iteration_count < max_iterations:
            merged_something = False
            iteration_count += 1
            
            # Re-calculate segments based on current state
            segments = calculate_segments(new_states, times)
            
            # Look for patterns with short stops
            i = 0
            while i < len(segments) - 2:
                # Map segment states to behavior names
                state1 = segments[i]['state']
                state2 = segments[i+1]['state']
                state3 = segments[i+2]['state']
                
                # Get durations (all in seconds)
                duration1 = segments[i]['duration']
                duration2 = segments[i+1]['duration']  # Stop duration
                duration3 = segments[i+2]['duration']
                
                # Check different patterns
                is_cast1 = state1 in [2.0, 1.5]  # large_cast or small_cast
                is_run1 = state1 in [1.0, 0.5]   # large_run or small_run
                is_stop = state2 in [3.0, 2.5]   # large_stop or small_stop
                is_cast3 = state3 in [2.0, 1.5]  # large_cast or small_cast
                is_run3 = state3 in [1.0, 0.5]   # large_run or small_run
                
                # Pattern 1: Cast → short stop → cast
                if is_cast1 and is_stop and is_cast3 and duration2 < min_stop_duration_cast:
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
                    
                    merged_something = True
                    break  # Break to recalculate segments
                    
                # Pattern 2: Run → short stop → run
                elif is_run1 and is_stop and is_run3 and duration2 < min_stop_duration_run:
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
                    
                    merged_something = True
                    break  # Break to recalculate segments
                    
                # Pattern 3: Mixed (run → short stop → cast OR cast → short stop → run)
                elif ((is_run1 and is_stop and is_cast3) or (is_cast1 and is_stop and is_run3)) and duration2 < min_stop_duration_cast:
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
                    
                    merged_something = True
                    break  # Break to recalculate segments
                
                # Move to next segment if no merge happened
                i += 1
        
        # Update the modified states back to the larva data
        larva_data['global_state_small_large_state'] = new_states
    
    # Print summary statistics
    total_merged = merge_summary["cast_stop_cast_count"] + merge_summary["run_stop_run_count"] + merge_summary["mixed_count"]
    print(f"Merged {total_merged} sequences with short stops:")
    print(f"  - {merge_summary['cast_stop_cast_count']} cast-stop-cast sequences")
    print(f"  - {merge_summary['run_stop_run_count']} run-stop-run sequences") 
    print(f"  - {merge_summary['mixed_count']} mixed sequences (run-stop-cast or cast-stop-run)")
    print(f"Total duration saved: {merge_summary['total_duration_saved']:.2f} seconds")
    
    return merged_data

def analyze_behavior_durations(trx_data):
    """
    Analyze durations of different behavioral states across all larvae.
    
    Args:
        trx_data: Dictionary of larva data
        
    Returns:
        dict: Dictionary containing duration data for each behavior type
    """
    import numpy as np
    
    # Define state value mapping (same as in plot_global_behavior_matrix)
    state_info = {
        # Large behaviors (integer values)
        1.0: ('run', 'large'), 2.0: ('cast', 'large'), 3.0: ('stop', 'large'),
        4.0: ('hunch', 'large'), 5.0: ('backup', 'large'), 6.0: ('roll', 'large'),
        # Small behaviors (half-integer values)
        0.5: ('run', 'small'), 1.5: ('cast', 'small'), 2.5: ('stop', 'small'),
        3.5: ('hunch', 'small'), 4.5: ('backup', 'small'), 5.5: ('roll', 'small'),
    }
    
    # Process larvae data
    if isinstance(trx_data, dict) and 'data' in trx_data:
        larvae_data = trx_data['data']
    else:
        larvae_data = trx_data
        
    larva_ids = sorted(larvae_data.keys())
    
    # Initialize duration storage
    duration_data = {}
    for behavior in ['run', 'cast', 'stop', 'hunch', 'backup', 'roll']:
        duration_data[behavior] = {'large': [], 'small': []}
    
    # Process each larva
    for larva_id in larva_ids:
        if ('global_state_small_large_state' not in larvae_data[larva_id] or 
            't' not in larvae_data[larva_id] or 
            len(larvae_data[larva_id]['t']) == 0):
            continue
            
        times = np.array(larvae_data[larva_id]['t']).flatten()
        states = np.array(larvae_data[larva_id]['global_state_small_large_state']).flatten()
        
        # Ensure same length and sort if needed
        min_len = min(len(times), len(states))
        times, states = times[:min_len], states[:min_len]
        
        if not np.all(np.diff(times) >= 0):
            sorted_idx = np.argsort(times)
            times, states = times[sorted_idx], states[sorted_idx]
        
        # Get this larva's actual end time
        larva_t_max = times[-1] if len(times) > 0 else times[0]
        
        # Create end times (next time point or this larva's max time)
        end_times = np.append(times[1:], larva_t_max)
        
        # Round states to nearest 0.5
        rounded_states = np.round(states * 2) / 2
        
        # Group consecutive identical states
        state_changes = np.where(np.diff(rounded_states) != 0)[0] + 1
        segment_starts = np.concatenate(([0], state_changes))
        segment_ends = np.concatenate((state_changes, [len(times)]))
        
        # Calculate durations for each segment
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            if start_idx >= len(times):
                continue
                
            state_val = rounded_states[start_idx]
            segment_start_time = times[start_idx]
            
            if end_idx - 1 < len(end_times):
                segment_end_time = end_times[end_idx - 1]
            else:
                segment_end_time = larva_t_max
                
            segment_duration = segment_end_time - segment_start_time
            
            # Store duration if state is recognized
            if state_val in state_info:
                behavior, size = state_info[state_val]
                duration_data[behavior][size].append(segment_duration)
    
    return duration_data